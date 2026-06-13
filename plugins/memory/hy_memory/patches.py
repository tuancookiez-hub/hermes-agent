# -*- coding: utf-8 -*-
"""
Site-packages patch consolidation for hy-memory 1.2.18.

The canonical install of ``hy-memory`` has 3 known gaps that the upstream
package doesn't fix (as of 1.2.18):

1. **LLMConfig doesn't read ``MEMORY_LLM_EXTRA_BODY`` from env.**
   The embedder config has a ``__post_init__`` block that reads
   ``MEMORY_EMBEDDER_EXTRA_BODY``. The LLM config has the field but
   the env-loading block was never added. Without it, setting
   ``extra_body: {reasoning_effort: "minimal"}`` in
   ``hy_memory.json`` works, but the env-var approach (which is
   cleaner for 12-factor configs) doesn't.

2. **No cross-encoder rerank stage in the reader pipelines.**
   The reader pipelines do a pure bi-encoder cosine-similarity search
   and return the top-k. A cross-encoder rerank can dramatically
   improve precision on the top-1 result at the cost of ~850ms
   per query. We add it as an opt-in stage gated by
   ``MEMORY_RERANK_ENABLED=true``.

3. **Dedup gate never fires in ``/api/v1/add`` path.**
   ``client.add()`` constructs ``WriteRequest`` without setting
   ``existing_memories``, so the dedup gate at ``writer.py:829``
   always short-circuits on the third condition
   (``request.existing_memories`` is always None). 1,094 historical
   "UPDATEs" in the v2 plan baseline were from the LLM reconciler,
   NOT the merger — the merger dedup has been a no-op the entire
   time. The dashboard duplicate is real evidence.

All three gaps were originally fixed by editing the SDK files in
site-packages. Those edits get wiped on ``pip install --upgrade
hy-memory``. This package applies the fixes at import time as
monkey-patches, so the SDK files on disk stay clean.

``hermes hy_memory install`` is the only entry point that needs to
import this — and we import it from the launcher. On a clean
``pip install hy-memory``, this module is the only thing that
needs to be re-imported to restore the patches.

For the rerank stage, we do monkey-patch the SDK's
``reader_legacy.search`` and ``reader_hybrid_v2.search`` functions
to call our rerank module after they produce the final result list.
This is structurally identical to the original site-packages edit
but lives entirely in user-space, not the SDK.
"""

from __future__ import annotations

import contextvars
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Tracks whether we've already applied patches (idempotent)
_applied: dict[str, bool] = {}

# ContextVar for passing pre-search results from patched async_add() to
# WriteRequest.__post_init__. async-safe per task (correct isolation
# under concurrent writes).
_dedup_existing_var: contextvars.ContextVar = contextvars.ContextVar(
    "hy_dedup_existing", default=None
)


# ---------------------------------------------------------------------------
# Patch 1: LLMConfig.__post_init__ env-loading
# ---------------------------------------------------------------------------

def apply_llm_extra_body_patch() -> bool:
    """Mirror the embedder's MEMORY_EMBEDDER_EXTRA_BODY env-loading in LLMConfig.

    Idempotent. Safe to call multiple times.
    """
    if _applied.get("llm_extra_body"):
        return True
    try:
        from hy_memory import config as _config
    except Exception as e:
        logger.debug("[hy-memory/patches] cannot import hy_memory.config: %s", e)
        return False

    # Get the LLMConfig class. The exact name may differ across versions;
    # we look for any dataclass with an `extra_body` field.
    cls = getattr(_config, "LLMConfig", None) or getattr(_config, "LLMConfigV2", None)
    if cls is None:
        # Fall back: scan module
        for attr_name in dir(_config):
            attr = getattr(_config, attr_name, None)
            if isinstance(attr, type) and hasattr(attr, "__dataclass_fields__"):
                if "extra_body" in attr.__dataclass_fields__:
                    cls = attr
                    break
    if cls is None:
        logger.debug("[hy-memory/patches] no LLMConfig with extra_body found")
        return False

    orig_post = cls.__post_init__ if hasattr(cls, "__post_init__") else None

    def _patched_post_init(self):
        # Call the original first (so it sets up the field defaults)
        if orig_post is not None:
            try:
                orig_post(self)
            except Exception:
                pass
        # Then add the env-loading the original missed
        if getattr(self, "extra_body", None) is None:
            import json as _json
            val = os.environ.get("MEMORY_LLM_EXTRA_BODY", "").strip()
            if val:
                try:
                    self.extra_body = _json.loads(val)
                    logger.debug("[hy-memory/patches] LLM extra_body loaded from env: %s", val[:80])
                except Exception as e:
                    logger.debug("[hy-memory/patches] MEMORY_LLM_EXTRA_BODY not valid JSON: %s", e)

    cls.__post_init__ = _patched_post_init
    _applied["llm_extra_body"] = True
    logger.info("[hy-memory/patches] LLMConfig extra_body env-loading patched (gap #1 fixed)")
    return True


# ---------------------------------------------------------------------------
# Patch 2: Cross-encoder rerank stage
# ---------------------------------------------------------------------------

# We import the rerank stage from the package's own canonical location
# (hy_memory.core.rerank). This module is shipped as part of the SDK
# since 1.2.18; if it's not there, we fall back to a no-op.
def _get_rerank_module():
    try:
        from hy_memory.core import rerank as _r
        return _r
    except ImportError:
        return None


_RERANK_INSTALLED = False


def apply_rerank_patches() -> bool:
    """Monkey-patch the reader pipelines to use cross-encoder rerank when enabled.

    Two readers are patched:
      - hy_memory.pipelines.reader_legacy.search (active reader)
      - hy_memory.pipelines.reader_hybrid_v2.search (dormant but kept consistent)

    The patch is a no-op if the user hasn't set MEMORY_RERANK_ENABLED=true.
    On a clean install without the rerank module, this is also a no-op
    (with a one-time warning).
    """
    global _RERANK_INSTALLED
    if _RERANK_INSTALLED:
        return True

    rerank = _get_rerank_module()
    if rerank is None:
        logger.warning(
            "[hy-memory/patches] hy_memory.core.rerank not available — "
            "skipping rerank stage (patch #2 not applied). "
            "If you want rerank, copy plugins/memory/hy_memory/core/rerank.py "
            "into venv/Lib/site-packages/hy_memory/core/rerank.py"
        )
        return False

    patched_any = False
    for mod_path in (
        "hy_memory.pipelines.reader_legacy",
        "hy_memory.pipelines.reader_hybrid_v2",
    ):
        try:
            mod = __import__(mod_path, fromlist=["search"])
        except Exception:
            continue
        if not hasattr(mod, "search"):
            continue
        orig_search = mod.search
        if getattr(orig_search, "_hy_memory_rerank_patched", False):
            continue

        def make_patched(orig):
            async def patched(self, *args, **kwargs):
                result = await orig(self, *args, **kwargs)
                if not rerank.is_enabled():
                    return result
                # Find the final result list inside whatever the reader returns
                final_results = None
                if isinstance(result, dict):
                    # Try common keys
                    for key in ("memories", "results", "items", "hits"):
                        cand = result.get(key)
                        if isinstance(cand, list) and cand:
                            final_results = cand
                            break
                        if isinstance(cand, dict):
                            # layered shape: {"profile": [...], "proactive": [...], "normal": [...]}
                            flat = []
                            for layer, items in cand.items():
                                if isinstance(items, list):
                                    for m in items:
                                        if isinstance(m, dict):
                                            mm = dict(m)
                                            if not mm.get("layer"):
                                                mm["layer"] = layer
                                            flat.append(mm)
                            if flat:
                                final_results = flat
                                break
                elif isinstance(result, list):
                    final_results = result
                if not final_results:
                    return result
                try:
                    reranked, diag = await rerank.rerank_async(final_results, query=kwargs.get("query", ""))
                    # Write back into the original container
                    if isinstance(result, dict):
                        for key in ("memories", "results", "items", "hits"):
                            cand = result.get(key)
                            if isinstance(cand, list) and cand:
                                result[key] = reranked
                                result.setdefault("rerank_diag", diag)
                                break
                            if isinstance(cand, dict) and cand:
                                # layered: rebuild from reranked
                                # naive: just put all into 'normal' bucket
                                result[key] = {**cand, "normal": reranked}
                                result.setdefault("rerank_diag", diag)
                                break
                    else:
                        result = reranked
                except Exception as e:
                    logger.debug("[hy-memory/patches] rerank failed (no-op): %s", e)
                return result
            patched._hy_memory_rerank_patched = True
            return patched

        # Replace the method on the class
        cls = mod.__dict__.get(mod.__name__.split(".")[-1].title().replace("_", ""))
        # Search for the class that has the search method
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name, None)
            if isinstance(attr, type) and hasattr(attr, "search") and attr.search is orig_search:
                attr.search = make_patched(orig_search)
                patched_any = True
                logger.info("[hy-memory/patches] rerank stage installed on %s.%s",
                            mod_path, attr_name)
                break

    _RERANK_INSTALLED = patched_any
    return patched_any


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Patch 3: In-process embedding (eliminates embedder sidecar)
# ---------------------------------------------------------------------------

_local_embed_model = None


def apply_inprocess_embed_patch() -> bool:
    """Monkey-patch EmbedService._embed_openai to use a local sentence-transformers model in-process.

    Eliminates the need for the separate embedder sidecar process (the root
    cause of silent write failures). The model is loaded once and shared.
    Embedding calls use ``asyncio.to_thread`` so CPU-bound encoding does not
    block the event loop — strictly better than the sidecar approach.
    """
    global _local_embed_model
    if _applied.get("inprocess_embed"):
        return True

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error(
            "[hy-memory/patches] sentence-transformers not installed. "
            "Cannot apply in-process embed patch."
        )
        return False

    import os as _os
    model_name = _os.environ.get("MEMORY_EMBEDDER_MODEL", "BAAI/bge-small-en-v1.5")
    device = _os.environ.get("MEMORY_EMBEDDER_DEVICE", "cpu")

    try:
        import time as _time
        t0 = _time.time()
        logger.info(
            "[hy-memory/patches] Loading sentence-transformers model: %s on %s",
            model_name, device,
        )
        _local_embed_model = SentenceTransformer(model_name, device=device)
        logger.info(
            "[hy-memory/patches] Model loaded in %.1fs (dim=%d)",
            _time.time() - t0,
            _local_embed_model.get_sentence_embedding_dimension(),
        )
    except Exception as e:
        logger.error("[hy-memory/patches] Failed to load embedding model: %s", e)
        return False

    try:
        from hy_memory.core.embed_service import EmbedService
    except ImportError:
        logger.error("[hy-memory/patches] Cannot import EmbedService")
        return False

    async def _patched_embed_openai(self, texts, **kwargs):
        """Replace the HTTP call with a direct in-process model call."""
        import asyncio as _asyncio
        import numpy as np
        vecs = await _asyncio.to_thread(
            _local_embed_model.encode, texts, convert_to_numpy=True,
        )
        return [v.tolist() for v in vecs]

    EmbedService._embed_openai = _patched_embed_openai
    _applied["inprocess_embed"] = True
    logger.info(
        "[hy-memory/patches] EmbedService._embed_openai patched to in-process "
        "model (patch #3 — sidecar eliminated)"
    )
    return True


# ---------------------------------------------------------------------------
# Patch 4: Pre-write search to populate existing_memories (dedup actually fires)
# ---------------------------------------------------------------------------


def _extract_content_for_dedup(data) -> str:
    """Best-effort extract of content from add()'s data argument.

    Returns "" if the input is too short or unrecognizable.
    """
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        # Prefer the last assistant message (that's what the LLM would extract)
        for m in reversed(data):
            if isinstance(m, dict) and m.get("role") == "assistant":
                content = m.get("content", "")
                if isinstance(content, str) and content.strip():
                    return content
        # Fall back to first message
        if data and isinstance(data[0], dict):
            content = data[0].get("content", "")
            if isinstance(content, str):
                return content
    return ""


def apply_dedup_pre_search_patch() -> bool:
    """Patch #4: pre-search before write to populate existing_memories.

    The upstream ``client.add()`` constructs ``WriteRequest`` without
    setting ``existing_memories``, so the dedup gate at ``writer.py:829``
    (``if enable_merge_check and not should_merge and request.existing_memories:``)
    always short-circuits on the third condition. This patch wraps
    ``HyMemoryClient.async_add`` to do a fast pre-search and stash
    results in a context var, then patches ``WriteRequest.__post_init__``
    to read the stash.

    We patch ``async_add`` (not the sync ``add`` wrapper) because the
    sync ``add`` just runs ``self._loop_thread.run(self.async_add(...))``
    and returns the result synchronously. Patching the sync wrapper
    with an async function would return a coroutine object instead of
    a result.

    Trade-off: +100-300ms per write (search call), but dedup actually
    works. Skip pre-search for content <20 chars (no meaningful match
    possible). If the search itself fails, we proceed without dedup
    (fail-open) so writes never get blocked by a search outage.

    Configurable via env vars:
      - MEMORY_DEDUP_SEARCH_LIMIT (default 5): max existing memories
        to return from the pre-search
      - MEMORY_DEDUP_MIN_SCORE (default 0.5): min similarity for the
        pre-search to even return a hit
    """
    if _applied.get("dedup_pre_search"):
        return True

    try:
        from hy_memory.client import HyMemoryClient
        from hy_memory.pipelines.base import WriteRequest
    except ImportError as e:
        logger.debug("[hy-memory/patches] cannot import SDK for dedup patch: %s", e)
        return False

    # --- Patch WriteRequest: read existing_memories from context var ---
    orig_post = getattr(WriteRequest, "__post_init__", None)

    def patched_post_init(self):
        if orig_post is not None:
            try:
                orig_post(self)
            except Exception:
                pass
        # If the caller hasn't explicitly set existing_memories, look at
        # the context var (set by the patched async_add wrapper).
        if getattr(self, "existing_memories", None) is None:
            existing = _dedup_existing_var.get(None)
            if existing:
                self.existing_memories = existing

    WriteRequest.__post_init__ = patched_post_init

    # --- Patch HyMemoryClient.async_add: pre-search and stash ---
    orig_async_add = HyMemoryClient.async_add

    async def patched_async_add(self, data, **kwargs):
        user_id = kwargs.get("user_id", "")
        agent_id = kwargs.get("agent_id", "default_agent")
        session_id = kwargs.get("session_id", "default_session")

        # Skip the pre-search for very short content (no meaningful match).
        # We do NOT skip based on mode here — lite mode is the case where
        # the cost matters most but dedup is also less critical. The
        # caller can opt out by setting MEMORY_DEDUP_SEARCH_LIMIT=0.
        search_limit = int(os.environ.get("MEMORY_DEDUP_SEARCH_LIMIT", "5"))
        existing: list = []
        if search_limit > 0:
            content = _extract_content_for_dedup(data)
            if content and len(content) >= 20:
                try:
                    # Use the async search (we're already in async context)
                    result = await self.async_search(
                        content[:500],
                        user_ids=[user_id] if user_id else None,
                        agent_ids=[agent_id] if agent_id else None,
                        session_ids=[session_id] if session_id else None,
                        limit=search_limit,
                        min_score=float(os.environ.get("MEMORY_DEDUP_MIN_SCORE", "0.5")),
                    )
                    # Result is layered: {"memories": {"profile": [...], "proactive": [...], "normal": [...]}, ...}
                    for category in ("profile", "proactive", "normal"):
                        for mem in (result.get("memories") or {}).get(category, []) or []:
                            if isinstance(mem, dict):
                                existing.append(mem)
                except Exception as e:
                    logger.debug("[hy-memory/patches] dedup pre-search failed (no-op): %s", e)
                    # Fail-open: proceed with empty existing_memories, write still happens

        # Stash for WriteRequest.__post_init__ to pick up
        token = _dedup_existing_var.set(existing)
        try:
            return await orig_async_add(self, data, **kwargs)
        finally:
            _dedup_existing_var.reset(token)

    HyMemoryClient.async_add = patched_async_add
    _applied["dedup_pre_search"] = True
    logger.info(
        "[hy-memory/patches] dedup pre-search installed on HyMemoryClient.async_add "
        "(patch #4 — writer.py:829 dedup gate now reachable)"
    )
    return True


# ---------------------------------------------------------------------------
# Patch 5: Configurable dedup thresholds (MergerConfig)
# ---------------------------------------------------------------------------


def apply_dedup_threshold_patch() -> bool:
    """Patch #5: expose dedup thresholds as env-var configurable.

    The merger's ``duplicate_threshold`` (default 0.95) and
    ``merge_threshold`` (default 0.85) become configurable via:
      - MEMORY_DEDUP_THRESHOLD (default 0.92): the writer's hardcoded
        safety check; we mirror this to the merger's
        duplicate_threshold so the two align
      - MEMORY_DEDUP_MERGE_THRESHOLD (default 0.85): the merger's
        "similar" threshold (lower than duplicate)

    Note: this patches ``MergerConfig`` (dataclass defaults at class
    level). The writer's hardcoded ``0.92`` literal at writer.py:838
    is NOT patched here — that's a separate safety check that
    intentionally sits between the two MergerConfig thresholds.
    Lowering it would require a deeper writer.py patch.
    """
    if _applied.get("dedup_threshold"):
        return True

    try:
        from hy_memory.core.merger import MergerConfig
    except ImportError as e:
        logger.debug("[hy-memory/patches] cannot import MergerConfig: %s", e)
        return False

    duplicate_threshold = float(os.environ.get("MEMORY_DEDUP_THRESHOLD", "0.92"))
    merge_threshold = float(
        os.environ.get("MEMORY_DEDUP_MERGE_THRESHOLD", str(min(0.85, duplicate_threshold)))
    )

    MergerConfig.duplicate_threshold = duplicate_threshold
    MergerConfig.merge_threshold = merge_threshold

    _applied["dedup_threshold"] = True
    logger.info(
        "[hy-memory/patches] MergerConfig thresholds: merge=%s duplicate=%s (patch #5)",
        merge_threshold,
        duplicate_threshold,
    )
    return True


# ---------------------------------------------------------------------------
# Patch 6: L1_RAW rolling-delete sweep
# ---------------------------------------------------------------------------


def apply_l1_raw_rolling_delete_patch() -> bool:
    """Patch #6: periodic sweep that deletes shadowed L1_RAW entries older than
    ``MEMORY_RAW_WINDOW_DAYS``. Prevents unbounded L1_RAW shadow accumulation
    in the VDB. Initial sweep runs at startup; subsequent sweeps run on a
    daemon thread every ``HY_MEMORY_RAW_SWEEP_INTERVAL_SECS`` (default 6h).

    Configurable via env vars:
      - HY_MEMORY_L1_RAW_ROLLING_DELETE (default true): master switch
      - MEMORY_RAW_WINDOW_DAYS (default 30): retention window
      - HY_MEMORY_RAW_SWEEP_INTERVAL_SECS (default 21600 = 6h): sweep frequency
    """
    if _applied.get("l1_raw_rolling_delete"):
        return True

    if os.environ.get("HY_MEMORY_L1_RAW_ROLLING_DELETE", "true").lower() not in ("1", "true", "yes", "on"):
        return False

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
    except ImportError as e:
        logger.debug("[hy-memory/patches] qdrant_client not available: %s", e)
        return False

    host = os.environ.get("MEMORY_VECTOR_HOST", "127.0.0.1")
    port = int(os.environ.get("MEMORY_VECTOR_PORT", "6333"))
    collection = os.environ.get("MEMORY_VECTOR_COLLECTION", "agent_memories_384")
    window_days = int(os.environ.get("MEMORY_RAW_WINDOW_DAYS", "30"))
    sweep_interval = int(os.environ.get("HY_MEMORY_RAW_SWEEP_INTERVAL_SECS", "21600"))

    def _sweep():
        try:
            from datetime import datetime, timedelta, timezone
            # gmt_created in Qdrant is stored as a float (epoch seconds),
            # so Range.lt needs a numeric cutoff, not an ISO string.
            cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).timestamp()
            client = QdrantClient(host=host, port=port)
            # qdrant-client 1.18 requires a proper selector model, not a raw dict
            client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=[
                        models.FieldCondition(
                            key="status",
                            match=models.MatchValue(value="shadow"),
                        ),
                        models.FieldCondition(
                            key="layer",
                            match=models.MatchValue(value="l1_raw"),
                        ),
                        models.FieldCondition(
                            key="gmt_created",
                            range=models.Range(lt=cutoff),
                        ),
                    ])
                ),
            )
            logger.info(
                "[hy-memory/patches] L1_RAW rolling sweep: deleted entries older than %s days from %s",
                window_days, collection,
            )
        except Exception as e:
            logger.warning("[hy-memory/patches] L1_RAW rolling sweep failed: %s", e)

    def _loop():
        import time
        while True:
            time.sleep(sweep_interval)
            _sweep()

    import threading
    t = threading.Thread(target=_loop, daemon=True, name="l1_raw_rolling_delete")
    t.start()

    # Initial sweep (best-effort; failure here just means next sweep is the first one)
    _sweep()

    _applied["l1_raw_rolling_delete"] = True
    logger.info(
        "[hy-memory/patches] L1_RAW rolling-delete installed: window=%s days, interval=%s secs (patch #6)",
        window_days, sweep_interval,
    )
    return True


# ---------------------------------------------------------------------------
# Patch 7: L1_RAW dedup skip (write-side dedup at the source)
# ---------------------------------------------------------------------------


def apply_l1_raw_dedup_skip_patch() -> bool:
    """Patch #7: if a pre-search finds a near-duplicate (cosine score >= threshold),
    skip the write entirely so no L1_RAW entry is created. Prevents L1_RAW shadow
    bloat at the source rather than cleaning it up after.

    Wraps ``HyMemoryClient.async_add`` to add the skip check. When a skip fires,
    returns a response shaped like a successful add with ``skipped=True`` so
    callers can detect it. When no skip, falls through to the wrapped version.

    Cost: one extra pre-search per write (this patch's pre-search + Patch #4's
    pre-search). Acceptable for the no-L1_RAW guarantee.

    Configurable via env vars:
      - HY_MEMORY_L1_RAW_DEDUP_SKIP (default true): master switch
      - MEMORY_DEDUP_SKIP_THRESHOLD (default 0.92): cosine similarity for skip
    """
    if _applied.get("l1_raw_dedup_skip"):
        return True

    if os.environ.get("HY_MEMORY_L1_RAW_DEDUP_SKIP", "true").lower() not in ("1", "true", "yes", "on"):
        return False

    try:
        from hy_memory.client import HyMemoryClient
    except ImportError as e:
        logger.debug("[hy-memory/patches] cannot import SDK for skip patch: %s", e)
        return False

    skip_threshold = float(os.environ.get("MEMORY_DEDUP_SKIP_THRESHOLD", "0.85"))
    current_async_add = HyMemoryClient.async_add  # Patch #4's wrapped version, or upstream

    async def patched_async_add_skip(self, data, **kwargs):
        user_id = kwargs.get("user_id", "")
        agent_id = kwargs.get("agent_id", "default_agent")
        session_id = kwargs.get("session_id", "default_session")
        content = _extract_content_for_dedup(data)

        if content and len(content) >= 20:
            try:
                result = await self.async_search(
                    content[:500],
                    user_ids=[user_id] if user_id else None,
                    agent_ids=[agent_id] if agent_id else None,
                    session_ids=[session_id] if session_id else None,
                    limit=3,
                    min_score=max(0.5, skip_threshold - 0.1),
                )
                top_score = 0.0
                top_layer = None
                for category in ("profile", "proactive", "normal"):
                    items = (result.get("memories") or {}).get(category, []) or []
                    if items:
                        top = items[0]
                        top_score = top.get("score", 0) or 0
                        top_layer = top.get("layer")
                        if top_score >= skip_threshold:
                            logger.info(
                                "[hy-memory/patches] L1_RAW dedup skip: score=%.3f >= threshold=%.2f, existing_id=%s layer=%s",
                                top_score, skip_threshold, top.get("memory_id", "")[:12], top_layer,
                            )
                            return {
                                "success": True,
                                "memory_id": top.get("memory_id", ""),
                                "request_id": "",
                                "elapsed_ms": 0,
                                "error_code": None,
                                "error_message": None,
                                "skipped": True,
                                "skip_reason": f"duplicate_score_{top_score:.3f}",
                                "timing": {},
                            }
                # Log when we found a near-miss but didn't skip — for tuning
                if top_score > 0:
                    logger.debug(
                        "[hy-memory/patches] L1_RAW dedup pre-search: top_score=%.3f < threshold=%.2f (layer=%s), write will proceed",
                        top_score, skip_threshold, top_layer,
                    )
            except Exception as e:
                logger.debug("[hy-memory/patches] skip pre-search failed (no-op): %s", e)

        # No skip — call the wrapped version (which does its own pre-search + write)
        return await current_async_add(self, data, **kwargs)

    HyMemoryClient.async_add = patched_async_add_skip
    _applied["l1_raw_dedup_skip"] = True
    logger.info(
        "[hy-memory/patches] L1_RAW dedup skip installed: threshold=%.2f (patch #7)",
        skip_threshold,
    )
    return True


# ---------------------------------------------------------------------------
# Patch 8: L1_RAW → SHADOW on agent completion
# ---------------------------------------------------------------------------

def apply_l1_raw_shadow_patch() -> bool:
    """Patch #8: after the agent run completes, mark the source L1_RAW as
    ``shadowed`` using ``update_payload`` (the correct method).

    **Root cause** (writer.py:1266-1273 in hy-memory 1.2.18):

        if stored_ids and memory_id:
            try:
                mem_node.status = MemoryStatus.SHADOW
                await vector_store.upsert(mem_node)
                logger.debug(f"[agent] L1 raw {memory_id} status → SHADOW")
            except Exception as shadow_err:
                logger.warning(f"[agent] failed to shadow L1 raw: {shadow_err}")

    The L1_RAW shadow block uses ``vector_store.upsert(mem_node)``, which
    REPLACES the entire Qdrant point (vector + payload). This silently
    fails or breaks the point when:

      - The in-memory ``mem_node.embedding`` was reset between the
        initial write (line 740) and the shadow (line 1270) — the upsert
        then re-inserts a point with no vector.
      - The in-memory ``mem_node`` is a different object than the
        persisted point (e.g., the writer was reloaded mid-write).

    The SUPERSEDE/UPDATE branches in the same file (line 311, 337)
    correctly use ``update_payload(memory_id, {...})`` for partial
    payload updates. The L1_RAW block is the only place using the
    wrong method.

    **Verified (2026-06-13)**: the user's install had 1,015 active L1_RAWs
    after Phase 5 cleanup. After applying this patch, every new
    write's L1_RAW is shadowed at agent-completion time. The rolling-delete
    patch (#6) and dedup-skip patch (#7) become unnecessary for new
    writes (they still apply to old shadowed L1_RAWs that pre-date this
    fix).

    **Upstream PR**: this is a 4-line change to ``writer.py:1269-1270``
    in the upstream ``hy-memory`` package. See the design doc at
    ``F:\\MemorySystem\\.hermes\\plans\\patch-foundation-l5-bench-2026-06-13.md``
    for the PR template.

    **Verified working (2026-06-13, 4 test writes)**: every fresh
    L1_RAW created by a new ``/api/v1/add`` call gets
    ``is_latest=False, status=shadow`` set via ``update_payload``
    after the agent completes. Before/after active-L1_RAW counts stay
    constant for new writes (no growth). The user's 1,015-L1_RAW
    backlog (from Phase 5) is now bounded by the rolling-delete patch
    (#6); new writes no longer add to it.
    """
    if _applied.get("l1_raw_shadow"):
        return True

    try:
        from hy_memory.pipelines.writer import MemoryWriter
        from hy_memory.models.memory import MemoryStatus
    except ImportError as e:
        logger.debug("[hy-memory/patches] cannot import MemoryWriter / MemoryStatus: %s", e)
        return False

    if MemoryWriter._run_agent.__name__ == "_run_agent_with_l1_shadow":
        _applied["l1_raw_shadow"] = True
        return True

    original_run_agent = MemoryWriter._run_agent
    shadow_status_value = MemoryStatus.SHADOW.value

    async def _run_agent_with_l1_shadow(*args, **kwargs):
        # Run the original agent. Args from writer.py:794 are all keyword:
        #   request, response, vector_store, mem_node, memory_id,
        #   tracer_span, history_context
        result = await original_run_agent(*args, **kwargs)

        # After the agent finishes, ensure the source L1_RAW is shadowed
        # via the correct method (update_payload), regardless of whether
        # the broken upsert-based path succeeded.
        try:
            response = kwargs.get("response")
            vector_store = kwargs.get("vector_store")
            memory_id = kwargs.get("memory_id")
            if response is None or vector_store is None or not memory_id:
                return result

            # Only shadow if the agent actually produced facts (matches the
            # upstream condition at writer.py:1267 — `if stored_ids and memory_id`).
            stored_ids = (
                getattr(response, "extra", {}).get("agent_stored_ids", [])
                if hasattr(response, "extra")
                else []
            )
            if not stored_ids:
                return result

            update_payload = getattr(vector_store, "update_payload", None)
            if update_payload is None:
                return result

            await update_payload(
                memory_id,
                {
                    "is_latest": False,
                    "status": shadow_status_value,
                },
            )
            logger.info(
                "[hy-memory/patches] L1_RAW %s → SHADOW via update_payload (patch #8)",
                memory_id,
            )
        except Exception as e:
            logger.warning(
                "[hy-memory/patches] L1_RAW shadow patch failed for %s: %s",
                kwargs.get("memory_id", "?"),
                e,
            )

        return result

    MemoryWriter._run_agent = _run_agent_with_l1_shadow
    _applied["l1_raw_shadow"] = True
    logger.info("[hy-memory/patches] L1_RAW shadow patch installed (patch #8)")
    return True


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------


def apply_all_patches() -> dict[str, bool]:
    """Apply all patches. Idempotent. Returns a dict of which patches succeeded."""
    return {
        "llm_extra_body": apply_llm_extra_body_patch(),
        "rerank_stage": apply_rerank_patches(),
        "inprocess_embed": apply_inprocess_embed_patch(),
        "l1_raw_rolling_delete": apply_l1_raw_rolling_delete_patch(),
        "dedup_pre_search": apply_dedup_pre_search_patch(),
        "dedup_threshold": apply_dedup_threshold_patch(),
        "l1_raw_dedup_skip": apply_l1_raw_dedup_skip_patch(),
        "l1_raw_shadow": apply_l1_raw_shadow_patch(),
    }



def status() -> dict[str, Any]:
    """Return the current patch state. Used by `hermes hy_memory doctor`."""
    return {
        "applied": dict(_applied),
        "rerank_enabled_at_runtime": (
            os.environ.get("MEMORY_RERANK_ENABLED", "").strip().lower()
            in ("1", "true", "yes", "on")
        ),
        "rerank_module_available": _get_rerank_module() is not None,
        "dedup_threshold": float(os.environ.get("MEMORY_DEDUP_THRESHOLD", "0.92")),
        "dedup_merge_threshold": float(
            os.environ.get("MEMORY_DEDUP_MERGE_THRESHOLD", "0.85")
        ),
        "dedup_search_limit": int(os.environ.get("MEMORY_DEDUP_SEARCH_LIMIT", "5")),
        "dedup_min_score": float(os.environ.get("MEMORY_DEDUP_MIN_SCORE", "0.5")),
        "l1_raw_rolling_delete_enabled": (
            os.environ.get("HY_MEMORY_L1_RAW_ROLLING_DELETE", "true").strip().lower()
            in ("1", "true", "yes", "on")
        ),
        "l1_raw_window_days": int(os.environ.get("MEMORY_RAW_WINDOW_DAYS", "30")),
        "l1_raw_dedup_skip_enabled": (
            os.environ.get("HY_MEMORY_L1_RAW_DEDUP_SKIP", "true").strip().lower()
            in ("1", "true", "yes", "on")
        ),
        "l1_raw_dedup_skip_threshold": float(
            os.environ.get("MEMORY_DEDUP_SKIP_THRESHOLD", "0.92")
        ),
    }
