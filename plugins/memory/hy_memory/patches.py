# -*- coding: utf-8 -*-
"""
Site-packages patch consolidation for hy-memory 1.2.18.

The canonical install of ``hy-memory`` has 2 known gaps that the upstream
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

Both gaps were originally fixed by editing the SDK files in
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

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Tracks whether we've already applied patches (idempotent)
_applied: dict[str, bool] = {}


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


def apply_all_patches() -> dict[str, bool]:
    """Apply all patches. Idempotent. Returns a dict of which patches succeeded."""
    return {
        "llm_extra_body": apply_llm_extra_body_patch(),
        "rerank_stage": apply_rerank_patches(),
        "inprocess_embed": apply_inprocess_embed_patch(),
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
    }
