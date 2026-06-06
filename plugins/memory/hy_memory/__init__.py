"""Hy-Memory memory plugin — MemoryProvider interface.

7-layer cognitive memory with System1/System2 dual processing, evolution
chains, and Kuzu graph for Hermes Agent.

Modes:
  lite  — embedding-only, zero LLM cost
  pro   — LLM fact extraction + reconciliation
  ultra — pro + System2 cognitive layer with Kuzu graph

Config via $HERMES_HOME/hy_memory.json or environment variables:
  HY_MEMORY_LLM_API_KEY      — LLM API key (pro/ultra)
  HY_MEMORY_EMBEDDER_API_KEY — Embedding API key (all modes)

Server: localhost:19527 by default, auto-started by the plugin.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker — after N consecutive failures, pause calls
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120

_DEFAULT_PORT = 19527


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from $HERMES_HOME/hy_memory.json, with env var fallbacks."""
    try:
        home = get_hermes_home()
    except Exception:
        home = Path.home() / ".hermes"

    config_path = home / "hy_memory.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Env var fallbacks for secrets
    if not config.get("llm", {}).get("api_key"):
        env_key = os.environ.get("HY_MEMORY_LLM_API_KEY", "")
        if env_key:
            config.setdefault("llm", {})["api_key"] = env_key

    if not config.get("embedder", {}).get("api_key"):
        env_key = os.environ.get("HY_MEMORY_EMBEDDER_API_KEY", "")
        if env_key:
            config.setdefault("embedder", {})["api_key"] = env_key

    return config


def _format_memories(memories: list[dict]) -> str:
    """Format search results into a readable context string."""
    if not memories:
        return ""
    lines = []
    for m in memories:
        content = m.get("content", "")
        layer = m.get("layer", "")
        score = m.get("score", 0)
        prefix = f"[{layer}] " if layer else ""
        lines.append(f"{prefix}{content} (score: {score:.2f})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class HyMemoryProvider(MemoryProvider):
    """Hermes memory provider backed by Hy-Memory."""

    def __init__(self):
        self._config: dict = {}
        self._client = None
        self._process = None
        self._user_id = "hermes-user"
        self._agent_id = "default"
        self._mode = "pro"
        # Prefetch
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
        # Sync
        self._sync_thread: threading.Thread | None = None
        # Circuit breaker
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "hy_memory"

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if Hy-Memory is configured with required credentials."""
        cfg = _load_config()
        if not cfg:
            return False
        # Need embedder key for all modes
        emb_key = (cfg.get("embedder", {}).get("api_key")
                   or os.environ.get("HY_MEMORY_EMBEDDER_API_KEY", ""))
        return bool(emb_key)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        """Start/connect to Hy-Memory server, store identity."""
        # Cron guard — skip for cron context
        agent_context = kwargs.get("agent_context", "")
        if agent_context in {"cron", "flush"}:
            logger.debug("Hy-Memory skipped: cron/flush context")
            return

        self._config = _load_config()
        self._mode = self._config.get("mode", "pro")

        # Identity mapping
        self._user_id = kwargs.get("user_id", "") or "hermes-user"
        self._agent_id = kwargs.get("agent_identity", "") or "default"

        # Auto-start server if configured
        if self._config.get("auto_start", True):
            from .process import HyMemoryProcess
            self._process = HyMemoryProcess(self._config)
            if not self._process.ensure_running():
                logger.error("[hy-memory] Server failed to start")
                return

        # Create client
        from .client import HyMemoryClient
        port = self._config.get("server_port", _DEFAULT_PORT)
        host = self._config.get("server_host", "127.0.0.1")
        self._client = HyMemoryClient(f"http://{host}:{port}")

        if self._client.is_reachable():
            logger.info("[hy-memory] Connected (mode=%s, user=%s)",
                        self._mode, self._user_id)
        else:
            logger.warning("[hy-memory] Server not reachable at %s:%d", host, port)

    def system_prompt_block(self) -> str:
        """Return static context about Hy-Memory status."""
        if not self._client or not self._client.is_reachable():
            return ""
        return (
            f"# Hy-Memory Active\n"
            f"Mode: {self._mode}. "
            f"7-layer memory with {'System1/System2 dual processing' if self._mode == 'ultra' else 'LLM extraction' if self._mode == 'pro' else 'embedding-only'}.\n"
            f"Memories persist across sessions. Use hy_memory_search to recall.\n"
        )

    # ------------------------------------------------------------------
    # Prefetch (background recall)
    # ------------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return cached prefetch result from background thread."""
        with self._prefetch_lock:
            return self._prefetch_result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Start background search for the next turn."""
        if not self._client:
            return

        # Circuit breaker
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            if time.time() < self._breaker_open_until:
                return
            self._consecutive_failures = 0

        def _do_prefetch():
            try:
                result = self._client.search(
                    query, user_ids=[self._user_id],
                    agent_ids=[self._agent_id], limit=5,
                )
                memories = result.get("memories", [])
                formatted = _format_memories(memories)
                with self._prefetch_lock:
                    self._prefetch_result = formatted
                self._consecutive_failures = 0
            except Exception as e:
                self._consecutive_failures += 1
                if self._consecutive_failures >= _BREAKER_THRESHOLD:
                    self._breaker_open_until = time.time() + _BREAKER_COOLDOWN_SECS
                    logger.warning("[hy-memory] Circuit breaker open: %s", e)
                else:
                    logger.debug("[hy-memory] Prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_do_prefetch, daemon=True)
        self._prefetch_thread.start()

    # ------------------------------------------------------------------
    # Sync (background write)
    # ------------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", messages: list | None = None) -> None:
        """Queue memory write in background."""
        if not self._client:
            return

        # Circuit breaker
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            if time.time() < self._breaker_open_until:
                return
            self._consecutive_failures = 0

        def _do_sync():
            try:
                data = messages or [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                self._client.add(
                    data, user_id=self._user_id,
                    agent_id=self._agent_id, session_id=session_id,
                )
                self._consecutive_failures = 0
            except Exception as e:
                self._consecutive_failures += 1
                if self._consecutive_failures >= _BREAKER_THRESHOLD:
                    self._breaker_open_until = time.time() + _BREAKER_COOLDOWN_SECS
                    logger.warning("[hy-memory] Circuit breaker open: %s", e)
                else:
                    logger.debug("[hy-memory] sync_turn failed: %s", e)

        self._sync_thread = threading.Thread(target=_do_sync, daemon=True)
        self._sync_thread.start()

    # ------------------------------------------------------------------
    # Session hooks
    # ------------------------------------------------------------------

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """Write final session snapshot on session end."""
        if not self._client or not messages:
            return
        try:
            self._client.add(
                messages, user_id=self._user_id,
                agent_id=self._agent_id, session_id="session-end",
            )
        except Exception as e:
            logger.debug("[hy-memory] on_session_end write failed: %s", e)

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """Extract insights before context compression."""
        # Let Hy-Memory's System1 handle this via sync_turn — no extra work needed
        return ""

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [{
            "name": "hy_memory_search",
            "description": (
                "Search persistent long-term memory across all past sessions. "
                "Returns memories about user preferences, facts, identity, "
                "and context that persist between conversations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — what to look for in memory",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        }]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        """Dispatch tool calls."""
        if tool_name == "hy_memory_search":
            return self._tool_search(args)
        return tool_error(f"Unknown Hy-Memory tool: {tool_name}")

    def _tool_search(self, args: dict) -> str:
        """Handle hy_memory_search tool call."""
        if not self._client:
            return json.dumps({"error": "Hy-Memory not connected"})

        query = args.get("query", "")
        limit = args.get("limit", 5)
        if not query:
            return json.dumps({"error": "query is required"})

        try:
            result = self._client.search(
                query, user_ids=[self._user_id],
                agent_ids=[self._agent_id], limit=limit,
            )
            memories = result.get("memories", [])
            if not memories:
                return json.dumps({"memories": [], "note": "No relevant memories found"})

            formatted = []
            for m in memories:
                formatted.append({
                    "content": m.get("content", ""),
                    "layer": m.get("layer", ""),
                    "score": round(m.get("score", 0), 3),
                })
            return json.dumps({"memories": formatted})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop the server if we started it."""
        if self._process:
            self._process.stop()
            self._process = None
        self._client = None

    # ------------------------------------------------------------------
    # Config schema (for hermes memory setup fallback)
    # ------------------------------------------------------------------

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {"key": "mode", "description": "Processing mode",
             "default": "pro", "choices": ["lite", "pro", "ultra"]},
            {"key": "server_port", "description": "Hy-Memory server port",
             "default": str(_DEFAULT_PORT)},
            {"key": "auto_start", "description": "Auto-start server with Hermes",
             "default": "true", "choices": ["true", "false"]},
            {"key": "llm_api_key", "description": "LLM API key for memory extraction",
             "secret": True, "env_var": "HY_MEMORY_LLM_API_KEY",
             "when": {"mode": ["pro", "ultra"]}},
            {"key": "llm_model", "description": "LLM model",
             "default": "gpt-4o-mini", "when": {"mode": ["pro", "ultra"]}},
            {"key": "llm_base_url", "description": "LLM API base URL",
             "default": "https://api.openai.com/v1",
             "when": {"mode": ["pro", "ultra"]}},
            {"key": "embedder_api_key", "description": "Embedding API key",
             "secret": True, "env_var": "HY_MEMORY_EMBEDDER_API_KEY"},
            {"key": "embedder_model", "description": "Embedding model",
             "default": "text-embedding-3-small"},
            {"key": "embedder_dims", "description": "Embedding dimensions",
             "default": "1536"},
            {"key": "vector_store", "description": "Vector store backend",
             "default": "chroma", "choices": ["chroma", "qdrant", "faiss"]},
        ]

    # ------------------------------------------------------------------
    # Config save
    # ------------------------------------------------------------------

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        """Write config to $HERMES_HOME/hy_memory.json."""
        config_path = Path(hermes_home) / "hy_memory.json"
        existing: dict[str, Any] = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        try:
            import stat
            config_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except (OSError, AttributeError):
            pass  # Windows

    # ------------------------------------------------------------------
    # post_setup — custom interactive wizard
    # ------------------------------------------------------------------

    def post_setup(self, hermes_home: str, config: dict) -> None:
        """Custom setup wizard for Hy-Memory."""
        import subprocess
        import shutil
        import sys

        from hermes_cli.config import save_config
        from hermes_cli.secret_prompt import masked_secret_prompt
        from hermes_cli.memory_setup import _curses_select

        print("\n  Configuring Hy-Memory memory:\n")

        existing_config = _load_config()

        # Step 1: Mode selection
        mode_values = ["lite", "pro", "ultra"]
        mode_items = [
            ("Lite", "Embedding only — zero LLM cost, fastest"),
            ("Pro", "LLM fact extraction — balanced quality and cost"),
            ("Ultra", "System1/System2 cognitive — highest quality, Kuzu graph"),
        ]
        existing_mode = existing_config.get("mode", "pro")
        mode_default = mode_values.index(existing_mode) if existing_mode in mode_values else 1
        mode_idx = _curses_select("  Select processing mode", mode_items, default=mode_default)
        mode = mode_values[mode_idx]

        provider_config: dict = dict(existing_config)
        provider_config["mode"] = mode
        env_writes: dict = {}

        # Step 2: Install dependencies
        print("\n  Checking dependencies...")
        uv_path = shutil.which("uv")
        deps = ["hy-memory", "kuzu", "chromadb"]
        if not uv_path:
            print(f"  ⚠ uv not found — install: curl -LsSf https://astral.sh/uv/install.sh | sh")
            print(f"  Then: uv pip install --python {sys.executable} {' '.join(deps)}")
        else:
            try:
                subprocess.run(
                    [uv_path, "pip", "install", "--python", sys.executable,
                     "--quiet", "--upgrade"] + deps,
                    check=True, timeout=180, capture_output=True,
                )
                print("  ✓ Dependencies up to date")
            except Exception as e:
                print(f"  ⚠ Install failed: {e}")
                print(f"  Run manually: uv pip install --python {sys.executable} {' '.join(deps)}")

        # Step 3: LLM config (pro/ultra only)
        if mode in ("pro", "ultra"):
            print("\n  LLM Configuration (for memory extraction):\n")
            llm_cfg = existing_config.get("llm", {})

            existing_key = llm_cfg.get("api_key", "") or os.environ.get("HY_MEMORY_LLM_API_KEY", "")
            if existing_key:
                masked = f"...{existing_key[-4:]}" if len(existing_key) > 4 else "set"
                sys.stdout.write(f"  LLM API key (current: {masked}, blank to keep): ")
                sys.stdout.flush()
                api_key = masked_secret_prompt("") if sys.stdin.isatty() else sys.stdin.readline().strip()
            else:
                sys.stdout.write("  LLM API key: ")
                sys.stdout.flush()
                api_key = masked_secret_prompt("") if sys.stdin.isatty() else sys.stdin.readline().strip()
            if api_key:
                env_writes["HY_MEMORY_LLM_API_KEY"] = api_key

            val = input(f"  LLM model [{llm_cfg.get('model', 'gpt-4o-mini')}]: ").strip()
            if val:
                provider_config.setdefault("llm", {})["model"] = val
            elif llm_cfg.get("model"):
                provider_config.setdefault("llm", {})["model"] = llm_cfg["model"]

            val = input(f"  LLM base URL [{llm_cfg.get('base_url', 'https://api.openai.com/v1')}]: ").strip()
            if val:
                provider_config.setdefault("llm", {})["base_url"] = val
            elif llm_cfg.get("base_url"):
                provider_config.setdefault("llm", {})["base_url"] = llm_cfg["base_url"]

        # Step 4: Embedding config
        print("\n  Embedding Configuration:\n")
        emb_cfg = existing_config.get("embedder", {})

        existing_emb_key = emb_cfg.get("api_key", "") or os.environ.get("HY_MEMORY_EMBEDDER_API_KEY", "")
        if existing_emb_key:
            masked = f"...{existing_emb_key[-4:]}" if len(existing_emb_key) > 4 else "set"
            same_as_llm = ""
            if mode in ("pro", "ultra") and env_writes.get("HY_MEMORY_LLM_API_KEY"):
                same_as_llm = " (blank to use same as LLM key)"
            sys.stdout.write(f"  Embedding API key (current: {masked}{same_as_llm}, blank to keep): ")
            sys.stdout.flush()
            emb_key = masked_secret_prompt("") if sys.stdin.isatty() else sys.stdin.readline().strip()
        else:
            if mode in ("pro", "ultra") and env_writes.get("HY_MEMORY_LLM_API_KEY"):
                sys.stdout.write("  Embedding API key (blank to use same as LLM key): ")
                sys.stdout.flush()
                emb_key = masked_secret_prompt("") if sys.stdin.isatty() else sys.stdin.readline().strip()
                if not emb_key:
                    emb_key = env_writes.get("HY_MEMORY_LLM_API_KEY", "")
            else:
                sys.stdout.write("  Embedding API key: ")
                sys.stdout.flush()
                emb_key = masked_secret_prompt("") if sys.stdin.isatty() else sys.stdin.readline().strip()
        if emb_key:
            env_writes["HY_MEMORY_EMBEDDER_API_KEY"] = emb_key

        val = input(f"  Embedding model [{emb_cfg.get('model', 'text-embedding-3-small')}]: ").strip()
        if val:
            provider_config.setdefault("embedder", {})["model"] = val
        elif emb_cfg.get("model"):
            provider_config.setdefault("embedder", {})["model"] = emb_cfg["model"]

        val = input(f"  Embedding dims [{emb_cfg.get('dims', 1536)}]: ").strip()
        if val:
            provider_config.setdefault("embedder", {})["dims"] = int(val)

        # Step 5: Vector store
        print("\n  Vector Store:\n")
        vs_values = ["chroma", "qdrant", "faiss"]
        vs_items = [
            ("Chroma", "Local, zero-config (recommended)"),
            ("Qdrant", "Remote or local Qdrant server"),
            ("FAISS", "Local, fast, Facebook AI Similarity Search"),
        ]
        existing_vs = existing_config.get("vector_store", {}).get("provider", "chroma")
        vs_default = vs_values.index(existing_vs) if existing_vs in vs_values else 0
        vs_idx = _curses_select("  Vector store backend", vs_items, default=vs_default)
        provider_config.setdefault("vector_store", {})["provider"] = vs_values[vs_idx]

        # Step 6: Server config
        val = input(f"  Server port [{existing_config.get('server_port', _DEFAULT_PORT)}]: ").strip()
        if val:
            provider_config["server_port"] = int(val)

        provider_config["auto_start"] = True

        # Step 7: Start server + health check
        print("\n  Starting Hy-Memory server...")
        # Temporarily set env vars so the server can find credentials
        for k, v in env_writes.items():
            os.environ[k] = v

        from .process import HyMemoryProcess
        proc = HyMemoryProcess(provider_config)
        if proc.start():
            from .client import HyMemoryClient
            client = HyMemoryClient(proc.base_url)
            try:
                status = client.status()
                checks = []
                for key in ("vdb", "embed", "llm"):
                    val = status.get(key, "?")
                    checks.append(f"{key}: {val}")
                print(f"  ✓ Server ready — {', '.join(checks)}")
            except Exception:
                print("  ✓ Server running (deep status check skipped)")
        else:
            print("  ⚠ Server failed to start — check logs and retry")
            print(f"    Logs: {get_hermes_home() / 'hy-memory-venv'}")

        # Step 8: Save & activate
        # Write secrets to .env
        if env_writes:
            env_path = Path(hermes_home) / ".env"
            _write_env_vars(env_path, env_writes)
            print(f"  ✓ API keys saved to .env")

        # Write provider config
        self.save_config(provider_config, hermes_home)

        # Activate in config.yaml
        config.setdefault("memory", {})["provider"] = "hy_memory"
        save_config(config)

        print(f"\n  ✓ Hy-Memory activated (mode: {mode})")
        print(f"  Start a new session to activate.\n")


def _write_env_vars(env_path: Path, env_writes: dict) -> None:
    """Append or update env vars in .env file."""
    env_path.parent.mkdir(parents=True, exist_ok=True)

    existing_lines = []
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()

    updated_keys = set()
    new_lines = []
    for line in existing_lines:
        key_match = line.split("=", 1)[0].strip() if "=" in line else ""
        if key_match in env_writes:
            new_lines.append(f"{key_match}={env_writes[key_match]}")
            updated_keys.add(key_match)
        else:
            new_lines.append(line)

    for key, val in env_writes.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={val}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    try:
        import stat
        env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass  # Windows
