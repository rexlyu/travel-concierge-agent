"""Context compaction, persistent session state, and async memory subsystem."""

from __future__ import annotations

from travel_concierge.memory.async_memory import (
    AsyncMemoryConsolidator,
    after_agent_memory_callback,
    get_default_memory_consolidator,
)
from travel_concierge.memory.compaction import (
    CachedContentConfigFallback,
    ConversationCompactor,
    EventsCompactionConfigFallback,
    build_adk_compaction_config,
    build_context_cache_config,
    create_genai_cached_content_config,
)
from travel_concierge.memory.session_store import (
    DEFAULT_SESSION_DB_PATH,
    LocalSqliteAdkSessionServiceAdapter,
    PersistentTravelSessionManager,
    compute_text_embedding,
    cosine_similarity,
)

__all__ = [
    "AsyncMemoryConsolidator",
    "CachedContentConfigFallback",
    "ConversationCompactor",
    "DEFAULT_SESSION_DB_PATH",
    "EventsCompactionConfigFallback",
    "LocalSqliteAdkSessionServiceAdapter",
    "PersistentTravelSessionManager",
    "after_agent_memory_callback",
    "build_adk_compaction_config",
    "build_context_cache_config",
    "compute_text_embedding",
    "cosine_similarity",
    "create_genai_cached_content_config",
    "get_default_memory_consolidator",
]
