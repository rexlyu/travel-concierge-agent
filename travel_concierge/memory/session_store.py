"""Persistent travel session state, SQLite + vector memory store, and cloud adapters.

Implements Rubric 2.3 (Persistent Session State):
- Durable SQLite persistence across turns and process restarts (``sessions``,
  ``conversation_turns``, and ``traveler_memories`` tables).
- Deterministic hashed n-gram + domain-concept embeddings for fast local
  cosine-similarity semantic memory retrieval.
- Optional Google Cloud Firestore session synchronization and Vertex AI Search
  (Discovery Engine) grounding/memory retrieval adapter.
- Native Google ADK ``SessionService`` factory (``VertexAiSessionService``,
  ``DatabaseSessionService``, ``InMemorySessionService``) with offline adapter.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Iterator

logger = logging.getLogger(__name__)

DEFAULT_SESSION_DB_PATH = "/tmp/travel_concierge_sessions.sqlite3"

try:
    from google.cloud import firestore as _firestore_module

    _FIRESTORE_CLIENT_CLS = getattr(_firestore_module, "Client", None)
except ImportError:
    _firestore_module = None
    _FIRESTORE_CLIENT_CLS = None

_DISCOVERY_SEARCH_CLIENT_CLS = None
_DISCOVERY_SEARCH_REQUEST_CLS = None
for _mod_name in ("google.cloud.discoveryengine_v1", "google.cloud.discoveryengine"):
    try:
        _de_mod = __import__(_mod_name, fromlist=["SearchServiceClient", "SearchRequest"])
        _client_cls = getattr(_de_mod, "SearchServiceClient", None)
        if _client_cls is not None:
            _DISCOVERY_SEARCH_CLIENT_CLS = _client_cls
            _DISCOVERY_SEARCH_REQUEST_CLS = getattr(_de_mod, "SearchRequest", None)
            break
    except ImportError:
        continue

try:
    from google.adk import sessions as _adk_sessions

    _ADK_SESSIONS_AVAILABLE = True
except ImportError:
    _adk_sessions = None
    _ADK_SESSIONS_AVAILABLE = False


# Deterministic fallback PII patterns to guarantee memory hygiene even when
# called standalone before observability initialization.
_FALLBACK_PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|AIza[0-9A-Za-z_-]{20,})\b"), "[REDACTED_SECRET]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{12,}\b", re.IGNORECASE), "[REDACTED_SECRET]"),
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        re.compile(r"\b(?:\d{4}[-\s]){3}\d{4}\b|\b\d{15,16}\b"),
        "[REDACTED_CARD]",
    ),
    (
        re.compile(r"\b(?!000|666|9\d\d)\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED_SSN]",
    ),
    (
        re.compile(
            r"(?<!\d)(?:\+\d{1,3}[-\s.]?)?(?:\(\d{3}\)|\b\d{3})[-\s.]\d{3}[-\s.]\d{4}\b"
        ),
        "[REDACTED_PHONE]",
    ),
    (
        re.compile(
            r"(?i)(\bpassport(?:\s*(?:number|no\.?|id|#))?\s*[:#-]?\s*)([A-Z0-9]{6,12})\b"
        ),
        r"\1[REDACTED_PASSPORT]",
    ),
)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)

_CONCEPT_ANCHORS: tuple[tuple[frozenset[str], tuple[str, ...]], ...] = (
    (
        frozenset(
            {
                "flight",
                "flights",
                "airline",
                "airlines",
                "aisle",
                "window",
                "seat",
                "seating",
                "seats",
                "cabin",
                "business",
                "economy",
                "first",
                "nonstop",
                "direct",
                "layover",
                "legroom",
                "ana",
                "jal",
                "united",
                "delta",
                "emirates",
                "lufthansa",
            }
        ),
        ("_concept_flight", "_concept_aviation_pref"),
    ),
    (
        frozenset(
            {
                "hotel",
                "hotels",
                "lodging",
                "accommodation",
                "accommodations",
                "stay",
                "ryokan",
                "boutique",
                "resort",
                "room",
                "suite",
                "king",
                "queen",
                "nightly",
                "breakfast",
                "transit",
                "quiet",
            }
        ),
        ("_concept_lodging", "_concept_stay_pref"),
    ),
    (
        frozenset(
            {
                "budget",
                "cost",
                "price",
                "usd",
                "dollar",
                "dollars",
                "spend",
                "spending",
                "limit",
                "cap",
                "max",
                "maximum",
                "under",
                "affordable",
                "luxury",
                "moderate",
            }
        ),
        ("_concept_budget", "_concept_financial"),
    ),
    (
        frozenset(
            {
                "diet",
                "dietary",
                "food",
                "meal",
                "meals",
                "vegetarian",
                "vegan",
                "gluten",
                "halal",
                "kosher",
                "allergy",
                "allergies",
                "allergic",
                "nut",
                "peanut",
                "dairy",
                "lactose",
                "pescatarian",
            }
        ),
        ("_concept_dietary", "_concept_meal_pref"),
    ),
    (
        frozenset(
            {
                "destination",
                "destinations",
                "city",
                "cities",
                "country",
                "trip",
                "visit",
                "visiting",
                "tokyo",
                "kyoto",
                "osaka",
                "paris",
                "london",
                "rome",
                "barcelona",
                "lisbon",
                "singapore",
                "seoul",
                "bangkok",
            }
        ),
        ("_concept_destination", "_concept_geography"),
    ),
)


def _utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _scrub_text_safely(text: str, redactor: Any = None) -> str:
    """Scrub PII using configured observability redactor plus local regex guard."""
    if not text:
        return text

    active_redactor = redactor
    if active_redactor is None:
        try:
            from travel_concierge.observability.pii_redactor import get_default_redactor

            active_redactor = get_default_redactor()
        except ImportError:
            active_redactor = None

    cleaned = text
    if active_redactor is not None and hasattr(active_redactor, "redact_text"):
        cleaned = str(active_redactor.redact_text(cleaned))

    for pattern, replacement in _FALLBACK_PII_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def _scrub_payload_safely(payload: Any, redactor: Any = None) -> Any:
    """Recursively scrub PII in arbitrary JSON-like metadata structures."""
    if payload is None:
        return None
    if isinstance(payload, str):
        return _scrub_text_safely(payload, redactor=redactor)
    if isinstance(payload, dict):
        sensitive_keys = {
            "password",
            "secret",
            "api_key",
            "token",
            "card_number",
            "credit_card",
            "ssn",
            "passport_number",
        }
        out: dict[str, Any] = {}
        for k, v in payload.items():
            if str(k).lower() in sensitive_keys and isinstance(v, str) and v:
                out[k] = "[REDACTED_SECRET]"
            else:
                out[k] = _scrub_payload_safely(v, redactor=redactor)
        return out
    if isinstance(payload, list):
        return [_scrub_payload_safely(item, redactor=redactor) for item in payload]
    if isinstance(payload, tuple):
        return [_scrub_payload_safely(item, redactor=redactor) for item in payload]
    return payload


def _simple_stem(token: str) -> str:
    """Lightweight deterministic English suffix normalization for retrieval."""
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("es") and not token.endswith("ses"):
        return token[:-1]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def compute_text_embedding(
    text: str,
    tags: list[str] | None = None,
    dim: int = 128,
) -> list[float]:
    """Compute a deterministic L2-normalized hashed n-gram & concept embedding.

    Combines word unigrams, stems, word bigrams, character 4-grams, and travel
    domain concept anchors projected via signed Weinberger feature hashing.

    Args:
        text: Input memory statement or search query.
        tags: Optional categorical memory tags to boost semantic alignment.
        dim: Dimensionality of the dense output vector (default 128).

    Returns:
        Unit-normalized ``list[float]`` of length ``dim``.
    """
    if dim < 8:
        raise ValueError(f"embedding dim must be >= 8, got {dim}")

    vec = [0.0] * dim
    normalized = re.sub(r"[^a-z0-9$]+", " ", text.lower()).strip()
    raw_tokens = [tok for tok in normalized.split() if tok]
    if tags:
        for tag in tags:
            tag_tokens = re.sub(r"[^a-z0-9]+", " ", str(tag).lower()).split()
            raw_tokens.extend(tag_tokens)

    if not raw_tokens:
        return vec

    weighted_features: list[tuple[str, float]] = []
    content_tokens: list[str] = []

    for tok in raw_tokens:
        weight = 0.25 if tok in _STOPWORDS else 1.0
        weighted_features.append((f"w:{tok}", weight))
        stem = _simple_stem(tok)
        if stem != tok:
            weighted_features.append((f"s:{stem}", weight * 0.9))
        if tok not in _STOPWORDS:
            content_tokens.append(stem)
            if len(tok) >= 4:
                padded = f"^{tok}$"
                for i in range(len(padded) - 3):
                    weighted_features.append((f"c4:{padded[i : i + 4]}", 0.35))

    # Word bigrams on salient content tokens.
    for idx in range(len(content_tokens) - 1):
        bigram = f"bg:{content_tokens[idx]}_{content_tokens[idx + 1]}"
        weighted_features.append((bigram, 1.15))

    # Travel domain concept projection so paraphrased queries align semantically.
    token_set = set(raw_tokens) | set(content_tokens)
    for trigger_words, concept_labels in _CONCEPT_ANCHORS:
        if token_set & trigger_words:
            for label in concept_labels:
                weighted_features.append((label, 0.85))

    for feature, weight in weighted_features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        h_val = int.from_bytes(digest, byteorder="little", signed=False)
        bucket = h_val % dim
        sign = 1.0 if ((h_val >> 32) & 1) == 0 else -1.0
        vec[bucket] += sign * weight

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 1e-12:
        vec = [round(v / norm, 6) for v in vec]
    return vec


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute bounded cosine similarity in [-1.0, 1.0] between two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    score = dot / (norm_a * norm_b)
    return max(-1.0, min(1.0, score))


class LocalSqliteAdkSessionServiceAdapter:
    """Offline ADK-compatible SessionService backed by ``PersistentTravelSessionManager``."""

    def __init__(self, manager: PersistentTravelSessionManager) -> None:
        self._manager = manager

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_id = session_id or (
            f"sess_{hashlib.sha256(f'{app_name}:{user_id}:{_utc_now_iso()}'.encode()).hexdigest()[:12]}"
        )
        merged_state = self._manager.update_session_state(
            session_id=resolved_id,
            updates=dict(state or {}),
            user_id=user_id,
        )
        return {
            "id": resolved_id,
            "app_name": app_name,
            "user_id": user_id,
            "state": merged_state,
            "events": self._manager.get_session_history(resolved_id),
        }

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        return {
            "id": session_id,
            "app_name": app_name,
            "user_id": user_id,
            "state": self._manager.get_session_state(session_id),
            "events": self._manager.get_session_history(session_id),
        }


class PersistentTravelSessionManager:
    """Durable multi-turn session state & semantic traveler memory store.

    Persists sessions, conversation turns, and vector-indexed traveler preferences
    in SQLite, with optional synchronization to Google Cloud Firestore and
    hybrid semantic lookup via Vertex AI Search (Discovery Engine).
    """

    def __init__(
        self,
        db_path: str | None = None,
        project_id: str | None = None,
        use_firestore: bool = False,
        use_vertex_search: bool = False,
        *,
        location: str = "global",
        firestore_collection: str = "travel_concierge_sessions",
        vertex_data_store_id: str | None = None,
        embedding_dim: int = 128,
        scrub_pii: bool = True,
        pii_redactor: Any = None,
    ) -> None:
        resolved_db = (
            db_path
            or os.environ.get("TRAVEL_SESSION_DB_PATH")
            or DEFAULT_SESSION_DB_PATH
        )
        self.db_path = resolved_db
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT") or None
        self.use_firestore = use_firestore
        self.use_vertex_search = use_vertex_search
        self.location = location
        self.firestore_collection = firestore_collection
        self.vertex_data_store_id = (
            vertex_data_store_id
            or os.environ.get("VERTEX_SEARCH_DATA_STORE_ID")
            or "travel-concierge-memory-store"
        )
        self.embedding_dim = embedding_dim
        self.scrub_pii = scrub_pii
        self.pii_redactor = pii_redactor

        self._lock = threading.RLock()
        self._memory_conn: sqlite3.Connection | None = None

        if self.db_path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._firestore_client: Any = None
        if self.use_firestore and _FIRESTORE_CLIENT_CLS is not None:
            try:
                self._firestore_client = (
                    _FIRESTORE_CLIENT_CLS(project=self.project_id)
                    if self.project_id
                    else _FIRESTORE_CLIENT_CLS()
                )
            except Exception as exc:  # pragma: no cover - cloud auth runtime guard
                logger.warning("Firestore client initialization skipped: %s", exc)
                self._firestore_client = None

        self._vertex_search_client: Any = None
        if self.use_vertex_search and _DISCOVERY_SEARCH_CLIENT_CLS is not None:
            try:
                self._vertex_search_client = _DISCOVERY_SEARCH_CLIENT_CLS()
            except Exception as exc:  # pragma: no cover - cloud auth runtime guard
                logger.warning("Vertex AI Search client initialization skipped: %s", exc)
                self._vertex_search_client = None

        self._initialize_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a thread-safe SQLite connection and commit/close cleanly."""
        with self._lock:
            if self._memory_conn is not None:
                try:
                    yield self._memory_conn
                    self._memory_conn.commit()
                except Exception:
                    self._memory_conn.rollback()
                    raise
            else:
                conn = sqlite3.connect(self.db_path, timeout=15.0)
                conn.row_factory = sqlite3.Row
                try:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=NORMAL;")
                    conn.execute("PRAGMA foreign_keys=ON;")
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()

    def close(self) -> None:
        """Close any held in-memory SQLite connection cleanly."""
        with self._lock:
            if self._memory_conn is not None:
                self._memory_conn.close()
                self._memory_conn = None

    def _initialize_schema(self) -> None:
        """Create durable SQLite tables and indexes if they do not yet exist."""
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT '',
                    state_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversation_turns (
                    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_turns_session_order
                    ON conversation_turns(session_id, turn_id ASC);

                CREATE TABLE IF NOT EXISTS traveler_memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_text TEXT NOT NULL,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memories_user
                    ON traveler_memories(user_id, updated_at DESC);
                """
            )

    def create_adk_session_service(self) -> Any:
        """Create an ADK ``SessionService`` instance aligned with runtime capabilities.

        Returns ``VertexAiSessionService``, ``DatabaseSessionService``, or
        ``InMemorySessionService`` when ``google.adk.sessions`` is installed, or
        ``LocalSqliteAdkSessionServiceAdapter`` in offline environments.
        """
        if _ADK_SESSIONS_AVAILABLE and _adk_sessions is not None:
            vertex_cls = getattr(_adk_sessions, "VertexAiSessionService", None)
            if self.project_id and self.use_firestore and vertex_cls is not None:
                try:
                    return vertex_cls(project=self.project_id, location=self.location)
                except Exception as exc:  # pragma: no cover
                    logger.warning("VertexAiSessionService fallback triggered: %s", exc)

            db_cls = getattr(_adk_sessions, "DatabaseSessionService", None)
            if db_cls is not None and self.db_path != ":memory:":
                try:
                    return db_cls(db_url=f"sqlite:///{self.db_path}")
                except Exception as exc:  # pragma: no cover
                    logger.warning("DatabaseSessionService fallback triggered: %s", exc)

            in_mem_cls = getattr(_adk_sessions, "InMemorySessionService", None)
            if in_mem_cls is not None:
                return in_mem_cls()

        return LocalSqliteAdkSessionServiceAdapter(self)

    def _sync_session_to_firestore(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Synchronize session document to Google Cloud Firestore when enabled."""
        if not self.use_firestore or self._firestore_client is None:
            return
        try:
            doc_ref = self._firestore_client.collection(self.firestore_collection).document(
                session_id
            )
            doc_ref.set(payload, merge=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed syncing session %s to Firestore: %s", session_id, exc)

    def save_turn(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist a single conversation turn to SQLite (and optional Firestore).

        Active PII scrubbing runs prior to storage so raw emails, phone numbers,
        payment cards, SSNs, passports, or secrets are never written at rest.

        Args:
            session_id: Unique session identifier.
            user_id: Unique traveler identifier.
            role: Speaker role (``"user"``, ``"assistant"``, ``"system"``, ``"tool"``).
            content: Message text content.
            metadata: Optional JSON-serializable turn metadata dictionary.
        """
        if not session_id or not str(session_id).strip():
            raise ValueError("session_id must be a non-empty string")
        if not user_id or not str(user_id).strip():
            raise ValueError("user_id must be a non-empty string")
        if not role or not str(role).strip():
            raise ValueError("role must be a non-empty string")

        clean_content = (
            _scrub_text_safely(str(content), redactor=self.pii_redactor)
            if self.scrub_pii
            else str(content)
        )
        clean_metadata = (
            _scrub_payload_safely(metadata or {}, redactor=self.pii_redactor)
            if self.scrub_pii
            else dict(metadata or {})
        )
        now = _utc_now_iso()
        metadata_json = json.dumps(clean_metadata, sort_keys=True, default=str)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, user_id, state_json, created_at, updated_at)
                VALUES (?, ?, '{}', ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id = CASE WHEN excluded.user_id != '' THEN excluded.user_id ELSE sessions.user_id END,
                    updated_at = excluded.updated_at
                """,
                (session_id, user_id, now, now),
            )
            conn.execute(
                """
                INSERT INTO conversation_turns (session_id, user_id, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_id, role, clean_content, metadata_json, now),
            )

        self._sync_session_to_firestore(
            session_id,
            {
                "session_id": session_id,
                "user_id": user_id,
                "last_turn_role": role,
                "last_turn_content": clean_content,
                "updated_at": now,
            },
        )

    def get_session_history(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve ordered conversation turns for ``session_id``.

        Args:
            session_id: Session identifier to load.
            limit: Optional cap returning the most recent ``limit`` turns in chronological order.

        Returns:
            Chronological list of turn dicts containing ``turn_id``, ``session_id``,
            ``user_id``, ``role``, ``content``, ``metadata``, and ``created_at``.
        """
        with self._connect() as conn:
            if limit is not None and limit > 0:
                rows = conn.execute(
                    """
                    SELECT turn_id, session_id, user_id, role, content, metadata_json, created_at
                    FROM (
                        SELECT turn_id, session_id, user_id, role, content, metadata_json, created_at
                        FROM conversation_turns
                        WHERE session_id = ?
                        ORDER BY turn_id DESC
                        LIMIT ?
                    )
                    ORDER BY turn_id ASC
                    """,
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT turn_id, session_id, user_id, role, content, metadata_json, created_at
                    FROM conversation_turns
                    WHERE session_id = ?
                    ORDER BY turn_id ASC
                    """,
                    (session_id,),
                ).fetchall()

        history: list[dict[str, Any]] = []
        for row in rows:
            try:
                parsed_meta = json.loads(row["metadata_json"])
            except (TypeError, json.JSONDecodeError):
                parsed_meta = {}
            history.append(
                {
                    "turn_id": row["turn_id"],
                    "session_id": row["session_id"],
                    "user_id": row["user_id"],
                    "role": row["role"],
                    "content": row["content"],
                    "metadata": parsed_meta,
                    "created_at": row["created_at"],
                }
            )
        return history

    def get_session_state(self, session_id: str) -> dict[str, Any]:
        """Return persisted session state dictionary for ``session_id``."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        if row is not None:
            try:
                data = json.loads(row["state_json"])
                if isinstance(data, dict):
                    return data
            except (TypeError, json.JSONDecodeError):
                pass

        if self.use_firestore and self._firestore_client is not None:
            try:
                snap = (
                    self._firestore_client.collection(self.firestore_collection)
                    .document(session_id)
                    .get()
                )
                if getattr(snap, "exists", False):
                    doc = snap.to_dict() or {}
                    remote_state = doc.get("state")
                    if isinstance(remote_state, dict):
                        return dict(remote_state)
            except Exception as exc:  # pragma: no cover
                logger.warning("Firestore get_session_state fallback: %s", exc)

        return {}

    def update_session_state(
        self,
        session_id: str,
        updates: dict[str, Any],
        user_id: str = "",
    ) -> dict[str, Any]:
        """Merge ``updates`` into the persisted state for ``session_id`` and return it."""
        if not session_id or not str(session_id).strip():
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(updates, dict):
            raise ValueError("updates must be a dict")

        clean_updates = (
            _scrub_payload_safely(updates, redactor=self.pii_redactor)
            if self.scrub_pii
            else dict(updates)
        )
        now = _utc_now_iso()

        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_id, state_json, created_at FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            if row is not None:
                try:
                    existing_state = json.loads(row["state_json"])
                    if not isinstance(existing_state, dict):
                        existing_state = {}
                except (TypeError, json.JSONDecodeError):
                    existing_state = {}
                merged_state = {**existing_state, **clean_updates}
                resolved_user = user_id or row["user_id"] or ""
                conn.execute(
                    """
                    UPDATE sessions
                    SET user_id = ?, state_json = ?, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (
                        resolved_user,
                        json.dumps(merged_state, sort_keys=True, default=str),
                        now,
                        session_id,
                    ),
                )
            else:
                merged_state = dict(clean_updates)
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, user_id, state_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        user_id,
                        json.dumps(merged_state, sort_keys=True, default=str),
                        now,
                        now,
                    ),
                )

        self._sync_session_to_firestore(
            session_id,
            {
                "session_id": session_id,
                "user_id": user_id,
                "state": merged_state,
                "updated_at": now,
            },
        )
        return merged_state

    def upsert_memory_entry(
        self,
        user_id: str,
        memory_text: str,
        tags: list[str] | None = None,
    ) -> str:
        """Upsert a semantic traveler memory entry with deterministic vector embedding.

        Applies PII scrubbing before hashing and storage, deduplicating identical
        normalized facts per ``user_id`` while merging tags.

        Args:
            user_id: Traveler identifier owning this memory.
            memory_text: Salient preference or trip constraint statement.
            tags: Optional categorical tags (e.g. ``["flight", "preference"]``).

        Returns:
            Deterministic ``memory_id`` string stored in ``traveler_memories``.
        """
        if not user_id or not str(user_id).strip():
            raise ValueError("user_id must be a non-empty string")
        if not memory_text or not str(memory_text).strip():
            raise ValueError("memory_text must be a non-empty string")

        normalized_raw = " ".join(str(memory_text).split())
        clean_text = (
            _scrub_text_safely(normalized_raw, redactor=self.pii_redactor)
            if self.scrub_pii
            else normalized_raw
        )
        clean_tags = sorted(
            {str(t).strip().lower() for t in (tags or []) if str(t).strip()}
        )

        digest = hashlib.sha256(
            f"{user_id.strip()}::{clean_text.lower()}".encode("utf-8")
        ).hexdigest()[:16]
        memory_id = f"mem_{digest}"
        now = _utc_now_iso()

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT tags_json, created_at FROM traveler_memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()

            if existing is not None:
                try:
                    prev_tags = json.loads(existing["tags_json"])
                    if not isinstance(prev_tags, list):
                        prev_tags = []
                except (TypeError, json.JSONDecodeError):
                    prev_tags = []
                merged_tags = sorted({*prev_tags, *clean_tags})
                embedding = compute_text_embedding(
                    clean_text, tags=merged_tags, dim=self.embedding_dim
                )
                conn.execute(
                    """
                    UPDATE traveler_memories
                    SET memory_text = ?, tags_json = ?, embedding_json = ?, updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (
                        clean_text,
                        json.dumps(merged_tags),
                        json.dumps(embedding),
                        now,
                        memory_id,
                    ),
                )
            else:
                embedding = compute_text_embedding(
                    clean_text, tags=clean_tags, dim=self.embedding_dim
                )
                conn.execute(
                    """
                    INSERT INTO traveler_memories
                        (memory_id, user_id, memory_text, tags_json, embedding_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        user_id.strip(),
                        clean_text,
                        json.dumps(clean_tags),
                        json.dumps(embedding),
                        now,
                        now,
                    ),
                )

        return memory_id

    def _query_vertex_ai_search(
        self,
        user_id: str,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Query optional Vertex AI Search (Discovery Engine) memory data store."""
        if (
            not self.use_vertex_search
            or self._vertex_search_client is None
            or not self.project_id
        ):
            return []

        serving_config = (
            f"projects/{self.project_id}/locations/{self.location}/"
            f"collections/default_collection/dataStores/{self.vertex_data_store_id}/"
            "servingConfigs/default_search"
        )
        try:
            if _DISCOVERY_SEARCH_REQUEST_CLS is not None:
                request = _DISCOVERY_SEARCH_REQUEST_CLS(
                    serving_config=serving_config,
                    query=query,
                    page_size=top_k,
                    filter=f'user_id: ANY("{user_id}")',
                )
            else:
                request = {
                    "serving_config": serving_config,
                    "query": query,
                    "page_size": top_k,
                }
            response = self._vertex_search_client.search(request=request)
            results: list[dict[str, Any]] = []
            for idx, item in enumerate(getattr(response, "results", [])):
                doc = getattr(item, "document", None)
                struct_data = getattr(doc, "struct_data", {}) or {}
                mem_text = str(
                    struct_data.get("memory_text")
                    or struct_data.get("content")
                    or ""
                ).strip()
                if not mem_text:
                    continue
                rank_score = round(max(0.1, 1.0 - idx * 0.1), 4)
                results.append(
                    {
                        "memory_id": str(getattr(doc, "id", f"vertex_{idx}")),
                        "user_id": user_id,
                        "memory_text": _scrub_text_safely(
                            mem_text, redactor=self.pii_redactor
                        ),
                        "tags": list(struct_data.get("tags", ["vertex_search"])),
                        "score": rank_score,
                        "similarity_score": rank_score,
                        "source": "vertex_ai_search",
                        "updated_at": str(struct_data.get("updated_at", _utc_now_iso())),
                    }
                )
            return results
        except Exception as exc:  # pragma: no cover
            logger.warning("Vertex AI Search retrieval skipped: %s", exc)
            return []

    def retrieve_relevant_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Retrieve top-k traveler memories ranked by cosine vector similarity.

        Combines local SQLite hashed-embedding cosine similarity with optional
        Vertex AI Search results when ``use_vertex_search=True``.

        Args:
            user_id: Traveler identifier whose memories should be searched.
            query: Natural-language query describing current planning context.
            top_k: Maximum number of ranked memory records to return (>= 1).

        Returns:
            Ranked list of memory dicts with ``memory_id``, ``user_id``,
            ``memory_text``, ``tags``, ``score``, ``similarity_score``, ``source``,
            and ``updated_at``.
        """
        if top_k <= 0:
            return []

        query_vec = compute_text_embedding(query or "", dim=self.embedding_dim)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT memory_id, user_id, memory_text, tags_json, embedding_json, updated_at
                FROM traveler_memories
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                (user_id.strip(),),
            ).fetchall()

        scored: list[dict[str, Any]] = []
        seen_texts: set[str] = set()

        for row in rows:
            try:
                emb = json.loads(row["embedding_json"])
                if not isinstance(emb, list) or len(emb) != self.embedding_dim:
                    emb = compute_text_embedding(
                        row["memory_text"], dim=self.embedding_dim
                    )
            except (TypeError, json.JSONDecodeError):
                emb = compute_text_embedding(row["memory_text"], dim=self.embedding_dim)

            try:
                tags = json.loads(row["tags_json"])
                if not isinstance(tags, list):
                    tags = []
            except (TypeError, json.JSONDecodeError):
                tags = []

            sim = round(cosine_similarity(query_vec, emb), 6)
            seen_texts.add(row["memory_text"].lower())
            scored.append(
                {
                    "memory_id": row["memory_id"],
                    "user_id": row["user_id"],
                    "memory_text": row["memory_text"],
                    "tags": tags,
                    "score": sim,
                    "similarity_score": sim,
                    "source": "sqlite_vector",
                    "updated_at": row["updated_at"],
                }
            )

        for remote_item in self._query_vertex_ai_search(user_id, query, top_k):
            if remote_item["memory_text"].lower() not in seen_texts:
                seen_texts.add(remote_item["memory_text"].lower())
                scored.append(remote_item)

        scored.sort(
            key=lambda item: (item["similarity_score"], item["updated_at"]),
            reverse=True,
        )
        return scored[:top_k]
