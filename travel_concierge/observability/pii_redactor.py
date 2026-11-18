"""Active PII redaction with Google Cloud DLP integration and deterministic local scrubbing.

Provides defense-in-depth PII protection across structured logs, distributed trace
attributes, intent/outcome audit records, and persistent traveler memory stores.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Mapping
from typing import Any

try:
    from google.cloud import dlp_v2  # type: ignore[import-not-found]

    HAS_CLOUD_DLP = True
except ImportError:
    dlp_v2 = None  # type: ignore[assignment]
    HAS_CLOUD_DLP = False


REDACTED_EMAIL = "[REDACTED_EMAIL]"
REDACTED_PHONE = "[REDACTED_PHONE]"
REDACTED_CARD = "[REDACTED_CARD]"
REDACTED_PASSPORT = "[REDACTED_PASSPORT]"
REDACTED_SSN = "[REDACTED_SSN]"
REDACTED_SECRET = "[REDACTED_SECRET]"
REDACTED_NAME = "[REDACTED_NAME]"

DLP_INFO_TYPES: tuple[str, ...] = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD_NUMBER",
    "PASSPORT",
    "US_SOCIAL_SECURITY_NUMBER",
    "PERSON_NAME",
)

_DLP_PLACEHOLDER_BY_INFOTYPE: dict[str, str] = {
    "EMAIL_ADDRESS": REDACTED_EMAIL,
    "PHONE_NUMBER": REDACTED_PHONE,
    "CREDIT_CARD_NUMBER": REDACTED_CARD,
    "PASSPORT": REDACTED_PASSPORT,
    "US_SOCIAL_SECURITY_NUMBER": REDACTED_SSN,
    "PERSON_NAME": REDACTED_NAME,
}

# Protected telemetry correlation identifiers that should preserve hex trace/span IDs.
_TELEMETRY_ID_KEYS: frozenset[str] = frozenset(
    {
        "trace_id",
        "span_id",
        "parent_span_id",
        "correlation_id",
        "logging.googleapis.com/trace",
        "logging.googleapis.com/spanid",
        "eval_id",
        "eval_set_id",
        "invocation_id",
    }
)

_EMAIL_KEYS: frozenset[str] = frozenset(
    {
        "email",
        "contact_email",
        "traveler_email",
        "user_email",
        "email_address",
        "recipient_email",
    }
)

_PHONE_KEYS: frozenset[str] = frozenset(
    {
        "phone",
        "phone_number",
        "contact_phone",
        "mobile",
        "mobile_number",
        "telephone",
        "traveler_phone",
    }
)

_CARD_KEYS: frozenset[str] = frozenset(
    {
        "card_number",
        "credit_card",
        "credit_card_number",
        "debit_card",
        "debit_card_number",
        "pan",
        "cvv",
        "cvc",
        "security_code",
    }
)

_PASSPORT_KEYS: frozenset[str] = frozenset(
    {
        "passport",
        "passport_number",
        "passport_id",
        "passport_no",
    }
)

_SSN_KEYS: frozenset[str] = frozenset(
    {
        "ssn",
        "social_security_number",
        "us_social_security_number",
        "tax_id",
    }
)

_SECRET_KEYS: frozenset[str] = frozenset(
    {
        "password",
        "passwd",
        "pwd",
        "secret",
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "auth_token",
        "bearer_token",
        "client_secret",
        "private_key",
        "payment_token",
        "gateway_token",
    }
)

# Deterministic local regex patterns ordered to prevent partial-match collisions.
_BEARER_TOKEN_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{6,}")
_PROVIDER_SECRET_KEY_RE = re.compile(
    r"\b(?:[s][k]-[A-Za-z0-9_-]{6,}|[A][I][z][a][0-9A-Za-z_-]{6,})\b"
)
_INLINE_SECRET_KV_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
    r"password|passwd|secret[_-]?key|private[_-]?key)"
    r"(\s*[:=]\s*)"
    r"(?!\[REDACTED_)(?:\"[^\"]+\"|'[^']+'|[^\s,;\"')\]}]+)"
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Credit/debit card patterns (13-16 contiguous digits, 4x4 groups, or 4-6-5 Amex).
_FORMATTED_CARD_RE = re.compile(
    r"(?<![\w-])(?:\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{1,4}|\d{4}[ -]\d{6}[ -]\d{5})(?![\w-])"
)
_CONTIGUOUS_CARD_RE = re.compile(r"(?<![\w.-])\d{13,16}(?![\w.-])")

# US Social Security Numbers (explicit context or XXX-XX-XXXX).
_CONTEXT_SSN_RE = re.compile(
    r"(?i)(\b(?:ssn|social\s+security(?:\s+number)?)\s*[:=#-]?\s*)"
    r"(\d{3}[ -]?\d{2}[ -]?\d{4})\b"
)
_HYPHEN_SSN_RE = re.compile(r"(?<![\w-])\d{3}-\d{2}-\d{4}(?![\w-])")

# Passport numbers (contextual label + alphanumeric ID, or standard 1-2 uppercase + 7-8 digits).
_CONTEXT_PASSPORT_RE = re.compile(
    r"(?i)(\bpassport(?:\s*(?:id|no\.?|number|#))?\s*[:=#]?\s*)"
    r"(?!\[REDACTED_)([A-Z0-9-]{6,12})\b"
)
_STANDALONE_PASSPORT_RE = re.compile(r"\b[A-Z]{1,2}\d{7,8}\b")

# US/NANP and E.164 international phone numbers.
_NANP_PHONE_RE = re.compile(
    r"(?<![\w+])(?:\+?1[-.\s]?)?(?:\(\d{3}\)[-.\s]?|\d{3}[-.\s])\d{3}[-.\s]\d{4}(?!\d)"
)
_INTL_PHONE_RE = re.compile(r"(?<!\w)\+\d{1,3}(?:[-.\s]?\(?\d{1,4}\)?){2,5}(?!\d)")


def _replace_intl_phone(match: re.Match[str]) -> str:
    candidate = match.group(0)
    digit_count = sum(ch.isdigit() for ch in candidate)
    if 8 <= digit_count <= 15:
        return REDACTED_PHONE
    return candidate


def _resolve_sensitive_key_placeholder(key: str) -> str | None:
    normalized = key.lower().strip()
    if normalized.endswith("_secret_id"):
        return None
    if normalized in _EMAIL_KEYS or normalized.endswith("_email"):
        return REDACTED_EMAIL
    if normalized in _PHONE_KEYS or normalized.endswith("_phone"):
        return REDACTED_PHONE
    if normalized in _CARD_KEYS or normalized.endswith("_card_number"):
        return REDACTED_CARD
    if normalized in _PASSPORT_KEYS or normalized.endswith("_passport"):
        return REDACTED_PASSPORT
    if normalized in _SSN_KEYS or normalized.endswith("_ssn"):
        return REDACTED_SSN
    if normalized in _SECRET_KEYS or normalized.endswith(("_api_key", "_password", "_secret")):
        return REDACTED_SECRET
    return None


class PiiRedactor:
    """Scrubs sensitive PII and credentials via Cloud DLP and deterministic regex rules."""

    def __init__(
        self,
        enable_cloud_dlp: bool = False,
        project_id: str | None = None,
        dlp_client: Any = None,
    ) -> None:
        self.enable_cloud_dlp = bool(enable_cloud_dlp)
        self.project_id = (
            project_id
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCP_PROJECT_ID")
        )
        self._dlp_client: Any = dlp_client
        self._client_lock = threading.Lock()

    def _get_dlp_client(self) -> Any:
        if not self.enable_cloud_dlp:
            return None
        if self._dlp_client is not None:
            return self._dlp_client
        if not HAS_CLOUD_DLP or dlp_v2 is None or not self.project_id:
            return None
        with self._client_lock:
            if self._dlp_client is None:
                try:
                    self._dlp_client = dlp_v2.DlpServiceClient()
                except Exception:
                    self._dlp_client = None
        return self._dlp_client

    def _deidentify_via_cloud_dlp(self, text: str) -> str:
        """Calls Google Cloud Sensitive Data Protection (DLP) deidentify_content API."""
        client = self._get_dlp_client()
        if client is None or not self.project_id:
            return text

        inspect_config = {
            "info_types": [{"name": info_type} for info_type in DLP_INFO_TYPES],
            "min_likelihood": "POSSIBLE",
        }
        transformations = [
            {
                "info_types": [{"name": info_type}],
                "primitive_transformation": {
                    "replace_config": {
                        "new_value": {"string_value": placeholder},
                    }
                },
            }
            for info_type, placeholder in _DLP_PLACEHOLDER_BY_INFOTYPE.items()
        ]
        deidentify_config = {
            "info_type_transformations": {
                "transformations": transformations,
            }
        }
        request = {
            "parent": f"projects/{self.project_id}/locations/global",
            "inspect_config": inspect_config,
            "deidentify_config": deidentify_config,
            "item": {"value": text},
        }
        try:
            response = client.deidentify_content(request=request)
            item = getattr(response, "item", None)
            redacted_value = getattr(item, "value", None)
            if isinstance(redacted_value, str):
                return redacted_value
        except Exception:
            return text
        return text

    def redact_text(self, text: str) -> str:
        """Scrubs emails, phones, cards, passports, SSNs, and secrets from a string."""
        if not isinstance(text, str) or not text:
            return text

        scrubbed = text
        if self.enable_cloud_dlp:
            scrubbed = self._deidentify_via_cloud_dlp(scrubbed)

        # 1. Secrets, bearer tokens, and provider API keys
        scrubbed = _BEARER_TOKEN_RE.sub(REDACTED_SECRET, scrubbed)
        scrubbed = _PROVIDER_SECRET_KEY_RE.sub(REDACTED_SECRET, scrubbed)
        scrubbed = _INLINE_SECRET_KV_RE.sub(rf"\1\2{REDACTED_SECRET}", scrubbed)

        # 2. Email addresses
        scrubbed = _EMAIL_RE.sub(REDACTED_EMAIL, scrubbed)

        # 3. Payment card numbers (run prior to phone matching)
        scrubbed = _FORMATTED_CARD_RE.sub(REDACTED_CARD, scrubbed)
        scrubbed = _CONTIGUOUS_CARD_RE.sub(REDACTED_CARD, scrubbed)

        # 4. Social Security Numbers
        scrubbed = _CONTEXT_SSN_RE.sub(rf"\1{REDACTED_SSN}", scrubbed)
        scrubbed = _HYPHEN_SSN_RE.sub(REDACTED_SSN, scrubbed)

        # 5. Passport identifiers
        scrubbed = _CONTEXT_PASSPORT_RE.sub(rf"\1{REDACTED_PASSPORT}", scrubbed)
        scrubbed = _STANDALONE_PASSPORT_RE.sub(REDACTED_PASSPORT, scrubbed)

        # 6. Telephone numbers (NANP and E.164 international)
        scrubbed = _NANP_PHONE_RE.sub(REDACTED_PHONE, scrubbed)
        scrubbed = _INTL_PHONE_RE.sub(_replace_intl_phone, scrubbed)

        return scrubbed

    def redact_payload(self, payload: Any) -> Any:
        """Recursively scrubs strings, collections, and sensitive mapping keys."""
        if payload is None or isinstance(payload, (bool, int, float)):
            return payload

        if isinstance(payload, str):
            return self.redact_text(payload)

        if isinstance(payload, Mapping):
            sanitized_dict: dict[Any, Any] = {}
            for raw_key, raw_val in payload.items():
                key_str = str(raw_key) if isinstance(raw_key, str) else None
                sanitized_key = self.redact_text(raw_key) if isinstance(raw_key, str) else raw_key

                if key_str is not None and key_str.lower().strip() in _TELEMETRY_ID_KEYS:
                    if isinstance(raw_val, str) and "@" not in raw_val:
                        sanitized_dict[sanitized_key] = raw_val
                    else:
                        sanitized_dict[sanitized_key] = self.redact_payload(raw_val)
                    continue

                if key_str is not None:
                    placeholder = _resolve_sensitive_key_placeholder(key_str)
                    if placeholder is not None:
                        if raw_val is None or raw_val == "":
                            sanitized_dict[sanitized_key] = raw_val
                        else:
                            sanitized_dict[sanitized_key] = placeholder
                        continue

                sanitized_dict[sanitized_key] = self.redact_payload(raw_val)
            return sanitized_dict

        if isinstance(payload, tuple):
            return tuple(self.redact_payload(item) for item in payload)

        if isinstance(payload, list):
            return [self.redact_payload(item) for item in payload]

        if isinstance(payload, set):
            return {self.redact_payload(item) for item in payload}

        return payload


_DEFAULT_REDACTOR_LOCK = threading.Lock()
_DEFAULT_REDACTOR: PiiRedactor | None = None


def get_default_redactor() -> PiiRedactor:
    """Returns the shared default PiiRedactor configured from app/environment settings."""
    global _DEFAULT_REDACTOR
    if _DEFAULT_REDACTOR is not None:
        return _DEFAULT_REDACTOR

    with _DEFAULT_REDACTOR_LOCK:
        if _DEFAULT_REDACTOR is None:
            enable_dlp_env = os.environ.get("ENABLE_CLOUD_DLP", "").strip().lower() in {
                "1",
                "true",
                "yes",
            }
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get(
                "GCP_PROJECT_ID"
            )
            try:
                from travel_concierge.config import get_config  # type: ignore[import-not-found]

                cfg = get_config()
                enable_dlp_env = enable_dlp_env or bool(getattr(cfg, "enable_cloud_dlp", False))
                project_id = project_id or getattr(cfg, "project_id", None)
            except Exception:
                pass
            _DEFAULT_REDACTOR = PiiRedactor(
                enable_cloud_dlp=enable_dlp_env,
                project_id=project_id,
            )
    return _DEFAULT_REDACTOR
