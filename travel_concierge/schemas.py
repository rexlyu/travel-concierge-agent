"""Strict input and output JSON schemas for Travel & Budget Concierge tools.

Implements Rubric 1.3 (Explicit JSON Schemas) using Pydantic v2 when installed
and a strict standard-library validation engine in offline/minimal runtimes.
Both modes enforce:
- ``additionalProperties = False`` (unknown fields rejected)
- Strict primitive types (no boolean-as-number or string-as-number coercion)
- Valid ISO-8601 calendar dates (``YYYY-MM-DD``) and chronological ordering
- Positive/non-negative monetary amounts and passenger/guest counts
- Validated 3-letter uppercase IATA airport codes and ISO-4217 currency codes
- Draft 2020-12 JSON Schema generation via ``.model_json_schema()``
"""

from collections.abc import Callable, Mapping
import copy
import datetime
import math
import re
import types
from typing import Any, Literal, Union, get_args, get_origin

try:
    from pydantic import (
        BaseModel as _PydanticBaseModel,
        ConfigDict as _PydanticConfigDict,
        Field as _PydanticField,
        ValidationError as _PydanticValidationError,
        model_validator as _pydantic_model_validator,
    )

    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover - active in stdlib-only sandbox
    _PydanticBaseModel = None
    _PydanticConfigDict = None
    _PydanticField = None
    _PydanticValidationError = None
    _pydantic_model_validator = None
    _HAS_PYDANTIC = False


JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_IATA_CODE_RE = re.compile(r"^[A-Z]{3}$")
_CURRENCY_CODE_RE = re.compile(r"^[A-Z]{3}$")
_COUNTRY_CODE_RE = re.compile(r"^[A-Z]{2,3}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SECRET_ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,128}$")

SUPPORTED_ISO_CURRENCIES: frozenset[str] = frozenset(
    {
        "USD",
        "EUR",
        "GBP",
        "JPY",
        "CAD",
        "AUD",
        "CHF",
        "CNY",
        "HKD",
        "SGD",
        "KRW",
        "INR",
        "MXN",
        "BRL",
        "NZD",
        "SEK",
        "NOK",
        "DKK",
        "THB",
        "TWD",
        "ZAR",
        "AED",
        "TRY",
        "PLN",
        "CZK",
        "HUF",
        "ILS",
        "PHP",
        "IDR",
        "MYR",
        "VND",
        "CLP",
        "COP",
        "PEN",
        "ARS",
        "EGP",
        "SAR",
        "QAR",
        "ISK",
    }
)

SUPPORTED_COUNTRY_CODES: frozenset[str] = frozenset(
    {
        # ISO-3166-1 alpha-2
        "US",
        "CA",
        "MX",
        "GB",
        "UK",
        "IE",
        "FR",
        "DE",
        "IT",
        "ES",
        "PT",
        "NL",
        "BE",
        "CH",
        "AT",
        "SE",
        "NO",
        "DK",
        "FI",
        "IS",
        "GR",
        "PL",
        "CZ",
        "HU",
        "HR",
        "TR",
        "JP",
        "KR",
        "CN",
        "TW",
        "HK",
        "SG",
        "TH",
        "VN",
        "MY",
        "ID",
        "PH",
        "IN",
        "AU",
        "NZ",
        "AE",
        "QA",
        "SA",
        "IL",
        "EG",
        "MA",
        "ZA",
        "KE",
        "BR",
        "AR",
        "CL",
        "CO",
        "PE",
        "CR",
        # ISO-3166-1 alpha-3
        "USA",
        "CAN",
        "MEX",
        "GBR",
        "IRL",
        "FRA",
        "DEU",
        "ITA",
        "ESP",
        "PRT",
        "NLD",
        "BEL",
        "CHE",
        "AUT",
        "SWE",
        "NOR",
        "DNK",
        "FIN",
        "ISL",
        "GRC",
        "POL",
        "CZE",
        "HUN",
        "HRV",
        "TUR",
        "JPN",
        "KOR",
        "CHN",
        "TWN",
        "HKG",
        "SGP",
        "THA",
        "VNM",
        "MYS",
        "IDN",
        "PHL",
        "IND",
        "AUS",
        "NZL",
        "ARE",
        "QAT",
        "SAU",
        "ISR",
        "EGY",
        "MAR",
        "ZAF",
        "KEN",
        "BRA",
        "ARG",
        "CHL",
        "COL",
        "PER",
        "CRI",
    }
)


class SchemaValidationError(ValueError):
    """Raised when tool input or output payloads violate strict JSON schemas."""

    def __init__(
        self,
        message: str,
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.errors: list[dict[str, Any]] = errors or []


def validate_iso_date(value: str, field_name: str) -> datetime.date:
    """Validates a strict YYYY-MM-DD calendar date string."""
    if not isinstance(value, str) or not _ISO_DATE_RE.fullmatch(value):
        raise SchemaValidationError(
            f"Field '{field_name}' must be an ISO-8601 date formatted as YYYY-MM-DD "
            f"(got {value!r}).",
            errors=[
                {
                    "field": field_name,
                    "code": "invalid_iso_date",
                    "message": "Expected valid YYYY-MM-DD calendar date.",
                }
            ],
        )
    try:
        return datetime.date.fromisoformat(value)
    except ValueError as exc:
        raise SchemaValidationError(
            f"Field '{field_name}' is not a valid calendar date: {value!r} ({exc}).",
            errors=[
                {
                    "field": field_name,
                    "code": "invalid_calendar_date",
                    "message": str(exc),
                }
            ],
        ) from exc


def validate_currency_code(value: str, field_name: str) -> str:
    """Validates a 3-letter uppercase ISO-4217 currency code."""
    if not isinstance(value, str) or not _CURRENCY_CODE_RE.fullmatch(value):
        raise SchemaValidationError(
            f"Field '{field_name}' must be a 3-letter uppercase ISO-4217 currency "
            f"code matching ^[A-Z]{{3}}$ (got {value!r}).",
            errors=[
                {
                    "field": field_name,
                    "code": "invalid_currency_format",
                    "message": "Expected 3-letter uppercase currency code (e.g., USD, EUR, JPY).",
                }
            ],
        )
    if value not in SUPPORTED_ISO_CURRENCIES:
        sample = ", ".join(sorted(list(SUPPORTED_ISO_CURRENCIES))[:10])
        raise SchemaValidationError(
            f"Field '{field_name}' has unsupported ISO-4217 currency code '{value}'. "
            f"Supported examples: {sample}.",
            errors=[
                {
                    "field": field_name,
                    "code": "unsupported_currency_code",
                    "message": f"Unsupported currency code '{value}'.",
                }
            ],
        )
    return value


def validate_iata_code(value: str, field_name: str) -> str:
    """Validates a 3-letter uppercase IATA airport or metropolitan code."""
    if not isinstance(value, str) or not _IATA_CODE_RE.fullmatch(value):
        raise SchemaValidationError(
            f"Field '{field_name}' must be a 3-letter uppercase IATA airport code "
            f"matching ^[A-Z]{{3}}$ (got {value!r}).",
            errors=[
                {
                    "field": field_name,
                    "code": "invalid_iata_code",
                    "message": "Expected 3-letter uppercase IATA code (e.g., SFO, JFK, NRT).",
                }
            ],
        )
    return value


_MISSING = object()


class _FieldSpec:
    """Declarative field constraint descriptor for stdlib schema mode."""

    def __init__(
        self,
        default: Any = _MISSING,
        *,
        default_factory: Callable[[], Any] | None = None,
        description: str = "",
        gt: float | None = None,
        ge: float | None = None,
        lt: float | None = None,
        le: float | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
    ) -> None:
        self.default = default
        self.default_factory = default_factory
        self.description = description
        self.gt = gt
        self.ge = ge
        self.lt = lt
        self.le = le
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern) if pattern else None

    @property
    def is_required(self) -> bool:
        return self.default is _MISSING and self.default_factory is None

    def get_default(self) -> Any:
        if self.default_factory is not None:
            return self.default_factory()
        return copy.deepcopy(self.default)


def schema_field(
    default: Any = _MISSING,
    *,
    default_factory: Callable[[], Any] | None = None,
    description: str = "",
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    pattern: str | None = None,
) -> Any:
    """Creates a schema field definition compatible with Pydantic v2 and stdlib."""
    if _HAS_PYDANTIC and _PydanticField is not None:
        kwargs: dict[str, Any] = {"description": description}
        if gt is not None:
            kwargs["gt"] = gt
        if ge is not None:
            kwargs["ge"] = ge
        if lt is not None:
            kwargs["lt"] = lt
        if le is not None:
            kwargs["le"] = le
        if min_length is not None:
            kwargs["min_length"] = min_length
        if max_length is not None:
            kwargs["max_length"] = max_length
        if pattern is not None:
            kwargs["pattern"] = pattern
        if default_factory is not None:
            return _PydanticField(default_factory=default_factory, **kwargs)
        if default is _MISSING:
            return _PydanticField(..., **kwargs)
        return _PydanticField(default=default, **kwargs)

    return _FieldSpec(
        default=default,
        default_factory=default_factory,
        description=description,
        gt=gt,
        ge=ge,
        lt=lt,
        le=le,
        min_length=min_length,
        max_length=max_length,
        pattern=pattern,
    )


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    """Unwraps Optional[T] / T | None into (inner_type, is_nullable)."""
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        args = get_args(tp)
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) < len(args):
            inner = non_none[0] if len(non_none) == 1 else Union[tuple(non_none)]
            return inner, True
    return tp, False


def _check_strict_primitive(field_name: str, expected_type: Any, value: Any) -> Any:
    """Enforces strict runtime primitive types without unsafe coercion."""
    inner_type, is_nullable = _unwrap_optional(expected_type)
    if value is None:
        if is_nullable:
            return None
        raise SchemaValidationError(
            f"Field '{field_name}' cannot be None.",
            errors=[
                {
                    "field": field_name,
                    "code": "null_not_allowed",
                    "message": "Null value is not permitted.",
                }
            ],
        )

    origin = get_origin(inner_type)
    if origin is Literal:
        allowed = get_args(inner_type)
        # Prevent True == 1 and False == 0 matching integer literals or vice versa.
        matched = any(
            value == candidate and type(value) is type(candidate)
            for candidate in allowed
        )
        if not matched:
            raise SchemaValidationError(
                f"Field '{field_name}' must be one of {list(allowed)!r} "
                f"(got {value!r}).",
                errors=[
                    {
                        "field": field_name,
                        "code": "invalid_enum",
                        "message": f"Allowed values: {list(allowed)!r}.",
                    }
                ],
            )
        return value

    if inner_type is bool:
        if not isinstance(value, bool):
            raise SchemaValidationError(
                f"Field '{field_name}' must be a boolean (got {type(value).__name__}).",
                errors=[
                    {
                        "field": field_name,
                        "code": "invalid_type",
                        "message": "Expected boolean (true/false).",
                    }
                ],
            )
        return value

    if inner_type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SchemaValidationError(
                f"Field '{field_name}' must be an integer (got {type(value).__name__}).",
                errors=[
                    {
                        "field": field_name,
                        "code": "invalid_type",
                        "message": "Expected integer.",
                    }
                ],
            )
        return value

    if inner_type is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SchemaValidationError(
                f"Field '{field_name}' must be a number (got {type(value).__name__}).",
                errors=[
                    {
                        "field": field_name,
                        "code": "invalid_type",
                        "message": "Expected finite numeric value.",
                    }
                ],
            )
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            raise SchemaValidationError(
                f"Field '{field_name}' must be a finite number (got {value!r}).",
                errors=[
                    {
                        "field": field_name,
                        "code": "non_finite_number",
                        "message": "NaN and Infinity are not permitted.",
                    }
                ],
            )
        return numeric

    if inner_type is str:
        if not isinstance(value, str):
            raise SchemaValidationError(
                f"Field '{field_name}' must be a string (got {type(value).__name__}).",
                errors=[
                    {
                        "field": field_name,
                        "code": "invalid_type",
                        "message": "Expected string.",
                    }
                ],
            )
        return value.strip()

    if origin is list or inner_type is list:
        if not isinstance(value, list):
            raise SchemaValidationError(
                f"Field '{field_name}' must be a list (got {type(value).__name__}).",
                errors=[
                    {
                        "field": field_name,
                        "code": "invalid_type",
                        "message": "Expected JSON array / list.",
                    }
                ],
            )
        item_args = get_args(inner_type)
        item_type = item_args[0] if item_args else Any
        validated_items: list[Any] = []
        for idx, item in enumerate(value):
            elem_name = f"{field_name}[{idx}]"
            if isinstance(item_type, type) and issubclass(item_type, StrictSchemaModel):
                if isinstance(item, item_type):
                    validated_items.append(item)
                elif isinstance(item, Mapping):
                    validated_items.append(item_type.model_validate(dict(item)))
                else:
                    raise SchemaValidationError(
                        f"Field '{elem_name}' must be an object matching "
                        f"{item_type.__name__} (got {type(item).__name__}).",
                        errors=[
                            {
                                "field": elem_name,
                                "code": "invalid_nested_object",
                                "message": f"Expected {item_type.__name__} object.",
                            }
                        ],
                    )
            elif get_origin(item_type) is dict or item_type is dict:
                if not isinstance(item, dict) or not item:
                    raise SchemaValidationError(
                        f"Field '{elem_name}' must be a non-empty dictionary.",
                        errors=[
                            {
                                "field": elem_name,
                                "code": "invalid_dict_item",
                                "message": "Expected non-empty JSON object.",
                            }
                        ],
                    )
                validated_items.append(copy.deepcopy(item))
            elif item_type is not Any:
                validated_items.append(
                    _check_strict_primitive(elem_name, item_type, item)
                )
            else:
                validated_items.append(copy.deepcopy(item))
        return validated_items

    if origin is dict or inner_type is dict:
        if not isinstance(value, dict):
            raise SchemaValidationError(
                f"Field '{field_name}' must be a dictionary (got {type(value).__name__}).",
                errors=[
                    {
                        "field": field_name,
                        "code": "invalid_type",
                        "message": "Expected JSON object / dictionary.",
                    }
                ],
            )
        return copy.deepcopy(value)

    if isinstance(inner_type, type) and issubclass(inner_type, StrictSchemaModel):
        if isinstance(value, inner_type):
            return value
        if isinstance(value, Mapping):
            return inner_type.model_validate(dict(value))
        raise SchemaValidationError(
            f"Field '{field_name}' must be an object matching {inner_type.__name__}.",
            errors=[
                {
                    "field": field_name,
                    "code": "invalid_nested_object",
                    "message": f"Expected {inner_type.__name__}.",
                }
            ],
        )

    return value


def _check_field_bounds(field_name: str, spec: _FieldSpec, value: Any) -> None:
    """Validates numeric bounds, string/list lengths, and regex patterns."""
    if value is None:
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if spec.gt is not None and not (value > spec.gt):
            raise SchemaValidationError(
                f"Field '{field_name}' must be strictly greater than {spec.gt} "
                f"(got {value}).",
                errors=[
                    {
                        "field": field_name,
                        "code": "greater_than",
                        "message": f"Must be > {spec.gt}.",
                    }
                ],
            )
        if spec.ge is not None and not (value >= spec.ge):
            raise SchemaValidationError(
                f"Field '{field_name}' must be greater than or equal to {spec.ge} "
                f"(got {value}).",
                errors=[
                    {
                        "field": field_name,
                        "code": "greater_than_equal",
                        "message": f"Must be >= {spec.ge}.",
                    }
                ],
            )
        if spec.lt is not None and not (value < spec.lt):
            raise SchemaValidationError(
                f"Field '{field_name}' must be strictly less than {spec.lt} "
                f"(got {value}).",
                errors=[
                    {
                        "field": field_name,
                        "code": "less_than",
                        "message": f"Must be < {spec.lt}.",
                    }
                ],
            )
        if spec.le is not None and not (value <= spec.le):
            raise SchemaValidationError(
                f"Field '{field_name}' must be less than or equal to {spec.le} "
                f"(got {value}).",
                errors=[
                    {
                        "field": field_name,
                        "code": "less_than_equal",
                        "message": f"Must be <= {spec.le}.",
                    }
                ],
            )

    if isinstance(value, (str, list, dict)):
        length = len(value)
        if spec.min_length is not None and length < spec.min_length:
            raise SchemaValidationError(
                f"Field '{field_name}' length must be at least {spec.min_length} "
                f"(got {length}).",
                errors=[
                    {
                        "field": field_name,
                        "code": "min_length",
                        "message": f"Minimum length is {spec.min_length}.",
                    }
                ],
            )
        if spec.max_length is not None and length > spec.max_length:
            raise SchemaValidationError(
                f"Field '{field_name}' length must be at most {spec.max_length} "
                f"(got {length}).",
                errors=[
                    {
                        "field": field_name,
                        "code": "max_length",
                        "message": f"Maximum length is {spec.max_length}.",
                    }
                ],
            )

    if isinstance(value, str) and spec.compiled_pattern is not None:
        if not spec.compiled_pattern.fullmatch(value):
            raise SchemaValidationError(
                f"Field '{field_name}' failed pattern '{spec.pattern}' (got {value!r}).",
                errors=[
                    {
                        "field": field_name,
                        "code": "pattern_mismatch",
                        "message": f"Must match pattern {spec.pattern}.",
                    }
                ],
            )


def _type_to_json_schema(tp: Any, spec: _FieldSpec) -> dict[str, Any]:
    """Generates Draft 2020-12 JSON Schema property descriptor for a field."""
    inner_type, is_nullable = _unwrap_optional(tp)
    origin = get_origin(inner_type)
    prop: dict[str, Any]

    if origin is Literal:
        allowed = list(get_args(inner_type))
        first = allowed[0] if allowed else ""
        lit_type = "string"
        if isinstance(first, bool):
            lit_type = "boolean"
        elif isinstance(first, int):
            lit_type = "integer"
        elif isinstance(first, float):
            lit_type = "number"
        prop = {"type": lit_type, "enum": allowed}
    elif inner_type is str:
        prop = {"type": "string"}
    elif inner_type is bool:
        prop = {"type": "boolean"}
    elif inner_type is int:
        prop = {"type": "integer"}
    elif inner_type is float:
        prop = {"type": "number"}
    elif origin is list or inner_type is list:
        item_args = get_args(inner_type)
        item_type = item_args[0] if item_args else Any
        if isinstance(item_type, type) and issubclass(item_type, StrictSchemaModel):
            nested = item_type.model_json_schema()
            nested.pop("$schema", None)
            items_schema = nested
        elif get_origin(item_type) is dict or item_type is dict:
            items_schema = {"type": "object"}
        elif item_type is str:
            items_schema = {"type": "string"}
        elif item_type is int:
            items_schema = {"type": "integer"}
        elif item_type is float:
            items_schema = {"type": "number"}
        else:
            items_schema = {}
        prop = {"type": "array", "items": items_schema}
    elif origin is dict or inner_type is dict:
        prop = {"type": "object"}
    elif isinstance(inner_type, type) and issubclass(inner_type, StrictSchemaModel):
        nested = inner_type.model_json_schema()
        nested.pop("$schema", None)
        prop = nested
    else:
        prop = {}

    if spec.gt is not None:
        prop["exclusiveMinimum"] = spec.gt
    if spec.ge is not None:
        prop["minimum"] = spec.ge
    if spec.lt is not None:
        prop["exclusiveMaximum"] = spec.lt
    if spec.le is not None:
        prop["maximum"] = spec.le
    if spec.min_length is not None:
        if prop.get("type") == "array":
            prop["minItems"] = spec.min_length
        else:
            prop["minLength"] = spec.min_length
    if spec.max_length is not None:
        if prop.get("type") == "array":
            prop["maxItems"] = spec.max_length
        else:
            prop["maxLength"] = spec.max_length
    if spec.pattern is not None:
        prop["pattern"] = spec.pattern

    if is_nullable:
        prop = {"anyOf": [prop, {"type": "null"}]}

    if spec.description:
        prop["description"] = spec.description
    if not spec.is_required and spec.default is not _MISSING:
        prop["default"] = spec.default

    return prop


if _HAS_PYDANTIC and _PydanticBaseModel is not None:

    class StrictSchemaModel(_PydanticBaseModel):
        """Pydantic v2-backed strict schema base class raising SchemaValidationError."""

        model_config = _PydanticConfigDict(
            extra="forbid",
            validate_assignment=True,
        )

        def __init__(self, **data: Any) -> None:
            try:
                super().__init__(**data)
            except SchemaValidationError:
                raise
            except _PydanticValidationError as exc:
                errors = [
                    {
                        "field": ".".join(str(p) for p in err.get("loc", ())),
                        "code": err.get("type", "validation_error"),
                        "message": err.get("msg", "Invalid value"),
                    }
                    for err in exc.errors()
                ]
                raise SchemaValidationError(str(exc), errors=errors) from exc

        @_pydantic_model_validator(mode="before")
        @classmethod
        def _enforce_strict_primitives(cls, data: Any) -> Any:
            if not isinstance(data, Mapping):
                raise SchemaValidationError(
                    f"{cls.__name__} expects a mapping/dict payload "
                    f"(got {type(data).__name__})."
                )
            annotations: dict[str, Any] = {}
            for base in reversed(cls.__mro__):
                annotations.update(getattr(base, "__annotations__", {}))
            cleaned = dict(data)
            unknown = [k for k in cleaned if k not in annotations]
            if unknown:
                raise SchemaValidationError(
                    f"Unexpected extra field(s) in {cls.__name__}: {unknown}. "
                    "Schema enforces additionalProperties=False.",
                    errors=[
                        {
                            "field": key,
                            "code": "extra_forbidden",
                            "message": f"Extra field '{key}' is not permitted.",
                        }
                        for key in unknown
                    ],
                )
            for key, raw_val in list(cleaned.items()):
                cleaned[key] = _check_strict_primitive(key, annotations[key], raw_val)
            return cleaned

        @_pydantic_model_validator(mode="after")
        def _run_domain_rules(self) -> "StrictSchemaModel":
            self._validate_domain_rules()
            return self

        def _validate_domain_rules(self) -> None:
            """Hook for multi-field and domain-specific invariants."""

        @classmethod
        def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(obj, cls):
                return cls(**obj.model_dump())
            if not isinstance(obj, Mapping):
                raise SchemaValidationError(
                    f"{cls.__name__}.model_validate expects a dict "
                    f"(got {type(obj).__name__})."
                )
            try:
                return super().model_validate(dict(obj), *args, **kwargs)
            except SchemaValidationError:
                raise
            except _PydanticValidationError as exc:
                errors = [
                    {
                        "field": ".".join(str(p) for p in err.get("loc", ())),
                        "code": err.get("type", "validation_error"),
                        "message": err.get("msg", "Invalid value"),
                    }
                    for err in exc.errors()
                ]
                raise SchemaValidationError(str(exc), errors=errors) from exc

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raw = super().model_json_schema(*args, **kwargs)
            schema: dict[str, Any] = {
                "$schema": JSON_SCHEMA_DRAFT,
                **raw,
                "type": "object",
                "additionalProperties": False,
            }
            schema.setdefault("required", [])
            for def_schema in schema.get("$defs", {}).values():
                if isinstance(def_schema, dict) and def_schema.get("type") == "object":
                    def_schema["additionalProperties"] = False
            return schema

else:

    class _StdlibSchemaMeta(type):
        """Metaclass collecting type annotations and _FieldSpec constraints."""

        def __new__(
            mcs,
            name: str,
            bases: tuple[type, ...],
            namespace: dict[str, Any],
        ) -> type:
            cls = super().__new__(mcs, name, bases, namespace)
            if name == "StrictSchemaModel":
                return cls

            merged_annotations: dict[str, Any] = {}
            merged_specs: dict[str, _FieldSpec] = {}
            for base in reversed(cls.__mro__):
                base_ann = getattr(base, "__annotations__", {})
                for field_name, field_type in base_ann.items():
                    if field_name.startswith("_"):
                        continue
                    merged_annotations[field_name] = field_type
                    attr = base.__dict__.get(field_name, _MISSING)
                    if isinstance(attr, _FieldSpec):
                        merged_specs[field_name] = attr
                    elif attr is not _MISSING and field_name not in merged_specs:
                        merged_specs[field_name] = _FieldSpec(default=attr)
                    elif field_name not in merged_specs:
                        merged_specs[field_name] = _FieldSpec()

            for field_name, spec in merged_specs.items():
                if isinstance(cls.__dict__.get(field_name), _FieldSpec):
                    delattr(cls, field_name)

            cls._field_types = merged_annotations  # type: ignore[attr-defined]
            cls._field_specs = merged_specs  # type: ignore[attr-defined]
            return cls

    class StrictSchemaModel(metaclass=_StdlibSchemaMeta):
        """Stdlib strict schema base class matching Pydantic v2 API & semantics."""

        _field_types: dict[str, Any] = {}
        _field_specs: dict[str, _FieldSpec] = {}

        def __init__(self, **data: Any) -> None:
            unknown = [k for k in data if k not in self._field_types]
            if unknown:
                raise SchemaValidationError(
                    f"Unexpected extra field(s) in {self.__class__.__name__}: "
                    f"{unknown}. Schema enforces additionalProperties=False.",
                    errors=[
                        {
                            "field": key,
                            "code": "extra_forbidden",
                            "message": f"Extra field '{key}' is not permitted.",
                        }
                        for key in unknown
                    ],
                )

            missing: list[str] = []
            for field_name, spec in self._field_specs.items():
                if field_name not in data and spec.is_required:
                    missing.append(field_name)
            if missing:
                raise SchemaValidationError(
                    f"Missing required field(s) in {self.__class__.__name__}: "
                    f"{missing}.",
                    errors=[
                        {
                            "field": key,
                            "code": "missing",
                            "message": f"Field '{key}' is required.",
                        }
                        for key in missing
                    ],
                )

            for field_name, field_type in self._field_types.items():
                spec = self._field_specs[field_name]
                raw_val = (
                    data[field_name] if field_name in data else spec.get_default()
                )
                checked_val = _check_strict_primitive(field_name, field_type, raw_val)
                _check_field_bounds(field_name, spec, checked_val)
                setattr(self, field_name, checked_val)

            self._validate_domain_rules()

        def _validate_domain_rules(self) -> None:
            """Hook for multi-field and domain-specific invariants."""

        @classmethod
        def model_validate(cls, obj: Any) -> "StrictSchemaModel":
            """Validates a dictionary (or model instance) against this schema."""
            if isinstance(obj, cls):
                return cls(**obj.model_dump())
            if not isinstance(obj, Mapping):
                raise SchemaValidationError(
                    f"{cls.__name__}.model_validate expects a dict "
                    f"(got {type(obj).__name__}).",
                    errors=[
                        {
                            "field": "__root__",
                            "code": "invalid_root_type",
                            "message": "Expected JSON object / dictionary.",
                        }
                    ],
                )
            return cls(**dict(obj))

        def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
            """Serializes the validated model into a plain Python dictionary."""
            result: dict[str, Any] = {}
            for field_name in self._field_types:
                val = getattr(self, field_name)
                if exclude_none and val is None:
                    continue
                if isinstance(val, StrictSchemaModel):
                    result[field_name] = val.model_dump(exclude_none=exclude_none)
                elif isinstance(val, list):
                    result[field_name] = [
                        item.model_dump(exclude_none=exclude_none)
                        if isinstance(item, StrictSchemaModel)
                        else copy.deepcopy(item)
                        for item in val
                    ]
                else:
                    result[field_name] = copy.deepcopy(val)
            return result

        @classmethod
        def model_json_schema(cls) -> dict[str, Any]:
            """Generates a Draft 2020-12 JSON Schema with additionalProperties=False."""
            properties: dict[str, Any] = {}
            required: list[str] = []
            for field_name, field_type in cls._field_types.items():
                spec = cls._field_specs[field_name]
                properties[field_name] = _type_to_json_schema(field_type, spec)
                if spec.is_required:
                    required.append(field_name)

            schema: dict[str, Any] = {
                "$schema": JSON_SCHEMA_DRAFT,
                "title": cls.__name__,
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            }
            if cls.__doc__:
                schema["description"] = cls.__doc__.strip()
            return schema

        def __eq__(self, other: object) -> bool:
            if type(self) is not type(other):
                return False
            return self.model_dump() == other.model_dump()  # type: ignore[attr-defined]

        def __repr__(self) -> str:
            fields_str = ", ".join(
                f"{k}={getattr(self, k)!r}" for k in self._field_types
            )
            return f"{self.__class__.__name__}({fields_str})"


# ---------------------------------------------------------------------------
# 1. Flight Domain Schemas
# ---------------------------------------------------------------------------


class FlightSearchInput(StrictSchemaModel):
    """Input schema for searching commercial flight inventory."""

    origin: str = schema_field(
        description="3-letter uppercase IATA origin airport code (e.g., SFO, JFK).",
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )
    destination: str = schema_field(
        description="3-letter uppercase IATA destination airport code (e.g., NRT, HND, LHR).",
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )
    departure_date: str = schema_field(
        description="Outbound departure date in ISO-8601 format (YYYY-MM-DD).",
        min_length=10,
        max_length=10,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    passengers: int = schema_field(
        default=1,
        description="Number of traveling passengers (1 to 9).",
        ge=1,
        le=9,
    )
    cabin_class: Literal["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"] = (
        schema_field(
            default="ECONOMY",
            description="Requested airline cabin class.",
        )
    )
    max_price_usd: float | None = schema_field(
        default=None,
        description="Optional maximum total fare filter in USD (> 0).",
        gt=0.0,
    )

    def _validate_domain_rules(self) -> None:
        validate_iata_code(self.origin, "origin")
        validate_iata_code(self.destination, "destination")
        if self.origin == self.destination:
            raise SchemaValidationError(
                f"Fields 'origin' and 'destination' must differ (both were '{self.origin}').",
                errors=[
                    {
                        "field": "destination",
                        "code": "identical_airports",
                        "message": "Origin and destination IATA codes cannot be identical.",
                    }
                ],
            )
        validate_iso_date(self.departure_date, "departure_date")


class FlightOption(StrictSchemaModel):
    """Single commercial flight itinerary option returned by flight search."""

    flight_id: str = schema_field(
        description="Unique carrier flight option identifier.",
        min_length=3,
        max_length=64,
    )
    airline: str = schema_field(
        description="Operating commercial carrier name.",
        min_length=2,
        max_length=80,
    )
    origin: str = schema_field(
        description="3-letter uppercase IATA origin airport code.",
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )
    destination: str = schema_field(
        description="3-letter uppercase IATA destination airport code.",
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )
    departure_date: str = schema_field(
        description="Scheduled departure date in YYYY-MM-DD format.",
        min_length=10,
        max_length=10,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    departure_time: str = schema_field(
        description="Local scheduled departure time (HH:MM).",
        min_length=4,
        max_length=10,
    )
    arrival_time: str = schema_field(
        description="Local scheduled arrival time with day offset if applicable.",
        min_length=4,
        max_length=12,
    )
    duration_hours: float = schema_field(
        description="Total gate-to-gate travel duration in hours.",
        gt=0.0,
    )
    stops: int = schema_field(
        description="Number of intermediate stops (0 for non-stop).",
        ge=0,
        le=4,
    )
    cabin_class: Literal["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"] = (
        schema_field(description="Booked airline cabin class.")
    )
    price_per_passenger_usd: float = schema_field(
        description="Fare per passenger in USD including taxes.",
        gt=0.0,
    )
    total_price_usd: float = schema_field(
        description="Total fare across all passengers in USD.",
        gt=0.0,
    )
    refundable: bool = schema_field(
        description="Whether the fare can be refunded prior to departure."
    )
    carbon_kg: float = schema_field(
        description="Estimated CO2 emissions per passenger in kilograms.",
        ge=0.0,
    )

    def _validate_domain_rules(self) -> None:
        validate_iata_code(self.origin, "origin")
        validate_iata_code(self.destination, "destination")
        validate_iso_date(self.departure_date, "departure_date")


class FlightSearchOutput(StrictSchemaModel):
    """Validated output schema for commercial flight search results."""

    status: Literal["success"] = schema_field(
        default="success",
        description="Execution status indicator.",
    )
    origin: str = schema_field(
        description="3-letter uppercase IATA origin code.",
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )
    destination: str = schema_field(
        description="3-letter uppercase IATA destination code.",
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )
    departure_date: str = schema_field(
        description="Departure date in YYYY-MM-DD format.",
        min_length=10,
        max_length=10,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    passengers: int = schema_field(
        description="Number of passengers priced.",
        ge=1,
        le=9,
    )
    cabin_class: Literal["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"] = (
        schema_field(description="Evaluated cabin class.")
    )
    currency: Literal["USD"] = schema_field(
        default="USD",
        description="ISO-4217 pricing currency.",
    )
    options_count: int = schema_field(
        description="Number of matching flight options returned.",
        ge=0,
    )
    flights: list[FlightOption] = schema_field(
        description="List of matching commercial flight options."
    )
    cheapest_total_usd: float | None = schema_field(
        default=None,
        description="Lowest total fare in USD among returned options, or null if none.",
        gt=0.0,
    )
    advisory_note: str = schema_field(
        default="",
        description="Guidance note summarizing filter outcomes.",
    )

    def _validate_domain_rules(self) -> None:
        validate_iata_code(self.origin, "origin")
        validate_iata_code(self.destination, "destination")
        validate_iso_date(self.departure_date, "departure_date")


class FlightHoldInput(StrictSchemaModel):
    """Input schema for placing a flight fare hold reservation."""

    flight_id: str = schema_field(
        description="Identifier of the flight option returned by search_commercial_flights.",
        min_length=3,
        max_length=64,
    )
    traveler_full_name: str = schema_field(
        description="Full legal name of the primary traveler.",
        min_length=2,
        max_length=120,
    )
    contact_email: str = schema_field(
        description="Contact email address for carrier itinerary notifications.",
        min_length=5,
        max_length=254,
        pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    )
    total_fare_usd: float = schema_field(
        description="Total quoted fare in USD (> 0).",
        gt=0.0,
        le=250000.0,
    )
    refundable: bool = schema_field(
        default=True,
        description="Whether the held fare carries full refund eligibility.",
    )

    def _validate_domain_rules(self) -> None:
        if not _EMAIL_RE.fullmatch(self.contact_email):
            raise SchemaValidationError(
                f"Field 'contact_email' must be a valid email address (got {self.contact_email!r}).",
                errors=[
                    {
                        "field": "contact_email",
                        "code": "invalid_email",
                        "message": "Expected valid user@domain.tld email address.",
                    }
                ],
            )


class FlightHoldOutput(StrictSchemaModel):
    """Output schema for a placed flight fare hold."""

    status: Literal["success"] = schema_field(
        default="success",
        description="Execution status indicator.",
    )
    hold_status: Literal["HELD"] = schema_field(
        default="HELD",
        description="Carrier hold reservation state.",
    )
    hold_reference: str = schema_field(
        description="Unique carrier hold confirmation code.",
        min_length=4,
        max_length=64,
    )
    flight_id: str = schema_field(
        description="Held flight option identifier.",
        min_length=3,
        max_length=64,
    )
    traveler_full_name: str = schema_field(
        description="Primary traveler name recorded on the hold.",
        min_length=2,
        max_length=120,
    )
    contact_email_masked: str = schema_field(
        description="Redacted/masked contact email representation for safe logging.",
        min_length=3,
        max_length=120,
    )
    total_fare_usd: float = schema_field(
        description="Locked total fare in USD.",
        gt=0.0,
    )
    currency: Literal["USD"] = schema_field(
        default="USD",
        description="ISO-4217 currency code.",
    )
    refundable: bool = schema_field(
        description="Whether the held reservation is refundable."
    )
    expires_in_hours: int = schema_field(
        default=24,
        description="Hours before the carrier fare hold expires if unpaid.",
        ge=1,
        le=168,
    )
    requires_payment_authorization: bool = schema_field(
        description="Whether explicit payment authorization is required to ticket."
    )


# ---------------------------------------------------------------------------
# 2. Lodging & Accommodation Schemas
# ---------------------------------------------------------------------------


class AccommodationSearchInput(StrictSchemaModel):
    """Input schema for searching safety-verified accommodations."""

    city: str = schema_field(
        description="Destination city name (e.g., Tokyo, Kyoto, Paris).",
        min_length=2,
        max_length=100,
    )
    check_in_date: str = schema_field(
        description="Check-in date in ISO-8601 format (YYYY-MM-DD).",
        min_length=10,
        max_length=10,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    check_out_date: str = schema_field(
        description="Check-out date in ISO-8601 format (YYYY-MM-DD), strictly after check_in_date.",
        min_length=10,
        max_length=10,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    guests: int = schema_field(
        default=1,
        description="Number of adult/child guests occupying the booking (1 to 16).",
        ge=1,
        le=16,
    )
    max_nightly_rate_usd: float | None = schema_field(
        default=None,
        description="Optional ceiling on nightly rate in USD (> 0).",
        gt=0.0,
    )

    def _validate_domain_rules(self) -> None:
        check_in = validate_iso_date(self.check_in_date, "check_in_date")
        check_out = validate_iso_date(self.check_out_date, "check_out_date")
        if check_out <= check_in:
            raise SchemaValidationError(
                f"Field 'check_out_date' ({self.check_out_date}) must be strictly "
                f"after 'check_in_date' ({self.check_in_date}).",
                errors=[
                    {
                        "field": "check_out_date",
                        "code": "invalid_date_range",
                        "message": "check_out_date must be at least 1 day after check_in_date.",
                    }
                ],
            )


class AccommodationOption(StrictSchemaModel):
    """Single verified accommodation option returned by lodging search."""

    property_id: str = schema_field(
        description="Unique verified lodging property identifier.",
        min_length=3,
        max_length=64,
    )
    name: str = schema_field(
        description="Property display name.",
        min_length=2,
        max_length=120,
    )
    city: str = schema_field(
        description="City in which the property is located.",
        min_length=2,
        max_length=100,
    )
    neighborhood: str = schema_field(
        description="District or neighborhood name.",
        min_length=2,
        max_length=100,
    )
    star_rating: float = schema_field(
        description="Verified hospitality rating from 1.0 to 5.0.",
        ge=1.0,
        le=5.0,
    )
    nightly_rate_usd: float = schema_field(
        description="Nightly room rate in USD.",
        gt=0.0,
    )
    total_stay_usd: float = schema_field(
        description="Total stay cost in USD including estimated taxes/fees.",
        gt=0.0,
    )
    nights: int = schema_field(
        description="Length of stay in nights.",
        ge=1,
        le=365,
    )
    refundable: bool = schema_field(
        description="Whether free cancellation is offered before check-in."
    )
    verified_safety_badge: bool = schema_field(
        default=True,
        description="Whether property passed 24/7 security and fire inspection.",
    )
    amenities: list[str] = schema_field(
        description="Included property amenities (e.g., Wi-Fi, Breakfast, Transit Access)."
    )


class AccommodationSearchOutput(StrictSchemaModel):
    """Validated output schema for accommodation search results."""

    status: Literal["success"] = schema_field(
        default="success",
        description="Execution status indicator.",
    )
    city: str = schema_field(
        description="Searched destination city.",
        min_length=2,
        max_length=100,
    )
    check_in_date: str = schema_field(
        description="Check-in date in YYYY-MM-DD format.",
        min_length=10,
        max_length=10,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    check_out_date: str = schema_field(
        description="Check-out date in YYYY-MM-DD format.",
        min_length=10,
        max_length=10,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    nights: int = schema_field(
        description="Computed number of stay nights.",
        ge=1,
        le=365,
    )
    guests: int = schema_field(
        description="Number of guests.",
        ge=1,
        le=16,
    )
    currency: Literal["USD"] = schema_field(
        default="USD",
        description="ISO-4217 pricing currency.",
    )
    options_count: int = schema_field(
        description="Number of matching properties.",
        ge=0,
    )
    accommodations: list[AccommodationOption] = schema_field(
        description="Matching verified accommodation options."
    )
    lowest_total_stay_usd: float | None = schema_field(
        default=None,
        description="Lowest total stay price in USD among returned options, or null.",
        gt=0.0,
    )

    def _validate_domain_rules(self) -> None:
        check_in = validate_iso_date(self.check_in_date, "check_in_date")
        check_out = validate_iso_date(self.check_out_date, "check_out_date")
        if check_out <= check_in:
            raise SchemaValidationError(
                "check_out_date must be strictly after check_in_date."
            )


class LodgingReservationInput(StrictSchemaModel):
    """Input schema for reserving a verified accommodation property."""

    property_id: str = schema_field(
        description="Identifier of the property from search_verified_accommodations.",
        min_length=3,
        max_length=64,
    )
    guest_full_name: str = schema_field(
        description="Primary registered guest full name.",
        min_length=2,
        max_length=120,
    )
    nights: int = schema_field(
        description="Number of nights to reserve (1 to 90).",
        ge=1,
        le=90,
    )
    nightly_rate_usd: float = schema_field(
        description="Quoted nightly rate in USD (> 0).",
        gt=0.0,
        le=50000.0,
    )
    refundable: bool = schema_field(
        default=True,
        description="Whether to book under a refundable cancellation tariff.",
    )


class LodgingReservationOutput(StrictSchemaModel):
    """Output schema for a confirmed lodging reservation."""

    status: Literal["success"] = schema_field(
        default="success",
        description="Execution status indicator.",
    )
    reservation_status: Literal["RESERVED"] = schema_field(
        default="RESERVED",
        description="Lodging booking state.",
    )
    reservation_id: str = schema_field(
        description="Unique lodging reservation reference code.",
        min_length=4,
        max_length=64,
    )
    property_id: str = schema_field(
        description="Reserved lodging property identifier.",
        min_length=3,
        max_length=64,
    )
    guest_full_name: str = schema_field(
        description="Registered primary guest full name.",
        min_length=2,
        max_length=120,
    )
    nights: int = schema_field(
        description="Reserved number of nights.",
        ge=1,
        le=90,
    )
    nightly_rate_usd: float = schema_field(
        description="Base nightly rate in USD.",
        gt=0.0,
    )
    taxes_and_fees_usd: float = schema_field(
        description="Hospitality taxes and service fees in USD.",
        ge=0.0,
    )
    total_cost_usd: float = schema_field(
        description="Total stay commitment in USD including taxes and fees.",
        gt=0.0,
    )
    currency: Literal["USD"] = schema_field(
        default="USD",
        description="ISO-4217 currency code.",
    )
    refundable: bool = schema_field(
        description="Whether free cancellation applies."
    )
    cancellation_policy: str = schema_field(
        description="Human-readable cancellation policy terms.",
        min_length=5,
        max_length=300,
    )


# ---------------------------------------------------------------------------
# 3. Budget, FX & High-Value Payment Schemas
# ---------------------------------------------------------------------------


class BudgetAllocationInput(StrictSchemaModel):
    """Input schema for calculating itemized trip budget allocations."""

    total_budget_usd: float = schema_field(
        description="Traveler's total trip budget ceiling in USD (> 0).",
        gt=0.0,
        le=1000000.0,
    )
    duration_days: int = schema_field(
        description="Total trip duration in days (1 to 365).",
        ge=1,
        le=365,
    )
    travelers: int = schema_field(
        default=1,
        description="Number of travelers sharing or scaling the trip budget (1 to 50).",
        ge=1,
        le=50,
    )
    travel_style: Literal["BUDGET", "MODERATE", "LUXURY"] = schema_field(
        default="MODERATE",
        description="Desired spending tier.",
    )
    estimated_flight_usd: float = schema_field(
        default=0.0,
        description="Known or estimated total airfare in USD (>= 0).",
        ge=0.0,
    )
    estimated_lodging_usd: float = schema_field(
        default=0.0,
        description="Known or estimated total lodging cost in USD (>= 0).",
        ge=0.0,
    )


class BudgetAllocationOutput(StrictSchemaModel):
    """Output schema for itemized trip budget allocation and feasibility check."""

    status: Literal["success"] = schema_field(
        default="success",
        description="Execution status indicator.",
    )
    total_budget_usd: float = schema_field(
        description="Traveler budget ceiling in USD.",
        gt=0.0,
    )
    duration_days: int = schema_field(
        description="Trip duration in days.",
        ge=1,
    )
    travelers: int = schema_field(
        description="Number of travelers.",
        ge=1,
    )
    travel_style: Literal["BUDGET", "MODERATE", "LUXURY"] = schema_field(
        description="Applied travel style tier."
    )
    within_budget: bool = schema_field(
        description="True when projected_total_usd <= total_budget_usd."
    )
    allocated_flights_usd: float = schema_field(
        description="Allocated or estimated airfare budget in USD.",
        ge=0.0,
    )
    allocated_lodging_usd: float = schema_field(
        description="Allocated or estimated lodging budget in USD.",
        ge=0.0,
    )
    allocated_dining_usd: float = schema_field(
        description="Allocated food and dining budget in USD.",
        ge=0.0,
    )
    allocated_activities_usd: float = schema_field(
        description="Allocated tours, museum tickets, and experiences in USD.",
        ge=0.0,
    )
    allocated_local_transit_usd: float = schema_field(
        description="Allocated rail, metro, and airport transfer budget in USD.",
        ge=0.0,
    )
    contingency_reserve_usd: float = schema_field(
        description="Reserved emergency buffer in USD.",
        ge=0.0,
    )
    projected_total_usd: float = schema_field(
        description="Sum of all allocated categories in USD.",
        gt=0.0,
    )
    remaining_buffer_usd: float = schema_field(
        description="total_budget_usd minus projected_total_usd (negative when over budget)."
    )
    daily_per_person_allowance_usd: float = schema_field(
        description="Available daily per-traveler ground spend (dining + activities + transit).",
        ge=0.0,
    )
    recommendations: list[str] = schema_field(
        description="Actionable financial optimization notes.",
        min_length=1,
    )


class CurrencyConversionInput(StrictSchemaModel):
    """Input schema for converting monetary amounts between ISO-4217 currencies."""

    amount: float = schema_field(
        description="Monetary amount to convert (> 0).",
        gt=0.0,
        le=100000000.0,
    )
    from_currency: str = schema_field(
        description="Source 3-letter uppercase ISO-4217 currency code (e.g., USD).",
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )
    to_currency: str = schema_field(
        description="Target 3-letter uppercase ISO-4217 currency code (e.g., JPY, EUR).",
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )

    def _validate_domain_rules(self) -> None:
        validate_currency_code(self.from_currency, "from_currency")
        validate_currency_code(self.to_currency, "to_currency")


class CurrencyConversionOutput(StrictSchemaModel):
    """Output schema for currency exchange rate conversion."""

    status: Literal["success"] = schema_field(
        default="success",
        description="Execution status indicator.",
    )
    amount: float = schema_field(
        description="Original amount in source currency.",
        gt=0.0,
    )
    from_currency: str = schema_field(
        description="Source 3-letter ISO-4217 currency code.",
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )
    to_currency: str = schema_field(
        description="Target 3-letter ISO-4217 currency code.",
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )
    exchange_rate: float = schema_field(
        description="Applied unit conversion rate (1 from_currency -> exchange_rate to_currency).",
        gt=0.0,
    )
    converted_amount: float = schema_field(
        description="Converted amount in target currency.",
        gt=0.0,
    )
    rate_source: str = schema_field(
        default="reference_treasury_fx_table",
        description="Identifier of the reference FX table.",
    )

    def _validate_domain_rules(self) -> None:
        validate_currency_code(self.from_currency, "from_currency")
        validate_currency_code(self.to_currency, "to_currency")


class PaymentAuthorizationInput(StrictSchemaModel):
    """Input schema for authorizing a travel payment via Secret Manager gateway credentials."""

    booking_reference: str = schema_field(
        description="Reservation or hold reference being settled.",
        min_length=3,
        max_length=64,
    )
    amount_usd: float = schema_field(
        description="Payment authorization amount in USD (> 0).",
        gt=0.0,
        le=500000.0,
    )
    currency: str = schema_field(
        default="USD",
        description="Settlement 3-letter uppercase ISO-4217 currency code.",
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )
    merchant_name: str = schema_field(
        default="TravelPartner",
        description="Verified travel merchant or carrier name.",
        min_length=2,
        max_length=100,
    )
    payment_token_secret_id: str = schema_field(
        default="travel-payment-gateway-token",
        description="Google Cloud Secret Manager secret ID storing the payment gateway token.",
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
    )

    def _validate_domain_rules(self) -> None:
        validate_currency_code(self.currency, "currency")
        if not _SECRET_ID_RE.fullmatch(self.payment_token_secret_id):
            raise SchemaValidationError(
                "Field 'payment_token_secret_id' must contain only alphanumeric, "
                "hyphen, or underscore characters (3-128 chars)."
            )


class PaymentAuthorizationOutput(StrictSchemaModel):
    """Output schema for a completed travel payment authorization."""

    status: Literal["success"] = schema_field(
        default="success",
        description="Execution status indicator.",
    )
    authorization_status: Literal["AUTHORIZED"] = schema_field(
        default="AUTHORIZED",
        description="Gateway settlement authorization state.",
    )
    transaction_id: str = schema_field(
        description="Unique payment authorization transaction identifier.",
        min_length=6,
        max_length=64,
    )
    booking_reference: str = schema_field(
        description="Associated booking or hold reference.",
        min_length=3,
        max_length=64,
    )
    amount_usd: float = schema_field(
        description="Authorized amount in USD.",
        gt=0.0,
    )
    currency: str = schema_field(
        description="3-letter ISO-4217 settlement currency.",
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )
    merchant_name: str = schema_field(
        description="Merchant receiving authorization.",
        min_length=2,
        max_length=100,
    )
    high_value_flagged: bool = schema_field(
        description="True when amount_usd exceeds high_value_payment_threshold_usd."
    )
    gateway_key_fingerprint: str = schema_field(
        description="Non-reversible SHA-256 fingerprint prefix of the gateway secret.",
        min_length=8,
        max_length=32,
    )
    secret_source_id: str = schema_field(
        description="Secret Manager identifier used to authenticate with gateway.",
        min_length=3,
        max_length=128,
    )

    def _validate_domain_rules(self) -> None:
        validate_currency_code(self.currency, "currency")


# ---------------------------------------------------------------------------
# 4. Travel Advisory & Itinerary Persistence Schemas
# ---------------------------------------------------------------------------


class TravelAdvisoryInput(StrictSchemaModel):
    """Input schema for retrieving destination safety and entry advisories."""

    country_code: str = schema_field(
        description="2-letter or 3-letter uppercase ISO-3166 country code (e.g., JP, JPN, FR, US).",
        min_length=2,
        max_length=3,
        pattern=r"^[A-Z]{2,3}$",
    )
    city: str = schema_field(
        description="Target city within the country (e.g., Tokyo, Kyoto, Paris).",
        min_length=2,
        max_length=100,
    )

    def _validate_domain_rules(self) -> None:
        if not _COUNTRY_CODE_RE.fullmatch(self.country_code):
            raise SchemaValidationError(
                f"Field 'country_code' must be a 2- or 3-letter uppercase ISO-3166 "
                f"country code (got {self.country_code!r}).",
                errors=[
                    {
                        "field": "country_code",
                        "code": "invalid_country_code",
                        "message": "Expected uppercase ISO-3166 code such as JP, US, FR, GB.",
                    }
                ],
            )
        if self.country_code not in SUPPORTED_COUNTRY_CODES:
            raise SchemaValidationError(
                f"Field 'country_code' ('{self.country_code}') is not a recognized "
                "ISO-3166 country code (examples: JP, JPN, US, USA, FR, GB, IT, SG).",
                errors=[
                    {
                        "field": "country_code",
                        "code": "unrecognized_country_code",
                        "message": f"Unrecognized ISO-3166 country code '{self.country_code}'.",
                    }
                ],
            )


class TravelAdvisoryOutput(StrictSchemaModel):
    """Output schema for destination safety, visa, and advisory lookup."""

    status: Literal["success"] = schema_field(
        default="success",
        description="Execution status indicator.",
    )
    country_code: str = schema_field(
        description="Normalized uppercase ISO-3166 country code.",
        min_length=2,
        max_length=3,
        pattern=r"^[A-Z]{2,3}$",
    )
    city: str = schema_field(
        description="Evaluated destination city.",
        min_length=2,
        max_length=100,
    )
    advisory_level: int = schema_field(
        description="Official safety level from 1 (Normal Precautions) to 4 (Do Not Travel).",
        ge=1,
        le=4,
    )
    advisory_summary: str = schema_field(
        description="Concise summary of current safety posture.",
        min_length=5,
        max_length=400,
    )
    entry_requirements: list[str] = schema_field(
        description="Passport validity, visa, and arrival registration requirements.",
        min_length=1,
    )
    health_and_safety_notes: list[str] = schema_field(
        description="Local health, weather, and neighborhood guidance.",
        min_length=1,
    )
    local_emergency_number: str = schema_field(
        description="Local emergency services telephone number.",
        min_length=2,
        max_length=80,
    )
    recommended_precautions: list[str] = schema_field(
        description="Actionable traveler safety precautions.",
        min_length=1,
    )


class ItineraryPersistenceInput(StrictSchemaModel):
    """Input schema for persisting a finalized and critic-approved trip itinerary."""

    session_id: str = schema_field(
        description="Active conversation session identifier.",
        min_length=1,
        max_length=128,
    )
    traveler_id: str = schema_field(
        description="Unique pseudonymous traveler identifier.",
        min_length=1,
        max_length=128,
    )
    destination: str = schema_field(
        description="Primary destination or multi-city route label.",
        min_length=2,
        max_length=120,
    )
    total_cost_usd: float = schema_field(
        description="Verified total itinerary cost in USD (> 0).",
        gt=0.0,
        le=1000000.0,
    )
    itinerary_items: list[dict[str, Any]] = schema_field(
        description="Non-empty list of structured itinerary segment dictionaries.",
        min_length=1,
    )


class ItineraryPersistenceOutput(StrictSchemaModel):
    """Output schema confirming durable itinerary persistence."""

    status: Literal["success"] = schema_field(
        default="success",
        description="Execution status indicator.",
    )
    persistence_status: Literal["PERSISTED"] = schema_field(
        default="PERSISTED",
        description="Storage commit state.",
    )
    itinerary_id: str = schema_field(
        description="Generated durable itinerary record identifier.",
        min_length=6,
        max_length=64,
    )
    session_id: str = schema_field(
        description="Session identifier associated with the itinerary.",
        min_length=1,
        max_length=128,
    )
    traveler_id: str = schema_field(
        description="Scrubbed traveler identifier.",
        min_length=1,
        max_length=128,
    )
    destination: str = schema_field(
        description="Persisted destination label.",
        min_length=2,
        max_length=120,
    )
    total_cost_usd: float = schema_field(
        description="Total verified itinerary cost in USD.",
        gt=0.0,
    )
    items_count: int = schema_field(
        description="Number of persisted itinerary segments.",
        ge=1,
    )
    checksum_sha256: str = schema_field(
        description="Deterministic SHA-256 integrity digest of persisted items.",
        min_length=16,
        max_length=64,
    )
    summary: str = schema_field(
        description="Confirmation message summarizing persisted itinerary.",
        min_length=5,
        max_length=300,
    )


# ---------------------------------------------------------------------------
# 5. Guided Tool Error Output Schema (Rubric 1.4)
# ---------------------------------------------------------------------------


class ToolErrorOutput(StrictSchemaModel):
    """Structured guided-recovery error payload returned to the LLM on tool errors."""

    status: Literal["error"] = schema_field(
        default="error",
        description="Constant 'error' status distinguishing failures from valid results.",
    )
    error_code: str = schema_field(
        description="Machine-readable error classification code.",
        min_length=2,
        max_length=64,
    )
    message: str = schema_field(
        description="Clear diagnostic description of what failed validation or execution.",
        min_length=3,
        max_length=600,
    )
    recovery_instructions: str = schema_field(
        description="Explicit step-by-step instructions guiding the LLM on exact parameter fixes.",
        min_length=10,
        max_length=800,
    )
    retryable: bool = schema_field(
        default=True,
        description="Whether the LLM should retry after fixing parameters or prerequisites.",
    )
    tool_name: str = schema_field(
        default="",
        description="Name of the domain tool that produced this error response.",
        max_length=100,
    )
    invalid_fields: list[str] = schema_field(
        default_factory=list,
        description="Names of specific arguments that failed schema validation.",
    )


ALL_SCHEMA_MODELS: tuple[type[StrictSchemaModel], ...] = (
    FlightSearchInput,
    FlightOption,
    FlightSearchOutput,
    FlightHoldInput,
    FlightHoldOutput,
    AccommodationSearchInput,
    AccommodationOption,
    AccommodationSearchOutput,
    LodgingReservationInput,
    LodgingReservationOutput,
    BudgetAllocationInput,
    BudgetAllocationOutput,
    CurrencyConversionInput,
    CurrencyConversionOutput,
    PaymentAuthorizationInput,
    PaymentAuthorizationOutput,
    TravelAdvisoryInput,
    TravelAdvisoryOutput,
    ItineraryPersistenceInput,
    ItineraryPersistenceOutput,
    ToolErrorOutput,
)
