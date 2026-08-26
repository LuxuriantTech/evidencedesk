import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RedactionResult:
    text: str
    counts: dict[str, int]


_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "email",
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        "[EMAIL REDACTED]",
    ),
    (
        "phone",
        re.compile(r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\d[ .-]?){8,11}\d(?!\w)"),
        "[PHONE REDACTED]",
    ),
    (
        "identifier",
        re.compile(r"\bSYN-ID-[A-Z0-9-]{4,}\b", re.IGNORECASE),
        "[IDENTIFIER REDACTED]",
    ),
)


def redact_pii(text: str) -> RedactionResult:
    counts: dict[str, int] = {}
    redacted = text
    for name, pattern, replacement in _PATTERNS:
        redacted, count = pattern.subn(replacement, redacted)
        if count:
            counts[name] = count
    return RedactionResult(text=redacted, counts=counts)
