import math
import re

_PATTERNS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
     "[REDACTED_PRIVATE_KEY]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[REDACTED_TOKEN]"),
    (re.compile(r"\bsk[-_](?:live|test|proj)[-_][A-Za-z0-9_\-]{6,}\b"), "[REDACTED_TOKEN]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"), "[REDACTED_TOKEN]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]{16,}=*"), "Bearer [REDACTED]"),
    (re.compile(r"(?i)\b([\w-]*(?:api[_-]?key|token|secret|password|passwd|authorization)[\w-]*)(\s*[:=]\s*)([^\s,;]+)"),
     r"\1\2[REDACTED]"),
]

_CANDIDATE_TOKEN = re.compile(r"\S{28,}")


def _entropy(s: str) -> float:
    counts = {c: s.count(c) for c in set(s)}
    return -sum((n / len(s)) * math.log2(n / len(s)) for n in counts.values())


# Note: this entropy fallback will sometimes redact benign high-entropy strings
# (e.g. base64-encoded prose, hashes, or random-looking identifiers). That's an
# intentional, conservative tradeoff for this module: since redact() runs before
# an LLM ever sees the transcript, a false negative (a real secret slipping
# through) is a security failure, while a false positive (over-redacting
# harmless text) only costs some readability. When in doubt, redact.
def _looks_like_secret(token: str) -> bool:
    if "/" in token or "\\" in token:
        return False  # paths and URLs
    has_digit = any(c.isdigit() for c in token)
    has_alpha = any(c.isalpha() for c in token)
    return has_digit and has_alpha and _entropy(token) > 4.0


def redact(text: str) -> str:
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return _CANDIDATE_TOKEN.sub(
        lambda m: "[REDACTED_HIGH_ENTROPY]" if _looks_like_secret(m.group()) else m.group(),
        text,
    )
