from __future__ import annotations

import re
from collections import Counter

_FULLWIDTH = str.maketrans(
    {
        "Ａ": "A",
        "Ｂ": "B",
        "Ｃ": "C",
        "Ｄ": "D",
        "：": ":",
        "（": "(",
        "）": ")",
    }
)

_PATTERNS = [
    re.compile(r"(?:最终答案|正确答案|答案)\s*(?:是|为|选|选择)?\s*:?\s*([ABCD])\b"),
    re.compile(r"(?:选择|选)\s*([ABCD])\b"),
    re.compile(r"^\s*([ABCD])\s*[。.]?\s*$"),
    re.compile(r"^\s*([ABCD])\s*[)）].*$"),
]


def last_nonempty_line(text: str) -> str:
    for line in reversed(text.replace("\r\n", "\n").split("\n")):
        if line.strip():
            return line.strip()
    return ""


def parse_answer(text: str) -> str | None:
    """Extract the final option from the last non-empty line only.

    The rule is intentionally conservative: letters mentioned inside reasoning
    are never treated as the final answer. If the last line does not contain a
    uniquely identifiable option, the answer is invalid.
    """
    if not text:
        return None
    line = last_nonempty_line(text).translate(_FULLWIDTH).strip()
    for pattern in _PATTERNS:
        match = pattern.search(line)
        if match:
            return match.group(1).upper()
    return None


def majority_vote(answers: list[str | None]) -> str | None:
    """Majority vote with a deterministic tie-break: earliest valid sample wins."""
    valid = [answer for answer in answers if answer in {"A", "B", "C", "D"}]
    if not valid:
        return None
    counts = Counter(valid)
    best_count = max(counts.values())
    for answer in valid:
        if counts[answer] == best_count:
            return answer
    return None
