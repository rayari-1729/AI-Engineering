"""
PII pseudonymizer — companion code for Chapter 9 (Q20).

Detects structured PII (emails, Indian phone numbers, Aadhaar with Verhoeff
checksum, PAN, payment cards with Luhn checksum, API keys), replaces each value
with a typed, consistent placeholder like <EMAIL_1>, and restores placeholders
in model output.

Regex + checksums only: names, addresses and contextual sensitive facts need an
NER / classifier layer (e.g. Microsoft Presidio) on top of this.

Run:  python pii_pseudonymizer.py
"""
import re
from collections import defaultdict

# --- checksums -------------------------------------------------------------
_D = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],
      [3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
      [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],
      [9,8,7,6,5,4,3,2,1,0]]
_P = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],
      [8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
      [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]

def verhoeff_ok(digits: str) -> bool:          # Aadhaar check digit
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0

def luhn_ok(digits: str) -> bool:              # payment cards
    nums = [int(d) for d in reversed(digits)]
    total = sum(nums[0::2]) + sum(sum(divmod(2 * d, 10)) for d in nums[1::2])
    return total % 10 == 0

only_digits = lambda s: re.sub(r"\D", "", s)

# --- detectors: (entity, regex, validator) — order matters ---------------
DETECTORS = [
    ("API_KEY",    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16})\b"), None),
    ("EMAIL",      re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), None),
    ("IN_AADHAAR", re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)"),
                   lambda m: verhoeff_ok(only_digits(m))),
    ("CARD",       re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
                   lambda m: 13 <= len(only_digits(m)) <= 19 and luhn_ok(only_digits(m))),
    ("IN_PAN",     re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), None),
    ("PHONE",      re.compile(r"(?<!\d)(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?!\d)"), None),
]

class Pseudonymizer:
    """Replace PII with typed, consistent placeholders; restore them in model output."""

    def __init__(self):
        self.forward: dict[str, str] = {}      # original -> token
        self.reverse: dict[str, str] = {}      # token -> original
        self.counts = defaultdict(int)

    def _token(self, entity: str, value: str) -> str:
        if value not in self.forward:
            self.counts[entity] += 1
            tok = f"<{entity}_{self.counts[entity]}>"
            self.forward[value], self.reverse[tok] = tok, value
        return self.forward[value]

    def sanitize(self, text: str) -> str:
        for entity, rx, valid in DETECTORS:
            text = rx.sub(lambda m: self._token(entity, m.group())
                          if (valid is None or valid(m.group())) else m.group(), text)
        return text

    def restore(self, text: str) -> str:
        return re.sub(r"<[A-Z_]+_\d+>", lambda m: self.reverse.get(m.group(), m.group()), text)


if __name__ == "__main__":
    p = Pseudonymizer()
    text = ("Hi, I'm Priya (priya.sharma@example.com, +91 98765 43210). "
            "Aadhaar 2345 6789 0124, PAN ABCDE1234F, card 4111 1111 1111 1111. "
            "Order id 234567890125.")
    clean = p.sanitize(text)
    print("SANITIZED:", clean)
    print("RESTORED: ", p.restore("Thanks <EMAIL_1>, we'll call <PHONE_1>."))
    assert "<IN_AADHAAR_1>" in clean and "234567890125" in clean   # checksum works
