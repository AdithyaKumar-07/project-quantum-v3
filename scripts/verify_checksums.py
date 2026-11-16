from __future__ import annotations
"""Verify the SHA256SUMS.txt manifest from the project root."""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS.txt"


def main() -> int:
    if not MANIFEST.is_file():
        print(f"[FAIL] Missing checksum manifest: {MANIFEST}")
        return 2
    failures: list[str] = []
    checked = 0
    for line_number, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, rel = line.split("  ", 1)
        except ValueError:
            failures.append(f"line {line_number}: malformed manifest entry")
            continue
        path = (ROOT / rel).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError:
            failures.append(f"line {line_number}: path escapes project root: {rel}")
            continue
        if not path.is_file():
            failures.append(f"missing: {rel}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        checked += 1
        if actual.lower() != expected.lower():
            failures.append(f"checksum mismatch: {rel}")
    if failures:
        print("Checksum verification FAILED:")
        for item in failures:
            print(f" - {item}")
        return 1
    print(f"[OK] SHA-256 verified for {checked} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
