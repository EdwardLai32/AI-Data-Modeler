"""Scan source files for encoding damage.

Agent-written files reach the model as prompt text, so a mis-encoded em-dash is
not cosmetic — it ships garbage characters into every request built from that
string. This checks two distinct failures: bytes that are not valid UTF-8 at
all, and valid-UTF-8 text that already contains mojibake (UTF-8 bytes that were
decoded as latin-1 at some earlier point and re-encoded).
"""

from __future__ import annotations

import pathlib

# Sequences that only appear when UTF-8 was decoded as latin-1/cp1252.
SUSPECT = ("â", "Ã©", "â", "Â ", "â")

ROOTS = ("automl_architect", "tests", "scripts", "examples")


def main() -> int:
    bad_utf8: list[tuple[pathlib.Path, str]] = []
    mojibake: list[tuple[pathlib.Path, str]] = []
    scanned = 0

    this_file = pathlib.Path(__file__).resolve()

    for root_name in ROOTS:
        root = pathlib.Path(root_name)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            # This module necessarily contains the byte sequences it searches for.
            if path.resolve() == this_file:
                continue
            scanned += 1
            raw = path.read_bytes()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                bad_utf8.append((path, str(exc)[:90]))
                continue
            hit = next((s for s in SUSPECT if s in text), None)
            if hit:
                idx = text.find(hit)
                snippet = text[max(0, idx - 45) : idx + 25].replace("\n", " ")
                mojibake.append((path, snippet))

    print(f"files scanned          : {scanned}")
    print(f"not valid UTF-8        : {len(bad_utf8)}")
    for path, err in bad_utf8:
        print(f"    {path}: {err}")
    print(f"files with mojibake    : {len(mojibake)}")
    for path, snippet in mojibake[:15]:
        print(f"    {path}")
        print(f"        ...{snippet}...")

    return 1 if (bad_utf8 or mojibake) else 0


if __name__ == "__main__":
    raise SystemExit(main())
