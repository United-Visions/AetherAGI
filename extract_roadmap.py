#!/usr/bin/env python3
"""
populate_agi_final_fixed.py
Grab content **after** the literal 'Copy' line until the next filename or EOF.
"""

import re, sys
from pathlib import Path

SRC  = Path("add_agi_docs.md")
ROOT = Path("agi_roadmap")

txt = SRC.read_text(encoding="utf-8")

# ------------------------------------------------------------------
# regex: filename → 'Copy' → content → (next filename or EOF)
# ------------------------------------------------------------------
BLOCK_RE = re.compile(
    r"^\s*(?P<filepath>agi_roadmap[/\w\.\-]+)\s*\n"
    r"(?:Markdown|Python)?\s*\n"
    r"Copy\s*\n"
    r"(?P<content>(?:.*\n)*?)"
    r"(?=^\s*agi_roadmap[/\w\.\-]+|\Z)",
    re.MULTILINE,
)

# ------------------------------------------------------------------
# populate
# ------------------------------------------------------------------
written = 0
for m in BLOCK_RE.finditer(txt):
    file_path = m.group("filepath").strip()
    content   = m.group("content").rstrip()

    target = Path(file_path)
    if target.is_dir():
        print(f"⚠️  skip dir: {target}")
        continue
    if not target.exists():
        target.touch()
    target.write_text(content, encoding="utf-8")
    print(f"✅  {target.name:<35}  {len(content):>7} chars")
    written += 1

print(f"🎉  populated {written} files")