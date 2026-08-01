"""
make_user_guide.py — regenerate docs/USER_GUIDE.md from qt_help.py.

qt_help.py (the in-app F1 guide + About dialog) is the single source of the
user documentation; this script converts its restricted HTML subset to
markdown so the repo copy can never drift from what the app shows.

Run from the repo root after editing core/qt_help.py:
    python tools/make_user_guide.py
"""
from __future__ import annotations

import re

ENTITIES = {
    "&mdash;": "—", "&ndash;": "–", "&amp;": "&", "&nbsp;": " ",
    "&middot;": "·", "&rarr;": "→", "&larr;": "←", "&asymp;": "≈",
    "&sup2;": "²", "&theta;": "θ", "&lambda;": "λ", "&mu;": "μ",
    "&Delta;": "Δ", "&delta;": "δ", "&eta;": "η", "&sigma;": "σ",
    "&Aring;": "Å", "&deg;": "°", "&plusmn;": "±", "&times;": "×",
    "&gt;": ">", "&lt;": "<", "&#39;": "'", "&quot;": '"',
    "&rsaquo;": "›", "&lsaquo;": "‹", "&hellip;": "…",
}


def html_to_md(html: str) -> str:
    s = html
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r"<div[^>]*>|</div>", "", s)
    s = re.sub(r"<h1[^>]*>\s*", "\n# ", s)
    s = re.sub(r"<h2[^>]*>\s*", "\n## ", s)
    s = re.sub(r"<h3[^>]*>\s*", "\n### ", s)
    s = re.sub(r"\s*</h[123]>", "\n", s)
    s = re.sub(r"<(ul|ol)[^>]*>", "\n", s)
    s = re.sub(r"</(ul|ol)>", "\n", s)
    s = re.sub(r"<li[^>]*>\s*", "\n- ", s)
    s = re.sub(r"\s*</li>", "", s)
    s = re.sub(r"<p[^>]*>\s*", "\n", s)
    s = re.sub(r"\s*</p>", "\n", s)
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"</?b>", "**", s)
    s = re.sub(r"</?i>", "*", s)
    s = re.sub(r"</?code>", "`", s)
    s = re.sub(r"<[^>]+>", "", s)  # anything left (spans, style carriers)
    for ent, ch in ENTITIES.items():
        s = s.replace(ent, ch)
    # collapse intra-paragraph hard wraps the HTML never rendered
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip() + "\n"


def main() -> None:
    import os
    import sys
    # this file lives in tools/, so add repo root to sys.path to import core.qt_help
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import core.qt_help as qt_help

    parts = [
        "<!-- Generated from qt_help.py by make_user_guide.py. Edit qt_help.py, then regenerate. -->\n",
        html_to_md(qt_help.HELP_HTML),
        "\n---\n",
        html_to_md(qt_help.ABOUT_HTML),
    ]
    out = "\n".join(parts)
    with open("docs/USER_GUIDE.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    print(f"docs/USER_GUIDE.md regenerated ({len(out.splitlines())} lines).")


if __name__ == "__main__":
    main()
