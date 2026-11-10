import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import re as _re

_COMMENT_BLOCKS = _re.compile(
    r"<blockquote.*?</blockquote>|<ul class='clist'>.*?</ul>|<section class='oap'>.*?</section>",
    _re.S)


def strip_comment_text(html: str) -> str:
    """Remove the containers that hold employee comment text (quotes, the all-comments
    list, the stored action plan) so wording checks only look at the card's own copy."""
    return _COMMENT_BLOCKS.sub("", html)
