"""Enforced the version-sync convention introduced in OVERVIEW.md v4.1:
`Config.WANDB_PROJECT`'s trailing `_vN` suffix tracked OVERVIEW.md's
latest documented MAJOR version (not every point release -- see §21.1 for
the rationale on major-only tracking).

RETIRED (OVERVIEW.md v9.1): `WANDB_PROJECT` was deliberately renamed from
"NI_Review_v8" to "Persistence_Linear_v1" -- a rebrand, not a version
bump. Enforcing equality with OVERVIEW.md's version (then v9) against a
project deliberately restarting its own numbering at 1 would make this
test permanently, correctly-failingly wrong, not catch real drift. Kept
the "has SOME trailing _vN suffix" check (still a real, useful
convention -- every wandb project name should be versioned), dropped the
"and it must equal OVERVIEW.md's version" equality check.
"""
import os
import re
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MOD_DIR = os.path.dirname(_HERE)
if _MOD_DIR not in sys.path:
    sys.path.insert(0, _MOD_DIR)

import train_production_transformer_deep_dive as T  # noqa: E402

OVERVIEW_PATH = os.path.join(_MOD_DIR, "OVERVIEW.md")

# Matches headings like "## 20. v4.0 -- Mac shallow-sweep ..." -- the
# section number is deliberately NOT captured; only the version matters.
_VERSION_HEADING_RE = re.compile(r'^## \d+\.\s+v(\d+)\.(\d+)\b', re.MULTILINE)
_PROJECT_VERSION_RE = re.compile(r'_v(\d+)$')


def latest_documented_version(overview_path=OVERVIEW_PATH):
    """Return (major, minor) of the highest 'vX.Y' section heading found
    in OVERVIEW.md. Takes the max over ALL matches (not just the last one
    in the file), so this is robust to a version section ever being
    inserted out of strict chronological order.
    """
    with open(overview_path) as f:
        text = f.read()
    versions = [(int(maj), int(minr)) for maj, minr in _VERSION_HEADING_RE.findall(text)]
    if not versions:
        raise AssertionError(f"no '## N. vX.Y' heading found in {overview_path}")
    return max(versions)


def wandb_project_major_version(project_name):
    """Return the integer major version suffix of a '..._vN' project
    name, or None if the name has no such suffix at all."""
    m = _PROJECT_VERSION_RE.search(project_name)
    return int(m.group(1)) if m else None


class TestVersionSync(unittest.TestCase):

    def test_overview_md_has_at_least_one_version_heading(self):
        doc_major, doc_minor = latest_documented_version()
        self.assertGreaterEqual(doc_major, 1)

    def test_wandb_project_has_a_version_suffix(self):
        project_major = wandb_project_major_version(T.Config.WANDB_PROJECT)
        self.assertIsNotNone(
            project_major,
            f"Config.WANDB_PROJECT={T.Config.WANDB_PROJECT!r} has no "
            f"trailing '_vN' suffix. See OVERVIEW.md §21.1 for the "
            f"convention this is supposed to follow.")

    # test_wandb_project_major_version_matches_overview_md RETIRED
    # (OVERVIEW.md v9.1): see this module's docstring -- WANDB_PROJECT
    # was deliberately renamed/rebranded ("Persistence_Linear_v1"),
    # restarting its own version count independent of OVERVIEW.md's.


if __name__ == "__main__":
    unittest.main(verbosity=2)
