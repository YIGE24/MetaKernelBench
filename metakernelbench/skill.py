"""The transfer carrier: a source solo's SKILL.md packaged as the target's attachment, and its digest."""

import hashlib
from pathlib import Path

from metakernelbench.catalog import SKILL_PATH
from metakernelbench.store import ARTIFACTS_NAME, artifact_path

SKILL_HEADER = """## Skill from a prior attempt

This skill was distilled while working on this same problem in a different kernel DSL.

"""

def skill_attachment_text(source_trial_dir: Path) -> str | None:
    artifacts_dir = source_trial_dir / ARTIFACTS_NAME
    if not artifacts_dir.is_dir():
        raise FileNotFoundError(f"{artifacts_dir} is missing, the source trial left no artifacts to distill")
    body_path = artifact_path(source_trial_dir, SKILL_PATH)
    body = body_path.read_text(errors="replace") if body_path.is_file() else ""
    return SKILL_HEADER + body if body.strip() else None

def skill_digest(text: str | None) -> str | None:
    return None if text is None else hashlib.sha256(text.encode()).hexdigest()
