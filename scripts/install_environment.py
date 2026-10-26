#!/usr/bin/env python3
"""Retrieve the official engine at the exact experimental revision.

Upstream's flit metadata does not locate its module in this revision. Using its
unmodified src tree avoids that packaging issue and optional TensorFlow agents.
"""
from pathlib import Path
import subprocess
import sys

REVISION = "bf882e55eccd537d189f8b99fef1ad104339c92b"
ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / ".deps" / "ChefsHatGYM"
PATCH = ROOT / "docs" / "upstream_joker_fix.patch"

def main():
    TARGET.parent.mkdir(exist_ok=True)
    if not TARGET.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "https://github.com/pablovin/ChefsHatGYM.git", str(TARGET)], check=True)
    subprocess.run(["git", "-C", str(TARGET), "fetch", "origin", REVISION], check=True)
    subprocess.run(["git", "-C", str(TARGET), "checkout", "--detach", REVISION], check=True)
    revision = subprocess.check_output(["git", "-C", str(TARGET), "rev-parse", "HEAD"], text=True).strip()
    assert revision == REVISION
    check = subprocess.run(["git", "-C", str(TARGET), "apply", "--check", str(PATCH)], capture_output=True)
    if check.returncode == 0:
        subprocess.run(["git", "-C", str(TARGET), "apply", str(PATCH)], check=True)
    else:
        subprocess.run(["git", "-C", str(TARGET), "apply", "--reverse", "--check", str(PATCH)], check=True)
    sys.path.insert(0, str(TARGET / "src"))
    from core.game_env.game import Game
    from core.utils.rules import get_high_level_actions
    assert len(get_high_level_actions()) == 200
    print(f"Official engine ready: {Game.__module__}; revision {revision}")
    print("Applied documented compatibility patch: two-joker single discard (200 actions unchanged).")
    print("See upstream LICENSE and README use/distribution conditions before use.")

if __name__ == "__main__":
    main()
