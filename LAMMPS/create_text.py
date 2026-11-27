#!/usr/bin/env python3
"""Fixed texts of the LAMMPS environment page: the intro card (retrieve_welcome_message.sh) and the
thermo tip under the input file picker (retrieve_create_tips.sh). Same look as the other create-page
cards (accel_info.card). No cluster-specific text: availability shows in its own card."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accel_info import card  # noqa: E402

INTRO = (
    "<div class='intro'>"
    "<p>Analyzes your LAMMPS input file, suggests compatible LAMMPS builds and acceleration "
    "strategies, and generates the batch job.</p>"
    "<p class='muted foot'>Issues, suggestions, or no build available for your input? "
    "Use the feedback link in the footer.</p>"
    "</div>"
)

THERMO_TIP = (
    "<div class='muted'>Tip: write thermo output as YAML (<code>thermo_style yaml</code> or "
    "<code>thermo_modify line yaml</code>) to get live charts in Manage.</div>"
)

if __name__ == "__main__":
    print(card(THERMO_TIP if sys.argv[1:] == ["tip"] else INTRO))
