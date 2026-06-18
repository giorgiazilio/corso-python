# ============================================================
# config.py
# ============================================================
# Parameters shared between spike_inspector.py and the notebook.
# Edit here — changes automatically propagate
# to both programs.

from pathlib import Path

FOLDER             = Path("/Users/giorgia/Documents/AntennalMag_DEMO")
BAD_FRAMES_DIR     = FOLDER / "bad_frames"
FEATURES_OUTPUT    = FOLDER / "features_all_videos.csv"

FPS                = 29.97
LIKELIHOOD_THRESH  = 0.8
VELOCITY_THRESHOLD = 1000   # deg/s — suspicious spike threshold
SAVGOL_WINDOW      = 7      # must be an odd number
SAVGOL_POLYORDER   = 3
NAN_INTERP_LIMIT   = 5      # max consecutive frames to interpolate

BODYPARTS = ['HL', 'HR', 'AL', 'ALH', 'AR', 'ARH', 'B', 'F']
