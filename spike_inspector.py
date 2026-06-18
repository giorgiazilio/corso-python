"""
spike_inspector.py
------------------
Interactive video player for visual inspection of video frames containing
suspicious velocity spikes in antennal movement. Run from terminal using:

    python spike_inspector.py /path/to/video.mp4

If no video path argument is supplied, the script falls back to the default
video defined inside the global configuration parameters.

KEYBOARD CONTROLS:
    N  /  ->   Jump to next suspicious frame
    P  /  <-   Jump to previous suspicious frame
    .          Advance forward by 1 single frame (+1)
    ,          Step backward by 1 single frame (-1)
    M          Mark / Unmark the current frame as "bad tracking"
    Q          Quit inspection and save results
"""

import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from config import (FOLDER, BAD_FRAMES_DIR, FPS, LIKELIHOOD_THRESH,
                    VELOCITY_THRESHOLD, BODYPARTS)

# ============================================================
# 1. LOCAL PATH AND ARGUMENT HANDLING
# ============================================================

# Check if the user provided a specific video file path via the command line terminal
if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]
else:
    # Default fallback video if no argument is passed
    VIDEO_PATH = str(FOLDER / "antenna.up.mag.2.mp4")

video_stem = Path(VIDEO_PATH).stem
# Dynamically search the tracking folder for any CSV matching this video's base name
matching   = sorted(FOLDER.glob(f"{video_stem}*.csv"))

if len(matching) == 0:
    raise FileNotFoundError(
        f"No matching CSV tracking file found for '{video_stem}' inside {FOLDER}\n"
        f"Please verify that the DeepLabCut CSV file is placed in the same directory as the video."
    )
elif len(matching) > 1:
    # Safety feature: if multiple matching coordinate files exist, ask the student to pick one
    print(f"Multiple tracking CSV files found for this video:")
    for i, f in enumerate(matching):
        print(f"  {i}  {f.name}")
    idx      = int(input("Which file would you like to load? Enter the corresponding index number: "))
    csv_path = matching[idx]
else:
    csv_path = matching[0]

# Standardize output file destination name based on the analyzed video file name
OUTPUT_FILE = BAD_FRAMES_DIR / (video_stem + "_bad_frames.csv")

print(f"Video Source : {VIDEO_PATH}")
print(f"CSV Tracking : {csv_path.name}")
print(f"Output File  : {OUTPUT_FILE}")

# ============================================================
# 2. DESIGN CONSTANTS (BGR COLOR MAPPING FOR OPENCV)
# ============================================================

# OpenCV uses BGR (Blue, Green, Red) format instead of the standard RGB format
COLORS = {
    'F'  : (0,   200, 255),  # Front (Head)
    'B'  : (0,   200, 255),  # Back (Head)
    'AL' : (255, 200,   0),  # Antenna Left Tip
    'ALH': (200, 140,   0),  # Antenna Left Base/Head junction
    'AR' : (0,   255, 140),  # Antenna Right Tip
    'ARH': (0,   180,  80),  # Antenna Right Base/Head junction
    'HL' : (200,  60, 255),  # Left Head Reference
    'HR' : (200,  60, 255),  # Right Head Reference
}

# Anatomical skeleton connections to draw between points
CONNECTIONS = [
    ('B',   'F'),    # Head midline vector
    ('ALH', 'AL'),   # Left antenna segment
    ('ARH', 'AR'),   # Right antenna segment
]

# ============================================================
# 3. KINEMATIC COMPUTATION FUNCTIONS
# ============================================================

def load_data():
    """Loads the DeepLabCut coordinates DataFrame containing a 3-level MultiIndex hierarchy."""
    return pd.read_csv(csv_path, header=[0,1,2])

def get_coords(df, bp):
    """Extracts raw x, y coordinates and confidence scores (likelihood) for a single bodypart."""
    x = df.xs((bp, 'x'),          level=(1,2), axis=1).squeeze().to_numpy()
    y = df.xs((bp, 'y'),          level=(1,2), axis=1).squeeze().to_numpy()
    l = df.xs((bp, 'likelihood'), level=(1,2), axis=1).squeeze().to_numpy()
    return x, y, l

def angle_between_vectors(v1x, v1y, v2x, v2y):
    """Calculates the directional geometric angle in degrees between two 2D vectors."""
    dot   = v1x * v2x + v1y * v2y
    norm1 = np.sqrt(v1x**2 + v1y**2)
    norm2 = np.sqrt(v2x**2 + v2y**2)
    # Prevent system crashes by suppressing warnings during zero-division or null magnitudes
    with np.errstate(invalid='ignore', divide='ignore'):
        cos_a = np.where((norm1 > 0) & (norm2 > 0),
                         dot / (norm1 * norm2), np.nan)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

def compute_spikes(df):
    """
    Applies a likelihood filter threshold and detects unnatural velocity anomalies 
    (spikes) that suggest tracking errors.
    """
    df_f = df.copy()
    # Mask any coordinates falling below the acceptable confidence tracking threshold with NaN
    for bp in BODYPARTS:
        lk  = df_f.xs((bp, 'likelihood'), level=(1,2), axis=1).squeeze()
        bad = lk < LIKELIHOOD_THRESH
        df_f.loc[bad, (slice(None), bp, 'x')] = np.nan
        df_f.loc[bad, (slice(None), bp, 'y')] = np.nan

    def coord(bp, axis):
        return df_f.xs((bp, axis), level=(1,2), axis=1).squeeze().to_numpy()

    # Define head vectors as baseline orientation reference
    head_dx = coord('F','x') - coord('B','x')
    head_dy = coord('F','y') - coord('B','y')

    # Compute raw absolute angular position sequences relative to head axis
    left_angle  = angle_between_vectors(
        head_dx, head_dy,
        coord('AL','x') - coord('ALH','x'),
        coord('AL','y') - coord('ALH','y'))
    right_angle = angle_between_vectors(
        head_dx, head_dy,
        coord('AR','x') - coord('ARH','x'),
        coord('AR','y') - coord('ARH','y'))

    # Calculate instantaneous angular velocity profiles (differential changes scaled by FPS)
    left_vel  = np.diff(left_angle)  * FPS
    right_vel = np.diff(right_angle) * FPS

    # Filter index arrays identifying where instantaneous velocity exceeds expected biology thresholds
    left_spikes  = np.where(np.abs(left_vel)  > VELOCITY_THRESHOLD)[0]
    right_spikes = np.where(np.abs(right_vel) > VELOCITY_THRESHOLD)[0]

    return left_vel, right_vel, left_spikes, right_spikes

# ============================================================
# 4. COMPUTER VISION DRAWING AND HUD OVERLAY FUNCTIONS
# ============================================================

def draw_frame(frame, frame_n, coords_raw, left_vel, right_vel,
               left_spikes_set, right_spikes_set,
               spike_idx, total_spikes, total_frames,
               marked_frames):
    """Draws tracking skeletons, point confidence alerts, and telemetry logs directly on video frames."""

    # Draw a distinct thick red border if the frame is marked as bad tracking
    if frame_n in marked_frames:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 8)

    # Render structural connection lines (skeleton)
    for (bp1, bp2) in CONNECTIONS:
        x1, y1, _ = coords_raw[bp1]
        x2, y2, _ = coords_raw[bp2]
        if np.isnan(x1[frame_n]) or np.isnan(x2[frame_n]):
            continue
        pt1 = (int(x1[frame_n]), int(y1[frame_n]))
        pt2 = (int(x2[frame_n]), int(y2[frame_n]))
        cv2.line(frame, pt1, pt2, (255, 255, 255), 3)
        cv2.line(frame, pt1, pt2, COLORS[bp1],     1)

    # Render individual tracking keypoints
    for bp, (x, y, l) in coords_raw.items():
        if np.isnan(x[frame_n]):
            continue
        pt     = (int(x[frame_n]), int(y[frame_n]))
        color  = COLORS[bp]
        likely = l[frame_n] >= LIKELIHOOD_THRESH

        if likely:
            # Solid standard point for robust, reliable tracking coordinates
            cv2.circle(frame, pt, 6, (255, 255, 255), -1)
            cv2.circle(frame, pt, 5, color,            -1)
        else:
            # Clear red outer warning target indicator if likelihood is weak
            cv2.circle(frame, pt, 8, (0, 0, 255), 2)
            cv2.circle(frame, pt, 4, color,        1)

        # Draw clean, shadowed text labels near tracking points
        cv2.putText(frame, bp, (pt[0]+8, pt[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, bp, (pt[0]+8, pt[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color,     1, cv2.LINE_AA)

    is_left   = frame_n in left_spikes_set
    is_right  = frame_n in right_spikes_set
    marked_str = "  [M] BAD — marked" if frame_n in marked_frames else ""

    # Construct the tracking telemetry telemetry overlay text HUD rows
    lines = [
        f"Frame: {frame_n} / {total_frames-1}   t = {frame_n/FPS:.3f} s{marked_str}",
        f"Spike Frame {spike_idx+1} / {total_spikes}   |   Total marked bad: {len(marked_frames)}",
    ]
    if is_left and frame_n < len(left_vel):
        lines.append(f"LEFT  {left_vel[frame_n]:+.0f} deg/s  *** SPIKE ALERT ***")
    if is_right and frame_n < len(right_vel):
        lines.append(f"RIGHT {right_vel[frame_n]:+.0f} deg/s  *** SPIKE ALERT ***")
    lines += ["", "N/-> next   P/<- prev   ./,  +/-1frame   M mark   Q quit+save"]

    # Write text layout rows using contrasting background shadows for maximum visibility
    y0 = 30
    for line in lines:
        cv2.putText(frame, line, (15, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),      3, cv2.LINE_AA)
        cv2.putText(frame, line, (15, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y0 += 28

    return frame

# ============================================================
# 5. DATA EXPORT AND SAVING
# ============================================================

def save_bad_frames(marked_frames):
    """Generates structured CSV tracking files documenting marked anomalous intervals."""
    if not marked_frames:
        print("\nNo frames were marked as bad tracking — saving skipped.")
        return
    BAD_FRAMES_DIR.mkdir(exist_ok=True)
    pd.DataFrame({
        'frame' : sorted(marked_frames),
        'time_s': [f / FPS for f in sorted(marked_frames)],
        'video' : Path(VIDEO_PATH).name
    }).to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully saved {len(marked_frames)} bad tracking frames logs inside: {OUTPUT_FILE}")

# ============================================================
# 6. MAIN APPLICATION WORKFLOW LOOP
# ============================================================

def main():
    df         = load_data()
    coords_raw = {bp: get_coords(df, bp) for bp in BODYPARTS}
    left_vel, right_vel, left_spikes, right_spikes = compute_spikes(df)

    # Initialize video capture stream to extract accurate baseline stream dimensions
    cap          = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Context feature: because velocity represents a step shift from frame N to N+1, 
    # we force the list to include both adjacent indices to allow clear inspection of the jump.
    spike_centers = sorted(set(left_spikes.tolist() + right_spikes.tolist()))
    all_spikes    = sorted(set(
        f for center in spike_centers
        for f in [center, center + 1]
        if 0 <= f < total_frames
    ))

    if not all_spikes:
        print(f"Excellent tracking quality! No suspicious anomalies detected (Threshold: {VELOCITY_THRESHOLD} deg/s).")
        cap.release()
        return

    left_spikes_set  = set(left_spikes.tolist())
    right_spikes_set = set(right_spikes.tolist())

    print(f"\nFound {len(spike_centers)} velocity spikes → Showing {len(all_spikes)} inspection frames (showing frame N and N+1 for each spike).")
    print("Launching inspection player window... (Press M to mark/unmark bad tracking, Q to quit and save results)\n")

    spike_idx     = 0
    cur_frame     = all_spikes[0]
    marked_frames = set()

    # Instantiate graphical frame user interface window wrapper
    cv2.namedWindow("Spike Inspector", cv2.WINDOW_NORMAL)

    # ============================================================
    # 7. INTERACTIVE KEYBOARD LISTENER AND VIDEO NAVIGATION LOOP
    # ============================================================
    while True:
        # Seek video stream location pointer precisely to the requested tracking index position
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
        ret, frame = cap.read()
        if ret:
            # Overlap graphics summaries on pixel array buffers
            frame = draw_frame(frame, cur_frame, coords_raw,
                               left_vel, right_vel,
                               left_spikes_set, right_spikes_set,
                               spike_idx, len(all_spikes), total_frames,
                               marked_frames)
            cv2.imshow("Spike Inspector", frame)

        # Execution halts indefinitely until a valid key event registers in the application buffer
        key = cv2.waitKey(0) & 0xFF

        # Match byte value results to keyboard interactions
        if key == ord('q') or key == ord('Q') or key == 27:  # 'Q' or ESC key
            break
        elif key == ord('n') or key == ord('N') or key == 83 or key == 3:  # 'N' or Right Arrow key
            spike_idx = min(spike_idx + 1, len(all_spikes) - 1)
            cur_frame = all_spikes[spike_idx]
        elif key == ord('p') or key == ord('P') or key == 81 or key == 2:  # 'P' or Left Arrow key
            spike_idx = max(spike_idx - 1, 0)
            cur_frame = all_spikes[spike_idx]
        elif key == ord('.'):  # Fine tuning advance single step forward (+1 frame)
            cur_frame = min(cur_frame + 1, total_frames - 1)
        elif key == ord(','):  # Fine tuning step single step backward (-1 frame)
            cur_frame = max(cur_frame - 1, 0)
        elif key == ord('m') or key == ord('M'):  # Toggle bad tracking assignment tag on current frame
            if cur_frame in marked_frames:
                marked_frames.discard(cur_frame)
                print(f"  Frame {cur_frame} UN-MARKED (Total marked bad frames: {len(marked_frames)})")
            else:
                marked_frames.add(cur_frame)
                print(f"  Frame {cur_frame} MARKED as bad tracking (Total marked bad frames: {len(marked_frames)})")

    # Memory Cleanup and Resource Deallocation
    cap.release()
    cv2.destroyAllWindows()
    save_bad_frames(marked_frames)

if __name__ == "__main__":
    main()