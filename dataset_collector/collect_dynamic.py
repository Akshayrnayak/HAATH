"""
╔══════════════════════════════════════════════════════════════╗
║  HAATH — DYNAMIC ISL DATASET COLLECTOR                       ║
║  Uses MediaPipe Tasks API (works on ALL Windows versions)    ║
╚══════════════════════════════════════════════════════════════╝

BEFORE RUNNING:
    1. Download the hand landmark model file:
       URL: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
       Save it as: backend/hand_landmarker.task

    2. Run this script:
       python dataset_collector/collect_dynamic.py

WHAT IT CAPTURES:
    Each .npy file = 30 frames of one gesture performance
    Shape per file: (30, 63) = 30 frames × 21 landmarks × (x,y,z)
    Target: 200 sequences per gesture × 15 gestures = 3000 files total

CONTROLS:
    SPACE  — Record one sequence (perform gesture naturally)
    N      — Next gesture
    Q      — Quit and finish
"""

import cv2
import numpy as np
import os
import time
import urllib.request

# ── MediaPipe Tasks API import ────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─── CONFIG ────────────────────────────────────────────────
GESTURES = [
    "HELLO",    "THANK_YOU", "SORRY",    "PLEASE",   "HELP",
    "YES",      "NO",        "STOP",     "COME",     "GO",
    "WATER",    "FOOD",      "DOCTOR",   "WASHROOM", "EMERGENCY"
]
SEQUENCE_LENGTH     = 30     # frames per sequence (~1 second at 30fps)
SAMPLES_PER_GESTURE = 200    # sequences per gesture
DATASET_DIR         = "dataset/sequences"
MODEL_PATH          = "hand_landmarker.task"
MODEL_URL           = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
# ───────────────────────────────────────────────────────────


def download_model():
    """Auto-download the hand landmark model if not present."""
    if os.path.exists(MODEL_PATH):
        print(f"  ✓ Model found: {MODEL_PATH}")
        return True
    print(f"  Downloading hand landmark model...")
    print(f"  From: {MODEL_URL}")
    print(f"  To:   {MODEL_PATH}")
    print(f"  This is a one-time download (~25 MB). Please wait...")
    try:
        def progress(count, block_size, total_size):
            pct = int(count * block_size * 100 / total_size)
            pct = min(pct, 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}%", end="", flush=True)

        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=progress)
        print(f"\n  ✓ Download complete!")
        return True
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        print(f"\n  Please download manually:")
        print(f"  1. Open this URL in browser:")
        print(f"     {MODEL_URL}")
        print(f"  2. Save the file as: {MODEL_PATH}")
        print(f"     (inside your backend/ folder)")
        return False


def create_landmarker():
    """Create MediaPipe Tasks hand landmarker."""
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def extract_landmarks(frame_rgb, landmarker):
    """
    Extract 63 floats from a frame using Tasks API.
    Returns numpy array of shape (63,) or None if no hand detected.
    """
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    coords = []
    for lm in result.hand_landmarks[0]:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def normalize_landmarks(flat):
    """
    Make landmarks relative to wrist (landmark 0).
    This makes the model position-invariant — gesture works
    anywhere in frame, not just center.
    """
    wx, wy, wz = flat[0], flat[1], flat[2]
    out = []
    for i in range(0, len(flat), 3):
        out.extend([
            flat[i]   - wx,
            flat[i+1] - wy,
            flat[i+2] - wz
        ])
    return np.array(out, dtype=np.float32)


def draw_hand_connections(frame, landmarks_raw):
    """
    Draw hand skeleton using raw landmark coordinates.
    (Tasks API does not have built-in drawing utils like solutions did)
    """
    h, w = frame.shape[:2]

    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),          # thumb
        (0,5),(5,6),(6,7),(7,8),          # index
        (0,9),(9,10),(10,11),(11,12),     # middle
        (0,13),(13,14),(14,15),(15,16),   # ring
        (0,17),(17,18),(18,19),(19,20),   # pinky
        (5,9),(9,13),(13,17),             # palm
    ]

    # Convert flat 63-array back to (21, 3) for drawing
    pts = []
    for i in range(0, 63, 3):
        x = int(landmarks_raw[i]   * w)
        y = int(landmarks_raw[i+1] * h)
        pts.append((x, y))

    # Draw connections
    for a, b in CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 229, 160), 2)

    # Draw joints
    TIP_IDS = {4, 8, 12, 16, 20}
    for i, (x, y) in enumerate(pts):
        color  = (255, 255, 255) if i in TIP_IDS else (0, 229, 160)
        radius = 6 if i in TIP_IDS else 3
        cv2.circle(frame, (x, y), radius, color, -1)


def draw_ui(frame, gesture_name, gesture_idx, total,
            sample_count, target, state, frame_num):
    """Draw the recording UI overlay on the frame."""
    h, w = frame.shape[:2]

    # Dark header
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 135), (8, 12, 22), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    progress_str = f"{gesture_idx + 1}/{total}"

    # Gesture name
    cv2.putText(
        frame,
        f"Gesture {progress_str}: {gesture_name}",
        (24, 46),
        cv2.FONT_HERSHEY_SIMPLEX, 1.1,
        (0, 229, 160), 2, cv2.LINE_AA
    )

    # State text
    if state == "WAITING":
        status = f"Ready  [{sample_count}/{target}]  —  Press SPACE to record one sequence"
        color  = (100, 200, 255)
    elif state == "COUNTDOWN":
        status = f"Get ready... {frame_num}"
        color  = (255, 200, 50)
    elif state == "RECORDING":
        status = f"RECORDING  frame {frame_num}/{SEQUENCE_LENGTH}  — Perform gesture naturally!"
        color  = (0, 229, 160)
        # Pulsing red recording border
        t = int(time.time() * 8) % 2
        bw = 4 + t * 2
        cv2.rectangle(
            frame, (bw, bw), (w - bw, h - bw),
            (0, 60, 220), bw
        )
    else:
        status = f"✓ Saved! [{sample_count}/{target}]"
        color  = (100, 255, 150)

    cv2.putText(
        frame, status,
        (24, 92),
        cv2.FONT_HERSHEY_SIMPLEX, 0.78,
        color, 2, cv2.LINE_AA
    )

    # Sample progress bar
    bar_x, bar_y = 24, 114
    bar_w = w - 48
    bar_h = 10
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 45, 65), -1)
    filled = int(bar_w * sample_count / target)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), (0, 229, 160), -1)

    # Bottom controls hint
    cv2.putText(
        frame,
        "SPACE = record sequence    N = next gesture    Q = quit",
        (24, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
        (70, 90, 120), 1, cv2.LINE_AA
    )

    return frame


def collect():
    # ── Setup ──────────────────────────────────────────────
    if not download_model():
        print("\n  Cannot continue without model file. Exiting.")
        return

    # Create output directories for all gestures
    for gesture in GESTURES:
        os.makedirs(os.path.join(DATASET_DIR, gesture), exist_ok=True)

    print("\n" + "=" * 62)
    print("  HAATH — DYNAMIC ISL DATASET COLLECTOR")
    print("=" * 62)
    print(f"  Gestures        : {len(GESTURES)}")
    print(f"  Frames/sequence : {SEQUENCE_LENGTH}  (~1 second per gesture)")
    print(f"  Target          : {SAMPLES_PER_GESTURE} sequences per gesture")
    print(f"  Total to collect: {SAMPLES_PER_GESTURE * len(GESTURES)} sequences")
    print("\n  CONTROLS:")
    print("    SPACE = Start recording one sequence")
    print("    N     = Skip to next gesture")
    print("    Q     = Quit and save all collected data")
    print("=" * 62 + "\n")

    # ── Initialize MediaPipe Landmarker ───────────────────
    print("  Loading MediaPipe hand landmarker...")
    try:
        landmarker = create_landmarker()
        print("  ✓ MediaPipe Tasks API loaded successfully\n")
    except Exception as e:
        print(f"  ✗ Failed to load landmarker: {e}")
        return

    # ── Initialize camera ─────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("  ✗ Cannot open camera. Check webcam is connected.")
        return

    # ── State machine variables ───────────────────────────
    gesture_idx  = 0
    state        = "WAITING"   # WAITING | COUNTDOWN | RECORDING
    countdown    = 0
    sequence_buf = []
    sample_count = 0

    # Count existing samples (resume if restarting)
    existing = len([
        f for f in os.listdir(
            os.path.join(DATASET_DIR, GESTURES[gesture_idx])
        ) if f.endswith('.npy')
    ])
    if existing > 0:
        sample_count = existing
        print(f"  Resuming {GESTURES[gesture_idx]}: {existing} sequences already collected")

    print(f"  Starting with: {GESTURES[gesture_idx]}")
    print(f"  Press SPACE in the camera window to begin recording\n")

    while cap.isOpened() and gesture_idx < len(GESTURES):
        ret, frame = cap.read()
        if not ret:
            print("  ✗ Frame capture failed")
            break

        # Flip horizontally so it mirrors the user
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract landmarks using Tasks API
        raw_landmarks = extract_landmarks(rgb, landmarker)
        hand_ok       = raw_landmarks is not None

        # Draw hand skeleton if detected
        if hand_ok:
            draw_hand_connections(frame, raw_landmarks)

        gesture_name = GESTURES[gesture_idx]

        # ── State transitions ──────────────────────────────
        if state == "COUNTDOWN":
            countdown -= 1
            if countdown <= 0:
                state        = "RECORDING"
                countdown    = SEQUENCE_LENGTH
                sequence_buf = []

        elif state == "RECORDING":
            if hand_ok:
                norm = normalize_landmarks(raw_landmarks)
                sequence_buf.append(norm)
            else:
                # Pad with zeros if hand briefly disappears
                sequence_buf.append(np.zeros(63, dtype=np.float32))

            countdown -= 1

            # Check if sequence is complete
            if countdown <= 0 or len(sequence_buf) >= SEQUENCE_LENGTH:
                # Ensure exact length
                while len(sequence_buf) < SEQUENCE_LENGTH:
                    sequence_buf.append(np.zeros(63, dtype=np.float32))
                sequence_buf = sequence_buf[:SEQUENCE_LENGTH]

                # Save as .npy file
                arr      = np.array(sequence_buf, dtype=np.float32)  # (30, 63)
                savepath = os.path.join(
                    DATASET_DIR, gesture_name,
                    f"{sample_count:04d}.npy"
                )
                np.save(savepath, arr)
                sample_count += 1
                print(f"  ✓ {gesture_name:15} [{sample_count:3}/{SAMPLES_PER_GESTURE}] saved")

                if sample_count >= SAMPLES_PER_GESTURE:
                    print(f"\n  ✅ {gesture_name} COMPLETE! Press N to go to next gesture.\n")
                    state = "WAITING"
                else:
                    state = "WAITING"

                countdown = 0

        # ── Draw UI ───────────────────────────────────────
        ui_countdown = countdown if state in ("RECORDING", "COUNTDOWN") else 0
        frame = draw_ui(
            frame, gesture_name, gesture_idx,
            len(GESTURES), sample_count,
            SAMPLES_PER_GESTURE, state, ui_countdown
        )

        # No hand warning
        if not hand_ok and state == "RECORDING":
            cv2.putText(
                frame, "⚠ No hand detected — keep hand in frame",
                (frame.shape[1] // 2 - 200, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (0, 80, 255), 2, cv2.LINE_AA
            )

        # Show frame
        cv2.imshow("Haath — Dynamic ISL Collector  (press Q to quit)", frame)

        # ── Keyboard input ────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if state == "WAITING":
                if sample_count >= SAMPLES_PER_GESTURE:
                    print(f"  Already collected {SAMPLES_PER_GESTURE} for {gesture_name}. Press N for next.")
                else:
                    state     = "COUNTDOWN"
                    countdown = 5   # small countdown buffer
                    print(f"  ▶ Recording {gesture_name}... perform the gesture!")

        elif key in (ord('n'), ord('N')):
            print(f"  → Moving to next gesture (saved {sample_count} for {gesture_name})")
            gesture_idx  += 1
            sample_count  = 0
            state         = "WAITING"
            countdown     = 0
            sequence_buf  = []

            if gesture_idx < len(GESTURES):
                # Count existing for this gesture
                existing = len([
                    f for f in os.listdir(
                        os.path.join(DATASET_DIR, GESTURES[gesture_idx])
                    ) if f.endswith('.npy')
                ])
                if existing > 0:
                    sample_count = existing
                    print(f"  Resuming {GESTURES[gesture_idx]}: {existing} already collected")
                else:
                    print(f"  Now collecting: {GESTURES[gesture_idx]}")

        elif key in (ord('q'), ord('Q')):
            print("\n  Q pressed — quitting and saving data collected so far.")
            break

    # ── Cleanup ───────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    # ── Final summary ─────────────────────────────────────
    print("\n" + "=" * 62)
    print("  COLLECTION SUMMARY")
    print("=" * 62)
    total_collected = 0
    for g in GESTURES:
        folder = os.path.join(DATASET_DIR, g)
        if os.path.exists(folder):
            count = len([f for f in os.listdir(folder) if f.endswith('.npy')])
        else:
            count = 0
        total_collected += count
        bar    = "█" * (count // 10)
        status = "✅" if count >= SAMPLES_PER_GESTURE else "⚠️ "
        print(f"  {status} {g:15} {count:4} sequences  {bar}")

    print("=" * 62)
    print(f"  Total sequences collected: {total_collected}")
    print(f"  Dataset saved to: {DATASET_DIR}/")
    print(f"\n  Next step: python ml_model/train_lstm.py")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    collect()