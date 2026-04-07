"""
Iron Man Drawing System — Real-time Gesture Drawing
====================================================

Implements:
    Phase 1-3: hand detection, fingertip tracking, per-finger drawing
    Phase 4: smoothing, draw/erase/idle modes, multi-finger control
    Phase 5: glow/bloom, UI panel, save-to-file, sound hooks
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque
import time
import os
import urllib.request

# ──────────────────────────────────────────────────────────────
# CONSTANTS AND TUNING
# ──────────────────────────────────────────────────────────────

FINGERTIP_IDS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP_IDS       = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
MCP_IDS       = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
FINGER_COLORS = {
    "thumb":  (0, 0, 255),     # red
    "index":  (255, 0, 0),     # blue
    "middle": (0, 255, 0),     # green
    "ring":   (0, 255, 255),   # yellow
    "pinky":  (255, 0, 255),   # purple
}
SMOOTHING_ALPHA   = 0.35
TARGET_FPS        = 30
MIN_MOVE_PX       = 2
FINGER_EXTEND_PX  = 6
THUMB_EXTEND_PX   = 4
CAPTURE_WIDTH     = 960
CAPTURE_HEIGHT    = 540
MODEL_URL          = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH         = os.path.join("models", "hand_landmarker.task")


# ══════════════════════════════════════════════════════════════
# PHASE 4.1 — MODE FINITE STATE MACHINE
# ══════════════════════════════════════════════════════════════

class DrawMode:
    DRAW  = "draw"
    ERASE = "erase"
    IDLE  = "idle"


class ModeFSM:
    """
    Finite state machine for switching between draw / erase / idle.

    State transitions:
      any state  →  ERASE  if open palm detected
      ERASE      →  DRAW   if palm closed (cooldown enforced)
      DRAW       →  IDLE   if no hand for N frames
      IDLE       →  DRAW   when hand reappears
    """

    ERASE_COOLDOWN  = 1.2   # seconds before returning to DRAW after erase
    IDLE_THRESHOLD  = 15    # missed frames before switching to IDLE

    def __init__(self):
        self.mode           = DrawMode.IDLE
        self.just_erased    = False
        self._erase_ts      = 0.0
        self._no_hand_count = 0

    def update(self, has_hand, open_palm):
        self.just_erased = False
        now = time.time()

        if not has_hand:
            self._no_hand_count += 1
            if self._no_hand_count >= self.IDLE_THRESHOLD:
                self.mode = DrawMode.IDLE
            return self.mode

        self._no_hand_count = 0

        if open_palm:
            if self.mode != DrawMode.ERASE:
                self._erase_ts = now
                self.just_erased = True
            self.mode = DrawMode.ERASE
        else:
            if self.mode == DrawMode.ERASE:
                # Enforce cooldown before going back to DRAW
                if now - self._erase_ts >= self.ERASE_COOLDOWN:
                    self.mode = DrawMode.DRAW
            else:
                self.mode = DrawMode.DRAW

        return self.mode


# ══════════════════════════════════════════════════════════════
# PHASE 4.2 — MULTI-FINGER INDEPENDENCE & STABILITY
# ══════════════════════════════════════════════════════════════

class FingerState:
    """
    Per-finger state tracker.
    Manages: active status, velocity, and jitter suppression.
    """

    def __init__(self, name):
        self.name      = name
        self.active    = False
        self.last_pt   = None
        self.velocity  = (0, 0)
        self._history  = deque(maxlen=5)   # last 5 smoothed positions

    def update(self, x, y):
        if self.last_pt:
            vx = x - self.last_pt[0]
            vy = y - self.last_pt[1]
            self.velocity = (vx, vy)
        self._history.append((x, y))
        self.last_pt = (x, y)
        self.active  = True

    def deactivate(self):
        self.active   = False
        self.last_pt  = None
        self.velocity = (0, 0)

    @property
    def speed(self):
        vx, vy = self.velocity
        return (vx**2 + vy**2) ** 0.5

    def predict_next(self):
        """
        Simple linear extrapolation — useful when a finger is briefly occluded.
        Helps maintain continuity without phantom points.
        """
        if self.last_pt and self.velocity != (0, 0):
            px = self.last_pt[0] + self.velocity[0]
            py = self.last_pt[1] + self.velocity[1]
            return px, py
        return self.last_pt


# ══════════════════════════════════════════════════════════════
# PHASE 5.1 — GLOW / BLOOM VISUAL EFFECT
# ══════════════════════════════════════════════════════════════

def apply_glow_bloom(canvas, intensity=0.6, blur_ksize=21):
    """
    Simulate a laser glow/bloom by:
    1. Gaussian-blurring the canvas (spreads the light).
    2. Blending the blurred result back onto the original.
    This creates a halo around every drawn line.

    Parameters:
        canvas      : BGR drawing canvas (numpy array)
        intensity   : blend strength of the glow layer (0–1)
        blur_ksize  : kernel size for blur (must be odd); larger = wider halo

    Returns:
        Canvas with glow effect applied.
    """
    blurred = cv2.GaussianBlur(canvas, (blur_ksize, blur_ksize), 0)
    glowed  = cv2.addWeighted(canvas, 1.0, blurred, intensity, 0)
    return glowed


def apply_line_glow(canvas, p1, p2, color, core_t=2, glow_layers=3):
    """
    Draw a single line segment with multi-layer glow.
    Draws progressively thicker, dimmer rings outward, then a bright core.

    Parameters:
        canvas     : BGR canvas to draw on (modified in-place)
        p1, p2     : endpoints
        color      : BGR core color tuple
        core_t     : thickness of the bright core line
        glow_layers: number of outer halos
    """
    b, g, r = color
    for i in range(glow_layers, 0, -1):
        thickness = core_t + i * 4
        alpha_factor = 0.12 * (glow_layers - i + 1)
        dim_color = (
            int(b * alpha_factor * 2),
            int(g * alpha_factor * 2),
            int(r * alpha_factor * 2),
        )
        cv2.line(canvas, p1, p2, dim_color, thickness, cv2.LINE_AA)
    # Core bright line
    cv2.line(canvas, p1, p2, color, core_t, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════
# PHASE 5.2 — UI PANEL (color picker bar)
# ══════════════════════════════════════════════════════════════

class UIPanel:
    """
    Renders a sidebar showing:
    - Active mode indicator
    - Per-finger status (dot = drawing, ring = idle)
    - Hint text
    """

    PANEL_W = 170

    def __init__(self, frame_w, frame_h):
        self.fw = frame_w
        self.fh = frame_h

    def draw(self, frame, mode, active_fingers, fps):
        pw = self.PANEL_W
        panel = frame[:, self.fw - pw:]

        # Dark transparent panel background
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, 0), (pw, self.fh), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.60, panel, 0.40, 0, panel)
        frame[:, self.fw - pw:] = panel

        x0   = self.fw - pw + 14
        mode_colors = {
            DrawMode.DRAW:  (0, 220, 0),
            DrawMode.ERASE: (0, 0, 220),
            DrawMode.IDLE:  (140, 140, 140),
        }

        # Mode indicator
        mcol = mode_colors.get(mode, (200, 200, 200))
        cv2.putText(frame, "MODE", (x0, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(frame, mode.upper(), (x0, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, mcol, 2, cv2.LINE_AA)

        # FPS
        cv2.putText(frame, f"{fps:.0f} fps", (x0, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 100), 1, cv2.LINE_AA)

        # Separator
        sx = self.fw - pw
        cv2.line(frame, (sx + 8, 95), (self.fw - 8, 95), (60, 60, 60), 1)

        # Per-finger status
        cv2.putText(frame, "FINGERS", (x0, 114),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1, cv2.LINE_AA)
        for i, (name, _) in enumerate(FINGERTIP_IDS.items()):
            cy = 136 + i * 38
            color = FINGER_COLORS[name]
            drawing = name in active_fingers

            if drawing:
                cv2.circle(frame, (x0 + 10, cy), 10, color, -1)
                cv2.circle(frame, (x0 + 10, cy), 12, (255,255,255), 1)
            else:
                cv2.circle(frame, (x0 + 10, cy), 7,  (40,40,40), -1)
                cv2.circle(frame, (x0 + 10, cy), 7,  color, 1)

            cv2.putText(frame, name.capitalize(), (x0 + 26, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        color if drawing else (100, 100, 100),
                        1, cv2.LINE_AA)

        # Hint section
        cv2.line(frame, (sx + 8, 340), (self.fw - 8, 340), (60, 60, 60), 1)
        hints = ["Q - quit", "S - save", "D - save canvas", "C - clear", "PALM - erase"]
        for j, h in enumerate(hints):
            cv2.putText(frame, h, (x0, 360 + j * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 100, 100), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════
# PHASE 5.3 — SAVE DRAWINGS AS IMAGE
# ══════════════════════════════════════════════════════════════

class CanvasSaver:
    """
    Save the current drawing canvas (or composited frame) to disk.
    Supports PNG (lossless) and JPEG.
    """

    def __init__(self, output_dir="gesture_saves"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.count = 0

    def save(self, frame, canvas_only=False, canvas=None):
        """
        frame       : the composited video frame (full output)
        canvas_only : if True, saves only the drawing layer (black background)
        canvas      : the pure drawing numpy array (needed if canvas_only=True)
        """
        ts  = time.strftime("%Y%m%d_%H%M%S")
        img = canvas if (canvas_only and canvas is not None) else frame
        fname = os.path.join(self.output_dir, f"draw_{ts}_{self.count:03d}.png")
        cv2.imwrite(fname, img)
        self.count += 1
        print(f"Saved: {fname}")
        return fname


# ══════════════════════════════════════════════════════════════
# PHASE 5.4 — OPTIONAL: SOUND EFFECTS HOOK
# ══════════════════════════════════════════════════════════════

class SoundEngine:
    """
    Plays a tone when a finger is actively drawing.
    Requires: pip install playsound  OR  pygame

    Currently implemented as a stub so the main system works
    without audio dependencies. Activate by uncommenting the import
    and filling in play_draw() / play_erase().
    """

    def __init__(self, enabled=False):
        self.enabled = enabled

    def play_draw(self, finger):
        """Called each frame a finger is drawing. Keep it short (< 20ms)."""
        if not self.enabled:
            return
        # Example using pygame:
        # pygame.mixer.Sound("laser_hum.wav").play()
        pass

    def play_erase(self):
        """Called once when erase gesture is detected."""
        if not self.enabled:
            return
        # pygame.mixer.Sound("whoosh.wav").play()
        pass


# ══════════════════════════════════════════════════════════════
# PHASE 5.5 — ADVANCED DRAWING ENGINE (extends base)
# ══════════════════════════════════════════════════════════════

class AdvancedDrawingEngine:
    """
    Full-featured engine combining:
    - Multi-layer glow drawing
    - EMA-smoothed input points
    - Per-finger velocity-based thickness
    - Optional bloom post-process
    """

    GLOW_INTENSITY = 0.55
    USE_BLOOM      = True    # Set False for faster rendering on slow machines
    VELOCITY_SCALE = True    # Thicker lines = slower = more intentional

    def __init__(self, width, height):
        self.width  = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # Store raw trail points for future smoothing or replays
        self.trails = {name: deque(maxlen=600) for name in FINGERTIP_IDS}
        self.prev   = {name: None for name in FINGERTIP_IDS}
        self.states = {name: FingerState(name) for name in FINGERTIP_IDS}

    def update(self, finger, x, y):
        state = self.states[finger]
        state.update(x, y)

        prev = self.prev[finger]
        if prev is None:
            self.prev[finger] = (x, y)
            self.trails[finger].append((x, y))
            return

        dist = ((x - prev[0])**2 + (y - prev[1])**2) ** 0.5
        if dist < MIN_MOVE_PX:
            return

        # Velocity-scaled line thickness
        speed     = state.speed
        thickness = max(1, min(6, int(4 - speed * 0.05)))   # slower = thicker

        color = FINGER_COLORS[finger]
        apply_line_glow(self.canvas, prev, (x, y), color,
                        core_t=thickness, glow_layers=2)

        self.trails[finger].append((x, y))
        self.prev[finger] = (x, y)

    def lift(self, finger):
        self.prev[finger] = None
        self.states[finger].deactivate()

    def clear(self):
        self.canvas[:] = 0
        for name in FINGERTIP_IDS:
            self.trails[name].clear()
            self.prev[name] = None
            self.states[name].deactivate()

    def composite(self, frame):
        draw_layer = self.canvas.copy()

        # Optional bloom post-process
        if self.USE_BLOOM:
            draw_layer = apply_glow_bloom(draw_layer,
                                          intensity=self.GLOW_INTENSITY,
                                          blur_ksize=15)

        mask  = cv2.cvtColor(draw_layer, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask3 = cv2.merge([mask, mask, mask])

        blended = cv2.addWeighted(draw_layer, 0.92, frame, 0.08, 0)
        result  = np.where(mask3 > 0, blended, frame)
        return result


# ══════════════════════════════════════════════════════════════
# ADVANCED MAIN LOOP
# ══════════════════════════════════════════════════════════════

def is_finger_extended(landmarks, tip_id, pip_id, mcp_id, margin=FINGER_EXTEND_PX):
    tip_y = landmarks[tip_id][1]
    pip_y = landmarks[pip_id][1]
    mcp_y = landmarks[mcp_id][1]
    return (mcp_y - pip_y) > margin and (pip_y - tip_y) > margin


def is_thumb_extended(landmarks, handedness, margin=THUMB_EXTEND_PX):
    tip_x = landmarks[FINGERTIP_IDS["thumb"]][0]
    ip_x  = landmarks[PIP_IDS["thumb"]][0]
    mcp_x = landmarks[MCP_IDS["thumb"]][0]

    if handedness == "Right":
        return tip_x > ip_x + margin and ip_x > mcp_x + margin
    if handedness == "Left":
        return tip_x < ip_x - margin and ip_x < mcp_x - margin

    tip_y = landmarks[FINGERTIP_IDS["thumb"]][1]
    ip_y  = landmarks[PIP_IDS["thumb"]][1]
    mcp_y = landmarks[MCP_IDS["thumb"]][1]
    return (mcp_y - ip_y) > margin and (ip_y - tip_y) > margin


def get_finger_states(landmarks, handedness):
    states = {}
    states["thumb"] = is_thumb_extended(landmarks, handedness)
    for name in ("index", "middle", "ring", "pinky"):
        states[name] = is_finger_extended(
            landmarks,
            FINGERTIP_IDS[name],
            PIP_IDS[name],
            MCP_IDS[name],
        )
    return states


def detect_open_palm(finger_states):
    return bool(finger_states) and all(finger_states.values())


def ensure_hand_model(model_path=MODEL_PATH, url=MODEL_URL):
    if os.path.exists(model_path):
        return model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print("Downloading MediaPipe hand landmarker model...")
    urllib.request.urlretrieve(url, model_path)
    return model_path


class HandDetector:
    def __init__(self):
        model_path = ensure_hand_model()
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.hand_landmarks:
            return None, None
        lm = result.hand_landmarks[0]
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        handedness = None
        if result.handedness and result.handedness[0]:
            handedness = result.handedness[0][0].category_name
        return pts, handedness

    def close(self):
        self.landmarker.close()


class EMAFilter:
    def __init__(self, alpha=SMOOTHING_ALPHA):
        self.alpha = alpha
        self._s = {}

    def smooth(self, finger, x, y):
        if finger not in self._s:
            self._s[finger] = (x, y)
            return x, y
        sx, sy = self._s[finger]
        sx = self.alpha*x + (1-self.alpha)*sx
        sy = self.alpha*y + (1-self.alpha)*sy
        self._s[finger] = (sx, sy)
        return int(sx), int(sy)

    def reset(self, finger=None):
        if finger: self._s.pop(finger, None)
        else: self._s.clear()


def main_advanced():
    print("Advanced Gesture Drawing System — with glow, FSM, and UI panel")

    if os.name == "nt":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = HandDetector()
    smoother = EMAFilter()
    engine   = AdvancedDrawingEngine(w, h)
    fsm      = ModeFSM()
    ui       = UIPanel(w, h)
    saver    = CanvasSaver()
    sound    = SoundEngine(enabled=False)

    frame_delay = 1.0 / TARGET_FPS
    prev_time   = time.time()
    fps_disp    = 0.0

    WIN = "Iron Man Drawing — Advanced"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(w, 1280), min(h, 720))

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Camera frame not received. Exiting.")
            break

        frame = cv2.flip(frame, 1)
        landmarks, handedness = detector.detect(frame)

        finger_states = get_finger_states(landmarks, handedness) if landmarks else {}
        open_palm     = detect_open_palm(finger_states)
        mode          = fsm.update(bool(landmarks), open_palm)
        active_fingers = set()

        if fsm.just_erased:
            engine.clear()
            smoother.reset()
            sound.play_erase()

        if mode == DrawMode.DRAW and landmarks:
            for name, tip_id in FINGERTIP_IDS.items():
                if not finger_states.get(name, False):
                    engine.lift(name)
                    smoother.reset(name)
                    continue

                rx, ry = landmarks[tip_id]
                if rx < 5 or ry < 55 or rx > w - UIPanel.PANEL_W - 5:
                    engine.lift(name)
                    smoother.reset(name)
                    continue

                sx, sy = smoother.smooth(name, rx, ry)
                engine.update(name, sx, sy)
                active_fingers.add(name)
                sound.play_draw(name)

        else:
            for name in FINGERTIP_IDS:
                engine.lift(name)
            smoother.reset()

        output = engine.composite(frame)

        # Erase flash
        if mode == DrawMode.ERASE:
            cv2.rectangle(output, (0, 0), (w, h), (0, 0, 200), 6)
            cv2.putText(output, "ERASING", (w//2 - 80, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 40, 255), 3, cv2.LINE_AA)

        # FPS
        now      = time.time()
        fps_disp = 0.9*fps_disp + 0.1*(1.0/(now - prev_time + 1e-9))
        prev_time = now

        # UI Panel
        ui.draw(output, mode, active_fingers, fps_disp)

        cv2.imshow(WIN, output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            saver.save(output)
        elif key == ord('d'):
            saver.save(output, canvas_only=True, canvas=engine.canvas)
        elif key == ord('c'):
            engine.clear()
            smoother.reset()

        sleep_t = frame_delay - (time.time() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_advanced()