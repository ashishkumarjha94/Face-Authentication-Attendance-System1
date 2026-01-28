import time
import random
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# FaceMesh landmark indices for eyes (commonly used subset)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Yaw estimation landmarks (approx)
NOSE_TIP = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

def _ear(landmarks, eye_idx):
    # Eye Aspect Ratio using 6 landmarks
    p = [landmarks[i] for i in eye_idx]
    def dist(a, b):
        return np.linalg.norm(np.array([a.x, a.y]) - np.array([b.x, b.y]))
    # vertical distances
    v1 = dist(p[1], p[5])
    v2 = dist(p[2], p[4])
    # horizontal
    h = dist(p[0], p[3])
    return (v1 + v2) / (2.0 * h + 1e-9)

def _approx_yaw(landmarks):
    # Very rough yaw proxy: compare nose x to midpoint between cheeks
    nose = landmarks[NOSE_TIP]
    lc = landmarks[LEFT_CHEEK]
    rc = landmarks[RIGHT_CHEEK]
    mid_x = (lc.x + rc.x) / 2.0
    return nose.x - mid_x  # positive => nose shifts right (head turns left-ish)

class LivenessChecker:
    """
    Basic challenge-response liveness:
    - Random challenge: BLINK or TURN_LEFT or TURN_RIGHT
    - Must be completed within time limit
    """
    def __init__(self):
        self.mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.reset()

    def reset(self):
        self.challenge = random.choice(["BLINK", "TURN_LEFT", "TURN_RIGHT"])
        self.start_time = time.time()
        self.blink_count = 0
        self.eye_closed = False
        self.completed = False

    def instruction(self):
        return f"Liveness: {self.challenge}"

    def update(self, frame_bgr):
        if self.completed:
            return True, "OK"

        if time.time() - self.start_time > 8.0:
            return False, "LIVENESS_TIMEOUT"

        frame_rgb = frame_bgr[:, :, ::-1]
        res = self.mesh.process(frame_rgb)
        if not res.multi_face_landmarks:
            return False, "NO_FACE_MESH"

        landmarks = res.multi_face_landmarks[0].landmark

        if self.challenge == "BLINK":
            left = _ear(landmarks, LEFT_EYE)
            right = _ear(landmarks, RIGHT_EYE)
            ear = (left + right) / 2.0

            # thresholds depend on camera/user; tune if needed
            if ear < 0.19 and not self.eye_closed:
                self.eye_closed = True
            if ear > 0.23 and self.eye_closed:
                self.eye_closed = False
                self.blink_count += 1

            if self.blink_count >= 1:
                self.completed = True
                return True, "LIVENESS_OK"

            return False, f"BLINKS={self.blink_count}"

        yaw = _approx_yaw(landmarks)
        # yaw thresholds to tune
        if self.challenge == "TURN_LEFT":
            # head turned left => nose shifts right => yaw positive
            if yaw > 0.03:
                self.completed = True
                return True, "LIVENESS_OK"
            return False, f"YAW={yaw:.3f}"

        if self.challenge == "TURN_RIGHT":
            if yaw < -0.03:
                self.completed = True
                return True, "LIVENESS_OK"
            return False, f"YAW={yaw:.3f}"

        return False, "UNKNOWN_CHALLENGE"