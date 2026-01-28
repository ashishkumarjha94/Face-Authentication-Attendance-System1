import numpy as np
import cv2

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / (n + eps)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def enhance_lighting(frame_bgr):
    """
    Basic lighting normalization:
    - Convert to YCrCb
    - Apply CLAHE on Y channel
    """
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y2 = clahe.apply(y)
    out = cv2.merge([y2, cr, cb])
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)