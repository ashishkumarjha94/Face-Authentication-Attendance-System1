import numpy as np
from insightface.app import FaceAnalysis
from utils import l2_normalize

class FaceEngine:
    def __init__(self, det_size=(640, 640), providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0, det_size=det_size)

    def get_largest_face(self, frame_bgr):
        faces = self.app.get(frame_bgr)
        if not faces:
            return None

        def area(f):
            x1, y1, x2, y2 = f.bbox
            return (x2 - x1) * (y2 - y1)
        return sorted(faces, key=area, reverse=True)[0]

    def embedding_from_frame(self, frame_bgr):
        face = self.get_largest_face(frame_bgr)
        if face is None:
            return None, None
        emb = face.embedding
        emb = l2_normalize(emb)

        return emb, face
