import argparse
import time
import cv2
import numpy as np

from face_engine import FaceEngine
from storage import Storage
from utils import cosine_similarity, enhance_lighting
from liveness import LivenessChecker

SIM_THRESHOLD_DEFAULT = 0.50  

def draw_face_box(frame, face):
    if face is None:
        return
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

def enroll(name: str, cam_index: int, samples: int):
    db = Storage()
    engine = FaceEngine()
    live = LivenessChecker()

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    embeddings = []
    last_add = 0

    print(f"[ENROLL] User: {name}")
    print("[ENROLL] Look at camera. Complete liveness challenge first.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        enhanced = enhance_lighting(frame)

        live_ok, live_msg = live.update(frame)
        instr = live.instruction()

        emb, face = (None, None)
        if live_ok:
            emb, face = engine.embedding_from_frame(enhanced)
            now = time.time()
            if emb is not None and (now - last_add) > 0.15:
                embeddings.append(emb)
                last_add = now

        draw_face_box(frame, face)

        cv2.putText(frame, f"ENROLL: {name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, instr, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"Liveness: {live_ok} ({live_msg})", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if live_ok else (0,0,255), 2)
        cv2.putText(frame, f"Samples: {len(embeddings)}/{samples}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, "Press Q to quit", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        cv2.imshow("Face Attendance - Enroll", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in [ord('q'), ord('Q')]:
            break

        if len(embeddings) >= samples:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) < max(8, samples // 2):
        print("[ENROLL] Not enough good samples collected. Try again.")
        return

    template = np.mean(np.stack(embeddings, axis=0), axis=0)
    template = template / (np.linalg.norm(template) + 1e-12)
    db.upsert_user(name, template)
    print(f"[ENROLL] Saved user '{name}' with {len(embeddings)} samples.")

def attend(cam_index: int, sim_threshold: float):
    db = Storage()
    engine = FaceEngine()
    live = LivenessChecker()

    users = db.get_users()
    if not users:
        print("[ATTEND] No users enrolled. Run enroll first.")
        return

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("[ATTEND] Starting attendance. Complete liveness challenge when prompted.")

    last_mark_time = 0
    cooldown_sec = 3.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        enhanced = enhance_lighting(frame)

        live_ok, live_msg = live.update(frame)
        instr = live.instruction()

        status_line = "No match"
        color = (0, 0, 255)

        emb, face = (None, None)
        if live_ok:
            emb, face = engine.embedding_from_frame(enhanced)

        draw_face_box(frame, face)

        if emb is not None:

            best = None
            for uid, name, uemb in users:
                sim = cosine_similarity(emb, uemb)
                if best is None or sim > best[0]:
                    best = (sim, uid, name)

            if best and best[0] >= sim_threshold:
                sim, uid, name = best
                status_line = f"Match: {name} (sim={sim:.2f})"
                color = (0, 255, 0)

                now = time.time()
                if (now - last_mark_time) > cooldown_sec:
                    event, ts = db.mark_attendance(uid)
                    last_mark_time = now
                    status_line = f"{name}: {event} @ {ts} (sim={sim:.2f})"
                    live.reset()

        cv2.putText(frame, "ATTENDANCE MODE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, instr, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"Liveness: {live_ok} ({live_msg})", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if live_ok else (0,0,255), 2)
        cv2.putText(frame, status_line, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, f"Threshold={sim_threshold:.2f} | Press Q to quit", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 2)

        cv2.imshow("Face Attendance - Attend", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in [ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_enroll = sub.add_parser("enroll")
    p_enroll.add_argument("--name", required=True)
    p_enroll.add_argument("--cam", type=int, default=0)
    p_enroll.add_argument("--samples", type=int, default=20)

    p_att = sub.add_parser("attend")
    p_att.add_argument("--cam", type=int, default=0)
    p_att.add_argument("--threshold", type=float, default=SIM_THRESHOLD_DEFAULT)

    args = parser.parse_args()

    if args.cmd == "enroll":
        enroll(args.name, args.cam, args.samples)
    elif args.cmd == "attend":
        attend(args.cam, args.threshold)

if __name__ == "__main__":

    main()
