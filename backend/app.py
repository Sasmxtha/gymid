
import os, sys, base64, logging, traceback
from datetime import datetime, date
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from dotenv import load_dotenv; load_dotenv()
except: pass

from database.db import Database
from models.face_engine import FaceEngine
from models.face_matcher import FaceMatcher
from models.augmentor import Augmentor

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("gymid")

FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend")
application = Flask(__name__, static_folder=FRONTEND, static_url_path="") # Changed app to application
CORS(application, resources={r"/api/*": {"origins": "*"}})
app = application # Alias for compatibility

db          = Database()
face_engine = FaceEngine()
matcher     = FaceMatcher(backend=face_engine.backend)
augmentor   = Augmentor()

def decode_img(data_url):
    import cv2
    if "," in data_url: data_url = data_url.split(",", 1)[1]
    arr = np.frombuffer(base64.b64decode(data_url), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# ── Static ────────────────────────────────────────────────────────────────────
@application.route("/")
def index(): return send_from_directory(FRONTEND, "index.html")

# ── Health ────────────────────────────────────────────────────────────────────
@application.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "members": db.count_members(),
        "engine": face_engine.backend,
        "threshold": matcher.threshold,
        "timestamp": datetime.now().isoformat()
    })

# ── Register ──────────────────────────────────────────────────────────────────
@application.route("/api/register", methods=["POST"])
def register():
    try:
        body = request.get_json(force=True)
        for field in ["id","name","email","phone","plan","photos"]:
            if field not in body: return jsonify({"error": f"Missing: {field}"}), 400
        if len(body["photos"]) < 3:
            return jsonify({"error": "Need at least 3 photos"}), 400
        if db.member_exists_by_email(body["email"]):
            return jsonify({"error": "Email already registered"}), 409

        # Decode images
        images = [img for img in (decode_img(p) for p in body["photos"]) if img is not None]

        # Augment: 5 photos × 20 = 100 samples
        augmented = []
        for img in images:
            augmented.append(img)
            augmented.extend(augmentor.augment(img, n=19))

        # Extract embeddings
        embeddings = [e for e in (face_engine.extract_embedding(img) for img in augmented) if e is not None]
        if len(embeddings) < 5:
            return jsonify({"error": "Could not detect face. Better lighting, no mask, face centred."}), 422

        # Save to DB
        db.insert_member({
            "id": body["id"], "name": body["name"], "email": body["email"],
            "phone": body["phone"], "plan": body["plan"],
            "photo_count": len(body["photos"]), "embedding_count": len(embeddings)
        })
        for emb in embeddings:
            db.insert_embedding(body["id"], emb.tolist(), is_mean=False)
        # Store mean embedding too
        mean_emb = np.mean(embeddings, axis=0)
        mean_emb /= (np.linalg.norm(mean_emb) + 1e-9)
        db.insert_embedding(body["id"], mean_emb.tolist(), is_mean=True)

        log.info(f"Registered {body['name']} — {len(embeddings)} embeddings (buffalo_l)")
        return jsonify({"success": True, "member_id": body["id"], "embeddings_stored": len(embeddings)})

    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# ── Detect ────────────────────────────────────────────────────────────────────
@application.route("/api/detect", methods=["POST"])
def detect():
    try:
        body = request.get_json(force=True)
        img  = decode_img(body.get("frame", ""))
        if img is None:
            return jsonify({"error": "Bad frame"}), 400

        q_emb = face_engine.extract_embedding(img)
        if q_emb is None:
            return jsonify({"matched": False, "confidence": 0.0,
                            "message": "No face detected. Look at the camera."})

        all_embs = db.get_all_embeddings()
        if not all_embs:
            return jsonify({"matched": False, "confidence": 0.0, "message": "No members registered."})

        member_id, score, debug = matcher.match(q_emb, all_embs)
        log.info(f"Detect → {member_id} score={score:.3f} reason={debug.get('reason')}")

        if member_id:
            member = db.get_member(member_id)
            if member:
                cid = db.log_checkin(member_id, score)
                return jsonify({
                    "matched": True, "member": member,
                    "confidence": round(score, 4), "checkin_id": cid,
                    "message": f"Welcome, {member['name']}!", "debug": debug
                })

        reason = debug.get("reason", "")
        if "below_threshold" in reason:
            best = debug.get("best_weighted", 0)
            thr  = debug.get("threshold", matcher.threshold)
            msg  = f"Not recognised ({best/thr*100:.0f}% of threshold)."
        elif "ambiguous" in reason:
            msg = "Ambiguous — try again."
        else:
            msg = "Face not recognised."

        return jsonify({"matched": False, "confidence": round(score,4),
                        "reason": reason, "message": msg, "debug": debug})

    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# ── Members ───────────────────────────────────────────────────────────────────
@application.route("/api/members")
def members():
    return jsonify({"members": db.list_members(), "count": db.count_members()})

@application.route("/api/members/<mid>", methods=["DELETE"])
def del_member(mid):
    db.delete_member(mid)
    return jsonify({"success": True})

# ── Check-ins ─────────────────────────────────────────────────────────────────
@application.route("/api/checkins")
def checkins():
    day = request.args.get("date", date.today().isoformat())
    return jsonify({"checkins": db.get_checkins_for_date(day), "date": day})

# ── Debug probe (calibration) ─────────────────────────────────────────────────
@application.route("/api/debug/probe", methods=["POST"])
def probe():
    """Returns raw scores per member. Use the Calibrate tab to tune threshold."""
    try:
        body = request.get_json(force=True)
        img  = decode_img(body.get("frame", ""))
        emb  = face_engine.extract_embedding(img) if img is not None else None
        if emb is None:
            return jsonify({"face_detected": False})

        all_embs = db.get_all_embeddings()
        results  = []
        for mid, emb_list in all_embs.items():
            m    = db.get_member(mid)
            sims = sorted(
                [float(np.dot(emb, matcher._norm(np.array(e, dtype=np.float32)))) for e in emb_list],
                reverse=True
            )
            top5     = sims[:5]
            weighted = float(np.mean(top5))*0.7 + float(sims[0])*0.3
            results.append({
                "member_id": mid, "name": m["name"] if m else "?",
                "weighted": round(weighted, 4),
                "mean":     round(float(np.mean(top5)), 4),
                "max":      round(float(sims[0]), 4),
                "n":        len(sims)
            })
        results.sort(key=lambda x: x["weighted"], reverse=True)
        rec = round(results[0]["weighted"]*0.80, 3) if results else 0
        return jsonify({
            "face_detected": True,
            "current_threshold": matcher.threshold,
            "scores": results,
            "recommended_threshold": rec,
            "note": "Recommended = 80% of your top score. Lower if you get false negatives."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Live threshold update ─────────────────────────────────────────────────────
@application.route("/api/threshold", methods=["POST"])
def set_threshold():
    """Update threshold live without restarting Flask."""
    try:
        val = float(request.get_json(force=True).get("value", 0))
        if not 0.05 <= val <= 0.99:
            return jsonify({"error": "Must be 0.05–0.99"}), 400
        matcher.threshold = val
        os.environ["MATCH_THRESHOLD"] = str(val)
        log.info(f"Threshold updated to {val}")
        return jsonify({"success": True, "threshold": val})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
