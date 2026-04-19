
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

db = Database()

# Lazy-load heavy components
_face_engine = None
_matcher = None
_augmentor = None
_initialization_lock = __import__('threading').Lock()
_initialization_error = None

def get_face_engine():
    """Lazy initialize face engine on first use."""
    global _face_engine, _initialization_error
    if _face_engine is not None:
        return _face_engine
    with _initialization_lock:
        if _face_engine is not None:
            return _face_engine
        if _initialization_error:
            raise _initialization_error
        try:
            log.info("Initialising FaceEngine — model download may occur on first run...")
            _face_engine = FaceEngine()
            log.info(f"FaceEngine initialised successfully (backend={_face_engine.backend})")
            return _face_engine
        except Exception as e:
            _initialization_error = e
            log.error(f"FaceEngine failed to initialise: {e}", exc_info=True)
            raise

def get_matcher():
    """Lazy initialize matcher on first use."""
    global _matcher
    if _matcher is None:
        engine = get_face_engine()
        _matcher = FaceMatcher(backend=engine.backend)
    return _matcher

def get_augmentor():
    """Lazy initialize augmentor on first use."""
    global _augmentor
    if _augmentor is None:
        _augmentor = Augmentor()
    return _augmentor

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
    try:
        engine_backend = get_face_engine().backend if _face_engine else "initializing"
    except Exception as e:
        engine_backend = f"error: {str(e)[:30]}"
    
    return jsonify({
        "status": "ok",
        "members": db.count_members(),
        "engine": engine_backend,
        "threshold": get_matcher().threshold if _face_engine else None,
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
        try:
            images = [img for img in (decode_img(p) for p in body["photos"]) if img is not None]
            log.info(f"Register: Decoded {len(images)} images from {len(body['photos'])} photos")
        except Exception as e:
            log.error(f"Register: Image decode failed: {e}", exc_info=True)
            return jsonify({"error": f"Failed to decode photos: {str(e)[:50]}"}), 400

        # Augment: 5 photos × 20 = 100 samples
        try:
            augmented = []
            for img in images:
                augmented.append(img)
                augmented.extend(get_augmentor().augment(img, n=19))
            log.info(f"Register: Augmented to {len(augmented)} images")
        except Exception as e:
            log.error(f"Register: Augmentation failed: {e}", exc_info=True)
            return jsonify({"error": f"Image augmentation failed: {str(e)[:50]}"}), 500

        # Extract embeddings
        try:
            engine = get_face_engine()
            log.info(f"Register: FaceEngine loaded (backend={engine.backend})")
            embeddings = [e for e in (engine.extract_embedding(img) for img in augmented) if e is not None]
            log.info(f"Register: Extracted {len(embeddings)} embeddings from {len(augmented)} augmented images")
        except Exception as e:
            log.error(f"Register: Embedding extraction failed: {e}", exc_info=True)
            return jsonify({"error": f"Face detection failed: {str(e)[:50]}"}), 500
        
        if len(embeddings) < 5:
            return jsonify({"error": "Could not detect face. Better lighting, no mask, face centred."}), 422

        # Save to DB
        try:
            log.info(f"Register: Inserting member {body['id']}")
            db.insert_member({
                "id": body["id"], "name": body["name"], "email": body["email"],
                "phone": body["phone"], "plan": body["plan"],
                "photo_count": len(body["photos"]), "embedding_count": len(embeddings)
            })
            log.info(f"Register: Inserted member, now saving {len(embeddings)} embeddings")
            for i, emb in enumerate(embeddings):
                db.insert_embedding(body["id"], emb.tolist(), is_mean=False)
            
            # Store mean embedding too
            mean_emb = np.mean(embeddings, axis=0)
            mean_emb /= (np.linalg.norm(mean_emb) + 1e-9)
            db.insert_embedding(body["id"], mean_emb.tolist(), is_mean=True)
            log.info(f"Register: Saved all embeddings for {body['id']}")
        except Exception as e:
            log.error(f"Register: Database save failed: {e}", exc_info=True)
            # Try to clean up
            try:
                db.delete_member(body["id"])
            except:
                pass
            return jsonify({"error": f"Failed to save member: {str(e)[:50]}"}), 500

        log.info(f"Registered {body['name']} — {len(embeddings)} embeddings")
        return jsonify({"success": True, "member_id": body["id"], "embeddings_stored": len(embeddings)})

    except Exception as e:
        log.error(f"Register: Unexpected error: {traceback.format_exc()}")
        return jsonify({"error": f"Unexpected error: {str(e)[:50]}"}), 500

# ── Detect ────────────────────────────────────────────────────────────────────
@application.route("/api/detect", methods=["POST"])
def detect():
    try:
        body = request.get_json(force=True)
        img  = decode_img(body.get("frame", ""))
        if img is None:
            return jsonify({"error": "Bad frame"}), 400

        q_emb = get_face_engine().extract_embedding(img)
        if q_emb is None:
            return jsonify({"matched": False, "confidence": 0.0,
                            "message": "No face detected. Look at the camera."})

        all_embs = db.get_all_embeddings()
        if not all_embs:
            return jsonify({"matched": False, "confidence": 0.0, "message": "No members registered."})

        member_id, score, debug = get_matcher().match(q_emb, all_embs)
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
            thr  = debug.get("threshold", get_matcher().threshold)
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
        emb  = get_face_engine().extract_embedding(img) if img is not None else None
        if emb is None:
            return jsonify({"face_detected": False})

        all_embs = db.get_all_embeddings()
        results  = []
        for mid, emb_list in all_embs.items():
            m    = db.get_member(mid)
            sims = sorted(
                [float(np.dot(emb, get_matcher()._norm(np.array(e, dtype=np.float32)))) for e in emb_list],
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
            "current_threshold": get_matcher().threshold,
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
        get_matcher().threshold = val
        os.environ["MATCH_THRESHOLD"] = str(val)
        log.info(f"Threshold updated to {val}")
        return jsonify({"success": True, "threshold": val})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
