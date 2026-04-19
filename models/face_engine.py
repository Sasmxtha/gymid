
import os, logging
import numpy as np
from typing import Optional

log = logging.getLogger("gymid.face")

class FaceEngine:
    """
    Face detection + 512-dim ArcFace embedding extraction.
    Primary:  InsightFace buffalo_l  (ResNet100, 99.4% accuracy)
    Fallback: InsightFace buffalo_sc (ResNet34,  91% accuracy)
    CLAHE preprocessing applied to every image before embedding.
    """
    def __init__(self):
        self.backend = None
        self._model  = None
        self._init()

    def _init(self):
        # ── Try buffalo_l first (best accuracy, needs ~300MB download) ──
        for model_name in ["buffalo_l", "buffalo_sc"]:
            try:
                from insightface.app import FaceAnalysis
                log.info(f"Loading InsightFace {model_name}...")
                app = FaceAnalysis(
                    name=model_name,
                    root="/content/insightface_models",
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                # buffalo_l works best at 640x640, buffalo_sc at 320x320
                det_size = (640, 640) if model_name == "buffalo_l" else (320, 320)
                app.prepare(ctx_id=0, det_size=det_size)
                self._model  = app
                self.backend = model_name
                log.info(f"✅ Face engine: {model_name} (det_size={det_size})")
                return
            except Exception as e:
                log.warning(f"{model_name} failed: {e}")

        # ── Fallback: face_recognition (dlib) ──
        try:
            import face_recognition
            self._model  = "face_recognition"
            self.backend = "face_recognition"
            log.info("✅ Face engine: face_recognition (dlib)")
            return
        except Exception as e:
            log.warning(f"face_recognition failed: {e}")

        # ── Last resort: OpenCV Haar ──
        import cv2
        self._model  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.backend = "opencv_cascade"
        log.warning("⚠️  Face engine: OpenCV Haar cascade (limited accuracy)")

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_embedding(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Returns a normalised 512-dim embedding, or None if no face found."""
        if img_bgr is None or img_bgr.size == 0:
            return None
        # Always apply CLAHE — fixes all brightness/lighting issues
        img_bgr = self._clahe(img_bgr)
        try:
            if self.backend in ("buffalo_l", "buffalo_sc"):
                return self._embed_insightface(img_bgr)
            elif self.backend == "face_recognition":
                return self._embed_fr(img_bgr)
            else:
                return self._embed_cv(img_bgr)
        except Exception as e:
            log.debug(f"Embedding error: {e}")
            return None

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _clahe(self, img: np.ndarray) -> np.ndarray:
        """
        Contrast Limited Adaptive Histogram Equalisation.
        Normalises lighting per-region — fixes:
          - Dark faces (evening / dim corridor)
          - Backlit faces (person in front of bright entrance)
          - Overexposed faces (harsh fluorescent / midday sun)
          - One-sided shadow (door frame, hat brim)
        """
        try:
            import cv2
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        except:
            return img

    # ── Backend implementations ───────────────────────────────────────────────

    def _embed_insightface(self, img: np.ndarray) -> Optional[np.ndarray]:
        faces = self._model.get(img)
        if not faces:
            return None
        # Pick the largest face (closest to camera)
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )
        emb = faces[0].embedding.astype(np.float32)
        return self._norm(emb)

    def _embed_fr(self, img: np.ndarray) -> Optional[np.ndarray]:
        import face_recognition, cv2
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb, model="large")
        return self._norm(np.array(encs[0], dtype=np.float32)) if encs else None

    def _embed_cv(self, img: np.ndarray) -> Optional[np.ndarray]:
        import cv2
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._model.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if not len(faces):
            return None
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        crop = cv2.resize(gray[y:y+h, x:x+w], (64, 64))
        hist = cv2.calcHist([crop], [0], None, [128], [0, 256]).flatten()
        return self._norm(hist.astype(np.float32))

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n + 1e-9)
