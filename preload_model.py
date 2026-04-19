#!/usr/bin/env python3
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("preload")

sys.path.insert(0, os.path.dirname(__file__))

try:
    log.info("Preloading InsightFace model...")
    from models.face_engine import FaceEngine
    engine = FaceEngine()
    log.info(f"Model preloaded successfully: {engine.backend}")
except Exception as e:
    log.warning(f"Model preload failed (this is okay for constrained environments): {e}")
    log.info("The model will be loaded on first use instead")