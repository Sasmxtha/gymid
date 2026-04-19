#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models.face_engine import FaceEngine

print("Preloading InsightFace model...")
engine = FaceEngine()
print(f"Model loaded: {engine.backend}")