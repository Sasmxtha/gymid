
import random
import numpy as np
from typing import List

class Augmentor:
    """
    Generates 19 augmented variants per photo.
    5 original photos × 20 (1 + 19) = 100 training samples per person.
    Covers: brightness extremes, backlighting, shadows, colour casts,
            motion blur, CCTV quality, glare, perspective shift.
    """

    def augment(self, img_bgr: np.ndarray, n: int = 19) -> List[np.ndarray]:
        results = []
        # Core lighting scenarios — always included
        core = [
            self._very_bright, self._very_dark, self._backlit,
            self._cool_fluoro, self._warm_tungsten,
            self._shadow_left, self._shadow_right, self._shadow_top,
            self._haze, self._high_contrast,
        ]
        for i, fn in enumerate(core):
            if i >= n: break
            aug = img_bgr.copy()
            try: aug = fn(aug); aug = self._geo(aug)
            except: pass
            results.append(aug)

        # Extra scenarios — random fill
        extras = core + [
            self._motion_blur, self._low_res, self._overexposed,
            self._underexposed, self._green_tint, self._yellow_tint,
            self._blue_tint, self._partial_shadow, self._glare,
        ]
        while len(results) < n:
            aug = img_bgr.copy()
            try:
                aug = random.choice(extras)(aug)
                aug = self._geo(aug)
                aug = self._noise(aug)
            except: pass
            results.append(aug)
        return results

    # ── Lighting scenarios ────────────────────────────────────────────────────

    def _very_bright(self, img):
        import cv2
        h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h[:,:,2] = np.clip(h[:,:,2]*random.uniform(1.6,2.2)+random.uniform(30,60), 0, 255)
        h[:,:,1] = np.clip(h[:,:,1]*random.uniform(0.5,0.75), 0, 255)
        return cv2.cvtColor(h.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _very_dark(self, img):
        import cv2
        h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h[:,:,2] = np.clip(h[:,:,2]*random.uniform(0.18,0.42), 0, 255)
        return cv2.cvtColor(h.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _backlit(self, img):
        import cv2
        h, w = img.shape[:2]
        dark = np.clip(img.astype(np.float32)*random.uniform(0.15,0.38), 0, 255).astype(np.uint8)
        cx, cy = w//2, h//2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
        mask = np.clip((dist/np.sqrt(cx**2+cy**2))*200, 0, 120).astype(np.uint8)
        m3 = cv2.merge([mask, mask, mask])
        return np.clip(dark.astype(np.float32)+m3.astype(np.float32)*random.uniform(0.5,1.0), 0, 255).astype(np.uint8)

    def _cool_fluoro(self, img):
        r = img.astype(np.float32); b = random.uniform(1.1,1.4)
        r[:,:,0] = np.clip(r[:,:,0]*b*random.uniform(1.05,1.18), 0, 255)  # boost blue
        r[:,:,1] = np.clip(r[:,:,1]*b*1.0, 0, 255)
        r[:,:,2] = np.clip(r[:,:,2]*b*random.uniform(0.86,0.96), 0, 255)  # reduce red
        return r.astype(np.uint8)

    def _warm_tungsten(self, img):
        r = img.astype(np.float32); b = random.uniform(0.85,1.2)
        r[:,:,2] = np.clip(r[:,:,2]*b*random.uniform(1.15,1.38), 0, 255)  # boost red
        r[:,:,1] = np.clip(r[:,:,1]*b*1.0, 0, 255)
        r[:,:,0] = np.clip(r[:,:,0]*b*random.uniform(0.58,0.80), 0, 255)  # reduce blue
        return r.astype(np.uint8)

    def _shadow_left(self, img):  return self._side_shadow(img, "left",  random.uniform(0.18,0.50))
    def _shadow_right(self, img): return self._side_shadow(img, "right", random.uniform(0.18,0.50))

    def _shadow_top(self, img):
        import cv2
        h, w = img.shape[:2]
        mask = np.ones((h,w), dtype=np.float32)
        sh = int(h*random.uniform(0.25,0.45))
        for row in range(sh):
            mask[row,:] = random.uniform(0.18,0.42) + (row/sh)*0.58
        return np.clip(img.astype(np.float32)*cv2.merge([mask,mask,mask]), 0, 255).astype(np.uint8)

    def _haze(self, img):
        haze = np.full_like(img, random.randint(140,205), dtype=np.float32)
        a = random.uniform(0.28,0.55)
        return np.clip(img.astype(np.float32)*(1-a)+haze*a, 0, 255).astype(np.uint8)

    def _high_contrast(self, img):
        f = random.uniform(1.8,2.8)
        return np.clip((img.astype(np.float32)-128)*f+128, 0, 255).astype(np.uint8)

    def _motion_blur(self, img):
        import cv2
        k = random.choice([5,7,9,11])
        kern = np.zeros((k,k))
        if random.random() > 0.5: kern[k//2,:] = 1/k
        else: kern[:,k//2] = 1/k
        return cv2.filter2D(img, -1, kern)

    def _low_res(self, img):
        import cv2
        h, w = img.shape[:2]
        s  = random.uniform(0.22,0.50)
        sm = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_LINEAR)
        up = cv2.resize(sm, (w,h), interpolation=cv2.INTER_NEAREST)
        _, buf = cv2.imencode(".jpg", up, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(30,65)])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def _overexposed(self, img):
        import cv2
        g = random.uniform(0.25,0.52)
        t = np.array([((i/255)**g)*255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, t)

    def _underexposed(self, img):
        import cv2
        g   = random.uniform(2.2,3.8)
        t   = np.array([((i/255)**g)*255 for i in range(256)], dtype=np.uint8)
        res = cv2.LUT(img, t)
        nse = np.random.normal(0, random.uniform(8,20), img.shape).astype(np.float32)
        return np.clip(res.astype(np.float32)+nse, 0, 255).astype(np.uint8)

    def _green_tint(self, img):
        r = img.astype(np.float32)
        r[:,:,1] = np.clip(r[:,:,1]*random.uniform(1.10,1.30), 0, 255)
        r[:,:,0] = np.clip(r[:,:,0]*random.uniform(0.80,0.93), 0, 255)
        return r.astype(np.uint8)

    def _yellow_tint(self, img):
        r = img.astype(np.float32)
        r[:,:,2] = np.clip(r[:,:,2]*random.uniform(1.10,1.28), 0, 255)
        r[:,:,1] = np.clip(r[:,:,1]*random.uniform(1.05,1.15), 0, 255)
        r[:,:,0] = np.clip(r[:,:,0]*random.uniform(0.58,0.76), 0, 255)
        return r.astype(np.uint8)

    def _blue_tint(self, img):
        r = img.astype(np.float32)
        r[:,:,0] = np.clip(r[:,:,0]*random.uniform(1.15,1.38), 0, 255)
        r[:,:,2] = np.clip(r[:,:,2]*random.uniform(0.76,0.92), 0, 255)
        return r.astype(np.uint8)

    def _partial_shadow(self, img):
        import cv2
        h, w = img.shape[:2]
        mask = np.ones((h,w), dtype=np.float32)
        cx   = random.randint(w//5, 4*w//5)
        cy   = random.randint(h//5, 4*h//5)
        rx   = random.randint(w//6, w//2)
        ry   = random.randint(h//6, h//2)
        Y, X = np.ogrid[:h, :w]
        mask[((X-cx)/rx)**2 + ((Y-cy)/ry)**2 <= 1] = random.uniform(0.18,0.50)
        mask = cv2.GaussianBlur(mask, (51,51), 0)
        return np.clip(img.astype(np.float32)*cv2.merge([mask,mask,mask]), 0, 255).astype(np.uint8)

    def _glare(self, img):
        import cv2
        h, w = img.shape[:2]
        g = img.astype(np.float32).copy()
        for _ in range(random.randint(1,2)):
            cx = random.randint(w//5, 4*w//5)
            cy = random.randint(h//8, h//2)
            r  = random.randint(w//12, w//4)
            Y, X = np.ogrid[:h, :w]
            spot = np.clip(1.0 - np.sqrt((X-cx)**2+(Y-cy)**2)/r, 0, 1)**2
            s3   = cv2.merge([spot.astype(np.float32)]*3)
            g    = np.clip(g + s3*random.uniform(100,215), 0, 255)
        return g.astype(np.uint8)

    # ── Geometric + noise helpers ─────────────────────────────────────────────

    def _geo(self, img):
        ops = [self._flip, self._rotate, self._crop, self._warp]
        for op in random.sample(ops, k=random.randint(1,2)):
            try: img = op(img)
            except: pass
        return img

    def _noise(self, img):
        n = np.random.normal(0, random.uniform(2,14), img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32)+n, 0, 255).astype(np.uint8)

    def _flip(self, img):
        import cv2; return cv2.flip(img, 1)

    def _rotate(self, img):
        import cv2
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), random.uniform(-14,14), 1.0)
        return cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REFLECT)

    def _crop(self, img):
        import cv2
        h, w = img.shape[:2]
        m = int(min(h,w)*random.uniform(0.03,0.10))
        t, l = random.randint(0,m), random.randint(0,m)
        b, r = h-random.randint(0,m), w-random.randint(0,m)
        return cv2.resize(img[t:b, l:r], (w,h))

    def _warp(self, img):
        import cv2
        h, w = img.shape[:2]
        m   = int(min(h,w)*random.uniform(0.02,0.08))
        src = np.float32([[0,0],[w,0],[0,h],[w,h]])
        dst = np.float32([
            [random.randint(0,m), random.randint(0,m)],
            [w-random.randint(0,m), random.randint(0,m)],
            [random.randint(0,m), h-random.randint(0,m)],
            [w-random.randint(0,m), h-random.randint(0,m)],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, (w,h), borderMode=cv2.BORDER_REFLECT)

    def _side_shadow(self, img, side, intensity):
        import cv2
        h, w  = img.shape[:2]
        mask  = np.ones((h,w), dtype=np.float32)
        sw    = int(w*random.uniform(0.28,0.55))
        fw    = int(w*random.uniform(0.10,0.22))
        if side == "left":
            for c in range(sw): mask[:,c] = intensity + (1-intensity)*min(c/max(fw,1), 1)
        else:
            for c in range(w-sw, w): mask[:,c] = 1-(1-intensity)*min((c-(w-sw))/max(fw,1), 1)
        mask = cv2.GaussianBlur(mask, (31,31), 0)
        return np.clip(img.astype(np.float32)*cv2.merge([mask,mask,mask]), 0, 255).astype(np.uint8)
