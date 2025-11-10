"""
Author: VANDAN TANK
"""

import numpy as np
from PIL import Image
import utils
from scipy.ndimage import binary_closing, binary_dilation, label as ndi_label, find_objects


class HandwritingTranslator:
    """Handles image preprocessing, symbol segmentation, and expression translation."""

    def __init__(self, model=1):
        self.M_SIZE = 28
        self.PADDING = 8
        self.MAP_SYMBOLS = {10: '+', 11: '-', 12: '*', 13: '/', 14: '(', 15: ')', 16: '.'}
        if model == 1:
            import keras
            self.model = keras.models.load_model('src/model.h5')

    def image_centering(self, img):
        """Centers a symbol in a fixed-size (28×28) canvas."""
        cx, cy = utils.find_image_center(img)
        p = int(self.M_SIZE / 2)
        img = np.pad(img, ((p, p), (p, p)), 'constant')
        cx += p
        cy += p
        img = img[cy - p:cy + p, cx - p:cx + p]
        return img

    def image_resize(self, img):
        """Resizes symbol while maintaining aspect ratio."""
        h, w = img.shape
        if h == max(h, w):
            h1 = self.M_SIZE - self.PADDING
            w1 = round((h1 / h) * w) if h != 0 else 1
        else:
            w1 = self.M_SIZE - self.PADDING
            h1 = round((w1 / w) * h) if w != 0 else 1
        w1, h1 = max(1, int(w1)), max(1, int(h1))
        image = Image.fromarray(np.uint8(img), 'L')
        image = image.resize((w1, h1))
        return np.array(image)

    def detect_components(self, bin_img):
        """Detects connected components and returns (crop, center_x, bbox)."""
        labeled, n = ndi_label(bin_img)
        if n == 0:
            return []
        slices = find_objects(labeled)
        components = []
        for i, slc in enumerate(slices):
            if slc is None:
                continue
            row_slice, col_slice = slc
            mask = (labeled[row_slice, col_slice] == (i + 1))
            if mask.sum() == 0:
                continue
            cx_local = int(np.round(np.mean(np.where(mask)[1])))
            cx_global = cx_local + (col_slice.start or 0)
            crop = bin_img[row_slice, col_slice].astype(np.uint8)
            bbox = (
                row_slice.start or 0,
                row_slice.stop or bin_img.shape[0],
                col_slice.start or 0,
                col_slice.stop or bin_img.shape[1],
            )
            components.append((crop, cx_global, bbox))
        return components

    def thresholding(self, image):
        """Converts image to binary (1=ink,0=background) and applies morphological smoothing."""
        img = np.array(image)
        bin_img = np.where(img == 0, 1, 0).astype(np.uint8)
        struct5 = np.ones((5, 5), dtype=np.uint8)
        bin_img = binary_dilation(bin_img, structure=struct5).astype(np.uint8)
        bin_img = binary_closing(bin_img, structure=struct5).astype(np.uint8)
        return bin_img

    def predict_batch(self, imgs_list):
        """Predicts a batch of preprocessed symbol images."""
        if len(imgs_list) == 0:
            return []

        arrs = []
        for img in imgs_list:
            a = np.array(img, dtype=np.float32)
            if a.max() <= 1.0:
                a *= 255.0
            a = a.reshape(self.M_SIZE * self.M_SIZE) / 255.0
            arrs.append(a)

        batch = np.stack(arrs, axis=0).reshape(len(arrs), 1, self.M_SIZE * self.M_SIZE)
        res = self.model.predict(batch, verbose=0)

        results = []
        for r in res:
            num_idx = int(r.argmax())
            conf = round(float(r.max()) * 100, 2)
            num_str = self.MAP_SYMBOLS.get(num_idx, str(num_idx)) if num_idx >= 10 else str(num_idx)
            results.append((num_str, conf))
        return results

    def calculate(self, calculation):
        """Evaluates the recognized mathematical expression."""
        try:
            result = eval(calculation)
            if calculation != str(result):
                result = int(result) if int(result) == result else np.round(result, 2)
                return f" {calculation} = {result}"
        except Exception:
            pass
        return f" {calculation}"

    def translate(self, image):
        """Main pipeline: threshold, segment, filter, and recognize handwritten expressions."""
        bin_img = self.thresholding(image)
        comps = self.detect_components(bin_img)
        if not comps:
            return ' '

        comps_sorted = sorted(comps, key=lambda x: x[1])
        clusters = []
        cluster_thresh = 18
        for comp in comps_sorted:
            crop, cx, bbox = comp
            if not clusters:
                clusters.append([comp])
                continue
            last = clusters[-1]
            last_mean = np.mean([c[1] for c in last])
            if abs(cx - last_mean) <= cluster_thresh:
                last.append(comp)
            else:
                clusters.append([comp])

        imgs_ok, cxs = [], []
        h_img, w_img = bin_img.shape
        for cluster in clusters:
            touches_border = any(
                r1 == 0 or c1 == 0 or r2 == h_img or c2 == w_img
                for _, _, (r1, r2, c1, c2) in cluster
            )
            if touches_border:
                continue

            best, best_ink, best_cx, best_bbox = None, -1, None, None
            for crop, cx, bbox in cluster:
                ink = int(np.sum(crop))
                if ink > best_ink:
                    best_ink, best, best_cx, best_bbox = ink, crop, cx, bbox
            if best is None:
                continue

            r1, r2, c1, c2 = best_bbox
            bbox_h, bbox_w = r2 - r1, c2 - c1
            if bbox_w < 14 or bbox_h < 14 or bbox_w / max(1, bbox_h) > 3.5 or bbox_h / max(1, bbox_w) > 3.5:
                continue

            resized = self.image_resize(best)
            centered = self.image_centering(resized)
            if centered.shape != (self.M_SIZE, self.M_SIZE):
                tmp = np.zeros((self.M_SIZE, self.M_SIZE), dtype=np.uint8)
                h0, w0 = min(self.M_SIZE, centered.shape[0]), min(self.M_SIZE, centered.shape[1])
                tmp[:h0, :w0] = centered[:h0, :w0]
                centered = tmp

            if int(np.sum(centered)) < 30:
                continue

            imgs_ok.append(centered)
            cxs.append(best_cx)

        if not imgs_ok:
            return ' '

        ord_idx = np.argsort(np.array([float(x) for x in cxs], dtype=np.float32))
        sorted_imgs = [imgs_ok[i] for i in ord_idx]

        pred_pairs = self.predict_batch(sorted_imgs)
        nums = [num for num, _ in pred_pairs]
        calculation = ''.join(nums)
        return self.calculate(calculation)

    def prep(self, image):
        """Prepares a single symbol to match MNIST format (used for model training/testing)."""
        img = self.thresholding(image)
        comps = self.detect_components(img)
        if not comps:
            return np.zeros((self.M_SIZE, self.M_SIZE), dtype=np.uint8)
        crop, _, _ = comps[0]
        img_resized = self.image_resize(crop)
        return self.image_centering(img_resized)
