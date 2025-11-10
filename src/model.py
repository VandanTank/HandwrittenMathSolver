"""
Author: VANDAN TANK
(final cleaned src/model.py: robust preprocessing + label-based detection +
batched predict + clustering + bbox & ink filtering)
"""

import numpy as np
from PIL import Image
import utils

# stronger morphological ops and labeling
from scipy.ndimage import binary_closing, binary_dilation, label as ndi_label, find_objects


class HandwritingTranslator(object):
	"""Preprocessing handwritten image data and translate to pattern."""

	def __init__(self, model=1):
		self.M_SIZE = 28     # size of each sample (MNIST: 28x28)
		self.PADDING = 8     # number of empty pixel row / column (the smallest one)
		# mapping for symbols (>=10 are operators in this project's encoding)
		self.MAP_SYMBOLS = {10: '+', 11: '-', 12: '*', 13: '/', 14: '(', 15: ')', 16: '.'}
		if model == 1:
			import keras
			self.model = keras.models.load_model('src/model.h5')     # keras model for predict symbol

	# ------------------------- image helpers -------------------------
	def image_centering(self, img):
		"""Return M_SIZE x M_SIZE array of img on center with padding."""
		cx, cy = utils.find_image_center(img)
		p = int(self.M_SIZE / 2)
		img = np.pad(img, ((p,p),(p,p)), 'constant')
		cx += p
		cy += p
		img = img[cy-p:cy+p, cx-p:cx+p]
		return img

	def image_resize(self, img):
		"""Resize image keeping aspect ratio to MNIST sample scale."""
		h, w = img.shape
		if h == max(h, w):
			h1 = self.M_SIZE - self.PADDING
			w1 = round((h1 / h) * w) if h != 0 else 1
		else:
			w1 = self.M_SIZE - self.PADDING
			h1 = round((w1 / w) * h) if w != 0 else 1
		# ensure minimum size 1x1
		w1 = max(1, int(w1))
		h1 = max(1, int(h1))
		image = Image.fromarray(np.uint8(img), 'L')
		image = image.resize((w1, h1))
		img = np.array(image)
		return img

	# ------------------------- detection using scipy label (returns bbox info) -------------------------
	def detect_components(self, bin_img):
		"""
		Detect connected components using scipy.ndimage.label.
		Returns list of tuples (crop_array, center_x, bbox)
		where bbox = (row_start,row_stop,col_start,col_stop) in image coords.
		`bin_img` expected to be binary with 1 = ink, 0 = bg.
		"""
		# Label connected components
		labeled, n = ndi_label(bin_img)
		if n == 0:
			return []

		# find slices bounding boxes for each label
		slices = find_objects(labeled)

		components = []
		for i, slc in enumerate(slices):
			if slc is None:
				continue
			row_slice, col_slice = slc
			# extract boolean mask of this component
			mask = (labeled[row_slice, col_slice] == (i + 1))
			if mask.sum() == 0:
				continue
			# compute local center x and global cx
			cx_local = int(np.round(np.mean(np.where(mask)[1])))
			cx_global = cx_local + (col_slice.start if col_slice.start is not None else 0)
			# extract the actual crop in original image coordinates (1=ink)
			crop = bin_img[row_slice, col_slice].astype(np.uint8)
			bbox = (
				row_slice.start if row_slice.start is not None else 0,
				row_slice.stop if row_slice.stop is not None else bin_img.shape[0],
				col_slice.start if col_slice.start is not None else 0,
				col_slice.stop if col_slice.stop is not None else bin_img.shape[1]
			)
			components.append((crop, cx_global, bbox))
		return components

	# ------------------------- threshold + morphological smoothing -------------------------
	def thresholding(self, image):
		"""Convert image to binary (1=ink,0=background) and smooth it strongly."""
		img = np.array(image)  # grayscale 0..255

		# Map original project's convention: 0 => ink (black). Keep same mapping.
		bin_img = np.where(img == 0, 1, 0).astype(np.uint8)

		# aggressive smoothing: dilation followed by closing (fills gaps and joins strokes)
		struct5 = np.ones((5, 5), dtype=np.uint8)
		bin_img = binary_dilation(bin_img, structure=struct5).astype(np.uint8)
		bin_img = binary_closing(bin_img, structure=struct5).astype(np.uint8)

		return bin_img

	# ------------------------- batch prediction (robust scaling) -------------------------
	def predict_batch(self, imgs_list):
		"""
		Batch-predict a list of preprocessed imgs (each M_SIZE x M_SIZE arrays).
		Returns list of (num_str, confidence_float).
		"""
		if len(imgs_list) == 0:
			return []

		arrs = []
		for img in imgs_list:
			a = np.array(img, dtype=np.float32)
			# if image is binary (0/1) scale to 0-255 first so dividing by 255 yields 0-1
			if a.max() <= 1.0:
				a = a * 255.0
			# now normalize to 0-1 (model trained on values in this range)
			a = a.reshape(self.M_SIZE * self.M_SIZE) / 255.0
			arrs.append(a)

		batch = np.stack(arrs, axis=0)            # shape (n, 784)
		# add time axis -> (n, 1, 784)
		batch = batch.reshape(batch.shape[0], 1, batch.shape[1])

		# single batched predict (silent)
		res = self.model.predict(batch, verbose=0)

		results = []
		for r in res:
			num_idx = int(r.argmax())
			conf = round(float(r.max()) * 100, 2)
			if num_idx >= 10:
				num_str = self.MAP_SYMBOLS.get(num_idx, '?')
			else:
				num_str = str(num_idx)
			results.append((num_str, conf))

		return results

	# ------------------------- calculation helper -------------------------
	def calculate(self, calculation):
		"""Return text containing calculation and answer for displaying to user."""
		try:
			result = eval(calculation)
			if calculation == str(result):
				result = ''
			else:
				if int(result) == result:
					result = int(result)
				else:
					result = np.round(result, 2)
				result = ' = ' + str(result)
		except Exception:
			result = ''
		result = ' ' + calculation + result
		return result

	# ------------------------- main translate pipeline (clustering + bbox & ink filters) -------------------------
	def translate(self, image):
		"""
		Return formatted display string for the whole image (one shot).
		Steps:
		  - threshold + smooth
		  - detect components via labeling
		  - cluster nearby components horizontally (to avoid split-digit repeats)
		  - from each cluster pick best subcomponent (largest ink), apply bbox + border + noise filters
		  - sort by horizontal center and batch-predict
		"""
		# 1) threshold & smoothing
		bin_img = self.thresholding(image)

		# 2) detect components using scipy label (crop, cx, bbox)
		comps = self.detect_components(bin_img)
		if not comps:
			return ' '

		# 3) sort components left->right and cluster by horizontal proximity
		comps_sorted = sorted(comps, key=lambda x: x[1])  # sort by cx
		clusters = []
		cluster_thresh = 18  # pixels; if centers within this, group them
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

		# 4) from each cluster pick the best (most ink), apply border/noise and bbox-size filtering
		imgs_ok = []
		cxs = []
		h_img, w_img = bin_img.shape

		for cluster in clusters:
			# If any member of this cluster touches the border, skip the whole cluster
			touches_border = False
			for _, _, bbox in cluster:
				row_s, row_e, col_s, col_e = bbox
				if row_s == 0 or col_s == 0 or row_e == h_img or col_e == w_img:
					touches_border = True
					break
			if touches_border:
				continue

			# choose the member with largest ink (most likely main stroke)
			best = None
			best_ink = -1
			best_cx = None
			best_bbox = None
			for crop, cx, bbox in cluster:
				ink = int(np.sum(crop))
				if ink > best_ink:
					best_ink = ink
					best = crop
					best_cx = cx
					best_bbox = bbox
			if best is None:
				continue

			# bounding-box size filter: ignore too-small components or very long-thin fragments
			row_s, row_e, col_s, col_e = best_bbox
			bbox_h = row_e - row_s
			bbox_w = col_e - col_s
			if bbox_w < 14 or bbox_h < 14 or bbox_w / max(1, bbox_h) > 3.5 or bbox_h / max(1, bbox_w) > 3.5:
				continue

			# resize and center chosen crop
			resized = self.image_resize(best)
			centered = self.image_centering(resized)
			if centered.shape != (self.M_SIZE, self.M_SIZE):
				tmp = np.zeros((self.M_SIZE, self.M_SIZE), dtype=np.uint8)
				h0 = min(self.M_SIZE, centered.shape[0])
				w0 = min(self.M_SIZE, centered.shape[1])
				tmp[:h0, :w0] = centered[:h0, :w0]
				centered = tmp

			# filter tiny/noisy crops (stricter)
			if int(np.sum(centered)) < 30:
				continue

			imgs_ok.append(centered)
			cxs.append(best_cx)

		# if no symbols found
		if not imgs_ok:
			return ' '

		# 5) sort by horizontal center and batch-predict
		ord_idx = np.argsort(np.array([float(x) for x in cxs], dtype=np.float32))
		sorted_imgs = [imgs_ok[i] for i in ord_idx]

		pred_pairs = self.predict_batch(sorted_imgs)
		nums = [num for num, conf in pred_pairs]
		calculation = ''.join(nums)
		display = self.calculate(calculation)
		return display

	# ------------------------- prep helper -------------------------
	def prep(self, image):
		"""Prepare single image to be formatted image like MNIST sample image."""
		img = self.thresholding(image)
		# use labeling to find first component if any
		comps = self.detect_components(img)
		if not comps:
			return np.zeros((self.M_SIZE, self.M_SIZE), dtype=np.uint8)
		crop, _, _ = comps[0]
		img_resized = self.image_resize(crop)
		img_centered = self.image_centering(img_resized)
		return img_centered
