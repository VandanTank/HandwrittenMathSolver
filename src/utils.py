"""
Author: VANDAN TANK
"""

import numpy as np

def find_image_center(img):
    """Return tuple: (cx, cy) coordinate center of img.
    Safe if the image is all zeros (returns center of array).
    """
    total = np.sum(img)
    if total == 0:
        # empty image — return geometric center
        h, w = img.shape
        return int(round(w / 2.0)), int(round(h / 2.0))
    m = img / total
    dx = np.sum(m, axis=0)
    dy = np.sum(m, axis=1)
    cx = int(round(np.sum(dx * np.arange(len(dx))), 0))
    cy = int(round(np.sum(dy * np.arange(len(dy))), 0))
    return cx, cy

def clean_equivalency_dict(d):
    """Clean equivalency dict from connected component algorithm.
    Ensures every key maps to its final representative value.
    """
    # Resolve chains: a -> b -> c becomes a -> c
    for key in list(d.keys()):
        val = d[key]
        while d.get(val, val) != val:
            val = d[val]
        d[key] = val
    return d

def sort_by_other_list(this_list, by_this_list):
    """
    Return `this_list` sorted according to `by_this_list`.

    Handles cases where elements of by_this_list may be numpy arrays
    (e.g. small arrays from image-centering). In that case we convert
    the key to a scalar (try float(), otherwise use mean()) so sorting
    is well-defined.

    If lengths mismatch, returns `this_list` unchanged.
    """
    if len(this_list) != len(by_this_list):
        return this_list

    pairs = []
    for key, val in zip(by_this_list, this_list):
        # try direct float conversion
        try:
            k = float(key)
        except Exception:
            arr = np.asarray(key)
            if arr.size == 0:
                k = 0.0
            elif arr.size == 1:
                k = float(arr.reshape(-1)[0])
            else:
                # take the mean as a fallback scalar
                k = float(arr.mean())
        pairs.append((k, val))

    # stable sort by key
    pairs.sort(key=lambda t: t[0])
    return [v for _, v in pairs]
