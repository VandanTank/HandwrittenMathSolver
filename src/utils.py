"""
Author: VANDAN TANK
Utility helpers for image centering, connected-component cleaning, and sorting.
"""

import numpy as np


def find_image_center(img):
    """
    Return (cx, cy): the center of mass of `img`.
    If the image has no ink (all zeros), returns the geometric center.
    """
    total = np.sum(img)
    if total == 0:
        h, w = img.shape
        return int(round(w / 2.0)), int(round(h / 2.0))
    m = img / total
    dx = np.sum(m, axis=0)
    dy = np.sum(m, axis=1)
    cx = int(round(np.sum(dx * np.arange(len(dx))), 0))
    cy = int(round(np.sum(dy * np.arange(len(dy))), 0))
    return cx, cy


def clean_equivalency_dict(d):
    """
    Resolve equivalency chains produced by a connected-component algorithm.
    Ensures each key maps to its final representative value.
    """
    for key in list(d.keys()):
        val = d[key]
        while d.get(val, val) != val:
            val = d[val]
        d[key] = val
    return d


def sort_by_other_list(this_list, by_this_list):
    """
    Return `this_list` sorted according to the numeric order of `by_this_list`.

    Converts keys in `by_this_list` to scalars (float). If a key is an array,
    its mean is used as a fallback. If lengths differ, returns `this_list` unchanged.
    """
    if len(this_list) != len(by_this_list):
        return this_list

    pairs = []
    for key, val in zip(by_this_list, this_list):
        try:
            k = float(key)
        except Exception:
            arr = np.asarray(key)
            if arr.size == 0:
                k = 0.0
            elif arr.size == 1:
                k = float(arr.reshape(-1)[0])
            else:
                k = float(arr.mean())
        pairs.append((k, val))

    pairs.sort(key=lambda t: t[0])
    return [v for _, v in pairs]
