"""Stage 1: foreground segmentation.

Separates the object from the background with U^2-Net (via rembg), then crops
and re-centres it. The same segmenter is reused after stage 2 to get a
silhouette mask for each generated view, which the reconstruction uses as
supervision.
"""

from __future__ import annotations

import threading

import numpy as np
from PIL import Image

from .errors import PreprocessError

_session = None
_session_lock = threading.Lock()


def _rembg_session():
    global _session
    with _session_lock:
        if _session is None:
            try:
                import rembg
            except ImportError as exc:  # pragma: no cover - depends on install
                raise PreprocessError(
                    "Background removal needs the 'rembg' package. Install it, "
                    "or send remove_background=false with an RGBA image."
                ) from exc
            # u2net is rembg's default and what TripoSR's own demo uses.
            _session = rembg.new_session("u2net")
        return _session


def remove_background(image: Image.Image) -> Image.Image:
    """Return an RGBA image whose alpha channel masks out the background."""
    import rembg

    return rembg.remove(image.convert("RGB"), session=_rembg_session()).convert("RGBA")


def resize_foreground(image: Image.Image, ratio: float) -> Image.Image:
    """Crop to the visible object, pad to a square, and leave a margin so the
    object fills `ratio` of the frame (same idea as TripoSR's helper)."""
    rgba = np.asarray(image.convert("RGBA"))
    ys, xs = np.nonzero(rgba[..., 3] > 0)
    if len(ys) == 0:
        raise PreprocessError("No foreground found in the image after background removal.")
    fg = rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = fg.shape[:2]
    size = max(h, w)
    canvas = int(round(size / ratio))
    out = np.zeros((canvas, canvas, 4), dtype=np.uint8)
    top = (canvas - h) // 2
    left = (canvas - w) // 2
    out[top : top + h, left : left + w] = fg
    return Image.fromarray(out, mode="RGBA")


def prepare(image: Image.Image, remove_bg: bool, foreground_ratio: float) -> Image.Image:
    """Return an RGBA image, object centred, ready for a backend."""
    image = image.convert("RGBA") if image.mode in ("RGBA", "LA", "P") else image.convert("RGB")
    if remove_bg:
        image = remove_background(image)
    if image.mode == "RGBA":
        return resize_foreground(image, foreground_ratio)
    # No alpha and no removal requested: use the picture as it is.
    return image.convert("RGBA")


def composite_on_gray(image: Image.Image) -> Image.Image:
    """Flatten RGBA onto 50% gray, which is the input TripoSR was trained on."""
    rgba = np.asarray(image.convert("RGBA")).astype(np.float32) / 255.0
    rgb = rgba[..., :3] * rgba[..., 3:4] + (1.0 - rgba[..., 3:4]) * 0.5
    return Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")


def view_mask(view: Image.Image) -> np.ndarray:
    """Foreground probability in [0, 1] for one generated view (H, W) float32."""
    alpha = np.asarray(remove_background(view).getchannel("A"), dtype=np.float32) / 255.0
    return alpha
