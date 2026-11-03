// Python helper code shared by generated steps. Steps request helpers with `ctx.helper(name)`; step files can
// register their own with `definePyHelper`. `Targets` and the postprocess_* inverses are always emitted.
import { dedent } from '../py';

interface PyHelper {
  code: string;
  deps?: string[];
}

const helpers = new Map<string, PyHelper>();

export function definePyHelper(name: string, code: string, deps: string[] = []): void {
  helpers.set(name, { code: dedent(code), deps });
}

/** Requested helper names plus their dependencies, in registration order. */
export function resolvePyHelpers(names: Iterable<string>): string[] {
  const want = new Set<string>();
  const visit = (n: string) => {
    if (want.has(n)) return;
    const h = helpers.get(n);
    if (!h) throw new Error(`Unknown Python helper: ${n}`);
    for (const d of h.deps ?? []) visit(d);
    want.add(n);
  };
  for (const n of names) visit(n);
  return [...helpers.keys()].filter((k) => want.has(k));
}

export function pyHelperCode(name: string): string {
  return helpers.get(name)!.code;
}

// ---------------------------------------------------------------------------
// Random numbers: only rng.random() is drawn so the browser preview reproduces the stream exactly.
// ---------------------------------------------------------------------------

definePyHelper(
  'uniform',
  `
  def _uniform(rng, lo, hi):
      """Float in [lo, hi). Built on rng.random() only, so the Studio preview draws the same numbers."""
      return lo + (hi - lo) * rng.random()
  `,
);

definePyHelper(
  'randint',
  `
  def _randint(rng, lo, hi):
      """Integer in [lo, hi] inclusive, from one rng.random() draw."""
      return lo + min(int(rng.random() * (hi - lo + 1)), hi - lo)
  `,
);

definePyHelper(
  'normal',
  `
  def _normal(rng, shape):
      """Standard normal samples (Box-Muller on two rng.random() arrays), float64."""
      u1 = rng.random(shape)
      u2 = rng.random(shape)
      return np.sqrt(-2.0 * np.log(1.0 - u1)) * np.cos(2.0 * np.pi * u2)
  `,
);

definePyHelper(
  'saturate',
  `
  def _saturate(x, dtype):
      """Round and clip a float array into an integer dtype (what OpenCV's saturate_cast does)."""
      info = np.iinfo(dtype)
      return np.clip(np.rint(x), info.min, info.max).astype(dtype)
  `,
);

definePyHelper(
  'luminance',
  `
  def _on_luminance(img, fn, space="BGR"):
      """Apply fn to the lightness channel only (L of LAB), so colours do not shift. Gray images: fn(img)."""
      if img.ndim == 2:
          return fn(img)
      to_lab = cv2.COLOR_RGB2LAB if space == "RGB" else cv2.COLOR_BGR2LAB
      from_lab = cv2.COLOR_LAB2RGB if space == "RGB" else cv2.COLOR_LAB2BGR
      lab = cv2.cvtColor(img, to_lab)
      lab[..., 0] = fn(np.ascontiguousarray(lab[..., 0]))
      return cv2.cvtColor(lab, from_lab)
  `,
);

// ---------------------------------------------------------------------------
// Targets and inverse transforms (always emitted)
// ---------------------------------------------------------------------------

export const TARGETS_PY = dedent(`
  MIN_BOX_VISIBILITY = 0.1  # boxes keeping less of their area after clipping are dropped

  # Pixel-index matrices (OpenCV) ↔ continuous coordinates (pixel edges at integers) used for boxes / keypoints.
  _TO_CONT = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
  _FROM_CONT = np.array([[1.0, 0.0, -0.5], [0.0, 1.0, -0.5], [0.0, 0.0, 1.0]])


  def _apply_h(H, pts):
      """Apply a 3×3 homography to (N, 2) points."""
      p = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ H.T
      return p[:, :2] / p[:, 2:3]


  def _box_corners(boxes):
      b = boxes
      return np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]], axis=1).reshape(-1, 2)


  class Targets:
      """Mask, boxes and keypoints that follow the image through geometric steps.

      mask:      (H, W) or (H, W, C) label image, always warped with INTER_NEAREST so class ids stay intact.
      boxes:     (N, 4+) array of x1, y1, x2, y2 in pixels; extra columns (e.g. class ids) are carried along.
      keypoints: (N, 2+) array of x, y in pixels.
      history:   every geometric transform applied, for postprocess_boxes / _mask / _keypoints.
      """

      def __init__(self, image_size, mask=None, boxes=None, keypoints=None):
          self.size = tuple(image_size)  # (width, height)
          self.mask = mask
          self.boxes = np.zeros((0, 4)) if boxes is None else np.asarray(boxes, np.float64).reshape(len(boxes), -1)
          self.keypoints = np.zeros((0, 2)) if keypoints is None else np.asarray(keypoints, np.float64).reshape(len(keypoints), -1)
          self.history = []

      def _points(self, H, dst):
          if len(self.boxes):
              pts = _apply_h(H, _box_corners(self.boxes)).reshape(-1, 4, 2)
              x1, y1 = pts[..., 0].min(1), pts[..., 1].min(1)
              x2, y2 = pts[..., 0].max(1), pts[..., 1].max(1)
              area = (x2 - x1) * (y2 - y1)
              cx1, cx2 = np.clip(x1, 0, dst[0]), np.clip(x2, 0, dst[0])
              cy1, cy2 = np.clip(y1, 0, dst[1]), np.clip(y2, 0, dst[1])
              clipped = (cx2 - cx1) * (cy2 - cy1)
              keep = (area > 0) & (clipped > 0) & (clipped >= MIN_BOX_VISIBILITY * area)
              self.boxes = np.concatenate([np.stack([cx1, cy1, cx2, cy2], 1), self.boxes[:, 4:]], 1)[keep]
          if len(self.keypoints):
              self.keypoints = np.concatenate([_apply_h(H, self.keypoints[:, :2]), self.keypoints[:, 2:]], 1)

      def _record(self, kind, H, dst, arg=None):
          self.history.append({"kind": kind, "H": H, "src": self.size, "dst": tuple(dst), "arg": arg})
          self.size = tuple(dst)

      def resize(self, dst):
          """Follow cv2.resize to dst = (width, height)."""
          if self.mask is not None:
              self.mask = cv2.resize(self.mask, dst, interpolation=cv2.INTER_NEAREST)
          H = np.array([[dst[0] / self.size[0], 0, 0], [0, dst[1] / self.size[1], 0], [0, 0, 1.0]])
          self._points(H, dst)
          self._record("resize", H, dst)

      def warp(self, M, dst):
          """Follow cv2.warpAffine (2×3 M) or cv2.warpPerspective (3×3 M) to dst = (width, height)."""
          M = np.asarray(M, np.float64)
          M3 = np.vstack([M, [0, 0, 1]]) if M.shape[0] == 2 else M
          if self.mask is not None:
              warp = cv2.warpAffine if M.shape[0] == 2 else cv2.warpPerspective
              self.mask = warp(self.mask, M, dst, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
          H = _TO_CONT @ M3 @ _FROM_CONT
          self._points(H, dst)
          self._record("warp", H, dst, M3)

      def flip(self, code):
          """Follow cv2.flip: 1 horizontal, 0 vertical, -1 both."""
          w, h = self.size
          if self.mask is not None:
              self.mask = cv2.flip(self.mask, code)
          fx, fy = code != 0, code != 1
          H = np.array([[-1 if fx else 1, 0, w if fx else 0], [0, -1 if fy else 1, h if fy else 0], [0, 0, 1.0]])
          self._points(H, self.size)
          self._record("flip", H, self.size, code)

      def crop(self, x, y, w, h):
          if self.mask is not None:
              self.mask = self.mask[y : y + h, x : x + w].copy()
          H = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1.0]])
          self._points(H, (w, h))
          self._record("crop", H, (w, h), (x, y, w, h))

      def pad(self, top, bottom, left, right):
          if self.mask is not None:
              self.mask = cv2.copyMakeBorder(self.mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
          dst = (self.size[0] + left + right, self.size[1] + top + bottom)
          H = np.array([[1, 0, left], [0, 1, top], [0, 0, 1.0]])
          self._points(H, dst)
          self._record("pad", H, dst, (top, bottom, left, right))

      def rot90(self, k):
          """k quarter turns clockwise (k = 1 is cv2.ROTATE_90_CLOCKWISE)."""
          k %= 4
          if not k:
              return
          w, h = self.size
          if self.mask is not None:
              self.mask = cv2.rotate(self.mask, [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE][k])
          H = {1: [[0, -1, h], [1, 0, 0]], 2: [[-1, 0, w], [0, -1, h]], 3: [[0, 1, 0], [-1, 0, w]]}[k]
          H = np.array(H + [[0, 0, 1]], np.float64)
          dst = (w, h) if k == 2 else (h, w)
          self._points(H, dst)
          self._record("rot90", H, dst, k)

      def remap(self, map_x, map_y):
          """Follow cv2.remap (elastic / grid / optical distortion). Not invertible by postprocess_*."""
          dh, dw = map_x.shape[:2]
          if self.mask is not None:
              self.mask = cv2.remap(self.mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
          flat_x, flat_y = map_x.ravel() + 0.5, map_y.ravel() + 0.5

          def nearest(pts):
              idx = [int(np.argmin((flat_x - x) ** 2 + (flat_y - y) ** 2)) for x, y in pts]
              return np.array([[i % dw + 0.5, i // dw + 0.5] for i in idx]).reshape(-1, 2)

          if len(self.keypoints):
              self.keypoints = np.concatenate([nearest(self.keypoints[:, :2]), self.keypoints[:, 2:]], 1)
          if len(self.boxes):
              pts = nearest(_box_corners(self.boxes)).reshape(-1, 4, 2)
              b = np.stack([pts[..., 0].min(1), pts[..., 1].min(1), pts[..., 0].max(1), pts[..., 1].max(1)], 1)
              keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
              self.boxes = np.concatenate([b, self.boxes[:, 4:]], 1)[keep]
          self._record("remap", None, (dw, dh))


  def _check_invertible(history):
      if any(r["kind"] == "remap" for r in history):
          raise ValueError("remap-based distortions (elastic / grid / optical) cannot be inverted; use the inference pipeline")


  def postprocess_boxes(boxes, history):
      """Map boxes (N, 4+) predicted on the preprocessed image back to original image pixels.

      Undoes letterbox, resize, crop, pad, flips, 90° rotations and warps in reverse order
      (for a letterbox: x_orig = (x - pad_left) / scale).
      """
      _check_invertible(history)
      boxes = np.asarray(boxes, np.float64).reshape(len(boxes), -1)
      for rec in reversed(history):
          if not len(boxes):
              break
          pts = _apply_h(np.linalg.inv(rec["H"]), _box_corners(boxes)).reshape(-1, 4, 2)
          w, h = rec["src"]
          b = np.stack([pts[..., 0].min(1), pts[..., 1].min(1), pts[..., 0].max(1), pts[..., 1].max(1)], 1)
          b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
          b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
          boxes = np.concatenate([b, boxes[:, 4:]], 1)
      return boxes


  def postprocess_keypoints(keypoints, history):
      """Map keypoints (N, 2+) back to original image pixels."""
      _check_invertible(history)
      kp = np.asarray(keypoints, np.float64).reshape(len(keypoints), -1)
      for rec in reversed(history):
          kp = np.concatenate([_apply_h(np.linalg.inv(rec["H"]), kp[:, :2]), kp[:, 2:]], 1)
      return kp


  def postprocess_mask(mask, history, interpolation=cv2.INTER_NEAREST):
      """Map a mask / depth map predicted on the preprocessed image back to the original image size.

      Use interpolation=cv2.INTER_LINEAR for continuous maps such as depth.
      """
      _check_invertible(history)
      for rec in reversed(history):
          kind, arg = rec["kind"], rec["arg"]
          w, h = rec["src"]
          if kind == "resize":
              mask = cv2.resize(mask, (w, h), interpolation=interpolation)
          elif kind == "flip":
              mask = cv2.flip(mask, arg)
          elif kind == "rot90":
              mask = np.ascontiguousarray(np.rot90(mask, arg))  # np.rot90 turns counter-clockwise
          elif kind == "crop":
              x, y, cw, ch = arg
              full = np.zeros((h, w) + mask.shape[2:], mask.dtype)
              full[y : y + ch, x : x + cw] = mask
              mask = full
          elif kind == "pad":
              top, _, left, _ = arg
              mask = mask[top : top + h, left : left + w]
          elif kind == "warp":
              mask = cv2.warpPerspective(mask, np.linalg.inv(arg), (w, h), flags=interpolation)
      return mask
`);
