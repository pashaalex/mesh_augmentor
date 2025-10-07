# MeshAugmentor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
try:
    from typing import Literal  # Python 3.8+
except ImportError:
    from typing_extensions import Literal  # Python 3.7 fallback

from pathlib import Path
import ctypes as C
import numpy as np
import sys
import math


# --------------------------- Public types & configs ---------------------------

Attachment = Literal["rgb", "alpha", "mask", "uv"]  # requested outputs
UVSpace = Literal["pixels", "normalized"]           # UV coordinate space


@dataclass
class Optics:
    """Lens/sensor geometry passed to native render()."""
    F: float = 50.0   # lense focal distance 
    L: float = 66.7   # from lense to sensor distance
    R: float = 130.0  # lens/aperture radius
    def get_best_distance(self):
        return 1 / ((1 / self.F) - (1 / self.L))


@dataclass
class Distortion:
    """Radial distortion k1 and optical center."""
    use: bool = False
    k1: float = 0.0
    cx: float = 0.0
    cy: float = 0.0


@dataclass
class Lighting:
    """Disk light position/intensity and global lighting mix."""
    use: bool = False
    x: float = 0.0
    y: float = -100.0
    z: float = -100.0
    intensity: float = 1.0
    diameter: float = 20.0
    shadow_y: float = 0.0
    light_mix_koef: float = 0.8



@dataclass
class BackgroundShadow:
    """Shadow cast on a background plane z = bg_z."""
    use: bool = False
    bg_z: float = 0.0
    bottom_shadow_koef: float = 0.0


@dataclass
class RectOccluder:
    """Rectangular occluder for the disk light."""
    use: bool = False
    cx: float = 0.0
    cy: float = 0.0
    cz: float = 30.0
    w: float = 20.0
    h: float = 20.0
    yaw: float = 0.0   # Z
    pitch: float = 0.0 # Y
    roll: float = 0.0  # X
    circle_segments: int = 32


@dataclass
class CameraPose:
    """Extra camera tilt around X (radians)."""
    tilt_x_rad: float = 0.0


@dataclass
class RenderOutputs:
    """Returned buffers (only requested fields are filled).

    Shapes / dtypes:
      - rgb:   uint8,  (H, W, 3)
      - alpha: uint8,  (H, W)        0..255
      - mask:  uint8,  (H, W)        0/255 (validity)
      - uv:    float32,(H, W, 2)     backward sampling grid
    UV conventions:
      - "pixels": (x,y) in source pixels, origin top-left, x→right, y→down.
      - "normalized": (u,v) in [0..1], same origin; (0,0)=top-left, (1,1)=bottom-right.
    """
    rgb: Optional[np.ndarray] = None
    alpha: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    uv: Optional[np.ndarray] = None


# --------------------------- ctypes mirrors of C structs ----------------------

class Vector3d(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float), ("z", C.c_float)]


class Vector2d(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float)]


class Triangle(C.Structure):
    _fields_ = [
        ("v0", C.POINTER(Vector3d)),
        ("v1", C.POINTER(Vector3d)),
        ("v2", C.POINTER(Vector3d)),
        ("imagePoint0", C.POINTER(Vector2d)),
        ("imagePoint1", C.POINTER(Vector2d)),
        ("imagePoint2", C.POINTER(Vector2d)),
    ]


class Mesh(C.Structure):
    _fields_ = [
        ("triangles", C.POINTER(Triangle)),
        ("points3d", C.POINTER(Vector3d)),
        ("points2d", C.POINTER(Vector2d)),
        ("triangleCount", C.c_int),
        ("pointCount", C.c_int),
        # Lighting / shadows / bg shadow
        ("use_light_info", C.c_bool),
        ("light_x", C.c_float), ("light_y", C.c_float), ("light_z", C.c_float),
        ("light_intensivity", C.c_float),
        ("use_shadow_info", C.c_bool),
        ("light_diameter", C.c_float),
        ("shadow_y", C.c_float),
        ("use_bg_shadow", C.c_bool),
        ("bg_z", C.c_float),
        # Rect occluder & sampling
        ("occ_cx", C.c_float), ("occ_cy", C.c_float), ("occ_cz", C.c_float),
        ("occ_w", C.c_float),  ("occ_h", C.c_float),
        ("occ_yaw", C.c_float), ("occ_pitch", C.c_float), ("occ_roll", C.c_float),
        ("occ_circle_segments", C.c_int),
        # Camera tilt and distortion
        ("camera_tilt_x_rad", C.c_float),
        ("use_distortion_k1", C.c_bool),
        ("k1", C.c_float),
        ("dist_norm", C.c_float),
        ("cx", C.c_float), ("cy", C.c_float),
        ("bottom_shadow_koef", C.c_float),
        ("light_mix_koef", C.c_float),
    ]

# --------------------------- Native loader & signatures -----------------------

def _load_native(lib_path: Optional[str] = None) -> C.CDLL:
    """Load mesh_render.{dll|so|dylib} from same folder or a given path."""
    if lib_path:
        return C.CDLL(lib_path)
    here = Path(__file__).resolve().parent
    here = here / "cpp"
    names = (
        ["mesh_render.dll"] if sys.platform.startswith("win")
        else ["libmesh_render.dylib"] if sys.platform == "darwin"
        else ["libmesh_render.so"]
    )
    for n in names:
        p = here / n
        if p.exists():
            return C.CDLL(str(p))
    raise FileNotFoundError(f"Native library not found next to {here}.")


class _Native:
    """ctypes bindings to C API."""
    def __init__(self, lib_path: Optional[str] = None):
        self.dll = _load_native(lib_path)

        # Mesh* create_mesh(int img_w, int img_h, int wcnt, int hcnt);
        self.dll.create_mesh.argtypes = [C.c_int, C.c_int, C.c_int, C.c_int]
        self.dll.create_mesh.restype  = C.POINTER(Mesh)

        # int render(
        #   unsigned char* input_image, int input_width, int input_stride_bytes, int input_height,
        #   Mesh* mesh, unsigned char* output_rgba, int out_width, int out_stride_bytes, int out_height,
        #   bool render_mask_image, unsigned char* mask_image, int mask_stride_pixels,
        #   bool render_displacement_map, float* displacement_map /*H*W*2*/,
        #   float R, float L, float F);
        self.dll.render.argtypes = [
            C.c_void_p, C.c_int, C.c_int, C.c_int,
            C.POINTER(Mesh), C.c_void_p, C.c_int, C.c_int, C.c_int,
            C.c_bool, C.c_void_p, C.c_int,
            C.c_bool, C.c_void_p,
            C.c_float, C.c_float, C.c_float,
        ]
        self.dll.render.restype = C.c_int

        # bool reproject_point(Mesh* mesh, float L, float src_x, float src_y, float* dst_x, float* dst_y);
        self.dll.reproject_point.argtypes = [
            C.POINTER(Mesh), C.c_float, C.c_float, C.c_float, C.POINTER(C.c_float), C.POINTER(C.c_float)
        ]
        self.dll.reproject_point.restype = C.c_bool

        # void delete_mesh(Mesh* mesh);
        self.dll.delete_mesh.argtypes = [C.POINTER(Mesh)]
        self.dll.delete_mesh.restype  = None

        # void mesh_set_rect_occluder(mesh*, cx,cy,cz, w,h, yaw,pitch,roll, segments)
        self.dll.mesh_set_rect_occluder.argtypes = [
            C.POINTER(Mesh),
            C.c_float, C.c_float, C.c_float,
            C.c_float, C.c_float,
            C.c_float, C.c_float, C.c_float,
            C.c_int,
        ]
        self.dll.mesh_set_rect_occluder.restype = None


# --------------------------- High-level Python wrapper ------------------------

class MeshAugmentor:
    """High-level wrapper mirroring the C Mesh API while staying pythonic.

    Single render() can return RGB/ALPHA and optionally MASK/UV:
    - RGB/ALPHA come from the native RGBA result.
    - MASK is written by C to a separate H×W uint8 buffer (stride in pixels).
    - UV is written by C to H×W×2 float32 as a backward sampling grid.
    """

    def __init__(
        self,
        input_width: int,
        input_height: int,
        grid_w: int,
        grid_h: int,
        *,
        optics: Optics = Optics(),
        distortion: Distortion = Distortion(),
        lighting: Lighting = Lighting(),
        bg_shadow: BackgroundShadow = BackgroundShadow(),
        occluder: RectOccluder = RectOccluder(),
        pose: CameraPose = CameraPose(),
        lib_path: Optional[str] = None,
    ):
        self._api = _Native(lib_path)
        self._mesh = self._api.dll.create_mesh(int(input_width), int(input_height), int(grid_w), int(grid_h))
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        if not self._mesh:
            raise RuntimeError("create_mesh returned NULL")
        self.optics = optics
        self.distortion = distortion
        self.lighting = lighting
        self.bg_shadow = bg_shadow
        self.occluder = occluder
        self.pose = pose
        self._apply_configs()
        m = self._mesh.contents
        self.point_count = int(m.pointCount)
        self.points = [m.points3d[i] for i in range(self.point_count)]
    # ---------------------- push python configs into C Mesh -------------------

    def _apply_configs(self) -> None:
        """Push dataclass configs into the native Mesh struct."""
        m = self._mesh.contents
        # Distortion & optical center
        m.use_distortion_k1 = bool(self.distortion.use)
        m.k1 = float(self.distortion.k1)
        m.dist_norm = float(self.optics.L)
        m.cx = float(self.distortion.cx)
        m.cy = float(self.distortion.cy)
        # Lighting
        m.use_light_info = bool(self.lighting.use)
        m.light_x = float(self.lighting.x)
        m.light_y = float(self.lighting.y)
        m.light_z = float(self.lighting.z)
        m.light_intensivity = float(self.lighting.intensity)
        m.light_diameter = float(self.lighting.diameter)
        m.shadow_y = float(self.lighting.shadow_y)
        m.light_mix_koef = float(self.lighting.light_mix_koef)
        # Background shadow
        m.use_bg_shadow = bool(self.bg_shadow.use)
        m.bg_z = float(self.bg_shadow.bg_z)
        m.bottom_shadow_koef = float(self.bg_shadow.bottom_shadow_koef)
        # Camera pose
        m.camera_tilt_x_rad = float(self.pose.tilt_x_rad)
        # Rect occluder
        m.use_shadow_info = bool(self.occluder.use)
        self._api.dll.mesh_set_rect_occluder(
            self._mesh,
            float(self.occluder.cx), float(self.occluder.cy), float(self.occluder.cz),
            float(self.occluder.w), float(self.occluder.h),
            float(self.occluder.yaw), float(self.occluder.pitch), float(self.occluder.roll),
            int(self.occluder.circle_segments),
        )

    # Optional sugar setters
    def set_optics(self, v: Optics) -> None:        self.optics = v
    def set_distortion(self, v: Distortion) -> None: self.distortion = v; self._apply_configs()
    def set_lighting(self, v: Lighting) -> None:     self.lighting = v;  self._apply_configs()
    def set_background_shadow(self, v: BackgroundShadow) -> None:
        self.bg_shadow = v; self._apply_configs()
    def set_occluder(self, v: RectOccluder) -> None: self.occluder = v; self._apply_configs()
    def set_pose(self, v: CameraPose) -> None:       self.pose = v;      self._apply_configs()

    # --------------------------------- distort --------------------------------
    def cylynder_horizontal(self, R = None, dA = 0):
        h, w = self.input_height, self.input_width
        dx, dy = w // 2, h // 2
        if R == None: R = w * 3
        A = w / R
        for point in self.points:
            Ai = - A / 2 + A * point.x / h + dA
            point.z = (R - R * math.cos(Ai))
            point.x = R * math.sin(Ai)
            point.y -= dy

    def cylynder_vertical(self, R = None, dA = 0):
        h, w = self.input_height, self.input_width
        dx, dy = w // 2, h // 2
        if R == None: R = h * 3
        A = h / R
        for point in self.points:
            Ai = - A / 2 + A * point.y / w + dA
            point.z = (R - R * math.cos(Ai))
            point.x -= dx
            point.y = R * math.sin(Ai)            

    def shift(self, dx, dy, dz):
        for point in self.points:            
            point.x += dx
            point.y += dy
            point.z += dz

    def get_mass_center(self):
        x = 0
        y = 0
        z = 0    
        for point in self.points:
            x += point.x
            y += point.y
            z += point.z

        return x/len(self.points), y/len(self.points), z/len(self.points)

    def rotate_z(self, a):
        cx, cy, cz = self.get_mass_center()
        for point in self.points:
            translated_x = point.x - cx
            translated_y = point.y - cy
            rotated_x = translated_x * math.cos(a) - translated_y * math.sin(a)
            rotated_y = translated_x * math.sin(a) + translated_y * math.cos(a)
            point.x = rotated_x + cx
            point.y = rotated_y + cy

    def rotate_x(self, a):
        cx, cy, cz = self.get_mass_center()
        for point in self.points:
            translated_z = point.z - cz
            translated_y = point.y - cy
            rotated_z = translated_z * math.cos(a) - translated_y * math.sin(a)
            rotated_y = translated_z * math.sin(a) + translated_y * math.cos(a)
            point.z = rotated_z + cz
            point.y = rotated_y + cy

    def rotate_y(self, a):
        cx, cy, cz = self.get_mass_center()
        for point in self.points:
            translated_z = point.z - cz
            translated_x = point.x - cx
            rotated_x = translated_x * math.cos(a) - translated_z * math.sin(a)
            rotated_z = translated_x * math.sin(a) + translated_z * math.cos(a)
            point.x = rotated_x + cx
            point.z = rotated_z + cz


    # --------------------------------- render ---------------------------------

    def render(
        self,
        input_image: np.ndarray,
        out_size: Tuple[int, int],
        background = None, # optional
        attachments: Tuple[Attachment, ...] = ("rgb", "alpha"),
        *,
        uv_space: UVSpace = "pixels",          # reserved for future conversion on Python-side
        out_dtype_uv: np.dtype = np.float32,   # float32 recommended
    ) -> RenderOutputs:
        """Render RGB/ALPHA and optionally MASK/UV.

        Args:
          input_image: uint8 (H,W,3) source image.
          out_size: (W_out, H_out) of the result.
          attachments: any subset of {"rgb","alpha","mask","uv"}.
          uv_space: "pixels" or "normalized" (your C writes pixel UV; convert on Python if needed).
          out_dtype_uv: dtype for UV map (float32/float16).

        Returns:
          RenderOutputs with requested fields filled.
        """
        if input_image.dtype != np.uint8 or input_image.ndim != 3 or input_image.shape[2] != 3:
            raise TypeError("input_image must be uint8 (H,W,3).")
        Wout, Hout = int(out_size[0]), int(out_size[1])
        if Wout <= 0 or Hout <= 0:
            raise ValueError("out_size must be positive.")

        want_rgb  = "rgb"   in attachments
        want_a    = "alpha" in attachments
        want_mask = "mask"  in attachments
        want_uv   = "uv"    in attachments

        # Allocate output RGBA buffer (native writes RGBA)
        out_stride_bytes = Wout * 4
        out_rgba = np.zeros((Hout, out_stride_bytes), dtype=np.uint8, order="C")
        
        if background is not None and background.ndim == 3 and background.shape[2] == 3:
            bgr = background
            if bgr.shape[0] != Hout or bgr.shape[1] != Wout:
                bgr = cv2.resize(bgr, (Wout, Hout), interpolation=cv2.INTER_AREA)

            rgba_view = out_rgba[:, :Wout*4].reshape(Hout, Wout, 4)
            rgba_view[...] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)

        # Optional MASK (uint8 H×W, stride in PIXELS per your C code)
        mask = None
        mask_ptr = None
        mask_stride_pixels = 0
        if want_mask:
            mask = np.zeros((Hout, Wout), dtype=np.uint8, order="C")
            mask_ptr = mask.ctypes.data_as(C.c_void_p)
            mask_stride_pixels = int(mask.strides[0] // mask.itemsize)

        # Optional UV (float32 H×W×2)
        uv = None
        uv_ptr = None
        if want_uv:
            if out_dtype_uv not in (np.float32, np.float16):
                raise TypeError("UV dtype must be float32 or float16.")
            uv = np.zeros((Hout, Wout, 2), dtype=out_dtype_uv, order="C")
            uv_ptr = uv.ctypes.data_as(C.c_void_p)

        # Input pointers and strides
        in_ptr   = input_image.ctypes.data_as(C.c_void_p)
        in_w     = int(input_image.shape[1])
        in_h     = int(input_image.shape[0])
        in_stride_bytes = int(input_image.strides[0])  # bytes per row (safer than w*3)

        # Output pointer
        out_ptr = out_rgba.ctypes.data_as(C.c_void_p)

        # Call native render
        ret = self._api.dll.render(
            in_ptr, in_w, in_stride_bytes, in_h,
            self._mesh, out_ptr, Wout, out_stride_bytes, Hout,
            bool(want_mask), mask_ptr, mask_stride_pixels,
            bool(want_uv), uv_ptr,
            C.c_float(self.optics.R), C.c_float(self.optics.L), C.c_float(self.optics.F),
        )
        if ret != 0:
            raise RuntimeError(f"render() failed with code {ret}")

        # Convert RGBA → (rgb, alpha) if requested
        rgb = None
        alpha = None
        if want_rgb or want_a:
            rgba = out_rgba.reshape((Hout, Wout, 4))
            if want_rgb:
                rgb = np.ascontiguousarray(rgba[..., :3])  # view → contiguous copy
                rgb = cv2.flip(rgb,  -1)
            if want_a:
                alpha = np.ascontiguousarray(rgba[..., 3])
                alpha = cv2.flip(alpha,  -1)

        if want_uv:
            uv = cv2.flip(uv,  -1)
            # Optional: convert UV to normalized space if requested (user can also do it later)
            if uv_space == "normalized":
                # (x,y) pixels → (u,v) in [0..1]
                uv[..., 0] /= float(in_w - 1) if in_w > 1 else 1.0
                uv[..., 1] /= float(in_h - 1) if in_h > 1 else 1.0
                uv = uv.astype(out_dtype_uv, copy=False)

        if want_mask:
            mask = mask[::-1, ::-1]

        return RenderOutputs(rgb=rgb, alpha=alpha, mask=mask, uv=uv)

    # -------------------------------- utilities -------------------------------

    def reproject_point(self, src_x: float, src_y: float, output_width, output_height) -> Optional[Tuple[float, float]]:
        """Project a source pixel (x,y) into the output plane using current mesh and L."""
        dx = C.c_float(0.0)
        dy = C.c_float(0.0)
        ok = self._api.dll.reproject_point(
            self._mesh, C.c_float(self.optics.L),
            C.c_float(src_x), C.c_float(src_y),
            C.byref(dx), C.byref(dy)
        )
        
        return ((output_width / 2) - dx.value, (output_height / 2) - dy.value) if ok else None

    # -------------------------------- lifecycle -------------------------------

    def close(self) -> None:
        """Free native Mesh. Safe to call multiple times."""
        if getattr(self, "_mesh", None):
            self._api.dll.delete_mesh(self._mesh)
            self._mesh = None

    def __enter__(self) -> "MeshAugmentor":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
