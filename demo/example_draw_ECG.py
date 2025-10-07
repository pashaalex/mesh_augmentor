import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import ctypes as C

from MeshAugmentor import (
    MeshAugmentor,
    Optics, Distortion, Lighting, BackgroundShadow, RectOccluder, CameraPose,
    Mesh, Vector3d
)

signal = [26,20,12,4,-2,-6,-8,-9,-8,-8,-8,-8,-8,-9,-8,-7,-8,-7,-6,-6,
          -7,-6,-5,-4,-5,-4,-4,-3,-4,-3,-2,-1,-2,-2,0,-1,-2,-2,-1,-2,
          -2,-3,-2,-3,-4,-4,-3,-4,-5,-4,-5,-6,-6,-6,-5,-6,-7,-6,-6,-6,
          -7,-6,-6,-7,-7,-6,-6,-5,-6,-5,-4,-5,-4,-4,-4,-3,-3,-2,-2,-2,
          -1,-2,-4,-4,-4,-6,-7,-7,-7,-6,-6,-6,-8,-8,-8,-7,-8,-9,-8,-9,
          -8,-8,-2,6,14,20,28] * 10

# --------------------------- Helpers ---------------------------

def book_profile(segment_count: int, segment_length: float = 10.0):
    """Generate a 'book page' profile along X."""
    points = [(0.0, 0.0)]
    theta_deg = 139.0
    for k in range(segment_count):
        t = k / max(1, segment_count)
        dtheta = 15.0 * math.sin(2 * math.pi * 0.9 * t) * math.exp(-2.5 * t)
        theta_deg += dtheta
        x_prev, y_prev = points[-1]
        x_next = x_prev + segment_length * math.cos(math.radians(theta_deg))
        y_next = y_prev + segment_length * math.sin(math.radians(theta_deg))
        points.append((x_next, y_next))
    return points


def draw_ecg_pillow(
    signal,
    *,
    signal_scale_mm=0.5,
    img_width=512,
    img_height=256,
    pixels_per_mm=8,
    paper_speed_mm_s=25,
    sampling_rate=80,  # Hz
    thickness=2,
    draw_big_lines=True,
    start_x=0,
    start_y=128,
    grid_line_thickness=1,
    draw_dotted=True,
    background_color=(244, 244, 244),
    mm_line_color=(147, 148, 252),
    signal_color=(50, 50, 50),
    supersample=4,
):
    """Draw ECG with supersampling on a Pillow canvas. Returns (np.uint8 HxWx3, cross_points list)."""
    w_hr = img_width * supersample
    h_hr = img_height * supersample

    img_hr = Image.new("RGB", (w_hr, h_hr), background_color)
    draw = ImageDraw.Draw(img_hr)

    ppm_hr = pixels_per_mm * supersample

    grid_shift_x = random.random() * ppm_hr
    grid_shift_y = random.random() * ppm_hr

    cross_points = []
    n_x = int(math.ceil((w_hr - supersample - grid_shift_x) / ppm_hr))
    n_y = int(math.ceil((h_hr - supersample - grid_shift_y) / ppm_hr))
    for ix in range(0, n_x, 5):
        x_hr = grid_shift_x + ix * ppm_hr
        for iy in range(0, n_y, 5):
            y_hr = grid_shift_y + iy * ppm_hr
            cross_points.append((x_hr / supersample, y_hr / supersample))

    half_dot = (grid_line_thickness * supersample) / 2.0

    width_thin = int(max(1, grid_line_thickness * supersample))
    for ix in range(n_x + 1):
        x_hr = grid_shift_x + ix * ppm_hr
        draw.line((x_hr, 0, x_hr, h_hr), fill=mm_line_color, width=width_thin)
    for iy in range(n_y + 1):
        y_hr = grid_shift_y + iy * ppm_hr
        draw.line((0, y_hr, w_hr, y_hr), fill=mm_line_color, width=width_thin)

    if draw_big_lines:
        width_big = int(max(1, grid_line_thickness * 2 * supersample))
        for ix in range(int(math.ceil((w_hr - grid_shift_x) / (ppm_hr * 5))) + 1):
            x_hr = grid_shift_x + ix * ppm_hr * 5
            draw.line((x_hr, 0, x_hr, h_hr), fill=mm_line_color, width=width_big)
        for iy in range(int(math.ceil((h_hr - grid_shift_y) / (ppm_hr * 5))) + 1):
            y_hr = grid_shift_y + iy * ppm_hr * 5
            draw.line((0, y_hr, w_hr, y_hr), fill=mm_line_color, width=width_big)

    # ECG curve
    x_prev = start_x * supersample
    y_prev = start_y * supersample - signal[0] * signal_scale_mm * ppm_hr
    for i in range(len(signal) - 1):
        x_curr = start_x * supersample + (i + 1) * paper_speed_mm_s * supersample * pixels_per_mm / sampling_rate
        y_curr = start_y * supersample - signal[i + 1] * signal_scale_mm * ppm_hr
        draw.line((x_prev, y_prev, x_curr, y_curr), fill=signal_color, width=thickness * supersample)
        x_prev, y_prev = x_curr, y_curr
        if x_curr > w_hr:
            break

    img_final = img_hr.resize((img_width, img_height), resample=Image.LANCZOS)
    if img_final.mode not in ("RGB", "RGBA"):
        img_final = img_final.convert("RGB")
    img = np.array(img_final, dtype=np.uint8)  # RGB
    return img, cross_points

def get_ecg_with_points():
    """Generate an ECG strip and pad it vertically (white margins)."""    
    img, keypoints = draw_ecg_pillow(
        signal=signal,
        signal_scale_mm=0.3,
        img_width=1200,
        img_height=70 * 8,
        pixels_per_mm=8,
        paper_speed_mm_s=15,
        sampling_rate=75,
        thickness=5,
        draw_big_lines=True,
        start_x=10,
        start_y=300,
        grid_line_thickness=1,
        draw_dotted=False,
        background_color=(244, 244, 244),
        mm_line_color=(147, 148, 252),
        signal_color=(50, 50, 50),
    )
    pad = 8 * 5
    new_points = [(x, y + pad) for x, y in keypoints]
    img = cv2.copyMakeBorder(img, top=pad, bottom=pad, left=0, right=0,
                             borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return img, new_points  # RGB


# --------------------------- Distortion via MeshAugmentor ---------------------------

def render_distort(input_rgb: np.ndarray, out_width: int, out_height: int, cross_points):
    """Bend the 'paper' like a book page and render with light/shadow/distortion."""
    h, w, _ = input_rgb.shape

    # grid resolution (like old code)
    wcnt = max(4, w // 60)
    hcnt = max(4, h // 64)

    # Configure optics/lighting/distortion/etc.
    optics = Optics(F=35.0, L=66.7, R=14.0)
    distortion = Distortion(use=True, k1=-0.5)
    lighting = Lighting(
        use=True,
        x=0.0,
        y=0.0,
        z=10.0,
        intensity=0.99,
        diameter=170.0,
        shadow_y=0.0,
        light_mix_koef=0.5 
    )

    z_distance = optics.get_best_distance()
    bg_shadow = BackgroundShadow(use=True, bg_z=z_distance, bottom_shadow_koef=0.4)
    
    occluder = RectOccluder(use=True, cx=0.0, cy=150.0, cz=60.0, w=240.0, h=340.0,
                            yaw=math.radians(30.0), pitch=0.0, roll=0.0, circle_segments=32)
    pose = CameraPose(tilt_x_rad=math.radians(2.0))

    # Create mesh
    mesh = MeshAugmentor(
        input_width=w, input_height=h,
        grid_w=wcnt, grid_h=hcnt,
        optics=optics, distortion=distortion, lighting=lighting,
        bg_shadow=bg_shadow, occluder=occluder, pose=pose
    )
    
    # Build profile along X and bend the sheet
    stride = wcnt + 1
    profile_points = book_profile(stride, segment_length = 60)
    for index, point in enumerate(mesh.points):
        x = index % stride
        y = index // stride
        point.z = z_distance - profile_points[x][1] / 30.0
        point.x = -550 - profile_points[x][0]
        point.y = point.y - 350

    mesh.rotate_z(math.radians(5))
    
    # Render (RGB + optional alpha/mask/uv)
    outs = mesh.render(
        input_image=input_rgb,
        out_size=(out_width, out_height),
        background = np.full((out_height, out_width, 3), (255, 255, 255), dtype=np.uint8),
        attachments=("rgb", "mask", "uv"),
    )
    
    new_keypoints = [mesh.reproject_point(x, y, out_width, out_height) for x, y in cross_points]

    mesh.close()
    return outs.rgb, outs.mask, outs.uv, new_keypoints


# --------------------------- Main / demo ---------------------------

def render_uv_isolines_cv(
    uv: np.ndarray,
    mask: np.ndarray,
    n_levels: int = 12,
    thickness: int = 3,
    bg_gray: int = 245
) -> np.ndarray:
    assert uv.ndim == 3 and uv.shape[2] == 2, "uv must be (H, W, 2)"
    H, W = uv.shape[:2]
    u = uv[..., 0].astype(np.float32, copy=False)
    v = uv[..., 1].astype(np.float32, copy=False)

    if mask is None:
        valid = np.isfinite(u) & np.isfinite(v)
    else:
        valid = (mask.astype(bool)) & np.isfinite(u) & np.isfinite(v)

    def levels(arr: np.ndarray, n: int):
        vals = arr[valid]
        if vals.size == 0:
            return []
        lo, hi = float(vals.min()), float(vals.max())
        if not (hi > lo):
            return [lo]
        return np.linspace(lo, hi, n, dtype=np.float32).tolist()

    lev_u = levels(u, n_levels)
    lev_v = levels(v, n_levels)

    out = np.full((H, W, 3), int(bg_gray), dtype=np.uint8)

    def draw_isolines(scalar, levs, color_bgr):
        if len(levs) == 0:
            return
        for L in levs:
            bin_img = np.zeros((H, W), dtype=np.uint8)
            bin_img[valid] = (scalar[valid] >= L).astype(np.uint8) * 255
            contours, _ = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(out, contours, -1, color_bgr, thickness)

    draw_isolines(u, lev_u, (0, 0, 255))
    draw_isolines(v, lev_v, (255, 0, 0))

    return out


def main():
    ecg_img, cross_points = get_ecg_with_points()
    cv2.imwrite("ecg_source.png", ecg_img)
    
    rgb, mask, uv, cross_points = render_distort(ecg_img, 1024, 768, cross_points)
    isolines = render_uv_isolines_cv(uv, mask, n_levels=12)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    for x, y in cross_points:
        cv2.circle(mask, (int(x), int(y)), radius=4, color=(255,0,0), thickness=-1)

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))

    axes[0, 0].imshow(cv2.cvtColor(ecg_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Source image")
    axes[0, 1].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Processed image")
    axes[1, 0].imshow(isolines)
    axes[1, 0].set_title("UV-map")
    axes[1, 1].imshow(mask, cmap="gray")
    axes[1, 1].set_title("Mask and control points")

    plt.tight_layout()
    plt.show()

    cv2.imwrite("ecg_processed.png", rgb)
    cv2.imwrite("control_points.png", control_points)
    cv2.imwrite("mask.png", mask)
    cv2.imwrite("isolines.png", isolines)

if __name__ == "__main__":
    main()
