import numpy as np
import cv2
import matplotlib.pyplot as plt
from mesh_augmentor import *

def make_grid_image(size=640, step=10, bg_color=(255, 255, 255), line_color=(180, 180, 180), thickness=1):
    img = np.full((size, size, 3), bg_color, dtype=np.uint8)
    for x in range(0, size, step):
        cv2.line(img, (x, 0), (x, size-1), line_color, thickness=thickness, lineType=cv2.LINE_AA)
    for y in range(0, size, step):
        cv2.line(img, (0, y), (size-1, y), line_color, thickness=thickness, lineType=cv2.LINE_AA)
    return img

def render_crumpled_grid(
    size=640,
    grid_step=10,
    out_w=640,
    out_h=640,
    grid_w=32,
    grid_h=32,
    folds=12,
    angle_deg_range=(8, 22),
):
    """
    Create sample image with grid 640x640
    apply K=folds "origami", 
    returns (rgb, alpha, mask, uv).
    """
    src = make_grid_image(size=size, step=grid_step)

    ma = MeshAugmentor(
        input_width=src.shape[1],
        input_height=src.shape[0],
        grid_w=grid_w,
        grid_h=grid_h
    )

    ma.set_optics(Optics(F=35.0, L=66.7, R=4.0))
    ma.set_lighting(Lighting(use=True, x=0, y=0, z=12, intensity=1.5, diameter=12, light_mix_koef=0.99))    

    cylynder_horizontal(ma, R = 10000)
    apply_crumple_with_creases(
        ma,
        K=20,
        angle_deg_range=(8.0, 12.0),
        band_px=90,
        falloff_sigma=18.0,
        z_jitter_std=0.05,
        z_scale=0.25
    )

    ma.fit_best_geometric(W_out=out_w, H_out=out_h, margin=0.9)
    outs = ma.render(
        input_image=src,
        out_size=(out_w, out_h),
        background=np.full((out_h, out_w, 3), (128, 128, 128), dtype=np.uint8),
        attachments=("rgb", "alpha", "mask", "uv")
    )
    return outs.rgb, outs.alpha, outs.mask, outs.uv

if __name__ == "__main__":
    rgb, alpha, mask, uv = render_crumpled_grid(
        size=640,
        grid_step=10,
        out_w=640,
        out_h=640,
        grid_w=64,
        grid_h=64,
        folds=12,
        angle_deg_range=(8, 12)
    )
    cv2.imwrite("crumple.png", rgb)

    plt.imshow(rgb)
    plt.tight_layout()
    plt.show()

