import math
import random
import numpy as np
from mesh_augmentor.mesh_augmentor import MeshAugmentor

def cylynder_horizontal(ma: MeshAugmentor, R = None, dA = 0):
    h, w = ma.input_height, ma.input_width
    dx, dy = w // 2, h // 2
    if R == None: R = w * 3
    A = w / R
    for point in ma.points:
        Ai = - A / 2 + A * point.x / h + dA
        point.z = (R - R * math.cos(Ai))
        point.x = R * math.sin(Ai)
        point.y -= dy

def cylynder_vertical(ma: MeshAugmentor, R = None, dA = 0):
    h, w = ma.input_height, ma.input_width
    dx, dy = w // 2, h // 2
    if R == None: R = h * 3
    A = h / R
    for point in ma.points:
        Ai = - A / 2 + A * point.y / w + dA
        point.z = (R - R * math.cos(Ai))
        point.x -= dx
        point.y = R * math.sin(Ai)   

def apply_crumple_with_creases(
    ma: MeshAugmentor,
    K=8,
    angle_deg_range=(1.0, 5.0),
    crease_len_px=(20, 640),
    band_px=40,                       # width of the bend zone around the line
    falloff_sigma=18.0,               # smoothing the influence by distance
    z_jitter_std=0.05,                
    z_scale=0.25
):
    rng = random.Random()
    W, H = ma.input_width, ma.input_height

    z_center = sum(pt.z for pt in ma.points) / max(1, len(ma.points))

    def rodrigues_rotate_vec(p, a, b, angle):
        p = np.asarray(p, np.float64)
        a = np.asarray(a, np.float64)
        b = np.asarray(b, np.float64)
        k = b - a
        nk = np.linalg.norm(k)
        if nk < 1e-9:
            return p
        k = k / nk
        v = p - a
        c = math.cos(angle)
        s = math.sin(angle)
        K = np.array([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]], dtype=np.float64)
        R = c*np.eye(3) + s*K + (1-c)*np.outer(k, k)
        return a + R @ v

    def signed_dist_to_line(px, py, x0, y0, x1, y1):
        vx, vy = x1-x0, y1-y0
        L = math.hypot(vx, vy) + 1e-12
        # signed area / |v|
        return ((vx)*(py-y0) - (vy)*(px-x0)) / L

    for _ in range(K):
        Lc = rng.randint(*crease_len_px)
        x0 = rng.randint(-W//4, int(W*1.25))
        y0 = rng.randint(-H//4, int(H*1.25))
        phi = rng.random() * (2*math.pi)
        x1 = int(x0 + Lc*math.cos(phi))
        y1 = int(y0 + Lc*math.sin(phi))

        ang = math.radians(rng.uniform(*angle_deg_range))
        if rng.random() < 0.5:
            ang = -ang

        a3 = np.array([x0, y0, z_center], np.float64)
        b3 = np.array([x1, y1, z_center], np.float64)

        # We only bend the points WITHIN the band_px around the line, with a smooth falloff.
        for pt in ma.points:
            d = signed_dist_to_line(pt.x, pt.y, x0, y0, x1, y1)
            ad = abs(d)

            if ad <= band_px:
                # Gaussian weighting: 1 in the center of the line, ~0 at the edge of the band
                w = math.exp(-0.5 * (ad / max(1e-6, falloff_sigma))**2)
                ang_local = ang * w
                p_rot = rodrigues_rotate_vec([pt.x, pt.y, pt.z], a3, b3, ang_local)
                pt.x, pt.y, pt.z = float(p_rot[0]), float(p_rot[1]), float(p_rot[2])

    # Minor noise along the Z-axis (very weak)
    rng2 = np.random.default_rng()
    for pt in ma.points:
        pt.z += float(rng2.normal(0.0, z_jitter_std))

    # Final "compression" of the relief along the Z-axis around the center
    # (maintaining the composition, reducing the amplitude)
    for pt in ma.points:
        pt.z = z_center + (pt.z - z_center) * z_scale

