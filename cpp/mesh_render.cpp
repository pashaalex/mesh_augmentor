#include "pch.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <thread>
#include <vector>
#include <fstream>
#include "mesh_render.h"


#if defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif
// Begin Vector algebra

RgbF rgbf_from_u8(ColorRGB c) {
    RgbF o = { (float)c.r, (float)c.g, (float)c.b };
    return o;
}

FORCE_INLINE Vector3d vector_add(Vector3d a, Vector3d b) noexcept {
    Vector3d res = { a.x + b.x, a.y + b.y, a.z + b.z };
    return res;
}

FORCE_INLINE Vector3d vector_sub(Vector3d* a, Vector3d* b) noexcept {
    Vector3d res = { a->x - b->x, a->y - b->y, a->z - b->z };
    return res;
}

FORCE_INLINE Vector3d vector_mul(Vector3d a, float scalar) noexcept {
    Vector3d res = { a.x * scalar, a.y * scalar, a.z * scalar };
    return res;
}

FORCE_INLINE float vector_dot(Vector3d* a, Vector3d* b) noexcept {
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

FORCE_INLINE Vector3d vector_cross(Vector3d* a, Vector3d* b) noexcept {
    Vector3d res = { a->y * b->z - a->z * b->y, a->z * b->x - a->x * b->z, a->x * b->y - a->y * b->x };
    return res;
}

FORCE_INLINE Vector3d vector_normalize(Vector3d* a) noexcept {
    float length = sqrtf(vector_dot(a, a));
    Vector3d res = { a->x / length, a->y / length, a->z / length };
    return res;
}

Vector3d vector_reflect(Vector3d vector, Vector3d normal) {
    // Ensure the normal vector is normalized
    normal = vector_normalize(&normal);

    // Calculate the reflected vector
    float dot = vector_dot(&vector, &normal);
    Vector3d reflected = { vector.x - 2 * dot * normal.x, vector.y - 2 * dot * normal.y, vector.z - 2 * dot * normal.z };
    return reflected;
}

FORCE_INLINE Vector3d ray_point_at(Ray* ray, float t) noexcept {
    return vector_add(ray->origin, vector_mul(ray->direction, t));
}

// End Vector algebra

ColorRGB get_pixel_color(const unsigned char* image, int width, int stride, int height, int x, int y) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        int d = stride * y + 3 * x;
        ColorRGB res = { image[d + 0], image[d + 1], image[d + 2] };
        return res;
    }

    ColorRGB zero = { 0, 0, 0 };
    return zero;
}

ColorRGB get_bilinear_interpolated_color(const unsigned char* image, int width, int stride, int height, float x, float y) {
    if (x < 0 || x >= width - 1 || y < 0 || y >= height - 1) {
        ColorRGB zero = { 0, 0, 0 };
        return zero;
    }

    x = std::min(std::max(x, 0.0f), float(width - 1));
    y = std::min(std::max(y, 0.0f), float(height - 1));
    int x1 = int(std::floor(x)), y1 = int(std::floor(y));
    int x2 = std::min(x1 + 1, width - 1);
    int y2 = std::min(y1 + 1, height - 1);

    float dx = x - x1;
    float dy = y - y1;

    ColorRGB c11 = get_pixel_color(image, width, stride, height, x1, y1);
    ColorRGB c12 = get_pixel_color(image, width, stride, height, x1, y2);
    ColorRGB c21 = get_pixel_color(image, width, stride, height, x2, y1);
    ColorRGB c22 = get_pixel_color(image, width, stride, height, x2, y2);

    ColorRGB result;
    result.r = static_cast<unsigned char>((1 - dx) * (1 - dy) * c11.r + dx * (1 - dy) * c21.r + (1 - dx) * dy * c12.r + dx * dy * c22.r);
    result.g = static_cast<unsigned char>((1 - dx) * (1 - dy) * c11.g + dx * (1 - dy) * c21.g + (1 - dx) * dy * c12.g + dx * dy * c22.g);
    result.b = static_cast<unsigned char>((1 - dx) * (1 - dy) * c11.b + dx * (1 - dy) * c21.b + (1 - dx) * dy * c12.b + dx * dy * c22.b);

    return result;
}

ColorRGB get_RGB_from_barycentric(unsigned char* input_image, int width, int stride, int height, Triangle triangle, Vector3d barycentricCoordinates, float* textureX, float* textureY) {

    float newX = barycentricCoordinates.x * triangle.imagePoint0->x +
        barycentricCoordinates.y * triangle.imagePoint1->x +
        barycentricCoordinates.z * triangle.imagePoint2->x;

    float newY = barycentricCoordinates.x * triangle.imagePoint0->y +
        barycentricCoordinates.y * triangle.imagePoint1->y +
        barycentricCoordinates.z * triangle.imagePoint2->y;

    *textureX = newX;
    *textureY = newY;

    return get_bilinear_interpolated_color(input_image, width, stride, height, newX, newY);
}

Ray refract_ray_through_lens(Vector3d matrixPoint, Vector3d lensPoint, float F) {
    float k = -F / matrixPoint.z;
    float Hfy = k * (lensPoint.y - matrixPoint.y);
    float Hfx = k * (lensPoint.x - matrixPoint.x);
    Vector3d p2 = { Hfx - lensPoint.x, Hfy - lensPoint.y, F - lensPoint.z };
    float l = sqrtf(p2.x * p2.x + p2.y * p2.y + p2.z * p2.z);
    p2.x = p2.x / l;
    p2.y = p2.y / l;
    p2.z = p2.z / l;
    Ray res = { lensPoint, p2 };
    return res;
}

static inline Vector3d rotate_x_point(Vector3d p, float ang, Vector3d pivot) {
    p.x -= pivot.x; p.y -= pivot.y; p.z -= pivot.z;
    float c = cosf(ang), s = sinf(ang);
    float y = c * p.y - s * p.z;
    float z = s * p.y + c * p.z;
    Vector3d r = { p.x + pivot.x, y + pivot.y, z + pivot.z };
    return r;
}
static inline Vector3d rotate_x_dir(Vector3d v, float ang) {
    float c = cosf(ang), s = sinf(ang);
    return Vector3d{ v.x, c * v.y - s * v.z, s * v.y + c * v.z };
}

Ray refract_ray_through_lens_tilted(Vector3d matrixPoint,
    Vector3d lensPoint,
    float F,
    float tilt_x_rad)
{
    Ray r = refract_ray_through_lens(matrixPoint, lensPoint, F);

    Vector3d pivot = { 0.0f, 0.0f, 0.0f };
    r.origin = rotate_x_point(r.origin, tilt_x_rad, pivot);
    r.direction = rotate_x_dir(r.direction, tilt_x_rad);

    float L = sqrtf(r.direction.x * r.direction.x + r.direction.y * r.direction.y + r.direction.z * r.direction.z);
    if (L > 0.0f) { r.direction.x /= L; r.direction.y /= L; r.direction.z /= L; }
    return r;
}

int triangle_intersects(Triangle triangle, Ray ray, Vector3d* intersectionPoint) {
    Vector3d edge1 = vector_sub(triangle.v1, triangle.v0);
    Vector3d edge2 = vector_sub(triangle.v2, triangle.v0);
    Vector3d h = vector_cross(&ray.direction, &edge2);
    float a = vector_dot(&edge1, &h);

    if (a > -1e-6 && a < 1e-6)
        return 0;

    float f = 1.0f / a;
    Vector3d s = vector_sub(&ray.origin, triangle.v0);
    float u = f * vector_dot(&s, &h);

    if (u < 0.0 || u > 1.0)
        return 0;

    Vector3d q = vector_cross(&s, &edge1);
    float v = f * vector_dot(&ray.direction, &q);

    if (v < 0.0 || u + v > 1.0)
        return 0;

    float t = f * vector_dot(&edge2, &q);
    if (t > 1e-6) {
        *intersectionPoint = ray_point_at(&ray, (float)t);
        return 1;
    }

    return 0;
}

Vector3d triangle_get_barycentric_coordinates(Triangle triangle, Vector3d point) {
    Vector3d v0 = vector_sub(triangle.v1, triangle.v0);
    Vector3d v1 = vector_sub(triangle.v2, triangle.v0);
    Vector3d v2 = vector_sub(&point, triangle.v0);
    float d00 = vector_dot(&v0, &v0);
    float d01 = vector_dot(&v0, &v1);
    float d11 = vector_dot(&v1, &v1);
    float d20 = vector_dot(&v2, &v0);
    float d21 = vector_dot(&v2, &v1);
    float denom = d00 * d11 - d01 * d01;
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0F - v - w;
    Vector3d res = { u, v, w };
    return res;
}

// =================== SHADOWS ==================
typedef struct { float x, y; } Vec2;

static inline float cross2(Vec2 a, Vec2 b) { return a.x * b.y - a.y * b.x; }

static inline float poly_area(const Vec2* v, int n) {
    if (n < 3) return 0.0f;
    double s = 0.0;
    for (int i = 0, j = n - 1; i < n; j = i++) s += (double)v[j].x * v[i].y - (double)v[i].x * v[j].y;
    float A = (float)(0.5 * s);
    return (A >= 0 ? A : -A);
}

static Vec2 line_intersect(Vec2 A, Vec2 B, Vec2 C, Vec2 D) {
    Vec2 r = { B.x - A.x, B.y - A.y };
    Vec2 s = { D.x - C.x, D.y - C.y };
    float denom = cross2(r, s);
    float t = cross2(Vec2{ C.x - A.x, C.y - A.y }, s) / (denom == 0 ? 1e-30f : denom);
    Vec2 P = { A.x + t * r.x, A.y + t * r.y };
    return P;
}

// Sutherland–Hodgman
static int clip_polygon_convex(const Vec2* subj, int nSubj,
    const Vec2* clip, int nClip,
    Vec2* out, int outMax)
{
    if (nSubj < 3 || nClip < 3) return 0;
    Vec2 bufA[128], bufB[128];
    if (nSubj > 128 || nClip > 128 || outMax > 128) { return 0; }

    int nA = nSubj;
    for (int i = 0; i < nSubj; ++i) bufA[i] = subj[i];

    for (int c = 0; c < nClip; ++c) {
        Vec2 C = clip[c];
        Vec2 D = clip[(c + 1) % nClip];
        Vec2 e = { D.x - C.x, D.y - C.y };

        int nB = 0;
        if (nA == 0) return 0;

        Vec2 S = bufA[nA - 1];
        for (int i = 0; i < nA; ++i) {
            Vec2 E = bufA[i];
            Vec2 CS = { S.x - C.x, S.y - C.y };
            Vec2 CE = { E.x - C.x, E.y - C.y };
            float inS = cross2(e, CS);
            float inE = cross2(e, CE);
            int insideS = (inS >= 0.0f);
            int insideE = (inE >= 0.0f);

            if (insideE) {
                if (!insideS) {
                    Vec2 I = line_intersect(S, E, C, D);
                    if (nB < 128) bufB[nB++] = I;
                }
                if (nB < 128) bufB[nB++] = E;
            }
            else if (insideS) {
                Vec2 I = line_intersect(S, E, C, D);
                if (nB < 128) bufB[nB++] = I;
            }
            S = E;
        }
        nA = nB;
        for (int i = 0; i < nA; ++i) bufA[i] = bufB[i];
    }

    if (nA > outMax) nA = outMax;
    for (int i = 0; i < nA; ++i) out[i] = bufA[i];
    return nA;
}

static int build_circle_polygon(Vec2 center, float R, int N, Vec2* out, int outMax) {
    if (N < 3) N = 3;
    if (N > outMax) N = outMax;
    const float TWO_PI = 6.28318530718f;
    for (int i = 0; i < N; ++i) {
        float a = TWO_PI * (float)i / (float)N;
        out[i].x = center.x + R * cosf(a);
        out[i].y = center.y + R * sinf(a);
    }
    return N;
}

static inline Vector3d mat3_mul_vec3(const Mat3* M, const Vector3d* v) {
    Vector3d r;
    r.x = M->m[0][0] * v->x + M->m[0][1] * v->y + M->m[0][2] * v->z;
    r.y = M->m[1][0] * v->x + M->m[1][1] * v->y + M->m[1][2] * v->z;
    r.z = M->m[2][0] * v->x + M->m[2][1] * v->y + M->m[2][2] * v->z;
    return r;
}

// Order: R = Rz(yaw) * Ry(pitch) * Rx(roll)
static Mat3 mat3_from_euler(float yaw, float pitch, float roll) {
    float cy = cosf(yaw), sy = sinf(yaw);
    float cp = cosf(pitch), sp = sinf(pitch);
    float cr = cosf(roll), sr = sinf(roll);

    Mat3 Rz = { { { cy,-sy, 0 }, { sy, cy, 0 }, { 0, 0, 1 } } };
    Mat3 Ry = { { { cp, 0, sp }, { 0, 1, 0 }, { -sp, 0, cp } } };
    Mat3 Rx = { { { 1, 0, 0 }, { 0, cr,-sr }, { 0, sr, cr } } };

    // Rz * Ry
    Mat3 T;
    for (int i = 0;i < 3;++i) for (int j = 0;j < 3;++j) {
        T.m[i][j] = Rz.m[i][0] * Ry.m[0][j] + Rz.m[i][1] * Ry.m[1][j] + Rz.m[i][2] * Ry.m[2][j];
    }
    // (Rz*Ry)*Rx
    Mat3 R;
    for (int i = 0;i < 3;++i) for (int j = 0;j < 3;++j) {
        R.m[i][j] = T.m[i][0] * Rx.m[0][j] + T.m[i][1] * Rx.m[1][j] + T.m[i][2] * Rx.m[2][j];
    }
    return R;
}

static void rect_axes(const RectOccluder* Rocc, Vector3d* U, Vector3d* V, Vector3d* N) {
    Mat3 R = mat3_from_euler(Rocc->yaw, Rocc->pitch, Rocc->roll);
    Vector3d ex = { 1,0,0 }, ey = { 0,1,0 }, ez = { 0,0,1 };
    *U = mat3_mul_vec3(&R, &ex);
    *V = mat3_mul_vec3(&R, &ey);
    *N = mat3_mul_vec3(&R, &ez);
}

static int project_rect_to_light_plane(const Vector3d* P,
    const RectOccluder* Rocc,
    float zLight,
    Vec2* out2D, int outMax)
{
    Vector3d U, V, N;
    rect_axes(Rocc, &U, &V, &N);

    float hw = 0.5f * Rocc->width;
    float hh = 0.5f * Rocc->height;

    Vector3d C = Rocc->center;
    Vector3d Q[4] = {
        { C.x + U.x * hw + V.x * hh, C.y + U.y * hw + V.y * hh, C.z + U.z * hw + V.z * hh },
        { C.x - U.x * hw + V.x * hh, C.y - U.y * hw + V.y * hh, C.z - U.z * hw + V.z * hh },
        { C.x - U.x * hw - V.x * hh, C.y - U.y * hw - V.y * hh, C.z - U.z * hw - V.z * hh },
        { C.x + U.x * hw - V.x * hh, C.y + U.y * hw - V.y * hh, C.z + U.z * hw - V.z * hh }
    };

    int anyBetween = 0;
    for (int i = 0;i < 4;++i) {
        float a = (P->z - Q[i].z);
        float b = (zLight - Q[i].z);
        if (a * b <= 0.0f) { anyBetween = 1; break; }
    }
    if (!anyBetween) return 0;

    int n = 0;
    for (int i = 0;i < 4;++i) {
        float denom = (Q[i].z - P->z);
        if (denom == 0.0f) continue;
        float t = (zLight - P->z) / denom; // P + t*(Q - P)

        if (t <= 0.0f) continue;
        float x = P->x + t * (Q[i].x - P->x);
        float y = P->y + t * (Q[i].y - P->y);
        if (n < outMax) out2D[n++] = Vec2{ x, y };
    }
    return n;
}

float rect_occluder_coverage_on_disk(const Vector3d* P,
    const RectOccluder* Rocc,
    const Vector3d* lightC,
    float lightR,
    int circleSegments)
{
    if (lightR <= 0.0f) return 0.0f;

    Vec2 proj[8];
    int nProj = project_rect_to_light_plane(P, Rocc, lightC->z, proj, 8);
    if (nProj < 3) return 0.0f;

    Vec2 circlePoly[64];
    if (circleSegments > 64) circleSegments = 64;
    int nCircle = build_circle_polygon(Vec2{ lightC->x, lightC->y }, lightR,
        circleSegments, circlePoly, 64);

    float lightArea = 3.14159265359f * lightR * lightR;

    Vec2 inter[128];
    int nInter = clip_polygon_convex(proj, nProj, circlePoly, nCircle, inter, 128);
    if (nInter < 3) return 0.0f;

    float interArea = poly_area(inter, nInter);

    float occ = interArea / (lightArea > 0 ? lightArea : 1.0f);
    if (occ < 0.0f) occ = 0.0f;
    if (occ > 1.0f) occ = 1.0f;
    return occ;
}

void mesh_set_rect_occluder(Mesh* m,
    float cx, float cy, float cz,
    float w, float h,
    float yaw, float pitch, float roll,
    int circle_segments)
{
    m->occ_cx = cx;
    m->occ_cy = cy;
    m->occ_cz = cz;
    m->occ_w = w;
    m->occ_h = h;
    m->occ_yaw = yaw;
    m->occ_pitch = pitch;
    m->occ_roll = roll;
    m->occ_circle_segments = (circle_segments > 2) ? circle_segments : 32;
}
static RectOccluder mesh_make_rect_occluder(const Mesh* m) {
    RectOccluder r;
    r.center = Vector3d{ m->occ_cx, m->occ_cy, m->occ_cz };
    r.width = m->occ_w; r.height = m->occ_h;
    r.yaw = m->occ_yaw; r.pitch = m->occ_pitch; r.roll = m->occ_roll;
    return r;
}

static float mesh_visible_fraction_from_rect(const Mesh* m, const Vector3d* P)
{
    Vector3d lightC = { m->light_x, m->light_y, m->light_z };
    float lightR = m->light_diameter * 0.5f;
    RectOccluder occ = mesh_make_rect_occluder(m);

    float occ_frac = rect_occluder_coverage_on_disk(P, &occ, &lightC, lightR,
        (m->occ_circle_segments > 0) ? m->occ_circle_segments : 32);
    float visible = 1.0f - occ_frac;
    if (visible < 0.0f) visible = 0.0f;
    if (visible > 1.0f) visible = 1.0f;
    return visible;
}

// =================== SHADOWS ============

// =================== BVH ===================
struct AABB {
    Vector3d bmin, bmax;
};

static FORCE_INLINE AABB aabb_empty() {
    AABB b; b.bmin = { +INFINITY, +INFINITY, +INFINITY };
    b.bmax = { -INFINITY, -INFINITY, -INFINITY };
    return b;
}

static FORCE_INLINE void aabb_expand(AABB& b, const Vector3d& p) {
    b.bmin.x = std::min(b.bmin.x, p.x); b.bmin.y = std::min(b.bmin.y, p.y); b.bmin.z = std::min(b.bmin.z, p.z);
    b.bmax.x = std::max(b.bmax.x, p.x); b.bmax.y = std::max(b.bmax.y, p.y); b.bmax.z = std::max(b.bmax.z, p.z);
}

static FORCE_INLINE AABB aabb_of_triangle(const Triangle& t) {
    AABB b = aabb_empty();
    aabb_expand(b, *t.v0); aabb_expand(b, *t.v1); aabb_expand(b, *t.v2);
    return b;
}

static FORCE_INLINE Vector3d tri_centroid(const Triangle& t) {
    Vector3d c = { (t.v0->x + t.v1->x + t.v2->x) / 3.0f,
                   (t.v0->y + t.v1->y + t.v2->y) / 3.0f,
                   (t.v0->z + t.v1->z + t.v2->z) / 3.0f };
    return c;
}

static FORCE_INLINE bool aabb_intersect(const AABB& b, const Ray& r, float tMax, float& tnear_out) {
    float tmin = 0.0f, tmax = tMax;

    for (int axis = 0; axis < 3; ++axis) {
        float ro = (&r.origin.x)[axis];
        float rd = (&r.direction.x)[axis];
        float inv = 1.0f / rd;

        float t0 = ((&b.bmin.x)[axis] - ro) * inv;
        float t1 = ((&b.bmax.x)[axis] - ro) * inv;
        if (t0 > t1) std::swap(t0, t1);

        tmin = t0 > tmin ? t0 : tmin;
        tmax = t1 < tmax ? t1 : tmax;
        if (tmax < tmin) return false;
    }
    tnear_out = tmin;
    return true;
}

struct BVHNode {
    AABB bounds;
    int  left;
    int  right;
    int  start;
    int  count;
};

struct BVH {
    std::vector<BVHNode> nodes;
    std::vector<int>     triIdx;
};

static FORCE_INLINE float get_axis(const Vector3d& v, int axis) {
    return (axis == 0) ? v.x : (axis == 1) ? v.y : v.z;
}

static int build_node(BVH& bvh, const Mesh* mesh, int start, int end, int leafSize) {
    BVHNode node;
    node.start = start; node.count = end - start;
    node.left = node.right = -1;

    node.bounds = aabb_empty();
    for (int i = start; i < end; ++i) {
        const Triangle& t = mesh->triangles[bvh.triIdx[i]];
        AABB tb = aabb_of_triangle(t);
        aabb_expand(node.bounds, tb.bmin);
        aabb_expand(node.bounds, tb.bmax);
    }

    int nodeIndex = (int)bvh.nodes.size();
    bvh.nodes.push_back(node);

    if (node.count <= leafSize) {
        bvh.nodes[nodeIndex] = node;
        return nodeIndex;
    }

    AABB cb = aabb_empty();
    for (int i = start; i < end; ++i) {
        const Triangle& t = mesh->triangles[bvh.triIdx[i]];
        Vector3d c = tri_centroid(t);
        aabb_expand(cb, c);
    }
    Vector3d ext = { cb.bmax.x - cb.bmin.x, cb.bmax.y - cb.bmin.y, cb.bmax.z - cb.bmin.z };
    int axis = 0;
    if (ext.y > ext.x && ext.y >= ext.z) axis = 1;
    else if (ext.z > ext.x && ext.z >= ext.y) axis = 2;

    float splitPos = (get_axis(cb.bmin, axis) + get_axis(cb.bmax, axis)) * 0.5f;

    int mid = start;
    for (int i = start; i < end; ++i) {
        const Triangle& t = mesh->triangles[bvh.triIdx[i]];

        Vector3d c = tri_centroid(t);
        if (get_axis(c, axis) < splitPos) {
            std::swap(bvh.triIdx[i], bvh.triIdx[mid]);
            ++mid;
        }
    }

    if (mid == start || mid == end) mid = start + (end - start) / 2;

    int L = build_node(bvh, mesh, start, mid, leafSize);
    int R = build_node(bvh, mesh, mid, end, leafSize);

    bvh.nodes[nodeIndex].left = L;
    bvh.nodes[nodeIndex].right = R;
    return nodeIndex;
}

BVH* build_bvh(const Mesh* mesh) {
    BVH* bvh = new BVH();
    bvh->triIdx.resize(mesh->triangleCount);
    for (int i = 0; i < mesh->triangleCount; ++i) bvh->triIdx[i] = i;
    build_node(*bvh, mesh, 0, mesh->triangleCount, /*leafSize=*/8);
    return bvh;
}

void delete_bvh(BVH* bvh) {
    delete bvh;
}

bool bvh_intersect(const Mesh* mesh, const BVH* bvh,
    const Ray& ray, int* outTriangleId, Vector3d* outPoint)
{
    if (bvh->nodes.empty()) return false;

    const float INF = 1e30f;
    float bestT = INF;
    bool  hit = false;
    int   hitTri = -1;
    Vector3d hitPoint = {};

    int stack[64]; int sp = 0;
    stack[sp++] = 0;

    while (sp) {
        int ni = stack[--sp];
        const BVHNode& n = bvh->nodes[ni];

        float tnear;
        if (!aabb_intersect(n.bounds, ray, bestT, tnear)) continue;

        if (n.left == -1 && n.right == -1) {
            // Лист: проверяем треугольники
            for (int i = 0; i < n.count; ++i) {
                int triId = bvh->triIdx[n.start + i];
                Vector3d p;
                if (triangle_intersects(mesh->triangles[triId], ray, &p)) {
                    Vector3d d = vector_sub(&p, (Vector3d*)&ray.origin);
                    float t = vector_dot(&d, (Vector3d*)&ray.direction);
                    if (t > 1e-6f && t < bestT) {
                        bestT = t; hit = true; hitTri = triId; hitPoint = p;
                    }
                }
            }
        }
        else {
            float tnearL = 0.f, tnearR = 0.f;
            bool iL = n.left >= 0 && aabb_intersect(bvh->nodes[n.left].bounds, ray, bestT, tnearL);
            bool iR = n.right >= 0 && aabb_intersect(bvh->nodes[n.right].bounds, ray, bestT, tnearR);
            if (iL && iR) {
                if (tnearL > tnearR) { stack[sp++] = n.left;  stack[sp++] = n.right; }
                else { stack[sp++] = n.right; stack[sp++] = n.left; }
            }
            else if (iL) stack[sp++] = n.left;
            else if (iR) stack[sp++] = n.right;
        }
    }

    if (hit) {
        if (outTriangleId) *outTriangleId = hitTri;
        if (outPoint)      *outPoint = hitPoint;
        return true;
    }
    return false;
}
// ================= End BVH =================

// ================= Background shadow sampling (plane z = const) =================

static FORCE_INLINE bool intersect_ray_plane_z(const Ray& ray, float z_plane,
    Vector3d* outP, float* outT)
{
    float denom = ray.direction.z;
    if (denom > -1e-12f && denom < 1e-12f) return false;
    float t = (z_plane - ray.origin.z) / denom;
    if (t <= 0.0f) return false;
    if (outP) {
        outP->x = ray.origin.x + ray.direction.x * t;
        outP->y = ray.origin.y + ray.direction.y * t;
        outP->z = ray.origin.z + ray.direction.z * t;
    }
    if (outT) *outT = t;
    return true;
}

static inline void rect_axes_from_mesh(const Mesh* m, Vector3d* U, Vector3d* V, Vector3d* N) {
    Mat3 R = mat3_from_euler(m->occ_yaw, m->occ_pitch, m->occ_roll);
    Vector3d ex = { 1,0,0 }, ey = { 0,1,0 }, ez = { 0,0,1 };
    *U = mat3_mul_vec3(&R, &ex);
    *V = mat3_mul_vec3(&R, &ey);
    *N = mat3_mul_vec3(&R, &ez);
}

static inline Vector3d light_sample_point(int i, int n, const Vector3d* C, float R) {
    const float GA = 2.39996323f;
    float r = sqrtf((i + 0.5f) / (float)n) * R;
    float a = GA * i;
    return Vector3d{ C->x + r * cosf(a), C->y + r * sinf(a), C->z };
}

static inline int segment_hits_rect_mesh(const Mesh* m, const Vector3d* P, const Vector3d* S, float eps) {
    if (m->occ_w <= 0.0f || m->occ_h <= 0.0f) return 0;
    Vector3d U, V, N; rect_axes_from_mesh(m, &U, &V, &N);
    Vector3d C = { m->occ_cx, m->occ_cy, m->occ_cz };
    Vector3d PS = { S->x - P->x, S->y - P->y, S->z - P->z };
    float denom = PS.x * N.x + PS.y * N.y + PS.z * N.z;
    if (fabsf(denom) < 1e-8f) return 0;
    Vector3d CP = { C.x - P->x, C.y - P->y, C.z - P->z };
    float t = (CP.x * N.x + CP.y * N.y + CP.z * N.z) / denom;
    if (t <= eps || t >= 1.0f - eps) return 0;
    Vector3d X = { P->x + PS.x * t, P->y + PS.y * t, P->z + PS.z * t };
    Vector3d CX = { X.x - C.x, X.y - C.y, X.z - C.z };
    float u = CX.x * U.x + CX.y * U.y + CX.z * U.z;
    float v = CX.x * V.x + CX.y * V.y + CX.z * V.z;
    float hu = 0.5f * m->occ_w, hv = 0.5f * m->occ_h;
    return (fabsf(u) <= hu && fabsf(v) <= hv) ? 1 : 0;
}

static inline int segment_blocked_by_bvh(const Mesh* mesh, const BVH* bvh,
    const Vector3d* P, const Vector3d* S, float eps)
{
    Vector3d dir = { S->x - P->x, S->y - P->y, S->z - P->z };
    float dist = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    if (dist <= 1e-6f) return 0;
    dir.x /= dist; dir.y /= dist; dir.z /= dist;

    Ray r;
    r.origin = Vector3d{ P->x + dir.x * eps, P->y + dir.y * eps, P->z + dir.z * eps };
    r.direction = dir;

    int triId; Vector3d hit;
    if (!bvh_intersect(mesh, bvh, r, &triId, &hit)) return 0;

    Vector3d d = { hit.x - r.origin.x, hit.y - r.origin.y, hit.z - r.origin.z };
    float t = d.x * dir.x + d.y * dir.y + d.z * dir.z;
    return (t > 0.0f && t < dist - eps) ? 1 : 0;
}

RgbF shade_background_from_plane_weighted(const Mesh* mesh, const BVH* bvh,
    const Ray& ray, float z_plane,
    RgbF base_color,
    int samples, float shadow_strength,
    float eps,
    float cos_power)
{
    Vector3d P; float tplane;
    if (!intersect_ray_plane_z(ray, z_plane, &P, &tplane)) {
        return base_color;
    }

    Vector3d C = { mesh->light_x, mesh->light_y, mesh->light_z };
    float R = 0.5f * mesh->light_diameter;
    int N = (samples > 0) ? samples : 32;
    if (R <= 0.0f) { R = 0.0f; N = 1; }

    Vector3d n_plane = { 0, 0, (mesh->light_z >= z_plane) ? 1.0f : -1.0f };

    float sumW = 0.0f, blockedW = 0.0f;

    for (int i = 0; i < N; ++i) {
        Vector3d S = (R > 0.0f) ? light_sample_point(i, N, &C, R) : C;

        Vector3d L = { S.x - P.x, S.y - P.y, S.z - P.z };
        float Llen = sqrtf(L.x * L.x + L.y * L.y + L.z * L.z);
        if (Llen <= 1e-6f) continue;
        L.x /= Llen; L.y /= Llen; L.z /= Llen;

        float w = L.x * n_plane.x + L.y * n_plane.y + L.z * n_plane.z; // cos(theta)
        if (w < 0.0f) w = 0.0f;
        if (cos_power != 1.0f) w = powf(w, cos_power);
        if (w <= 0.0f) continue;

        int blocked = 0;
        if (mesh->use_shadow_info) {
            if (segment_hits_rect_mesh(mesh, &P, &S, eps)) blocked = 1;
            if (!blocked && bvh && mesh->triangleCount > 0) {
                if (segment_blocked_by_bvh(mesh, bvh, &P, &S, eps)) blocked = 1;
            }
        }

        if (blocked) blockedW += w;
        sumW += w;
    }

    float vis = (sumW > 0.0f) ? (1.0f - blockedW / sumW) : 0.0f;
    if (vis < 0.0f) vis = 0.0f;
    if (vis > 1.0f) vis = 1.0f;

    float factor = ((1.0f - shadow_strength) + shadow_strength * vis) * mesh->light_intensivity;
    if (factor < 0.0f) factor = 0.0f;
    if (factor > 1.0f) factor = 1.0f;

    RgbF out = { base_color.r * factor,
                 base_color.g * factor,
                 base_color.b * factor };
    return out;
}

// ================= End background shadows =================

// ================= Non linear ===================
void mesh_set_radial_k1(Mesh* m, float k1, float dist_norm, float cx, float cy) {
    m->use_distortion_k1 = 1;
    m->k1 = k1;
    m->dist_norm = (dist_norm != 0.0f) ? dist_norm : 1.0f;
    m->cx = cx; m->cy = cy;
}

static inline float invert_radial_k1_norm(float rd, float k1) {
    if (rd <= 0.0f) return 0.0f;
    if (fabsf(k1) < 1e-12f) return rd;

    if (k1 < 0.0f) {
        // r_d,max = (2/3)*sqrt(1/(-3*k1))
        float rd_max = (2.0f / 3.0f) * sqrtf(1.0f / (-3.0f * k1));
        if (rd >= rd_max) rd = rd_max * 0.9999f;
    }

    // x^3 + p x + q = 0
    float p = 1.0f / k1;
    float q = -rd / k1;

    float half_q = 0.5f * q;
    float third_p = p / 3.0f;
    float D = half_q * half_q + third_p * third_p * third_p;

    if (D >= 0.0f) {
        float sqrtD = sqrtf(D);
        float A = cbrtf(-half_q + sqrtD);
        float B = cbrtf(-half_q - sqrtD);
        float r = A + B;
        return (r > 0.0f) ? r : rd;
    }
    else {
        float rho = sqrtf(-third_p);            // >0
        float phi = acosf(-half_q / (rho * rho * rho)); // 0..pi
        float t = 2.0f * rho;
        float r1 = t * cosf(phi / 3.0f);
        float r2 = t * cosf((phi + 2.0f * 3.1415926535f) / 3.0f);
        float r3 = t * cosf((phi + 4.0f * 3.1415926535f) / 3.0f);

        float r = 1e30f;
        if (r1 > 0.0f && r1 < r) r = r1;
        if (r2 > 0.0f && r2 < r) r = r2;
        if (r3 > 0.0f && r3 < r) r = r3;
        if (r == 1e30f) r = rd;
        return r;
    }
}

// ---------- Forward pass (ideal -> distorted) ----------
static inline void distort_point_k1(const Mesh* m, float x, float y, float* xd, float* yd) {
    float X = x - m->cx, Y = y - m->cy;
    float s = (m->dist_norm != 0.0f) ? m->dist_norm : 1.0f;
    float xn = X / s, yn = Y / s;

    float r2 = xn * xn + yn * yn;
    float radial = 1.0f + m->k1 * r2;

    float xdn = xn * radial;
    float ydn = yn * radial;

    *xd = xdn * s + m->cx;
    *yd = ydn * s + m->cy;
}

// ---------- Backward pass (distorted -> ideal) ----------
static inline void undistort_point_k1_closed(const Mesh* m, float xd, float yd, float* x, float* y) {
    float Xd = xd - m->cx, Yd = yd - m->cy;
    float s = (m->dist_norm != 0.0f) ? m->dist_norm : 1.0f;

    float xdn = Xd / s, ydn = Yd / s;
    float rd = sqrtf(xdn * xdn + ydn * ydn);
    if (rd < 1e-12f) { *x = xd; *y = yd; return; }

    float r = invert_radial_k1_norm(rd, m->k1);
    float scale = r / rd;

    float xun = xdn * scale;
    float yun = ydn * scale;

    *x = xun * s + m->cx;
    *y = yun * s + m->cy;
}

Ray refract_ray_through_lens_tilted_distorted_k1(Vector3d matrixPoint,
    Vector3d lensPoint,
    float F,
    float tilt_x_rad,
    const Mesh* mesh)
{
    if (mesh && mesh->use_distortion_k1) {
        float ux, uy;
        undistort_point_k1_closed(mesh, matrixPoint.x, matrixPoint.y, &ux, &uy);
        matrixPoint.x = ux;
        matrixPoint.y = uy;
    }
    return refract_ray_through_lens_tilted(matrixPoint, lensPoint, F, tilt_x_rad);
}

void mesh_auto_set_dist_norm(Mesh* m, float dx, float dy, int imgW, int imgH) {
    float halfW = 0.5f * (imgW - 1) * dx;
    float halfH = 0.5f * (imgH - 1) * dy;
    float rmax = sqrtf(halfW * halfW + halfH * halfH);
    m->dist_norm = rmax;
    m->cx = 0.0f; m->cy = 0.0f;
}

static inline float image_corner_radius_pixels(int outW, int outH) {
    float halfW = 0.5f * (outW - 1);
    float halfH = 0.5f * (outH - 1);
    return sqrtf(halfW * halfW + halfH * halfH);
}

void mesh_calibrate_dist_norm_for_k1(Mesh* m, int outW, int outH) {
    float r_pix = image_corner_radius_pixels(outW, outH);

    if (m->k1 >= 0.0f) {
        m->dist_norm = r_pix;
    }
    else {
        float rd_max = (2.0f / 3.0f) * sqrtf(1.0f / (-3.0f * m->k1));
        float rd_corner = 0.90f * rd_max;
        if (rd_corner < 0.2f) rd_corner = 0.2f;
        m->dist_norm = r_pix / rd_corner;
    }
}

void mesh_k1_monotonic_guard(Mesh* m, int outW, int outH) {
    float rd_corner = image_corner_radius_pixels(outW, outH) / (m->dist_norm > 0 ? m->dist_norm : 1.0f);
    if (1.0f + 3.0f * m->k1 * rd_corner * rd_corner <= 0.0f) {
        float k1_min = -1.0f / (3.0f * rd_corner * rd_corner);
        m->k1 = k1_min + 1e-4f;
    }
}

// ================= END Non-linear ===============

static inline float saturate(float x) { return (x < 0.f) ? 0.f : ((x > 1.f) ? 1.f : x); }

static inline float diffuse_half_lambert(float ndotl, float k) {
    float d = (ndotl + k) / (1.f + k);
    return saturate(d);
}

#define CLAMP255(x) ((x) < 0.0f ? 0.0f : ((x) > 255.0f ? 255.0f : (x)))

int render(unsigned char* input_image,
    int input_width,
    int input_stride,
    int input_height,
    Mesh* mesh,
    unsigned char* output_image,
    int output_width,
    int output_stride,
    int output_height,
    bool render_mask_image,
    unsigned char* mask_image,
    int mask_image_stride,
    bool render_displacement_map,
    float* displacement_map,
    float R,
    float L,
    float F)
{
    const float PI_F = 3.14159265358979f;

    BVH* bvh = build_bvh(mesh);

    mesh_calibrate_dist_norm_for_k1(mesh, output_width, output_height);
    mesh_k1_monotonic_guard(mesh, output_width, output_height);

    const float min_k1 = -1.0f / 3.0f;
    if (mesh->k1 < min_k1) mesh->k1 = min_k1 + 1e-4f;

    // float R = 150; // Lens radius
    // float L = 66.66666F; // Distance from mtrix to lense
    // float F = 50; // Focal distance
    // Deph of field is about 200

    // For each point in matrix
#pragma omp parallel for collapse(2)
    for (int j = 0; j < output_height; ++j)
        for (int i = 0; i < output_width; ++i)
        {
            float x = i - 0.5f * (output_width - 1);
            float y = j - 0.5f * (output_height - 1);

            int N = 0; // Number of rays from this point
            float N_image = 0; // Number of rays that achieve image (include background is shadow case)
            float alpha_count = 0; // Number of rays that achieve image
            float colorR = 0;
            float colorG = 0;
            float colorB = 0;

            float textureX = 0;
            float textureY = 0;
            bool rayCrossTriangle = false;

            // Send a ray to the lense
            for (float r = 0; r < R; r = r + 10) // Send a rays into the points located concentric circles
            {
                float step = PI_F / 5.9f; // each 30 degree
                if (r < 5) step = 2.5f * PI_F;
                for (float ra = 0; ra < 2 * PI_F; ra = ra + step)
                {
                    N = N + 1;

                    float lensX = cosf(ra) * r;
                    float lensY = sinf(ra) * r;

                    Vector3d matrixPoint = { x, y, -L };
                    Vector3d lensPoint = { (float)lensX, (float)lensY, (float)0 };

                    Ray rayAfter = refract_ray_through_lens_tilted_distorted_k1(matrixPoint, lensPoint, F, mesh->camera_tilt_x_rad, mesh);

                    int triangleId;
                    Vector3d intersectionPoint;
                    if (bvh_intersect(mesh, bvh, rayAfter, &triangleId, &intersectionPoint)) {
                        Vector3d barycentricCoordinates =
                            triangle_get_barycentric_coordinates(mesh->triangles[triangleId], intersectionPoint);

                        float localTextureX;
                        float localTextureY;
                        ColorRGB color = get_RGB_from_barycentric(input_image, input_width, input_stride, input_height,
                            mesh->triangles[triangleId], barycentricCoordinates, &localTextureX, &localTextureY);
                        if (r < 1) { // from center of lense
                            if (render_displacement_map) {
                                textureX = localTextureX;
                                textureY = localTextureY;
                            }
                            rayCrossTriangle = true;
                        }

                        if (!mesh->use_light_info) {
                            colorR += color.r;
                            colorG += color.g;
                            colorB += color.b;
                            N_image = N_image + 1;
                            alpha_count = alpha_count + 1;
                        }
                        else {
                            Triangle triangle = mesh->triangles[triangleId];
                            Vector3d l1 = { triangle.v1->x - triangle.v0->x, triangle.v1->y - triangle.v0->y, triangle.v1->z - triangle.v0->z };
                            Vector3d l2 = { triangle.v2->x - triangle.v0->x, triangle.v2->y - triangle.v0->y, triangle.v2->z - triangle.v0->z };
                            Vector3d normal = vector_cross(&l1, &l2); normal = vector_normalize(&normal);

                            Vector3d lightDirection = { mesh->light_x - intersectionPoint.x, mesh->light_y - intersectionPoint.y, mesh->light_z - intersectionPoint.z };
                            lightDirection = vector_normalize(&lightDirection);

                            float material_r = color.r / 255.0f;
                            float material_g = color.g / 255.0f;
                            float material_b = color.b / 255.0f;

                            float diff = vector_dot(&normal, &lightDirection); if (diff < 0) diff = -diff;
                            //float base = diffuse_half_lambert(diff, 0.1f);
                            //diff = powf(base, 0.2f);
                            float base = diffuse_half_lambert(diff, 0.1f);
                            diff = powf(base, 0.5f);


                            if (mesh->use_shadow_info) {
                                float visible_frac = mesh_visible_fraction_from_rect(mesh, &intersectionPoint);
                                diff *= visible_frac;  // we scale the illumination by the visible fraction of the source
                            }

                            float light = mesh->light_intensivity;
                            float c = 255.0f * ((1.0f - mesh->light_mix_koef) + mesh->light_mix_koef * light * (diff + 0.5f) / 1.5f);
                            colorR += CLAMP255(material_r * c);
                            colorG += CLAMP255(material_g * c);
                            colorB += CLAMP255(material_b * c);
                            N_image = N_image + 1;
                            alpha_count = alpha_count + 1;
                        }
                    }
                    else {
                        // The ray did not intersect with any triangle
                        if (mesh->use_bg_shadow) {
                            float z_plane = mesh->bg_z;

                            int dt = output_stride * j + i * 4;
                            RgbF base_bg = { (float)output_image[dt + 0], (float)output_image[dt + 1], (float)output_image[dt + 2] };

                            RgbF shaded = shade_background_from_plane_weighted(
                                mesh, bvh, rayAfter,
                                z_plane,
                                base_bg,
                                16,    // samples
                                mesh->bottom_shadow_koef,  // 0..1, shadow affect
                                1e-3f, // eps
                                1.0f   // cos_power
                            );

                            colorR += shaded.r;
                            colorG += shaded.g;
                            colorB += shaded.b;
                            N_image = N_image + 1;
                        }
                    }
                }
            }

            // set image color
#pragma omp critical
            {
                int d = output_stride * j + i * 4;
                if (N_image > 0) {
                    output_image[d + 0] = (unsigned char)(colorR / N_image);
                    output_image[d + 1] = (unsigned char)(colorG / N_image);
                    output_image[d + 2] = (unsigned char)(colorB / N_image);
                }
                output_image[d + 3] = (unsigned char)((255 * alpha_count) / N);

                if (render_mask_image) {
                    if (rayCrossTriangle)
                        mask_image[mask_image_stride * j + i] = 255;
                    else
                        mask_image[mask_image_stride * j + i] = 0;
                }

                if ((render_displacement_map) && (rayCrossTriangle)) {
                    d = (output_width * j + i) * 2;
                    displacement_map[d + 0] = textureX;
                    displacement_map[d + 1] = textureY;
                }
            }
        }
    delete_bvh(bvh);
    return 0;
}

bool compute_barycentric_coordinates(Vector2d p, Vector2d a, Vector2d b, Vector2d c, float& u, float& v, float& w) {
    Vector2d v0 = { b.x - a.x, b.y - a.y };
    Vector2d v1 = { c.x - a.x, c.y - a.y };
    Vector2d v2 = { p.x - a.x, p.y - a.y };

    float d00 = v0.x * v0.x + v0.y * v0.y;
    float d01 = v0.x * v1.x + v0.y * v1.y;
    float d11 = v1.x * v1.x + v1.y * v1.y;
    float d20 = v2.x * v0.x + v2.y * v0.y;
    float d21 = v2.x * v1.x + v2.y * v1.y;

    float denom = d00 * d11 - d01 * d01;
    if (denom == 0) {
        return false;
    }

    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0f - v - w;

    return (u >= 0) && (v >= 0) && (w >= 0);  // The point is inside triangle. All coords are >= 0
}

bool reproject_point(Mesh* mesh, float L, float src_x, float src_y, float* dst_x, float* dst_y) {
    float u, v, w;
    Vector2d point = { src_x, src_y };

    const float tilt = mesh->camera_tilt_x_rad;
    const float c = cosf(tilt);
    const float s = sinf(tilt);

    for (int i = 0; i < mesh->triangleCount; ++i) {
        Triangle& tri = mesh->triangles[i];
        Vector2d a = *tri.imagePoint0;
        Vector2d b = *tri.imagePoint1;
        Vector2d c2 = *tri.imagePoint2;

        if (compute_barycentric_coordinates(point, a, b, c2, u, v, w)) {
            float x = u * tri.v0->x + v * tri.v1->x + w * tri.v2->x;
            float y = u * tri.v0->y + v * tri.v1->y + w * tri.v2->y;
            float z = u * tri.v0->z + v * tri.v1->z + w * tri.v2->z;

            float x_cam = x;
            float y_cam = c * y + s * z;
            float z_cam = -s * y + c * z;

            if (fabsf(z_cam) < 1e-6f) return false;

            float und_x = -x_cam * L / z_cam;
            float und_y = -y_cam * L / z_cam;

            if (mesh->use_distortion_k1) {
                float dx, dy;
                distort_point_k1(mesh, und_x, und_y, &dx, &dy);
                *dst_x = dx;
                *dst_y = dy;
            }
            else {
                *dst_x = und_x;
                *dst_y = und_y;
            }
            return true;
        }
    }
    return false;
}

int get_coord(int x, int y, int w) {
    return y * w + x;
}

Mesh* create_mesh(int img_width, int img_height, int wcnt, int hcnt) {
    Mesh* mesh = (Mesh*)malloc(sizeof(Mesh));
    mesh->use_bg_shadow = false;
    mesh->use_light_info = false;
    mesh->use_shadow_info = false;
    mesh->triangleCount = wcnt * hcnt * 2;
    float dx = (img_width - 1) / float(wcnt);
    float dy = (img_height - 1) / float(hcnt);
    mesh->pointCount = (hcnt + 1) * (wcnt + 1);
    mesh->points2d = (Vector2d*)malloc(mesh->pointCount * sizeof(Vector2d));
    mesh->points3d = (Vector3d*)malloc(mesh->pointCount * sizeof(Vector3d));
    mesh->triangles = (Triangle*)malloc(mesh->triangleCount * sizeof(Triangle));

    for (int x = 0; x < wcnt + 1; x++)
        for (int y = 0; y < hcnt + 1; y++) {
            int d = (wcnt + 1) * y + x;
            mesh->points2d[d].x = dx * x;
            mesh->points2d[d].y = dy * y;
            mesh->points3d[d].x = dx * x;
            mesh->points3d[d].y = dy * y;
            mesh->points3d[d].z = 0;
        }

    for (int x = 0; x < wcnt; x++)
        for (int y = 0; y < hcnt; y++) {
            int d = 2 * (x + y * wcnt);
            mesh->triangles[d].imagePoint0 = &mesh->points2d[get_coord(x, y, wcnt + 1)];
            mesh->triangles[d].imagePoint1 = &mesh->points2d[get_coord(x + 1, y, wcnt + 1)];
            mesh->triangles[d].imagePoint2 = &mesh->points2d[get_coord(x + 1, y + 1, wcnt + 1)];

            mesh->triangles[d].v0 = &mesh->points3d[get_coord(x, y, wcnt + 1)];
            mesh->triangles[d].v1 = &mesh->points3d[get_coord(x + 1, y, wcnt + 1)];
            mesh->triangles[d].v2 = &mesh->points3d[get_coord(x + 1, y + 1, wcnt + 1)];

            mesh->triangles[d + 1].imagePoint0 = &mesh->points2d[get_coord(x, y, wcnt + 1)];
            mesh->triangles[d + 1].imagePoint1 = &mesh->points2d[get_coord(x, y + 1, wcnt + 1)];
            mesh->triangles[d + 1].imagePoint2 = &mesh->points2d[get_coord(x + 1, y + 1, wcnt + 1)];

            mesh->triangles[d + 1].v0 = &mesh->points3d[get_coord(x, y, wcnt + 1)];
            mesh->triangles[d + 1].v1 = &mesh->points3d[get_coord(x, y + 1, wcnt + 1)];
            mesh->triangles[d + 1].v2 = &mesh->points3d[get_coord(x + 1, y + 1, wcnt + 1)];
        }
    return mesh;
}

void delete_mesh(Mesh* mesh) {
    free(mesh->points2d);
    free(mesh->points3d);
    free(mesh->triangles);
    free(mesh);
}