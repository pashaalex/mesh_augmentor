#pragma once
#ifndef MESH_RENDER_H
#define MESH_RENDER_H

#ifdef _WIN32
#ifdef BUILD_DLL
#define FORWARD_TRACER_API __declspec(dllexport)
#else
#define FORWARD_TRACER_API __declspec(dllimport)
#endif
#else
#define FORWARD_TRACER_API
#endif

typedef struct {
    float x, y, z;
} Vector3d;

typedef struct {
    float x, y;
} Vector2d;

typedef struct {
    Vector3d origin, direction;
} Ray;

// Матрица поворота из углов (yaw вокруг Z, pitch вокруг Y, roll вокруг X)
typedef struct {
    float m[3][3];
} Mat3;

typedef struct {
    Vector3d* v0;
    Vector3d* v1;
    Vector3d* v2;

    Vector2d* imagePoint0;
    Vector2d* imagePoint1;
    Vector2d* imagePoint2;
} Triangle;

typedef struct {
    Triangle* triangles;
    Vector3d* points3d;
    Vector2d* points2d;
    int triangleCount;
    int pointCount;
    bool use_light_info;
    float light_x;
    float light_y;
    float light_z;
    float light_intensivity;
    bool use_shadow_info;
    float light_diameter;
    float shadow_y;
    bool use_bg_shadow;
    float bg_z; // На каком расстоянии находится поверхность, на которую отбрасывать тень
    // --- Параметры прямоугольного окклюдера (всегда используем, если use_shadow_info=1) ---
    float occ_cx, occ_cy, occ_cz;    // центр
    float occ_w, occ_h;             // ширина/высота
    float occ_yaw, occ_pitch, occ_roll; // повороты (рад)
    int   occ_circle_segments;       // аппроксимация круга источника (напр. 32)
    float camera_tilt_x_rad; // Наклон камеры вокруг OX
    bool   use_distortion_k1;  // 0=выкл, 1=вкл
    float k1;                 // коэффициент радиальной дисторсии (k2=k3=0)
    float dist_norm;          // масштаб нормировки (удобно взять L)
    float cx, cy;             // оптический центр на матрице (обычно 0,0)
    float bottom_shadow_koef; // Влияние теней, котрые находятся под листом бумаги (чем больше значение тем более выражены тени)
    float light_mix_koef; // Влияние освещения на всё изображение. При большой величине влияние больше
} Mesh;

typedef struct {
    unsigned char r, g, b;
} ColorRGB;

typedef struct {
    float r, g, b;
} RgbF;

//Mesh* create_mesh(int img_width, int img_height, int wcnt, int hcnt);
//int render(unsigned char* input_image, int input_width, int input_stride, int input_height, Mesh* mesh, unsigned char* output_image, int output_width, int output_stride, int output_height, float R, float L, float F);
//void delete_mesh(Mesh* mesh);

extern "C" FORWARD_TRACER_API int render(unsigned char* input_image,
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
    float F);

extern "C" FORWARD_TRACER_API Mesh* create_mesh(int img_width, int img_height, int wcnt, int hcnt);

extern "C" FORWARD_TRACER_API bool reproject_point(Mesh* mesh, float L, float src_x, float src_y, float* dst_x, float* dst_y);

extern "C" FORWARD_TRACER_API void delete_mesh(Mesh* mesh);

extern "C" FORWARD_TRACER_API void mesh_set_rect_occluder(Mesh* m,
    float cx, float cy, float cz,
    float w, float h,
    float yaw, float pitch, float roll,
    int circle_segments);

struct BVH;

BVH* build_bvh(const Mesh* mesh);
void  delete_bvh(BVH* bvh);

bool  bvh_intersect(const Mesh* mesh, const BVH* bvh,
    const Ray& ray, int* outTriangleId, Vector3d* outPoint);

// Вариант с угловым весом (cos^power). rays_negative_z = 1 если полусфера "вниз".
RgbF shade_background_from_plane_weighted(const Mesh* mesh, const BVH* bvh,
    const Ray& ray, float z_plane,
    RgbF base_color,
    int samples, float shadow_strength,
    float eps,
    int rays_negative_z, float cos_power);

// --- Окклюдер-прямоугольник ---
typedef struct {
    Vector3d center;   // центр прямоугольника
    float width;       // ширина по локальной оси U
    float height;      // высота по локальной оси V
    // Повороты в радианах: yaw вокруг Z, pitch вокруг Y, roll вокруг X
    float yaw, pitch, roll;
} RectOccluder;

// Вычислить долю перекрытия дискового источника (0..1)
// P        — точка на поверхности, которую освещаем
// lightC   — центр источника (x,y,z) ; предполагаем плоскость z = lightC.z
// lightR   — радиус источника (light_diameter/2)
// circleSegments — сколько сторон у аппроксимации диска (напр. 32)
// Возвращает fraction в [0,1]
float rect_occluder_coverage_on_disk(const Vector3d* P,
    const RectOccluder* Rocc,
    const Vector3d* lightC,
    float lightR,
    int circleSegments);

#endif // MESH_RENDER_H
