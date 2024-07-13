#pragma once
#define FORWARD_TRACER_API __declspec(dllexport)

    typedef struct {
        float x, y, z;
    } Vector3d;

    typedef struct {
        float x, y;
    } Vector2d;

    typedef struct {
        Vector3d origin, direction;
    } Ray;

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
    } Mesh;

    typedef struct {
        unsigned char r, g, b;
    } ColorRGB;


    extern "C" FORWARD_TRACER_API int render(unsigned char* input_image, int input_width, int input_stride, int input_height, Mesh* mesh, unsigned char* output_image, int output_width, int output_stride, int output_height, float R, float L, float F);

    extern "C" FORWARD_TRACER_API Mesh* create_mesh(int img_width, int img_height, int wcnt, int hcnt);

    extern "C" FORWARD_TRACER_API bool reproject_point(Mesh * mesh, float L, float src_x, float src_y, float* dst_x, float* dst_y);
    
    extern "C" FORWARD_TRACER_API void delete_mesh(Mesh* mesh);

    
