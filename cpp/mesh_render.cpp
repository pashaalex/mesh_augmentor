#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <thread>
#include <vector>
#include <fstream>
#include "mesh_render.h"

// Begin Vector algebra

Vector3d vector_add(Vector3d a, Vector3d b) {
    Vector3d res = { a.x + b.x, a.y + b.y, a.z + b.z };
    return res;
}

Vector3d vector_sub(Vector3d* a, Vector3d* b) {
    Vector3d res = { a->x - b->x, a->y - b->y, a->z - b->z };
    return res;
}

Vector3d vector_mul(Vector3d a, float scalar) {
    Vector3d res = { a.x * scalar, a.y * scalar, a.z * scalar };
    return res;
}

float vector_dot(Vector3d* a, Vector3d* b) {
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

Vector3d vector_cross(Vector3d* a, Vector3d* b) {
    Vector3d res = { a->y * b->z - a->z * b->y, a->z * b->x - a->x * b->z, a->x * b->y - a->y * b->x };
    return res;
}

Vector3d vector_normalize(Vector3d* a) {
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

Vector3d ray_point_at(Ray* ray, float t) {
    return vector_add(ray->origin, vector_mul(ray->direction, t));
}

// End Vector algebra

ColorRGB get_pixel_color(const unsigned char* image, int width, int stride, int height, int x, int y) {
    if (x > 0 && x < width && y > 0 && y < height) {
        int d = stride * y + 3 * x;
        ColorRGB res = { image[d + 0], image[d + 1], image[d + 2] };
        return res;
    }

    return { 0, 0, 0 };
}

ColorRGB get_bilinear_interpolated_color(const unsigned char* image, int width, int stride, int height, float x, float y) {
    if (x < 0 || x >= width - 1 || y < 0 || y >= height - 1) {
        return { 0, 0, 0 };  
    }

    int x1 = static_cast<int>(x);
    int y1 = static_cast<int>(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

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

ColorRGB get_RGB_from_barycentric(unsigned char* input_image, int width, int stride, int height, Triangle triangle, Vector3d barycentricCoordinates) {

    float newX = barycentricCoordinates.x * triangle.imagePoint0->x +
        barycentricCoordinates.y * triangle.imagePoint1->x +
        barycentricCoordinates.z * triangle.imagePoint2->x;

    float newY = barycentricCoordinates.x * triangle.imagePoint0->y +
        barycentricCoordinates.y * triangle.imagePoint1->y +
        barycentricCoordinates.z * triangle.imagePoint2->y;

    return get_bilinear_interpolated_color(input_image, width, stride, height, newX, newY);
    //return get_pixel_color(input_image, width, stride, height, (int)newX, (int)newY);
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

int triangle_intersects(Triangle triangle, Ray ray, Vector3d* intersectionPoint) {
    Vector3d edge1 = vector_sub(triangle.v1, triangle.v0);
    Vector3d edge2 = vector_sub(triangle.v2, triangle.v0);
    Vector3d h = vector_cross(&ray.direction, &edge2);
    double a = vector_dot(&edge1, &h);

    if (a > -1e-6 && a < 1e-6)
        return 0;

    double f = 1.0 / a;
    Vector3d s = vector_sub(&ray.origin, triangle.v0);
    double u = f * vector_dot(&s, &h);

    if (u < 0.0 || u > 1.0)
        return 0;

    Vector3d q = vector_cross(&s, &edge1);
    double v = f * vector_dot(&ray.direction, &q);

    if (v < 0.0 || u + v > 1.0)
        return 0;

    double t = f * vector_dot(&edge2, &q);
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

int render(unsigned char* input_image, int input_width, int input_stride, int input_height, Mesh* mesh, unsigned char* output_image, int output_width, int output_stride, int output_height, float R, float L, float F)
{
    //{
    //    std::ofstream outfile;
    //    outfile.open("test.txt", std::ios_base::app); // append instead of overwrite
    //    outfile << "W=" << input_width << " H=" << input_height << " S=" << input_stride;
    //    outfile << "T=" << triangleCount;
    //    outfile << "W2=" << output_width << " H2=" << output_height << " S2=" << output_stride;
    //    outfile << " R=" << R << " L=" << L << " F=" << F;
    //    outfile << " B=" << (BYTE)input_image[0];
    //} 

    /*
    {        
        std::ofstream outfile;
        outfile.open("test.txt", std::ios_base::app); // append instead of overwrite    

        int triangleId = 0;

        Vector3d p2 = { mesh->triangles[triangleId].v1->x - 5, mesh->triangles[triangleId].v1->y + 5, mesh->triangles[triangleId].v1->z } ;
        float l = sqrtf(p2.x * p2.x + p2.y * p2.y + p2.z * p2.z);
        p2.x = p2.x / l;
        p2.y = p2.y / l;
        p2.z = p2.z / l;
        Vector3d p1 = { 0, 0, 0 };
        Ray ray = { p1, p2 };

        
        Vector3d intersectionPoint;
        if (triangle_intersects(mesh->triangles[triangleId], ray, &intersectionPoint) > 0) {
            Vector3d barycentricCoordinates = triangle_get_barycentric_coordinates(mesh->triangles[triangleId], intersectionPoint);

            Triangle triangle = mesh->triangles[triangleId];

            float newX = barycentricCoordinates.x * triangle.imagePoint0->x +
                barycentricCoordinates.y * triangle.imagePoint1->x +
                barycentricCoordinates.z * triangle.imagePoint2->x;

            float newY = barycentricCoordinates.x * triangle.imagePoint0->y +
                barycentricCoordinates.y * triangle.imagePoint1->y +
                barycentricCoordinates.z * triangle.imagePoint2->y;

            outfile << "3DC " << intersectionPoint.x << "," << intersectionPoint.y << "," << intersectionPoint.z << "\n";
            outfile << "newC " << newX << "," << newY << "\n";

            ColorRGB color = get_RGB_from_barycentric(input_image, input_width, input_stride, input_height, mesh->triangles[triangleId], barycentricCoordinates);
            outfile << "color " << color.r << "," << color.g << "," << color.b << "\n";
        }
        else
        {
            outfile << "not in";
        }
        return 0;


        
        for (int i = 0; i < mesh->triangleCount; i++)
        {
            outfile << "T(" << i << ") = [";
            outfile << "(" << mesh->triangles[i].v0->x << ", " << mesh->triangles[i].v0->y << ", " << mesh->triangles[i].v0->z << ") ";
            outfile << "(" << mesh->triangles[i].v1->x << ", " << mesh->triangles[i].v1->y << ", " << mesh->triangles[i].v1->z << ") ";
            outfile << "(" << mesh->triangles[i].v2->x << ", " << mesh->triangles[i].v2->y << ", " << mesh->triangles[i].v2->z << ") ";
            outfile << "]; \n";
        }
    }
    */
    
    



    const float PI_F = 3.14159265358979f;

    int dx = output_width / 2;
    int dy = output_height / 2;

    // float R = 150; // Lens radius
    // float L = 66.66666F; // Distance from mtrix to lense
    // float F = 50; // Focal distance
    // Deph of field is about 200

    bool is_log = false;


    // For each point in matrix
#pragma omp parallel for collapse(2)
    for (int y = -dy; y < dy; y++)
        for (int x = -dx; x < dx; x++)
        {
            int N = 0; // Number of rays from this point
            float N_image = 0; // Number of rays that achieve image
            float colorR = 0;
            float colorG = 0;
            float colorB = 0;
            float colorA = 0;

            // Send a ray to the lense      
            for (float r = 0; r < R; r = r + 10) // Send a rays into the points located concentric circles
            {
                float step = PI_F / 5.9f; // each 30 degree                
                if (r < 5) step = 2.5 * PI_F; 
                for (float ra = 0; ra < 2 * PI_F; ra = ra + step)
                {
                    N = N + 1;

                    float lensX = cosf(ra) * r;
                    float lensY = sinf(ra) * r;

                    Vector3d matrixPoint = { (float)x, (float)y, -(float)L };
                    Vector3d lensPoint = { (float)lensX, (float)lensY, (float)0 };

                    Ray rayAfter = refract_ray_through_lens(matrixPoint, lensPoint, F);

                    for (int triangleId = 0; triangleId < mesh->triangleCount; triangleId++)
                    {
                        Vector3d intersectionPoint; 
                        if (triangle_intersects(mesh->triangles[triangleId], rayAfter, &intersectionPoint) > 0) {
                            Vector3d barycentricCoordinates = triangle_get_barycentric_coordinates(mesh->triangles[triangleId], intersectionPoint);

                            ColorRGB color = get_RGB_from_barycentric(input_image, input_width, input_stride, input_height, mesh->triangles[triangleId], barycentricCoordinates);
                            if (!mesh->use_light_info)
                            {
                                // Just simple raytrace without light info
                                colorR += color.r;
                                colorG += color.g;
                                colorB += color.b;
                                N_image = N_image + 1;
                            }
                            else
                            {
                                // use light info
                                Triangle triangle = mesh->triangles[triangleId];
                                Vector3d l1 = { triangle.v1->x - triangle.v0->x, triangle.v1->y - triangle.v0->y, triangle.v1->z - triangle.v0->z };
                                Vector3d l2 = { triangle.v2->x - triangle.v0->x, triangle.v2->y - triangle.v0->y, triangle.v2->z - triangle.v0->z };                                
                                Vector3d normal = vector_cross(&l1, &l2);
                                normal = vector_normalize(&normal);

                                Vector3d lightDirection = { mesh->light_x - intersectionPoint.x, mesh->light_y - intersectionPoint.y, mesh->light_z - intersectionPoint.z };
                                lightDirection = vector_normalize(&lightDirection);

                                float material_r = color.r / 255.0;
                                float material_g = color.g / 255.0;
                                float material_b = color.b / 255.0;

                                // Diffuse component
                                float diff = vector_dot(&normal, &lightDirection);
                                if (diff < 0) diff = -diff;

                                if (mesh->use_shadow_info) {
                                    float t = intersectionPoint.z / (intersectionPoint.z + L);
                                    float light_y1 = mesh->light_y - mesh->light_diameter / 2.0f;
                                    float light_y2 = mesh->light_y + mesh->light_diameter / 2.0f;
                                    float y1 = light_y1 * t + intersectionPoint.y * (1 - t);
                                    float y2 = light_y2 * t + intersectionPoint.y * (1 - t);
                                    if ((mesh->shadow_y > y1) && (mesh->shadow_y < y2) && (y2 - y1 > 0.0001)) {
                                        diff = diff * (mesh->shadow_y - y1) / (y2 - y1);
                                    }
                                    if (mesh->shadow_y < y1) {
                                        diff = 0;
                                    }
                                }

                                // Final color
                                float light = mesh->light_intensivity;
                                colorR += 255.0 * material_r * light * (diff + 0.5) / 1.5;
                                colorG += 255.0 * material_g * light * (diff + 0.5) / 1.5;
                                colorB += 255.0 * material_b * light * (diff + 0.5) / 1.5;
                                N_image = N_image + 1;
                            }

                            break;
                        }
                    }
                }
            }

            // set image color
#pragma omp critical
            {
                int x2 = x + dx;
                int y2 = y + dy;
                int d = output_stride * y2 + x2 * 4;
                output_image[d + 0] = (unsigned char)(colorR / N_image);
                output_image[d + 1] = (unsigned char)(colorG / N_image);
                output_image[d + 2] = (unsigned char)(colorB / N_image);
                output_image[d + 3] = (unsigned char)((255 * N_image) / N);

            }
        }
    return 1;
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

    for (int i = 0; i < mesh->triangleCount; ++i) {
        Triangle& tri = mesh->triangles[i];
        Vector2d a = *tri.imagePoint0;
        Vector2d b = *tri.imagePoint1;
        Vector2d c = *tri.imagePoint2;

        if (compute_barycentric_coordinates(point, a, b, c, u, v, w)) {

            float x = u * tri.v0->x + v * tri.v1->x + w * tri.v2->x;
            float y = u * tri.v0->y + v * tri.v1->y + w * tri.v2->y;
            float z = u * tri.v0->z + v * tri.v1->z + w * tri.v2->z;

            *dst_x = -x * L / z;
            *dst_y = -y * L / z;            
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
    mesh->triangleCount = wcnt * hcnt * 2;
    float dx = img_width / (float)wcnt;
    float dy = img_height / (float)hcnt;
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
