#pragma once
#include "geometry.cuh"
#include "robot.cuh"
#include "kinematics.cuh"
#include <vector>

struct Scene {
    std::vector<Primitive> primitives;

    Primitive* primitives_ptr;
    int num_primitives_val;
    
    __host__ __device__ int num_primitives() const {
        #ifdef __CUDA_ARCH__
        return num_primitives_val;
        #else
        return primitives.size();
        #endif
    }
    
    __host__ __device__ const Primitive& get_primitive(int idx) const {
        #ifdef __CUDA_ARCH__
        return primitives_ptr[idx];
        #else
        return primitives[idx];
        #endif
    }
    
    Primitive* send_primitives_to_device() const;

    void add_sphere(vec3 center, double radius);
    void add_box(vec3 center, vec3 size, quat4 orientation);
    void add_cylinder(vec3 center, double radius, double half_height, quat4 orientation);
};


Scene load_scene_json(const char* filename);
Robot load_urdf(const char* filename);


