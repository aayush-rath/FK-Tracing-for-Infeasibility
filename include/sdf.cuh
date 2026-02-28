#pragma once
#include "geometry.cuh"
#include "robot.cuh"
#include "kinematics.cuh"
#include "math.cuh"
#include <curand_kernel.h>

#define MAX_LINKS 10

// Device-compatible structures (NO std::string or std::vector)
struct DeviceLink {
    Primitive shape;
};

struct DeviceJoint {
    JOINT_TYPE type;
    vec3 origin_xyz;
    quat4 origin_rpy;
    vec3 axis;
    double lower_limit;
    double upper_limit;
    int parent_link_idx;
    int child_link_idx;
};

struct DeviceRobotData {
    DeviceLink* links;
    int num_links;
    DeviceJoint* joints;
    int num_joints;
    int root_link_idx;
};

struct DeviceSceneData {
    Primitive* obstacles;
    int num_obstacles;
};

struct DeviceSDFContext {
    DeviceRobotData robot;
    DeviceSceneData scene;
};

__device__ 
inline void compute_fk_device(
    const DeviceRobotData& robot,
    const double* joint_positions,
    Transform* link_transforms
) {
    link_transforms[robot.root_link_idx] = Transform();

    for (int i = 0; i < robot.num_joints; i++) {
        const DeviceJoint& joint = robot.joints[i];

        Transform parent_tf = link_transforms[joint.parent_link_idx];

        Transform joint_tf;
        joint_tf.translation = joint.origin_xyz;
        joint_tf.rotation    = joint.origin_rpy;

        double q = joint_positions[i];

        if (joint.type == REVOLUTE) {
            quat4 motion = quat_from_axis_angle(joint.axis, q);
            joint_tf.rotation = joint_tf.rotation * motion;
        } else if (joint.type == PRISMATIC) {
            joint_tf.translation = joint_tf.translation + joint.axis * q;
        }

        link_transforms[joint.child_link_idx] = parent_tf * joint_tf;
    }
}



__device__ 
inline Primitive transform_primitive_device(
    const Primitive& prim,
    const Transform& tf
) {
    Primitive result = prim;
    
    switch (prim.type) {
        case PRIM_SPHERE:
            result.data.sphere.center = tf.translation + rotate(prim.data.sphere.center, tf.rotation);
            break;
        case PRIM_BOX:
            result.data.box.center = tf.translation + rotate(prim.data.box.center, tf.rotation);
            result.data.box.orientation = tf.rotation * prim.data.box.orientation;
            break;
        case PRIM_CYLINDER:
            result.data.cylinder.center = tf.translation + rotate(prim.data.cylinder.center, tf.rotation);
            result.data.cylinder.orientation = tf.rotation * prim.data.cylinder.orientation;
            break;
    }
    
    return result;
}

__device__ 
static inline double compute_sdf(
    DeviceSDFContext* ctx,
    const double* config,
    int num_dof
) {
    if (ctx == nullptr) return 1e10;
    Transform local_link_transforms[MAX_LINKS];
    compute_fk_device(ctx->robot, config, local_link_transforms);
    double min_clearance = 1e10;

    for (int link_idx = 0; link_idx < ctx->robot.num_links; link_idx++) {
        Primitive robot_prim = transform_primitive_device(
            ctx->robot.links[link_idx].shape,
            local_link_transforms[link_idx]
        );

        for (int obs_idx = 0; obs_idx < ctx->scene.num_obstacles; obs_idx++) {
            Primitive obstacle = ctx->scene.obstacles[obs_idx];
            double dist = primitive_to_primitive_distance(robot_prim, obstacle);
            min_clearance = min(min_clearance, dist);
        }
    }
    
    return min_clearance;
}

struct RobotSDF {
    DeviceSDFContext* d_ctx;
    int dof;
    __host__ __device__ __forceinline__
    double operator()(const double* config) const {
        #ifdef __CUDA_ARCH__
            return compute_sdf(d_ctx, config, dof);
        #else
            return 1e30;
        #endif
    }

    bool get_line_intersections(
        const double* start_point,
        const double* goal_point,
        std::vector<double>& intersections,
        int samples = 500,
        int max_bisection_iters = 500,
        double tol = 1e-11
    ) const;
    bool get_line_rays(
        const double* start_point,
        const double* goal_point,
        std::vector<double>& intersections,
        int num_rays = 40,
        int samples = 100,
        int max_bisection_iters = 100,
        double tol = 1e-8
    ) const;
};


__global__ void ray_intersection_kernel(
    RobotSDF sdf,
    const double* d_start,
    const double* d_goal,
    const double* d_main_dir,
    double max_dist,
    int num_rays,
    int samples,
    int max_bisection_iters,
    double tol,
    double* d_out_intersections,
    int* d_out_count,
    int max_outputs,
    int dof
);

__global__
void eval_sdf_kernel(
    RobotSDF sdf,
    const double* q,
    double* out
);