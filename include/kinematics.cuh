#pragma once
#include "robot.cuh"
#include <queue>

struct Transform {
    vec3 translation;
    quat4 rotation;

    __host__ __device__ Transform() : translation(0, 0, 0), rotation(1, 0, 0, 0) {}
    __host__ __device__ Transform(vec3 translation, quat4 rotation) : translation(translation), rotation(rotation) {}

    __host__ __device__ Transform operator*(const Transform& other) const {
        vec3 rotated_trans = rotate(other.translation, rotation);
        vec3 new_trans = translation + rotated_trans;
        quat4 new_rot = rotation * other.rotation;
        
        return Transform(new_trans, new_rot);
    }
};

class Kinematics{
public:
    Kinematics(Robot robot) : robot(robot) {
        joint_positions.resize(robot.num_dof(), 0.0);
        link_transforms.resize(robot.links.size());
        build_kinematic_tree();
        update_forward_kinematics(); // Initialize to zero configuration
    }

    __host__ __device__
    void set_joint_position(int joint_idx, double value) {
        if (joint_idx >= 0 && joint_idx < joint_positions.size()) {
            joint_positions[joint_idx] = std::clamp(
                value,
                robot.joints[joint_idx].lower_limit,
                robot.joints[joint_idx].upper_limit
            );
        }
    }

    void set_joint_position(std::string name, double value) {
        int joint_idx = robot.joint_name_to_idx[name];
        set_joint_position(joint_idx, value);
    }

    void set_joint_positions(const std::vector<double>& configuration) {
        if (configuration.size() != joint_positions.size()) {
            std::cerr << "[ERROR]: Expected " << joint_positions.size()
                    << " joint positions but got " << configuration.size() << std::endl;
            return;
        }
        for (int i = 0; i < configuration.size(); i++) {
            set_joint_position(i, configuration[i]);
        }
    }

    const std::vector<double>& get_joint_positions() const {
        return joint_positions;
    }

    __host__ __device__
    void update_forward_kinematics() {
        link_transforms[robot.root_link_idx] = Transform();

        for(int link_idx : traversal_order) {
            if (link_idx == robot.root_link_idx) continue;

            int joint_idx = link_to_parent_joint[link_idx];
            const Joint& joint = robot.joints[joint_idx];

            Transform parent_transform = link_transforms[joint.parent_link_idx];
            Transform joint_transform = compute_joint_transform(joint, joint_positions[joint_idx]);

            link_transforms[link_idx] = parent_transform * joint_transform;
        }
    }

    __host__
    std::vector<Primitive> get_transformed_primitives() const {
        std::vector<Primitive> transformed_prims;

        for (int i = 0; i < robot.links.size(); i++) {
            Primitive prim = robot.links[i].shape;
            prim = transform_primitive(prim, link_transforms[i]);
            transformed_prims.push_back(prim);
        }

        return transformed_prims;
    }

    __host__ __device__ Transform get_link_transform_at(int idx) const {
        #ifdef __CUDA_ARCH__
        return link_transforms_ptr[idx];
        #else
        return link_transforms[idx];
        #endif
    }

    __host__ __device__ Primitive get_transformed_primitive(int link_idx) const {
        Primitive prim = robot.get_link(link_idx).shape;
        return transform_primitive(prim, get_link_transform_at(link_idx));
    }

    Transform get_link_transform(int link_idx) const {
        return link_transforms[link_idx];
    }
    
    Transform get_link_transform(const std::string& link_name) const {
        int idx = robot.get_link_idx(link_name);
        return link_transforms[idx];
    }

    __host__ __device__ const Robot& get_robot() const {
        return robot;
    }

private:
    Robot robot;
    std::vector<double> joint_positions;
    std::vector<Transform> link_transforms;

    std::vector<int> traversal_order;
    std::vector<int> link_to_parent_joint;

    double* joint_positions_ptr;
    Transform* link_transforms_ptr;
    int num_joints_val;
    int num_links_val;

    void build_kinematic_tree() {
        link_to_parent_joint.resize(robot.links.size(), -1);

        for (size_t i = 0; i < robot.joints.size(); i++) {
            link_to_parent_joint[robot.joints[i].child_link_idx] = i;
        }

        std::queue<int> queue;
        std::vector<bool> visited(robot.links.size(), false);

        queue.push(robot.root_link_idx);
        visited[robot.root_link_idx] = true;

        while (!queue.empty()) {
            int link_idx = queue.front();
            queue.pop();

            traversal_order.push_back(link_idx);

            for (const auto& joint : robot.joints) {
                if (joint.parent_link_idx == link_idx && !visited[joint.child_link_idx]) {
                    queue.push(joint.child_link_idx);
                    visited[joint.child_link_idx] = true;
                }
            }
        }
    }

    Transform compute_joint_transform(const Joint& joint, double q) const {
        Transform result;

        result.translation = joint.origin_xyz;
        result.rotation = joint.origin_rpy;

        if (joint.type == REVOLUTE) {
            quat4 joint_motion = quat_from_axis_angle(joint.axis, q);
            result.rotation = result.rotation * joint_motion;
        } else if (joint.type == PRISMATIC) {
            vec3 joint_motion = joint.axis * q;
            result.translation = result.translation + joint_motion;
        }

        return result;
    }

    __host__ __device__
    Primitive transform_primitive(const Primitive& prim, const Transform& transform) const {
        Primitive result = prim;
        
        switch(prim.type) {
            case PRIM_SPHERE:
                result.data.sphere.center = transform.translation + rotate(prim.data.sphere.center, transform.rotation);
                break;

            case PRIM_BOX:
                result.data.box.center = transform.translation + rotate(prim.data.box.center, transform.rotation);
                result.data.box.orientation = transform.rotation * prim.data.box.orientation;
                break;

            case PRIM_CYLINDER:
                result.data.cylinder.center = transform.translation + rotate(prim.data.cylinder.center, transform.rotation);
                result.data.cylinder.orientation = transform.rotation * prim.data.cylinder.orientation;
                break;
        } 

        return result;
    }
};