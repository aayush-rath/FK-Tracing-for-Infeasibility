#ifndef ROBOT_CUH
#define ROBOT_CUH

#include "geometry.cuh"
#include <vector>
#include <map>
#include <iostream>

enum JOINT_TYPE {
    REVOLUTE,
    PRISMATIC,
    FIXED
};

struct Joint {
    std::string name;
    JOINT_TYPE type;

    vec3 origin_xyz;
    quat4 origin_rpy;
    vec3 axis;

    double lower_limit;
    double upper_limit;

    int parent_link_idx;
    int child_link_idx;
};

struct Link {
    std::string name;
    Primitive shape;
};

struct Robot {
    std::string name;
    std::vector<Link> links;
    std::vector<Joint> joints;

    Link* links_ptr;
    Joint* joints_ptr;
    int num_links_val;
    int num_joints_val;
    
    std::map<std::string, int> link_name_to_idx;
    std::map<std::string, int> joint_name_to_idx;
    
    int root_link_idx;
    
    int get_link_idx(const std::string& name) const { return link_name_to_idx.at(name); };
    int get_joint_idx(const std::string& name) const { return joint_name_to_idx.at(name); };
    
    int num_dof() const {
        int count = 0;
        for (int i = 0; i < joints.size(); i++) {
            if (joints[i].type != FIXED) count++;
        } 
        return count;
    }

    __host__ __device__ int num_links() const {
        #ifdef __CUDA_ARCH__
        return num_links_val;
        #else
        return links.size();
        #endif
    }
    
    __host__ __device__ int num_joints() const {
        #ifdef __CUDA_ARCH__
        return num_joints_val;
        #else
        return joints.size();
        #endif
    }
    
    __host__ __device__ const Link& get_link(int idx) const {
        #ifdef __CUDA_ARCH__
        return links_ptr[idx];
        #else
        return links[idx];
        #endif
    }
    
    __host__ __device__ const Joint& get_joint(int idx) const {
        #ifdef __CUDA_ARCH__
        return joints_ptr[idx];
        #else
        return joints[idx];
        #endif
    }
};


inline std::ostream& operator<<(std::ostream& os, const Robot& robot) {
    os << "Robot '" << robot.name << "'" << std::endl;
    os << "         Links: " << robot.links.size() << std::endl;
    os << "         DOF: " << robot.num_dof() << std::endl;
    return os;
}

#endif