#include "loader.cuh"
#include <fstream>
#include <nlohmann/json.hpp>
#include <tinyxml2.h>

Primitive* Scene::send_primitives_to_device() const {
    if (primitives.empty()) return nullptr;

    Primitive* d_primitives;
    cudaMalloc(&d_primitives, primitives.size() * sizeof(Primitive));
    cudaMemcpy(d_primitives, primitives.data(), primitives.size() * sizeof(Primitive), cudaMemcpyHostToDevice);
    return d_primitives;
}

void Scene::add_sphere(vec3 center, double radius) {
    Primitive prim;
    prim.type = PRIM_SPHERE;
    prim.data.sphere.center = center;
    prim.data.sphere.radius = radius;
    primitives.push_back(prim);
}

void Scene::add_box(vec3 center, vec3 size, quat4 orientation) {
    Primitive prim;
    prim.type = PRIM_BOX;
    prim.data.box.center = center;
    prim.data.box.size = size;
    prim.data.box.orientation = orientation;
    primitives.push_back(prim);
}

void Scene::add_cylinder(vec3 center, double radius, double height, quat4 orientation) {
    Primitive prim;
    prim.type = PRIM_CYLINDER;
    prim.data.cylinder.center = center;
    prim.data.cylinder.radius = radius;
    prim.data.cylinder.height = height;
    prim.data.cylinder.orientation = orientation;
    primitives.push_back(prim);
}

Scene load_scene_json(const char* filename) {
    Scene scene;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open the scene file: " << filename << std::endl;
        return scene;
    }

    nlohmann::json j;
    file >> j;

    if (j.contains("spheres")) {
        for (const auto& sphere : j["spheres"]) {
            vec3 center(sphere["center"][0], sphere["center"][1], sphere["center"][2]);

            double radius = sphere["radius"];
            scene.add_sphere(center, radius);
        }
    }

    if (j.contains("boxes")) {
        for (const auto& box : j["boxes"]) {
            vec3 center(box["center"][0], box["center"][1], box["center"][2]);
            vec3 size(box["size"][0], box["size"][1], box["size"][2]);
            quat4 orientation;
            if (box.contains("orientation"))orientation = quat4(box["orientation"][0], box["orientation"][1], box["orientation"][2], box["orientation"][3]);
            else orientation = euler_to_quat(box["rpy"][0], box["rpy"][1], box["rpy"][2]);

            scene.add_box(center, size, orientation);
        }
    }

    if (j.contains("cylinders")) {
        for (const auto& cylinder : j["cylinders"]) {
            vec3 center(cylinder["center"][0], cylinder["center"][1], cylinder["center"][2]);
            double radius = cylinder["radius"];
            double height = cylinder["height"];
            quat4 orientation;
            if (cylinder.contains("orientation"))orientation = quat4(cylinder["orientation"][0], cylinder["orientation"][1], cylinder["orientation"][2], cylinder["orientation"][3]);
            else orientation = euler_to_quat(cylinder["rpy"][0], cylinder["rpy"][1], cylinder["rpy"][2]);

            scene.add_cylinder(center, radius, height, orientation);
        }
    }

    std::cout << "Loaded the scene with " << scene.num_primitives() << " primitives" << std::endl;
    return scene;
}

Robot load_urdf(const char* filename) {
    Robot robot;

    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(filename) != tinyxml2::XML_SUCCESS) {
        std::cerr << "Failed to load URDF file: " << filename << std::endl;
        return robot;
    }

    tinyxml2::XMLElement* robot_elem = doc.FirstChildElement("robot");
    if (!robot_elem) {
        std::cerr << "Robot element not found in URDF file: " << filename << std::endl;
        return robot;
    }

    robot.name = robot_elem->Attribute("name") ? robot_elem->Attribute("name") : "unnamed";
    for (tinyxml2::XMLElement* link_elem = robot_elem->FirstChildElement("link");
        link_elem != nullptr;
        link_elem = link_elem->NextSiblingElement("link"))  
    {
        Link link;
        link.name = link_elem->Attribute("name");

        tinyxml2::XMLElement* visual_elm = link_elem->FirstChildElement("visual");
        if (visual_elm) {
            tinyxml2::XMLElement* origin_elm = visual_elm->FirstChildElement("origin");
            tinyxml2::XMLElement* geom_elem = visual_elm->FirstChildElement("geometry");
            if (geom_elem) {
                tinyxml2::XMLElement* sphere_elm = geom_elem->FirstChildElement("sphere");
                tinyxml2::XMLElement* cylinder_elm = geom_elem->FirstChildElement("cylinder");
                tinyxml2::XMLElement* box_elm = geom_elem->FirstChildElement("box");

                if (sphere_elm) {
                    link.shape.type = PRIM_SPHERE;
                    double radius = sphere_elm->DoubleAttribute("radius");
                    link.shape.data.sphere.radius = radius;

                    vec3 center(0, 0, 0);
                    if (origin_elm && origin_elm->Attribute("xyz")) {
                        float x, y, z;
                        sscanf(origin_elm->Attribute("xyz"), "%f %f %f", &x, &y, &z);
                        center = vec3(x, y, z);
                    }
                    link.shape.data.sphere.center = center;
                }

                if (cylinder_elm) {
                    link.shape.type = PRIM_CYLINDER;
                    double length = cylinder_elm->DoubleAttribute("length");
                    link.shape.data.cylinder.height = length;
                    double radius = cylinder_elm->DoubleAttribute("radius");
                    link.shape.data.cylinder.radius = radius;

                    vec3 center(0, 0, 0);
                    quat4 orientation(1, 0, 0, 0);
                    if (origin_elm && origin_elm->Attribute("xyz")) {
                        float x, y, z;
                        sscanf(origin_elm->Attribute("xyz"), "%f %f %f", &x, &y, &z);
                        center = vec3(x, y, z);
                    }
                    if (origin_elm && origin_elm->Attribute("rpy")) {
                        float r, p, y;
                        sscanf(origin_elm->Attribute("rpy"), "%f %f %f", &r, &p, &y);
                        orientation = euler_to_quat(r, p, y);
                    }
                    link.shape.data.cylinder.center = center;
                    link.shape.data.cylinder.orientation = orientation;
                }


                if (box_elm) {
                    link.shape.type = PRIM_BOX;
                    vec3 size(0, 0, 0);
                    float size_x, size_y, size_z;
                    sscanf(box_elm->Attribute("size"), "%f %f %f", &size_x, &size_y, &size_z);
                    size = vec3(size_x, size_y, size_z);
                    link.shape.data.box.size = size;

                    vec3 center(0, 0, 0);
                    quat4 orientation(1, 0, 0, 0);
                    if (origin_elm && origin_elm->Attribute("xyz")) {
                        float x, y, z;
                        sscanf(origin_elm->Attribute("xyz"), "%f %f %f", &x, &y, &z);
                        center = vec3(x, y, z);
                    }
                    if (origin_elm && origin_elm->Attribute("rpy")) {
                        float r, p, y;
                        sscanf(origin_elm->Attribute("rpy"), "%f %f %f", &r, &p, &y);
                        orientation = euler_to_quat(r, p, y);
                    }
                    link.shape.data.box.center = center;
                    link.shape.data.box.orientation = orientation;
                }
            }
        }
        int link_idx = robot.links.size();
        robot.links.push_back(link);
        robot.link_name_to_idx[link.name] = link_idx;
    }


    for (tinyxml2::XMLElement* joint_elm = robot_elem->FirstChildElement("joint");
        joint_elm != nullptr;
        joint_elm = joint_elm->NextSiblingElement("joint")) 
    {
        Joint joint;
        joint.name = joint_elm->Attribute("name");
        const char* type_str = joint_elm->Attribute("type");
        // joint.type = REVOLUTE;
        if (strcmp(type_str, "revolute") == 0) {
            joint.type = REVOLUTE;
        } else if (strcmp(type_str, "prismatic") == 0) {
            joint.type = PRISMATIC;
        } else if (strcmp(type_str, "fixed") == 0){
            joint.type = FIXED;
        } else {
            std::cerr << "Unsupported joint type: " << type_str << std::endl;
            continue;
        }


        tinyxml2::XMLElement* origin_elm = joint_elm->FirstChildElement("origin");
        tinyxml2::XMLElement* parent_elm = joint_elm->FirstChildElement("parent");
        tinyxml2::XMLElement* child_elm = joint_elm->FirstChildElement("child");

        joint.child_link_idx = robot.link_name_to_idx[child_elm->Attribute("link")];
        joint.parent_link_idx = robot.link_name_to_idx[parent_elm->Attribute("link")];
        tinyxml2::XMLElement* axis_elm = joint_elm->FirstChildElement("axis");

        if (origin_elm && origin_elm->Attribute("xyz")) {
            float x, y, z;
            sscanf(origin_elm->Attribute("xyz"), "%f %f %f", &x, &y, &z);
            joint.origin_xyz = vec3(x, y, z);
        }
        if (origin_elm && origin_elm->Attribute("rpy")) {
            float r, p, y;
            sscanf(origin_elm->Attribute("rpy"), "%f %f %f", &r, &p, &y);
            joint.origin_rpy = euler_to_quat(r, p, y);
        }
        if (axis_elm) {
            float x, y, z;
            sscanf(axis_elm->Attribute("xyz"), "%f %f %f", &x, &y, &z);
            joint.axis = vec3(x, y, z);
        }

        tinyxml2::XMLElement* limit_elm = joint_elm->FirstChildElement("limit");
        if (limit_elm) {
            joint.lower_limit = limit_elm->DoubleAttribute("lower");
            joint.upper_limit = limit_elm->DoubleAttribute("upper");
        }
        robot.joints.push_back(joint);
        robot.joint_name_to_idx[joint.name] = robot.joints.size() - 1;
    }

    std::vector<bool> is_child(robot.links.size(), false);
    for (const auto& joint : robot.joints) {
        if (joint.child_link_idx >= 0) {
            is_child[joint.child_link_idx] = true;
        }
    }
    
    robot.root_link_idx = -1;
    for (size_t i = 0; i < robot.links.size(); i++) {
        if (!is_child[i]) {
            robot.root_link_idx = i;
            std::cout << "Root link: " << robot.links[i].name << std::endl;
            break;
        }
    }
    
    std::cout << "Robot '" << robot.name << "' loaded successfully" << std::endl;
    return robot;
}