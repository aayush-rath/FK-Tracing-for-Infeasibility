#include "loader.cuh"
#include "kinematics.cuh"
#include "sdf.cuh"
#include "manifold_tracing.cuh"
#include <chrono>
#include <fstream>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA ERROR: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

#include <iostream>

void simplex_vertices_cartesian(
    const FK_Triangulation& fk,
    const Permutahedral_Simplex& s,
    double verts[4][3]
) {
    int32_t v[MAX_D];
    for (int i = 0; i < fk.amb_dim; i++)
        v[i] = s.anchor[i];
    fk.cartesian_coordinates(v, verts[0]);

    for (int b = 0; b < s.num_blocks - 1; b++) {
        for (int j = 0; j < s.block_sizes[b]; j++) {
            uint8_t idx = s.blocks[b][j];
            if (idx < fk.amb_dim) {
                v[idx]++;
            }
        }
        fk.cartesian_coordinates(v, verts[b + 1]);
    }
}

void dump_intersecting_tetrahedra_from_components(
    const FK_Triangulation& fk,
    const Hashtable& d_Ls,
    const int* component_parent,
    const std::string& output_prefix
) {
    std::vector<Permutahedral_Simplex> h_keys(d_Ls.capacity);
    std::vector<Point> h_values(d_Ls.capacity);
    std::vector<int> h_occupied(d_Ls.capacity);
    std::vector<int> h_component(d_Ls.capacity);

    cudaMemcpy(h_keys.data(), d_Ls.simplices,
               d_Ls.capacity * sizeof(Permutahedral_Simplex),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(h_values.data(), d_Ls.coordinates,
               d_Ls.capacity * sizeof(Point),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(h_occupied.data(), d_Ls.occupied,
               d_Ls.capacity * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(h_component.data(), d_Ls.component,
               d_Ls.capacity * sizeof(int),
               cudaMemcpyDeviceToHost);

    std::unordered_map<
        int,
        std::unordered_map<
            Permutahedral_Simplex,
            std::vector<Point>,
            Permutahedral_Simplex_Hash
        >
    > component_maps;

    for (int i = 0; i < d_Ls.capacity; i++) {

        if (!h_occupied[i])
            continue;

        const Permutahedral_Simplex& edge = h_keys[i];
        const Point& p = h_values[i];

        int comp = h_component[i];

        int root = find_root(component_parent, comp);

        if (edge.num_blocks != 2)
            continue;

        Permutahedral_Simplex cofs[MAX_COFACES];
        int num_cofaces = cofaces(edge, cofs, fk.amb_dim);

        for (int j = 0; j < num_cofaces; j++) {
            component_maps[root][cofs[j]].push_back(p);
        }
    }

    for (auto& comp_pair : component_maps) {

        int root_id = comp_pair.first;
        auto& Ps = comp_pair.second;

        std::string filename =
            output_prefix + "_component_" +
            std::to_string(root_id) + ".txt";

        std::ofstream out(filename);
        if (!out) {
            std::cerr << "Failed to open " << filename << "\n";
            continue;
        }

        int written = 0;

        for (const auto& kv : Ps) {

            const Permutahedral_Simplex& simplex = kv.first;
            const std::vector<Point>& pts = kv.second;

            if (simplex.num_blocks != 4)
                continue;

            if (pts.empty())
                continue;

            double verts[4][3];
            simplex_vertices_cartesian(fk, simplex, verts);

            for (int i = 0; i < 4; i++) {
                out << verts[i][0] << " "
                    << verts[i][1] << " "
                    << verts[i][2] << " ";
            }

            out << "| ";

            for (const auto& p : pts) {
                out << p[0] << " "
                    << p[1] << " "
                    << p[2] << " ";
            }

            out << "\n";
            written++;
        }

        out.close();

        std::cout << "Root Component " << root_id
                  << " → wrote " << written
                  << " tetrahedra to "
                  << filename << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <robot_file> <scene_file> <num_rays>" << std::endl;
        return 0;
    }

    const char* robot_file = argv[1];
    const char* scene_file = argv[2];
    int num_rays = std::stoi(argv[3]);

    Robot robot = load_urdf(robot_file);
    Scene scene = load_scene_json(scene_file);

    std::vector<DeviceLink> h_links(robot.num_links());
    for (size_t i = 0; i < robot.links.size(); i++) h_links[i].shape = robot.links[i].shape;
    std::vector<DeviceJoint> h_joints(robot.num_joints());
    for (size_t i = 0; i < robot.joints.size(); i++) {
        auto& j = robot.joints[i];
        h_joints[i] = {j.type, j.origin_xyz, j.origin_rpy, j.axis, j.lower_limit, j.upper_limit, j.parent_link_idx, j.child_link_idx};
    }


    DeviceLink* d_links;
    DeviceJoint* d_joints;
    Primitive* d_obstacles;

    CUDA_CHECK(cudaMalloc(&d_links, h_links.size() * sizeof(DeviceLink)))
    CUDA_CHECK(cudaMalloc(&d_joints, h_joints.size() * sizeof(DeviceJoint)));
    CUDA_CHECK(cudaMalloc(&d_obstacles, scene.num_primitives() * sizeof(Primitive)));

    CUDA_CHECK(cudaMemcpy(d_links, h_links.data(), sizeof(DeviceLink) * h_links.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_joints, h_joints.data(), sizeof(DeviceJoint) * h_joints.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obstacles, scene.primitives.data(), sizeof(Primitive) * scene.primitives.size(), cudaMemcpyHostToDevice));

    DeviceSDFContext h_ctx = {
        {d_links, (int)robot.links.size(), d_joints, (int)robot.joints.size(), robot.root_link_idx},
        {d_obstacles, (int)scene.primitives.size()}
    };

    DeviceSDFContext* d_ctx;
    CUDA_CHECK(cudaMalloc(&d_ctx, sizeof(DeviceSDFContext)));
    CUDA_CHECK(cudaMemcpy(d_ctx, &h_ctx, sizeof(DeviceSDFContext), cudaMemcpyHostToDevice));

    RobotSDF sdf_functor;
    sdf_functor.d_ctx = d_ctx;
    sdf_functor.dof = robot.num_dof();

    FK_Triangulation fk(robot.num_dof());
    fk.scale = 10.0;

    std::vector<double> initial_guess = {0.0, 0.0, 0.0, 0.0};
    std::vector<double> final_guess = {0.0, 3.05, 0.0, 0.0};

    std::vector<double>seed;
    int dim = robot.num_dof();

    bool ok = sdf_functor.get_line_rays(initial_guess.data(), final_guess.data(), seed, num_rays);
    if (!ok) {
        std::cerr << "Failed to project seed onto manifold\n";
        return;
    }

    int* d_component_array;
    int num_seeds = seed.size() / dim;
    cudaMalloc(&d_component_array, num_seeds * sizeof(int));


    std::vector<int> h_comp(num_seeds);
    for(int i=0; i< num_seeds; i++) h_comp[i] = i;
    cudaMemcpy(d_component_array, h_comp.data(), num_seeds * sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "Component array: ";
    for (int i = 0; i < 4; i++) std::cout << h_comp[i] << " ";
    std::cout << std::endl;

    std::cout << "Seeds: ";
    for (int i = 0; i < num_seeds * dim; i++) {
        if (i % dim == 0) std::cout << "[";
        std::cout << seed[i] << ' ';
        if (i % dim == dim-1) std::cout << "]";
    }
    std::cout << std::endl;

    // double seed_flattened[MAX_NUM_SEEDS * MAX_D];
    // int num_seeds = seed.size();
    // for (int i = 0; i < num_seeds; i++) {
    //     for (int j = 0; j < seed[0].size(); j++) {
    //         seed_flattened[i * dim + j] = seed[i][j];
    //     }
    // }

    int hash_capacity = 1 << 20;
    Hashtable d_Ls = allocate_device_hash_table(hash_capacity);

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "I am in main1\n";
    traceManifold(fk, sdf_functor, d_Ls, seed.data(), d_component_array, num_seeds);
    std::cout << "I am in main2\n";
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Surface triangulation time: " << duration.count() << " milliseconds" << std::endl;

    int hcompo[num_seeds];
    cudaMemcpy(hcompo, d_component_array, num_seeds * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Component array: ";
    for (int i = 0; i < num_seeds; i++) std::cout << hcompo[i] << " ";
    std::cout << std::endl;

    dump_intersecting_tetrahedra_from_components(fk, d_Ls, hcompo,  "../plotting/intersecting_tetrahedra_co");

    cudaFree(d_Ls.simplices);
    cudaFree(d_Ls.coordinates);
    cudaFree(d_Ls.occupied);
    cudaFree(d_component_array);
    cudaFree(d_joints);
    cudaFree(d_links);
    cudaFree(d_obstacles);
    cudaFree(d_ctx);
    

    std::cout << "Done." << std::endl;
    return 0;
}


