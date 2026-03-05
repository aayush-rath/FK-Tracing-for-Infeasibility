#pragma once

/*
Manifold Tracing implementation for d-1 manifolds using the Permutahedral 
Representation and finding intersection with 1-simplex now on cuda

Aayush Rath
*/

#include "utils.cuh"
#include "permutahedral_simplex.cuh"
#include "fk_triangulation.cuh"
#include <unordered_map>
#include <omp.h>

template <typename Function>
__host__ __device__ __forceinline__
bool edge_intersection(
    Permutahedral_Simplex& s,
    const Function& function,
    const FK_Triangulation& fk,
    double* intersection_point
) {
    if(s.num_blocks != 2) return false;

    int32_t v0[MAX_D];
    for (int i = 0; i < s.amb_dim; ++i) v0[i] = s.anchor[i];
    
    int32_t v1[MAX_D];
    for (int i = 0; i < s.amb_dim; ++i) v1[i] = s.anchor[i];

    for (int j = 0; j < s.block_sizes[0]; ++j) {
        int idx = s.blocks[0][j];
        if (idx == s.amb_dim) {
            for (int k = 0; k < s.amb_dim; ++k) v1[k]++;
        } else {
            v1[idx]++;
        }
    }

    // Cartesian coordinates
    double p0[MAX_D], p1[MAX_D];
    fk.cartesian_coordinates(v0, p0);
    fk.cartesian_coordinates(v1, p1);

    double f0 = function(p0);
    double f1 = function(p1);

    const double vertex_eps = 1e-10;
    if (fabs(f0) < vertex_eps || fabs(f1) < vertex_eps) {
        return false;
    }

    if (f0 * f1 >= 0.0) return false;

    double bary_eps = 1e-8;

    double lambda0 = f1 / (f1 - f0);
    double lambda1 = -f0 / (f1 - f0);
    
    // Check barycentric coordinates are in (0, 1) - strictly interior
    if (lambda0 <= bary_eps || lambda0 >= 1.0 - bary_eps ||
        lambda1 <= bary_eps || lambda1 >= 1.0 - bary_eps) {
        // Intersection is too close to a vertex
        return false;
    }
    
    // Check sum = 1
    if (fabs(lambda0 + lambda1 - 1.0) > bary_eps) {
        return false;
    }

    // Compute intersection point
    for (int i = 0; i < fk.amb_dim; ++i) {
        intersection_point[i] = lambda0 * p0[i] + lambda1 * p1[i];
    }

    return true;
}

__device__
bool bound_check(
    const FK_Triangulation& fk,
    Point& point
) {
    for (int i = 0; i < fk.amb_dim; i++) if (point[i] > 3.14 || point[i] < -3.14) return false;
    return true;
}

struct FrontierNode {
    Permutahedral_Simplex simplex;
    int component;  
};

__host__ __device__ int find_root(const int* parent, int x) {
    while (parent[x] != x)
        x = parent[x];
    return x;
}


// template <typename Function>
// __global__
// void init_frontier_kernel(
//     const double* seed_points,
//     int num_seeds,
//     FrontierNode* d_frontier,
//     int* d_frontier_size,
//     Hashtable visited,
//     FK_Triangulation fk,
//     Function function,
//     int max_frontier_size,
//     int* component_array  // Added to initialize roots
// ) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < num_seeds) {
//         component_array[tid] = tid;
//     }
    
//     if (tid >= num_seeds) return;

//     double seed[MAX_D]; 
//     for (int i = 0; i < fk.amb_dim; i++) {
//         seed[i] = seed_points[tid * fk.amb_dim + i];
//     }

//     Permutahedral_Simplex initial_simplex = locate_simplex(fk, seed);

//     Permutahedral_Simplex edges[MAX_FACES]; 
//     int num_edges = faces(initial_simplex, edges, 1);

//     for (int j = 0; j < num_edges; j++) {
//         double intersection_point[MAX_D];

//         // if (edge_intersection(edges[j], function, fk, intersection_point)) {
            
//             Point p;
//             for (int d = 0; d < fk.amb_dim; d++) p[d] = intersection_point[d];

//             int insert_case = hash_insert(visited, edges[j], p, tid);
//             if (insert_case == -1) {
//                 if (!bound_check(fk, p)) continue;

//                 int idx = atomicAdd(d_frontier_size, 1);
//                 if (idx < max_frontier_size) {
//                     d_frontier[idx].simplex = edges[j];
//                     d_frontier[idx].component = tid;
//                 }
//             }
//         // }
//     }
// }

template <typename Function>
__global__
void initialize_kernel(
    const double* seeds,
    int num_seeds,
    FrontierNode* frontier,
    int* frontier_size,
    Hashtable d_Ls,
    FK_Triangulation fk,
    Function function,
    int* component_array
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_seeds) return;

    double seed[MAX_D];
    for (int d = 0; d < fk.amb_dim; d++) {
        seed[d] = seeds[tid * fk.amb_dim + d];
    }

    Permutahedral_Simplex simplex = locate_simplex(fk, seed);
    Permutahedral_Simplex edges[MAX_FACES];
    int num_edges = faces(simplex, edges, 1);

    for (int j = 0; j < num_edges; j++) {
        double intersection_point[MAX_D];
        if (!edge_intersection(edges[j], function, fk, intersection_point)) {
            continue;
        }

        Point p;
        for (int d = 0; d < fk.amb_dim; d++) {
            p[d] = intersection_point[d];
        }

        int insert_case = hash_insert(d_Ls, edges[j], p, tid);
        if (insert_case == -2 || insert_case == tid) continue;

        if (insert_case >= 0) {
            int r1 = find_root(component_array, tid);
            int r2 = find_root(component_array, insert_case);
            if (r1 != r2) {
                int high = max(r1, r2);
                int low  = min(r1, r2);
                atomicMin(&component_array[high], low);
            }
            continue;
        }

        if (!bound_check(fk, p)) continue;

        int idx = atomicAdd(frontier_size, 1);
        frontier[idx].simplex = edges[j];
        frontier[idx].component = tid;
    }
}

template <typename Function>
__global__
void expand_frontier_kernel (
    const FrontierNode* frontier,
    int frontier_size,
    FrontierNode* next_frontier,
    int* next_frontier_size,
    Hashtable visited,
    FK_Triangulation fk,
    Function function,
    int* component_array
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    const Permutahedral_Simplex& edge = frontier[tid].simplex;
    const int component = frontier[tid].component;
    Permutahedral_Simplex triangles[MAX_COFACES];
    int num_cofaces = cofaces(edge, triangles, 2);

        for (int i = 0; i < num_cofaces; i++) {

        Permutahedral_Simplex new_edges[MAX_FACES];
        int num_edges = faces(triangles[i], new_edges, 1);

        for (int j = 0; j < num_edges; j++) {

            double intersection_point[MAX_D];

            if (!edge_intersection(
                    new_edges[j],
                    function,
                    fk,
                    intersection_point)) {
                continue;
            }

            Point p;
            #pragma unroll
            for (int d = 0; d < fk.amb_dim; d++) {
                p[d] = intersection_point[d];
            }

            int insert_case = hash_insert(visited, new_edges[j], p, component);

            // There is no space left in the hash table or same seed with same component exists
            if (insert_case == -2 || insert_case == component) continue;

            // Component are now merging
            if (insert_case >= 0) {

                int r1 = find_root(component_array, component);
                int r2 = find_root(component_array, insert_case);

                if (r1 != r2) {
                    int high = max(r1, r2);
                    int low  = min(r1, r2);
                    atomicMin(&component_array[high], low);
                }

                continue;
            }

            if (!bound_check(fk, p)) continue;

            int idx = atomicAdd(next_frontier_size, 1);
            next_frontier[idx].simplex = new_edges[j];
            next_frontier[idx].component = component;
        }
    }
}

// template <typename Function>
// void traceManifold(
//     const FK_Triangulation& fk_host,
//     const Function& function,
//     Hashtable& d_Ls,
//     double* seed_host,
//     int* component_array,
//     int num_seeds
// ) {
//     std::cout << "I am here 0\n";
//     FrontierNode* d_frontier, *d_next_frontier;
//     int *d_frontier_size, *d_next_frontier_size;

//     int max_frontier_size = 500000; 

//     std::cout << "I am here 1\n";

//     cudaMalloc(&d_frontier, max_frontier_size * sizeof(FrontierNode));
//     cudaMalloc(&d_next_frontier, max_frontier_size * sizeof(FrontierNode));
//     cudaMalloc(&d_frontier_size, sizeof(int));
//     cudaMalloc(&d_next_frontier_size, sizeof(int));

//     Permutahedral_Simplex initial_simplex[MAX_NUM_SEEDS];

//     for (int i = 0; i < num_seeds; i++) {
//         double seed[MAX_D];
//         std::cout << "Seed " << i << ": ";
//         for (int j =  0; j < fk_host.amb_dim; j++) {
//             seed[j] = seed_host[i * fk_host.amb_dim + j];
//             std::cout << seed[j] << " ";
//         }
//         std::cout << " SDF: " << function(seed);
//         std::cout << std::endl;
//         initial_simplex[i] = locate_simplex(fk_host, seed);
//     }

//     Permutahedral_Simplex initial_edges[MAX_NUM_SEEDS][MAX_FACES];
//     int num_faces[MAX_NUM_SEEDS] = {0};
//     std::cout << "I am here 2\n";
//     for (int i = 0; i < num_seeds; i++) {
//         num_faces[i] += faces(initial_simplex[i], initial_edges[i], 1);
//     }

//     int total_faces = 0;
//     FrontierNode initial_frontier[MAX_NUM_SEEDS * MAX_FACES];
//     int offset = 0;
//     for (int i = 0; i < num_seeds; i++) {
//         for (int j = 0; j < num_faces[i]; j++) {
//             initial_frontier[offset].simplex = initial_edges[i][j];
//             initial_frontier[offset++].component = i;
//         }
//         total_faces += num_faces[i];
//     }
    
//     std::cout << "Initial simplex has " << total_faces << " edges\n";

//     cudaMemcpy(d_frontier, initial_frontier, total_faces * sizeof(FrontierNode), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_frontier_size, &total_faces, sizeof(int), cudaMemcpyHostToDevice);

//     FK_Triangulation fk_device = fk_host;

//     int frontier_size_host = total_faces;
//     int intersect_count = 0;
//     int iteration = 0;

//     while (frontier_size_host > 0) {
        
//         cudaMemset(d_next_frontier_size, 0, sizeof(int));

//         int threads = 256;
//         int blocks = (frontier_size_host + threads - 1) / threads;

//         expand_frontier_kernel<<<blocks, threads>>>(
//             d_frontier,
//             frontier_size_host,
//             d_next_frontier,
//             d_next_frontier_size,
//             d_Ls,
//             fk_device,
//             function,
//             component_array
//         );
        
//         // Check for kernel errors
//         cudaError_t err = cudaGetLastError();
//         if (err != cudaSuccess) {
//             std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
//             break;
//         }
        
//         // Wait for kernel to complete
//         cudaDeviceSynchronize();

//         cudaMemcpy(&frontier_size_host, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);
//         std::swap(d_frontier, d_next_frontier);
//         std::swap(d_frontier_size, d_next_frontier_size);

//         intersect_count += frontier_size_host;
//         iteration++;

//         if (iteration % 10 == 0) {
//             std::cout << "Iteration " << iteration << ", Frontier Size: " << frontier_size_host << ", Total Intersections: " << intersect_count << std::endl;
//         }
        
//         // Safety check to prevent infinite loops
//         if (iteration > 10000) {
//             std::cerr << "Warning: Exceeded maximum iterations (10000), stopping.\n";
//             break;
//         }
//     }

//     std::cout << "Intersection Count: " << intersect_count << std::endl;
//     std::cout << "Total iterations: " << iteration << std::endl;

//     cudaFree(d_frontier);
//     cudaFree(d_frontier_size);
//     cudaFree(d_next_frontier);
//     cudaFree(d_next_frontier_size);
// }

template <typename Function>
void traceManifold(
    const FK_Triangulation& fk_host,
    const Function& function,
    Hashtable& d_Ls,
    double* seed_host,
    int* component_array,
    int num_seeds
) {
    FrontierNode* d_frontier, *d_next_frontier;
    int *d_frontier_size, *d_next_frontier_size;
    FK_Triangulation fk_device = fk_host;

    int max_frontier_size = 5000000; 

    cudaMalloc(&d_frontier, max_frontier_size * sizeof(FrontierNode));
    cudaMalloc(&d_next_frontier, max_frontier_size * sizeof(FrontierNode));
    cudaMalloc(&d_frontier_size, sizeof(int));
    cudaMalloc(&d_next_frontier_size, sizeof(int));

    int frontier_size_host = 0;

    double* d_seeds;
    cudaMalloc(&d_seeds, num_seeds * fk_host.amb_dim * sizeof(double));
    cudaMemcpy(d_seeds, seed_host, num_seeds * fk_host.amb_dim * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemset(d_frontier_size, 0, sizeof(int));

    int threads = 256;
    int blocks = (num_seeds + threads - 1) / threads;
    initialize_kernel<<<blocks, threads>>>(
        d_seeds, num_seeds,
        d_frontier, d_frontier_size,
        d_Ls, fk_device, function, component_array
    );
    cudaDeviceSynchronize();

    cudaMemcpy(&frontier_size_host, d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_seeds);


    int intersect_count = 0;
    int iteration = 0;

    while (frontier_size_host > 0) {
        
        cudaMemset(d_next_frontier_size, 0, sizeof(int));

        int threads = 256;
        int blocks = (frontier_size_host + threads - 1) / threads;

        expand_frontier_kernel<<<blocks, threads>>>(
            d_frontier,
            frontier_size_host,
            d_next_frontier,
            d_next_frontier_size,
            d_Ls,
            fk_device,
            function,
            component_array
        );
        
        // Check for kernel errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
            break;
        }
        
        // Wait for kernel to complete
        cudaDeviceSynchronize();

        cudaMemcpy(&frontier_size_host, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);
        std::swap(d_frontier, d_next_frontier);
        std::swap(d_frontier_size, d_next_frontier_size);

        intersect_count += frontier_size_host;
        iteration++;

        if (iteration % 10 == 0) {
            std::cout << "Iteration " << iteration << ", Frontier Size: " << frontier_size_host << ", Total Intersections: " << intersect_count << std::endl;
        }
        
        // Safety check to prevent infinite loops
        if (iteration > 10000) {
            std::cerr << "Warning: Exceeded maximum iterations (10000), stopping.\n";
            break;
        }
    }

    std::cout << "Intersection Count: " << intersect_count << std::endl;
    std::cout << "Total iterations: " << iteration << std::endl;

    cudaFree(d_frontier);
    cudaFree(d_frontier_size);
    cudaFree(d_next_frontier);
    cudaFree(d_next_frontier_size);
}

struct Permutahedral_Simplex_Hash{
    std::size_t operator()(const Permutahedral_Simplex& s) const {
        std::size_t h = 0;

        auto hash_combine = [&](std::size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };

        hash_combine(s.amb_dim);
        hash_combine(s.num_blocks);

        for (int i = 0; i < s.amb_dim; i++) {
            hash_combine(std::hash<int32_t>{}(s.anchor[i]));
        }

        for (int i = 0; i < s.num_blocks; i++) {
            hash_combine(s.block_sizes[i]);
            for (int j = 0; j < s.block_sizes[i]; j++) hash_combine(s.blocks[i][j]); 
        }

        return h;
    }
};

bool same_point(
    const Point& a,
    const Point& b,
    double eps = 1e-2
) {
    for (int i = 0; i < 3; i++) {
        if (std::abs(a[i] - b[i]) > eps)
            return false;
    }
    return true;
}

void triangulate_surface(
    FK_Triangulation& fk,
    const Hashtable& d_Ls,
    std::unordered_map<
        Permutahedral_Simplex,
        std::vector<Point>,
        Permutahedral_Simplex_Hash
    >& Ps
) {
    std::cout << "=== TRIANGULATE SURFACE (POINT-DEDUP TEST) ===" << std::endl;

    std::vector<Permutahedral_Simplex> h_simplices(d_Ls.capacity);
    std::vector<Point>                h_coords(d_Ls.capacity);
    std::vector<int>                  h_occupied(d_Ls.capacity);

    cudaMemcpy(
        h_simplices.data(),
        d_Ls.simplices,
        d_Ls.capacity * sizeof(Permutahedral_Simplex),
        cudaMemcpyDeviceToHost
    );

    cudaMemcpy(
        h_coords.data(),
        d_Ls.coordinates,
        d_Ls.capacity * sizeof(Point),
        cudaMemcpyDeviceToHost
    );

    cudaMemcpy(
        h_occupied.data(),
        d_Ls.occupied,
        d_Ls.capacity * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    int input_edges = 0;
    for (int i = 0; i < d_Ls.capacity; i++)
        if (h_occupied[i])
            input_edges++;

    std::cout << "Input edges: " << input_edges << std::endl;

    for (int slot = 0; slot < d_Ls.capacity; slot++) {

        if (!h_occupied[slot])
            continue;

        const Permutahedral_Simplex& edge  = h_simplices[slot];
        const Point&                point = h_coords[slot];

        // Only edges (defensive, but cheap)
        if (edge.num_blocks != 2)
            continue;

        Permutahedral_Simplex d_simplices[MAX_COFACES];
        int num_cofaces = cofaces(edge, d_simplices, fk.amb_dim);

        for (int i = 0; i < num_cofaces; i++) {

            auto& vec = Ps[d_simplices[i]];

            bool found_duplicate = false;
            for (const auto& p : vec) {
                if (same_point(p, point)) {
                    found_duplicate = true;
                    break;
                }
            }

            if (!found_duplicate) {
                vec.push_back(point);
            }
        }
    }
    int num_tri = 0;
    for (const auto& kv : Ps) {
        if (kv.second.size() >= 3)
            num_tri++;
    }

    std::cout << "Number of triangles: " << num_tri << std::endl;
}
