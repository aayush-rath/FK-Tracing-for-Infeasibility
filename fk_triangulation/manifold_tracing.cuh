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
#include <unordered_set>
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

template <typename Function>
__global__
void initialize_kernel(
    const double* seeds,
    int num_seeds,
    FrontierNode* frontier,
    int* frontier_size,
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

        if (!bound_check(fk, p)) continue;

        int idx = atomicAdd(frontier_size, 1);
        frontier[idx].simplex = edges[j];
        frontier[idx].component = tid;
    }
}

template <typename Function>
__global__
void expand_frontier_kernel(
    const FrontierNode* frontier,
    int frontier_size,

    FrontierNode* next_frontier,
    int* next_frontier_size,

    Hashtable frontier_hash,
    Hashtable prev_frontier_hash,

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
            if (!edge_intersection(new_edges[j], function, fk, intersection_point))
                continue;

            Point p;
            #pragma unroll
            for (int d = 0; d < fk.amb_dim; d++)
                p[d] = intersection_point[d];

            if (hash_lookup(frontier_hash, new_edges[j]))
                continue;

            if (hash_lookup(prev_frontier_hash, new_edges[j]))
                continue;

            if (!bound_check(fk, p))
                continue;

            unsigned mask = __activemask();
            int leader = __ffs(mask) - 1;
            int lane = threadIdx.x & 31;

            int warp_offset;

            if (lane == leader)
                warp_offset = atomicAdd(next_frontier_size, __popc(mask));

            warp_offset = __shfl_sync(mask, warp_offset, leader);

            int my_offset = warp_offset + __popc(mask & ((1u << lane) - 1));

            next_frontier[my_offset].simplex = new_edges[j];
            next_frontier[my_offset].component = component;
        }
    }
}

__global__
void build_hash_kernel(
    const FrontierNode* frontier,
    int frontier_size,
    Hashtable table
) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size)
        return;

    const Permutahedral_Simplex& s = frontier[tid].simplex;

    Point dummy;
    #pragma unroll
    for (int i = 0; i < MAX_D; i++)
        dummy[i] = 0.0;

    hash_insert(
        table,
        s,
        dummy,
        frontier[tid].component
    );
}

template <typename Function>
void traceManifold(
    const FK_Triangulation& fk_host,
    const Function& function,
    double* seed_host,
    int* component_array,
    int num_seeds,
    std::unordered_set<
        Permutahedral_Simplex,
        Permutahedral_Simplex_Hash
    >& visited
) {

    FK_Triangulation fk_device = fk_host;

    FrontierNode* d_frontier;
    FrontierNode* d_next_frontier;

    int* d_frontier_size;
    int* d_next_frontier_size;

    int max_frontier_size = 20000000;

    cudaMalloc(&d_frontier, max_frontier_size * sizeof(FrontierNode));
    cudaMalloc(&d_next_frontier, max_frontier_size * sizeof(FrontierNode));

    cudaMalloc(&d_frontier_size, sizeof(int));
    cudaMalloc(&d_next_frontier_size, sizeof(int));

    cudaMemset(d_frontier_size, 0, sizeof(int));

    /* ---------------- GPU HASH TABLES ---------------- */

    Hashtable d_frontier_hash = allocate_device_hash_table(1 * max_frontier_size);
    Hashtable d_prev_frontier_hash = allocate_device_hash_table(1 * max_frontier_size);

    cudaMemset(d_frontier_hash.occupied, 0,
               d_frontier_hash.capacity * sizeof(int));

    cudaMemset(d_prev_frontier_hash.occupied, 0,
               d_prev_frontier_hash.capacity * sizeof(int));

    /* ---------------- HOST BUFFERS ---------------- */

    std::vector<FrontierNode> h_next_frontier(max_frontier_size);

    /* ---------------- SEED INITIALIZATION ---------------- */

    double* d_seeds;
    cudaMalloc(&d_seeds, num_seeds * fk_host.amb_dim * sizeof(double));

    cudaMemcpy(
        d_seeds,
        seed_host,
        num_seeds * fk_host.amb_dim * sizeof(double),
        cudaMemcpyHostToDevice
    );

    int threads = 256;
    int blocks = (num_seeds + threads - 1) / threads;

    initialize_kernel<<<blocks, threads>>>(
        d_seeds,
        num_seeds,
        d_frontier,
        d_frontier_size,
        fk_device,
        function,
        component_array
    );

    cudaDeviceSynchronize();

    int frontier_size_host = 0;

    cudaMemcpy(&frontier_size_host,
               d_frontier_size,
               sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(d_seeds);

    /* ---------------- BUILD INITIAL HASH ---------------- */

    cudaMemset(d_frontier_hash.occupied, 0,
               d_frontier_hash.capacity * sizeof(int));

    build_hash_kernel<<<
        (frontier_size_host + 255) / 256,
        256
    >>>(
        d_frontier,
        frontier_size_host,
        d_frontier_hash
    );

    cudaDeviceSynchronize();

    /* ---------------- BFS LOOP ---------------- */

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
            d_frontier_hash,
            d_prev_frontier_hash,
            fk_device,
            function,
            component_array
        );

        cudaDeviceSynchronize();

        int next_size = 0;

        cudaMemcpy(&next_size,
                   d_next_frontier_size,
                   sizeof(int),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(
            h_next_frontier.data(),
            d_next_frontier,
            next_size * sizeof(FrontierNode),
            cudaMemcpyDeviceToHost
        );

        /* -------- CPU GLOBAL VISITED FILTER -------- */

        std::vector<FrontierNode> filtered;
        filtered.reserve(next_size);

        for (int i = 0; i < next_size; i++) {

            const FrontierNode& node = h_next_frontier[i];

            if (visited.insert(node.simplex).second)
                filtered.push_back(node);
        }

        frontier_size_host = filtered.size();

        cudaMemcpy(
            d_next_frontier,
            filtered.data(),
            frontier_size_host * sizeof(FrontierNode),
            cudaMemcpyHostToDevice
        );

        /* -------- ROTATE HASH TABLES -------- */

        std::swap(d_prev_frontier_hash, d_frontier_hash);

        cudaMemset(
            d_frontier_hash.occupied,
            0,
            d_frontier_hash.capacity * sizeof(int)
        );

        build_hash_kernel<<<
            (frontier_size_host + 255) / 256,
            256
        >>>(
            d_next_frontier,
            frontier_size_host,
            d_frontier_hash
        );

        cudaDeviceSynchronize();

        std::swap(d_frontier, d_next_frontier);

        intersect_count += frontier_size_host;
        iteration++;

        if (iteration % 10 == 0) {
            std::cout
                << "Iteration " << iteration
                << ", Frontier Size: " << frontier_size_host
                << ", Total Intersections: " << intersect_count
                << std::endl;
        }

        if (iteration > 10000) {
            std::cerr << "Warning: Exceeded maximum iterations\n";
            break;
        }
    }

    std::cout << "Intersection Count: " << intersect_count << std::endl;
    std::cout << "Total iterations: " << iteration << std::endl;

    /* ---------------- CLEANUP ---------------- */

    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_frontier_size);
    cudaFree(d_next_frontier_size);

    cudaFree(d_frontier_hash.coordinates);
    cudaFree(d_frontier_hash.occupied);
    cudaFree(d_frontier_hash.simplices);

    cudaFree(d_prev_frontier_hash.coordinates);
    cudaFree(d_prev_frontier_hash.occupied);
    cudaFree(d_prev_frontier_hash.simplices);
}

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
