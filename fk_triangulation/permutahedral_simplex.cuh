#pragma once

/*
Permutahedral Representation to represent any dimensional simplex
in the Freudenthal-Kuhn Triangulation of the ambient space to run on
cuda

Aayush Rath
*/

#include "utils.cuh"

struct Point {
    double coords[MAX_D];

    __host__ __device__ __forceinline__ double& operator[](int i) {return coords[i];}
    __host__ __device__ __forceinline__ double operator[](int i) const {return coords[i];}
};

struct Permutahedral_Simplex {
    int32_t anchor[MAX_D];                                                                              // Minimal Vertex of the simplex representing the location of the simplex
    uint8_t amb_dim;                                                                                    // Ambient space dimension
    uint8_t num_blocks;                                                                                 // Number of ordered partitions in the set. Dimension of the simplex = number of blocks

    uint8_t block_sizes[MAX_D+1];                                                                       // The sizes of each of the partitions
    uint8_t blocks[MAX_D+1][MAX_D+1];                                                                   // The ordered partitions
};

// If two simplex have everything the same then they are equal
__host__ __device__ __forceinline__
bool operator==(
    const Permutahedral_Simplex& a,
    const Permutahedral_Simplex& b
) {
    if (a.amb_dim != b.amb_dim) return false;
    for (int i = 0; i < a.amb_dim; i++) {
        if (a.anchor[i] != b.anchor[i]) return false;
    }

    if (a.num_blocks != b.num_blocks) return false;

    for (int i = 0; i < a.num_blocks; i++) {
        if (a.block_sizes[i] != b.block_sizes[i]) return false;
        for (int j = 0; j < a.block_sizes[i]; j++) {
            if (a.blocks[i][j] != b.blocks[i][j]) return false;
        }
    }

    return true;
}

struct Hashtable {
    Permutahedral_Simplex* simplices;                                                                   // To store all the intersecting simplices
    Point* coordinates;                                                                                 // To store the coordinates of the intersection point on the simplex
    int* occupied;                                                                                      // If a given index is already occupied
    int* component;                                                                                     // the manifold component that the simplex corresponds to
    int capacity;                                                                                       // the size of the hash table
};

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

__device__ __forceinline__
uint64_t hash_permutahedral_simplex(const Permutahedral_Simplex& s) {
    uint64_t h = 1469598103934665603ULL;                                                                // FNV_offset basis

    auto mix = [&](uint64_t v) {                                                                        // This is the implementation of the Fowler-Noll-Vo hash algorithm (FNV-1a)
        h ^= v;     
        h *= 1099511628211ULL;                                                                          // This is an FNV prime
    };

    // Applying the hash on every byte of data of the Permutahedral Simplex struct
    mix(s.amb_dim);
    mix(s.num_blocks);

    for (int i = 0; i < s.amb_dim; i++) mix((uint64_t)s.anchor[i]);
    for (int i = 0; i < s.num_blocks; i++) {
        mix((uint64_t)s.block_sizes[i]);
        for (int j = 0; j < s.block_sizes[i]; j++) {
            mix((uint64_t)s.blocks[i][j]);
        }
    }

    return h;
}


__device__ __forceinline__
int hash_insert(
    Hashtable& h_table,
    const Permutahedral_Simplex& key,
    const Point& value,
    const int component
) {
    uint64_t hash = hash_permutahedral_simplex(key);
    int cap = h_table.capacity;

    for (int probe = 0; probe < cap; probe++) {
        int slot = (hash + probe) % cap;

        if (atomicCAS(&h_table.occupied[slot], 0, 1) == 0) {
            h_table.simplices[slot] = key;
            h_table.coordinates[slot] = value;
            h_table.component[slot] = component;
            return -1;
        }

        if (h_table.simplices[slot] == key) {
            return h_table.component[slot];
        }
    }

    return -2;
}

__device__ __forceinline__
bool hash_lookup(
    const Hashtable& h_table,
    const Permutahedral_Simplex& key
) {
    uint64_t hash = hash_permutahedral_simplex(key);
    int cap = h_table.capacity;

    for (int probe = 0; probe < cap; probe++) {

        int slot = (hash + probe) % cap;
        if (h_table.occupied[slot] == 0)
            return false;

        if (h_table.simplices[slot] == key)
            return true;
    }

    return false;
}

Hashtable allocate_device_hash_table(int capacity) {
    Hashtable table;
    table.capacity = capacity;

    cudaMalloc(&table.simplices,     capacity * sizeof(Permutahedral_Simplex));
    cudaMalloc(&table.coordinates,   capacity * sizeof(Point));
    cudaMalloc(&table.occupied, capacity * sizeof(int));
    cudaMalloc(&table.component, capacity * sizeof(int));

    cudaMemset(table.occupied, 0, capacity * sizeof(int));
    cudaMemset(table.component, 0, capacity * sizeof(int));

    return table;
}


// Returns the k-dimensional faces of an l-dimensional simplex
__host__ __device__ __forceinline__
int faces(
    const Permutahedral_Simplex& s,
    Permutahedral_Simplex *k_faces,
    uint8_t k
) {
    const uint8_t l = s.num_blocks - 1;                                                                     // The dimension l of the input simplex
    if (k > l) return 0;

    uint8_t comb[MAX_D];
    init_combination(comb, k);

    int face_index = 0;

    do {
        Permutahedral_Simplex k_face;
        k_face.amb_dim = s.amb_dim;

        for (int i = 0; i < s.amb_dim; i++) k_face.anchor[i] = s.anchor[i];
        for (int i = 0; i <= k; i++) k_face.block_sizes[i] = 0;

        for (int b = 0; b < comb[0]; b++) {
            for (int j = 0; j < s.block_sizes[b]; j++) {
                int idx = s.blocks[b][j];
                if (idx == s.amb_dim) {
                    for (int d = 0; d < s.amb_dim; ++d)
                        k_face.anchor[d]--;
                } else {
                    k_face.anchor[idx]++;
                }
            }
        }

        k_face.num_blocks = k + 1;

        for (int i = 0; i <= k; i++) {
            if (i < k) {
                k_face.block_sizes[i] = 0;
                for (int j = comb[i]; j < comb[i+1]; j++) {
                    for (int b = 0; b < s.block_sizes[j]; b++) k_face.blocks[i][k_face.block_sizes[i]++] = s.blocks[j][b];
                }
            }

            if (i == k) {
                k_face.block_sizes[k] = 0;
                for (int j = 0; j < comb[0]; j++) {
                    for(int b = 0; b < s.block_sizes[j]; b++) k_face.blocks[k][k_face.block_sizes[k]++] = s.blocks[j][b];
                }
                for (int j = comb[k]; j < l + 1; j++) {
                    for(int b = 0; b < s.block_sizes[j]; b++) k_face.blocks[k][k_face.block_sizes[k]++] = s.blocks[j][b];
                }
            }
        }
        k_faces[face_index++] = k_face;
    } while (next_combination(comb, k, l));

    return face_index;
}

__host__ __device__ __forceinline__
int cofaces(
    const Permutahedral_Simplex& s,
    Permutahedral_Simplex* l_faces,
    uint8_t l
) {
    const uint8_t k = s.num_blocks - 1;
    if (l < k) return 0;

    uint8_t a[MAX_D];
    uint8_t bounds[MAX_D];

    for (int i = 0; i <= k; i++)
        bounds[i] = s.block_sizes[i];

    int out_count = 0;

    if (!walsh_init(k, l, bounds, a))
        return 0;

    do {
        // RGS state for each block
        uint8_t rgs[MAX_D][MAX_D + 1];

        bool ok = true;
        for (int i = 0; i <= k; i++) {
            if (!rgs_init(s.block_sizes[i], a[i] + 1, rgs[i])) {
                ok = false;
                break;
            }
        }
        if (!ok) continue;

        bool done = false;
        while (!done) {
            // For each source block, create its sub-partitions
            uint8_t sub_blocks[MAX_D][MAX_D + 1][MAX_D + 1]; // [block_i][partition_p][elements]
            uint8_t sub_sizes[MAX_D][MAX_D + 1];              // [block_i][partition_p]
            uint8_t num_sub_parts[MAX_D];                     // how many parts each block splits into

            for (int i = 0; i <= k; i++) {
                num_sub_parts[i] = a[i] + 1;
                
                for (int p = 0; p < a[i] + 1; p++) {
                    sub_sizes[i][p] = 0;
                    
                    for (int j = 0; j < s.block_sizes[i]; j++) {
                        if (rgs[i][j] == p) {
                            uint8_t idx = s.blocks[i][j];
                            sub_blocks[i][p][sub_sizes[i][p]++] = idx;
                        }
                    }
                }
            }

            // Now generate all permutations of sub-partitions for each block
            uint8_t perm_indices[MAX_D][MAX_D + 1]; // permutation index for each block's sub-parts
            for (int i = 0; i <= k; i++) {
                perm_init(num_sub_parts[i], perm_indices[i]);
            }

            // Iterate through all combinations of permutations
            bool perm_done = false;
            while (!perm_done) {
                // Build the actual partition using current permutation
                uint8_t tmp_blocks[MAX_D + 1][MAX_D + 1];
                uint8_t tmp_sizes[MAX_D + 1];
                int block_cursor = 0;
                int diag_block = -1;

                for (int i = 0; i <= k; i++) {
                    for (int p = 0; p < num_sub_parts[i]; p++) {
                        int actual_p = perm_indices[i][p]; // use permuted index
                        tmp_sizes[block_cursor] = sub_sizes[i][actual_p];
                        
                        for (int j = 0; j < sub_sizes[i][actual_p]; j++) {
                            uint8_t idx = sub_blocks[i][actual_p][j];
                            tmp_blocks[block_cursor][j] = idx;
                            if (idx == s.amb_dim)
                                diag_block = block_cursor;
                        }
                        block_cursor++;
                    }
                }

                Permutahedral_Simplex out;
                out.amb_dim = s.amb_dim;
                out.num_blocks = l + 1;
                for (int i = 0; i < s.amb_dim; i++) out.anchor[i] = s.anchor[i];

                if (diag_block != l) {

                    for (int b = 0; b < diag_block; b++) {
                        for (int j = 0; j < tmp_sizes[b]; j++) {
                            uint8_t idx = tmp_blocks[b][j];
                            if (idx < s.amb_dim) {
                                out.anchor[idx]++;
                            } else {
                                for (int i = 0; i < s.amb_dim; i++) out.anchor[i]--;
                            }
                        }
                    }

                    for (int j = 0; j < tmp_sizes[diag_block]; j++) {
                        uint8_t idx = tmp_blocks[diag_block][j];
                        if (idx < s.amb_dim) {
                            out.anchor[idx]++;
                        } else {
                            for (int i = 0; i < s.amb_dim; i++) out.anchor[i]--;
                        }
                    }
                }

                int dst = 0;
                for (int b = diag_block + 1; b < l + 1; b++) {
                    out.block_sizes[dst] = tmp_sizes[b];
                    for (int j = 0; j < tmp_sizes[b]; j++)
                        out.blocks[dst][j] = tmp_blocks[b][j];
                    dst++;
                }
                for (int b = 0; b <= diag_block; b++) {
                    out.block_sizes[dst] = tmp_sizes[b];
                    for (int j = 0; j < tmp_sizes[b]; j++)
                            out.blocks[dst][j] = tmp_blocks[b][j];
                        dst++;
                    }


                l_faces[out_count++] = out;
                // Advance to next permutation combination
                bool perm_advanced = false;
                for (int i = k; i >= 0; i--) {
                    if (perm_next(num_sub_parts[i], perm_indices[i])) {
                        // Reset all higher blocks
                        for (int j = i + 1; j <= k; j++) {
                            for (int p = 0; p < num_sub_parts[j]; p++) {
                                perm_indices[j][p] = p;
                            }
                        }
                        perm_advanced = true;
                        break;
                    }
                }
                if (!perm_advanced)
                    perm_done = true;
            }

            // Advance RGS
            bool advanced = false;
            for (int i = k; i >= 0; i--) {
                if (rgs_next(s.block_sizes[i], a[i] + 1, rgs[i])) {
                    for (int j = i + 1; j <= k; j++)
                        rgs_init(s.block_sizes[j], a[j] + 1, rgs[j]);
                    advanced = true;
                    break;
                }
            }
            if (!advanced)
                done = true;
        }

    } while (walsh_next(k, bounds, a));

    return out_count;
}