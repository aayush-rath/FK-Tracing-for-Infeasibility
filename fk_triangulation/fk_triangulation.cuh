#pragma once 

/*
Freudenthal Kuhn Triangulation
for the ambient space (ℝ^d)

Aayush Rath
*/

#include "utils.cuh"
#include "permutahedral_simplex.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>

struct FK_Triangulation {
    uint8_t amb_dim;                                                                                    // Ambient space dimension
    double scale;                                                                                       // Lattice scaling

    double Lambda[MAX_D][MAX_D];                                                                        // Rotation matrix of skewing the latticr
    double Lambda_inv[MAX_D][MAX_D];                                                                    // Inverse to deskew
    double b[MAX_D];                                                                                    // Offset translation

    // Constructor for the FK Triangulation
    __host__ 
    FK_Triangulation(uint8_t d) {
        this->amb_dim = d;
        for (int i = 0; i < amb_dim; i++) {                                                             // Initialize with an identity rotation matrix
            for (int j = 0; j < amb_dim; j++) {
                if (i == j) Lambda[i][j] = 1.0;
                else Lambda[i][j] = 0.0;
            }
        }

        for (int i = 0; i < amb_dim; i++) {
            for (int j = 0; j < amb_dim; j++) {
                if (i == j) Lambda_inv[i][j] = 1.0;
                else Lambda_inv[i][j] = 0.0;
            }
        }

        for (int i = 0;i < d; i++) b[i] = 0.0;
    }

    __host__ __device__ __forceinline__
    void cartesian_coordinates(const int32_t *point, double *cartesian_point) const {
        for (int i = 0; i < amb_dim; i++) {
            cartesian_point[i] = 0.0;
            for (int j = 0; j < amb_dim; j++) {
                cartesian_point[i] += Lambda[i][j] * (double(point[j]) / scale);
            }
            cartesian_point[i] += b[i];
        } 
    }
};

struct C_Triangulation : public FK_Triangulation {
    __host__
    C_Triangulation(uint8_t d)
        : FK_Triangulation(d)
    {
        if (d == 3) {
            Lambda[0][0] = -0.3826834323650871; Lambda[0][1] =  0.3826834323650911; Lambda[0][2] =  0.9238795325112830;
            Lambda[1][0] = -0.7071067811865475; Lambda[1][1] = -0.7071067811865471; Lambda[1][2] =  0.7071067811865479;
            Lambda[2][0] = -0.9238795325112866; Lambda[2][1] =  0.9238795325112868; Lambda[2][2] = -0.3826834323650898;

            Lambda_inv[0][0] =  0.2705980500730996; Lambda_inv[0][1] = -0.7071067811865471; Lambda_inv[0][2] = -0.6532814824381884;
            Lambda_inv[1][0] =  0.6532814824381901; Lambda_inv[1][1] = -0.7071067811865459; Lambda_inv[1][2] =  0.2705980500730983;
            Lambda_inv[2][0] =  0.9238795325112886; Lambda_inv[2][1] =  0.0;                Lambda_inv[2][2] = -0.3826834323650899;

        }

        if (d == 4) {
            Lambda[0][0] =  0.5257311121191278; Lambda[0][1] =  -1.31183e-16;       Lambda[0][2] = -0.5257311121191275; Lambda[0][3] = -0.8506508083520354;
            Lambda[1][0] = -0.2763932022500212; Lambda[1][1] = -0.8944271909999156; Lambda[1][2] = -0.2763932022500208; Lambda[1][3] =  0.7236067977499788;
            Lambda[2][0] = -0.8506508083520403; Lambda[2][1] =  9.44412e-16;        Lambda[2][2] = 0.8506508083520395; Lambda[2][3] = -0.5257311121191329;
            Lambda[3][0] =  0.7236067977499785; Lambda[3][1] = -0.8944271909999157; Lambda[3][2] = 0.7236067977499795; Lambda[3][3] = -0.2763932022500213;

            Lambda_inv[0][0] = -0.1624598481164546; Lambda_inv[0][1] = -0.4999999999999999; Lambda_inv[0][2] = -0.6881909602355865; Lambda_inv[0][3] =  0.4999999999999998;
            Lambda_inv[1][0] = -0.4253254041760228; Lambda_inv[1][1] = -0.8090169943749472; Lambda_inv[1][2] = -0.2628655560595652; Lambda_inv[1][3] = -0.3090169943749473;
            Lambda_inv[2][0] = -0.6881909602355911; Lambda_inv[2][1] = -0.4999999999999992; Lambda_inv[2][2] =  0.1624598481164553; Lambda_inv[2][3] =  0.5000000000000006;
            Lambda_inv[3][0] = -0.8506508083520457; Lambda_inv[3][1] =  1.60803e-16;        Lambda_inv[3][2] = -0.5257311121191331; Lambda_inv[3][3] = -5.91151e-16;
        }

        if (d == 5) {
            Lambda[0][0] = -0.5773502692;   Lambda[0][1] = -0.2113248654;   Lambda[0][2] = 0.2113248654;    Lambda[0][3] = 0.5773502692;    Lambda[0][4] = 0.7886751346;	
            Lambda[1][0] = -0.0000000000;   Lambda[1][1] = 0.7071067812;    Lambda[1][2] = 0.7071067812;    Lambda[1][3] = 0.0000000000;    Lambda[1][4] = -0.7071067812;
            Lambda[2][0] = -0.5773502692;   Lambda[2][1] = -0.5773502692;   Lambda[2][2] = 0.5773502692;    Lambda[2][3] = 0.5773502692;    Lambda[2][4] = -0.5773502692;	
            Lambda[3][0] = -0.8164965809;   Lambda[3][1] = 0.4082482905;    Lambda[3][2] = 0.4082482905;    Lambda[3][3] = -0.8164965809;   Lambda[3][4] = 0.4082482905;	
            Lambda[4][0] = 0.5773502692;    Lambda[4][1] = -0.7886751346;   Lambda[4][2] = 0.7886751346;    Lambda[4][3] = -0.5773502692;   Lambda[4][4] = 0.2113248654;

            Lambda_inv[0][0] = 0.105662	; Lambda_inv[0][1] = 0.353553;       Lambda_inv[0][2] = -0.57735;     Lambda_inv[0][3] = -0.612372;	    Lambda_inv[0][4] = 0.394338;	
            Lambda_inv[1][0] = 0.288675	; Lambda_inv[1][1] = 0.707107;       Lambda_inv[1][2] = -0.57735;     Lambda_inv[1][3] = 3.06088e-16;	Lambda_inv[1][4] = -0.288675;	
            Lambda_inv[2][0] = 0.5	    ; Lambda_inv[2][1] = 0.707107;       Lambda_inv[2][2] = 1.45932e-16;  Lambda_inv[2][3] = -2.35514e-16;	Lambda_inv[2][4] = 0.5;
            Lambda_inv[3][0] = 0.683013	; Lambda_inv[3][1] = 0.353553;       Lambda_inv[3][2] = 3.25096e-16;  Lambda_inv[3][3] = -0.612372;	    Lambda_inv[3][4] = -0.183013;	
            Lambda_inv[4][0] = 0.788675	; Lambda_inv[4][1] = -8.75605e-17;   Lambda_inv[4][2] = -0.57735;     Lambda_inv[4][3] = -2.18901e-16;	Lambda_inv[4][4] = 0.211325;
        }

        for (int i = 0; i < d; i++)
            b[i] = 0.0;
    }
};

__host__ __device__ __forceinline__
double inf() {
#ifdef __CUDA_ARCH__
    return CUDART_INF;
#else
    return std::numeric_limits<double>::infinity();
#endif
}

__host__ __device__ __forceinline__
Permutahedral_Simplex locate_simplex(
    const FK_Triangulation& fk,
    const double *point
) {
    Permutahedral_Simplex s;
    s.amb_dim = fk.amb_dim;

    double x[MAX_D+1];                                                                                // The point transformed in the FK coordinate system
    double frac[MAX_D+1];                                                                             // Fractional part of the transformed point

    for (int i = 0; i < fk.amb_dim; i++) {
        double v = point[i] - fk.b[i];
        x[i] = 0.0;
        for (int j = 0; j < fk.amb_dim; j++) x[i] += fk.Lambda_inv[i][j] * v;
        x[i] *= fk.scale;
    }

    for (int i = 0; i < fk.amb_dim; i++) {
        int yi = (int)floor(x[i]);                                                                     // The interger points for the simplex anchor 
        s.anchor[i] = yi;
        frac[i] = x[i] - yi;
    }

    // frac[fk.amb_dim] = inf();
    frac[fk.amb_dim] = 0.0;
    
    uint8_t idx[MAX_D+1];

    // Set the idx to be the set {0, 1, ..., d}
    for (int i = 0; i <= fk.amb_dim; i++) idx[i] = i;

    // Sort the set according to the ascending order of the fractional part
    // for (int i = 1; i <= fk.amb_dim; ++i) {
    //     int key = idx[i];
    //     int j = i - 1;
    //     while (j >= 0 && frac[idx[j]] > frac[key]) {
    //         idx[j + 1] = idx[j];
    //         j--;
    //     }
    //     idx[j + 1] = key;
    // }

    for (int i = 1; i <= fk.amb_dim; ++i) {
        int key = idx[i];
        int j = i - 1;
        while (j >= 0 && frac[idx[j]] < frac[key]) {  // descending: move smaller elements right
            idx[j + 1] = idx[j];
            j--;
        }
        idx[j + 1] = key;
    }

    const double eps = 1e-12;
    s.num_blocks = 0;

    // Use the traversal order to set the ordered partition
    for (int i = 0; i <= fk.amb_dim; i++) {
        if (i == 0 || frac[idx[i-1]] - frac[idx[i]] > eps) {
            s.block_sizes[s.num_blocks] = 0;
            s.num_blocks++;
        }

        s.blocks[s.num_blocks - 1][s.block_sizes[s.num_blocks - 1]] = idx[i];
        s.block_sizes[s.num_blocks - 1]++;
    }

    return s;
}

__host__ __device__ __forceinline__
Permutahedral_Simplex locate_simplex(
    const C_Triangulation& fk,
    const double *point
) {
    Permutahedral_Simplex s;
    s.amb_dim = fk.amb_dim;

    double x[MAX_D+1];     // FK coordinates
    double frac[MAX_D+1];  // fractional parts

    // --- NEW: build RHS = (point - b) ---
    double rhs[MAX_D];
    for (int i = 0; i < fk.amb_dim; i++) {
        rhs[i] = point[i] - fk.b[i];
    }

    // --- NEW: solve Lambda * x = rhs ---
    bool ok = solve_linear_system(fk.amb_dim, fk.Lambda, rhs, x);

    #ifdef __CUDA_ARCH__
        if (!ok) {
            printf("Solve failed!\n");
        }
    #endif

    // --- Apply scaling ---
    for (int i = 0; i < fk.amb_dim; i++) {
        x[i] *= fk.scale;
    }

    // --- Compute anchor + fractional part ---
    for (int i = 0; i < fk.amb_dim; i++) {
        int yi = (int)floor(x[i]);
        s.anchor[i] = yi;
        frac[i] = x[i] - yi;
    }

    frac[fk.amb_dim] = 0.0;

    uint8_t idx[MAX_D+1];

    // Initialize indices {0, ..., d}
    for (int i = 0; i <= fk.amb_dim; i++) idx[i] = i;

    // Sort indices by descending fractional value
    for (int i = 1; i <= fk.amb_dim; ++i) {
        int key = idx[i];
        int j = i - 1;
        while (j >= 0 && frac[idx[j]] < frac[key]) {
            idx[j + 1] = idx[j];
            j--;
        }
        idx[j + 1] = key;
    }

    const double eps = 1e-12;
    s.num_blocks = 0;

    // Build ordered partition
    for (int i = 0; i <= fk.amb_dim; i++) {
        if (i == 0 || frac[idx[i-1]] - frac[idx[i]] > eps) {
            s.block_sizes[s.num_blocks] = 0;
            s.num_blocks++;
        }

        s.blocks[s.num_blocks - 1][s.block_sizes[s.num_blocks - 1]] = idx[i];
        s.block_sizes[s.num_blocks - 1]++;
    }

    return s;
}