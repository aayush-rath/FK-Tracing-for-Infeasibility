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
    double scale = 1.0;                                                                                 // Lattice scaling

    double Lambda[MAX_D][MAX_D];                                                                        // Rotation matrix of skewing the latticr
    double Lambda_inv[MAX_D][MAX_D];                                                                    // Inverse to deskew
    double b[MAX_D];                                                                                    // Offset translation

    // Constructor for the FK Triangulation
    __host__ 
    FK_Triangulation(uint8_t d) {
        this->amb_dim = d;
        this->scale = 1.0;
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
                cartesian_point[i] += Lambda[i][j] * (point[j] / scale);
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
            Lambda[0][0] =  0.70710678;  Lambda[0][1] = -0.70710678;  Lambda[0][2] =  0.0;
            Lambda[1][0] =  0.40824829;  Lambda[1][1] =  0.40824829;  Lambda[1][2] = -0.81649658;
            Lambda[2][0] =  0.57735027;  Lambda[2][1] =  0.57735027;  Lambda[2][2] =  0.57735027;

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    Lambda_inv[i][j] = Lambda[j][i];
        }

        else if (d == 4) {
            Lambda[0][0] =  0.70710678;  Lambda[0][1] = -0.70710678;  Lambda[0][2] =  0.0;         Lambda[0][3] =  0.0;
            Lambda[1][0] =  0.40824829;  Lambda[1][1] =  0.40824829;  Lambda[1][2] = -0.81649658; Lambda[1][3] =  0.0;
            Lambda[2][0] =  0.28867513;  Lambda[2][1] =  0.28867513;  Lambda[2][2] =  0.28867513; Lambda[2][3] = -0.86602540;
            Lambda[3][0] =  0.5;         Lambda[3][1] =  0.5;         Lambda[3][2] =  0.5;        Lambda[3][3] =  0.5;

            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    Lambda_inv[i][j] = Lambda[j][i];
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

    frac[fk.amb_dim] = inf();
    
    uint8_t idx[MAX_D+1];

    // Set the idx to be the set {0, 1, ..., d}
    for (int i = 0; i <= fk.amb_dim; i++) idx[i] = i;

    // Sort the set according to the ascending order of the fractional part
    for (int i = 1; i <= fk.amb_dim; ++i) {
        int key = idx[i];
        int j = i - 1;
        while (j >= 0 && frac[idx[j]] > frac[key]) {
            idx[j + 1] = idx[j];
            j--;
        }
        idx[j + 1] = key;
    }

    const double eps = 1e-12;
    s.num_blocks = 0;

    // Use the traversal order to set the ordered partition
    for (int i = 0; i <= fk.amb_dim; i++) {
        if (i == 0 || frac[idx[i]] - frac[idx[i-1]] > eps) {
            s.block_sizes[s.num_blocks] = 0;
            s.num_blocks++;
        }

        s.blocks[s.num_blocks - 1][s.block_sizes[s.num_blocks - 1]] = idx[i];
        s.block_sizes[s.num_blocks - 1]++;
    }

    return s;
}