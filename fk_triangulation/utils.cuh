#pragma once

/*
Utililty functions to help implement 
the Permutahedral Simplex implementatiton

Aayush Rath
*/


#include <cstdint>
#include <iostream>
#include <algorithm>
#include <functional>
#include <cmath>

constexpr int MAX_D = 6;
constexpr int MAX_FACES = 256;
constexpr int MAX_COFACES = 256;
constexpr int MAX_NUM_SEEDS = 60;

// Get the value of C(n, r) =  number of combinations
int binomial (int n, int r) {
    if (r < 0 || r > n) return 0;
    if (r > n) r = n - r;

    int res = 1;
    for (int i = 0; i < r; i++) res =  res * (n-i) / (i + 1);

    return res;
}

// Initialize the combination with 0, 1, 2, ... k
__host__ __device__ __forceinline__
void init_combination(uint8_t *comb, uint8_t k) {
    for (int i = 0; i <= k ; i++) comb[i] = i;
}

// Get the next combination from the current combination 
__host__ __device__ __forceinline__
bool next_combination(uint8_t *comb, uint8_t k, int l) {
    for (int i = k; i >=0; i--) {
        if (comb[i] < l - (k - i)) {                                                                    // Check if the max possible value is reached
            comb[i]++;                                                                                  // Increment by 1

            for (int j = i+1; j <= k; j++) comb[j] = comb[j-1] + 1;                                     // Reset all the elements right to the current element to the lowest possible value
            return true;
        }
    }
    return false;
}

// Initialize the permutation with 0, 1, 2, ... n-1 (Same as init_combination xD)
__host__ __device__ __forceinline__
void perm_init(uint8_t n, uint8_t *perm) {
    for (uint8_t i = 0; i < n; i++) perm[i] = i;
}

__host__ __device__ __forceinline__
bool perm_next(uint8_t n, uint8_t *perm) {
    if (n <= 1) return false;

    int i = n - 2;
    while (i >= 0 && perm[i] >= perm[i + 1]) i--;

    if (i < 0) return false;

    int j = n - 1;
    while (perm[j] <= perm[i]) j--;

    uint8_t tmp = perm[i];
    perm[i] = perm[j];
    perm[j] = tmp;

    int left = i + 1, right = n - 1;
    while (left < right) {
        tmp = perm[left];
        perm[left] = perm[right];
        perm[right] = tmp;
        left++;
        right--;
    }

    return true;
}

// Initialize the interger composition
__host__ __device__ __forceinline__
bool walsh_init(uint8_t k, uint8_t l, const uint8_t *bounds, uint8_t* a) {
    uint8_t remaining = l - k;

    for (uint8_t i = 0; i <= k; i++) {
        uint8_t v = remaining;
        if (v > bounds[i]-1) v = bounds[i] - 1;                                                         // Set the values to their bounds
        a[i] = v;                                                                                       // If no more remains the set the value to be the remaining value
        remaining -= v;
    }

    return (remaining == 0);
}

// Give the next integer composition
__host__ __device__ __forceinline__
bool walsh_next(uint8_t k, const uint8_t *bounds, uint8_t *a) {
    for (int i = 0; i < k; i++) {
        if (a[i] == 0) continue;                                                                        // Find the leftmost value that can give something

        for (int j = i + 1; j <= k; j++) {                                                              // Find the next index on right that can give something
            if (a[j] < bounds[j] - 1) {
                a[i]--;
                a[j]++;                                                                                 // Take 1 from a[i] and give to a[j]

                int s = 0;
                for (int t = 0; t < j; t++) {
                    s += a[t];
                    a[t] = 0;
                }                                                                                       // Collect all of the stuff from the left side including the current i-th element

                for (int t = 0; t < j && s > 0; t++) {                                                  // Redistribute the collected stuff including the current element i-th element
                    int v = s;
                    if (v > bounds[t] - 1) v = bounds[t] - 1;
                    a[t] = (uint8_t)v;
                    s -= v;
                }

                return true;
            }
        }
    }

    return false;
}

// Initialize the lexicographically smallest RGS of length omega
// using exactly a blocks
__host__ __device__ __forceinline__
bool rgs_init(uint8_t omega, uint8_t a, uint8_t* rgs) {
    if (a == 0 || a > omega) return false;

    // First omega-(a-1) elements are 0,
    // then 1,2,...,a-1
    uint8_t k = omega - (a - 1);

    for (uint8_t i = 0; i < k; ++i)
        rgs[i] = 0;

    for (uint8_t i = k; i < omega; ++i)
        rgs[i] = i - k + 1;

    return true;
}

__host__ __device__ __forceinline__
bool rgs_next(uint8_t omega, uint8_t a, uint8_t* rgs) {
    for (int i = omega - 1; i >= 1; --i) {

        // Compute prefix maximum
        uint8_t prefix_max = 0;
        for (int j = 0; j < i; ++j)
            prefix_max = std::max(prefix_max, rgs[j]);

        uint8_t limit = std::min<uint8_t>(prefix_max + 1, a - 1);

        if (rgs[i] < limit) {
            rgs[i]++;

            // Reset suffix minimally
            for (uint8_t j = i + 1; j < omega; ++j)
                rgs[j] = 0;

            // Check if we can still reach exactly a blocks
            uint8_t used_max = 0;
            for (uint8_t j = 0; j < omega; ++j)
                used_max = std::max(used_max, rgs[j]);

            uint8_t needed = (a - 1) - used_max;
            uint8_t remaining = omega - (i + 1);

            if (needed <= remaining) {
                // Force introduction of missing labels
                for (uint8_t j = omega - needed; j < omega; ++j)
                    rgs[j] = used_max + (j - (omega - needed)) + 1;

                return true;
            }
        }
    }
    return false;
}
