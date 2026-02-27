#pragma once

#include <iostream>
#include <set>
#include <fstream>

#include <unordered_map>
#include <map>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>

struct TriFloat3 {
    float v0[3];
    float v1[3];
    float v2[3];
};

// Union-Find data structure
class UnionFind {
private:
    std::vector<int> parent;
    std::vector<int> rank;
    
public:
    UnionFind(int n) : parent(n), rank(n, 0) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }
    
    int find(int x) {
        // Path compression
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }
    
    void union_sets(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        
        if (ra == rb) return;
        
        // Union by rank
        if (rank[ra] < rank[rb]) {
            parent[ra] = rb;
        } else if (rank[ra] > rank[rb]) {
            parent[rb] = ra;
        } else {
            parent[rb] = ra;
            rank[ra]++;
        }
    }
    
    std::vector<int> get_labels(int n) {
        std::vector<int> labels(n);
        for (int i = 0; i < n; i++) {
            labels[i] = find(i);
        }
        return labels;
    }
};

struct VertexEqual {
    bool operator()(const std::array<float, 3>& a, const std::array<float, 3>& b) const {
        const float eps = 1e-7f;
        const float pi = 3.14159265358979323846f;
        
        for (int i = 0; i < 3; i++) {
            float diff = std::abs(a[i] - b[i]);
            
            // Check if they're approximately equal
            if (diff < eps) continue;
            
            // Check if they wrap around at pi/-pi boundary
            // Two values wrap if one is ~pi and the other is ~-pi
            bool a_is_pi = std::abs(a[i] - pi) < eps;
            bool a_is_neg_pi = std::abs(a[i] + pi) < eps;
            bool b_is_pi = std::abs(b[i] - pi) < eps;
            bool b_is_neg_pi = std::abs(b[i] + pi) < eps;
            
            if ((a_is_pi && b_is_neg_pi) || (a_is_neg_pi && b_is_pi)) {
                continue;  // These wrap around, so they're equal
            }
            
            return false;  // Not equal
        }
        return true;
    }
};

struct VertexHash {
    std::size_t operator()(const std::array<float, 3>& v) const {
        const float pi = 3.14159265358979323846f;
        const float eps = 1e-7f;
        
        // Normalize pi and -pi to the same value (use pi)
        auto normalize = [&](float val) -> float {
            if (std::abs(val + pi) < eps) {
                return pi;  // Treat -pi as pi
            }
            return val;
        };
        
        // Round to 8 decimal places and hash
        long long x = static_cast<long long>(std::round(normalize(v[0]) * 1e8));
        long long y = static_cast<long long>(std::round(normalize(v[1]) * 1e8));
        long long z = static_cast<long long>(std::round(normalize(v[2]) * 1e8));
        
        // Combine hashes
        std::size_t h1 = std::hash<long long>{}(x);
        std::size_t h2 = std::hash<long long>{}(y);
        std::size_t h3 = std::hash<long long>{}(z);
        
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

// Edge represented as pair of vertices (ordered)
using Vertex = std::array<float, 3>;
using Edge = std::pair<Vertex, Vertex>;

struct EdgeHash {
    std::size_t operator()(const Edge& e) const {
        VertexHash vh;
        return vh(e.first) ^ (vh(e.second) << 1);
    }
};

struct EdgeEqual {
    bool operator()(const Edge& a, const Edge& b) const {
        VertexEqual ve;
        return ve(a.first, b.first) && ve(a.second, b.second);
    }
};

// Round vertex to avoid floating point precision issues
Vertex round_vertex(const float v[3], int decimals = 8) {
    float scale = std::pow(10.0f, decimals);
    return {
        std::round(v[0] * scale) / scale,
        std::round(v[1] * scale) / scale,
        std::round(v[2] * scale) / scale
    };
}

// Create ordered edge (smaller vertex first for consistency)
Edge make_edge(const Vertex& a, const Vertex& b) {
    // Compare vertices lexicographically
    if (a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]) || 
        (a[0] == b[0] && a[1] == b[1] && a[2] < b[2])) {
        return {a, b};
    } else {
        return {b, a};
    }
}

// Get triangles belonging to a specific component
std::vector<TriFloat3> get_component_triangles(
    const std::vector<TriFloat3>& triangles,
    const std::vector<int>& labels,
    int component_id)
{
    std::vector<TriFloat3> component_tris;
    for (size_t i = 0; i < triangles.size(); i++) {
        if (labels[i] == component_id) {
            component_tris.push_back(triangles[i]);
        }
    }
    return component_tris;
}

// Save mesh with component labels
void save_mesh_with_labels_csv(
    const std::string& filename,
    const std::vector<TriFloat3>& triangles,
    const std::vector<int>& labels)
{
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "# Mesh data with component labels\n";
    file << "# Format: v0_x,v0_y,v0_z,v1_x,v1_y,v1_z,v2_x,v2_y,v2_z,component_id\n";
    
    file << std::fixed;
    file.precision(8);
    
    for (size_t i = 0; i < triangles.size(); i++) {
        const auto& tri = triangles[i];
        file << tri.v0[0] << "," << tri.v0[1] << "," << tri.v0[2] << ","
             << tri.v1[0] << "," << tri.v1[1] << "," << tri.v1[2] << ","
             << tri.v2[0] << "," << tri.v2[1] << "," << tri.v2[2] << ","
             << labels[i] << "\n";
    }
    
    file.close();
    std::cout << "Saved " << triangles.size() << " triangles with labels to " << filename << std::endl;
}