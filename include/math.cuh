#pragma once
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>


class vec3 {
public:
    __host__ __device__ vec3() = default;
    __host__ __device__ vec3(double vec[3]) : e0(vec[0]), e1(vec[1]), e2(vec[2]) {}
    __host__ __device__ vec3(double v1, double v2, double v3) : e0(v1), e1(v2), e2(v3) {};
    __host__ __device__ vec3(const vec3& other) : e0(other.e0), e1(other.e1), e2(other.e2) {}

    __host__ __device__ double x() const { return e0; }
    __host__ __device__ double y() const { return e1; }
    __host__ __device__ double z() const { return e2; }

    __host__ __device__ double operator[](int i) const { return i == 0 ? e0 : i == 1 ? e1 : e2; }

    __host__ __device__ double norm2() const {return e0 * e0 + e1 * e1 + e2 * e2; }
    __host__ __device__ double norm() const {return sqrt(norm2()); }

    __host__ __device__ vec3 operator+=(const vec3& u) {
        e0 += u.x(); e1 += u.y(); e2 += u.z();
        return *this;
    }

    __host__ __device__ vec3 operator-=(const vec3& u) {
        e0 -= u.x(); e1 -= u.y(); e2 -= u.z();
        return *this;
    }

    __host__ __device__ vec3& operator=(const vec3& other) {
        e0 = other.e0;
        e1 = other.e1;
        e2 = other.e2;
        return *this;
    }


private:
    double e0, e1, e2;
};

class quat4 {
public:
    __host__ __device__ quat4() = default;
    __host__ __device__ quat4(double w, double x, double y, double z) : ew(w), ex(x), ey(y), ez(z) {}
    __host__ __device__ quat4(const quat4& other) : ew(other.ew), ex(other.ex), ey(other.ey), ez(other.ez) {}

    __host__ __device__ double w() const { return ew; }
    __host__ __device__ double x() const { return ex; }
    __host__ __device__ double y() const { return ey; }
    __host__ __device__ double z() const { return ez; }

    __host__ __device__ double& w() { return ew; }
    __host__ __device__ double& x() { return ex; }
    __host__ __device__ double& y() { return ey; }
    __host__ __device__ double& z() { return ez; }

    __host__ __device__ double norm2() const { return ew * ew + ex * ex + ey * ey + ez * ez;}

    __host__ __device__ quat4 conjugate() const { return quat4(ew, -ex, -ey, -ez); }
    __host__ __device__ quat4 inverse() const {
        double mag = norm2();
        quat4 q_c = conjugate();
        return quat4(q_c.w() / mag, q_c.x() / mag, q_c.y() / mag, q_c.z() / mag);
    }

    __host__ __device__ quat4& operator=(const quat4& other) {
        ew = other.ew;
        ex = other.ex;
        ey = other.ey;
        ez = other.ez;
        return *this;
    }

private:
    double ew, ex, ey, ez;
};

__host__ __device__ __forceinline__
double dot (const vec3& v, const vec3& u) {
    return v.x() * u.x() + v.y() * u.y() + v.z() * u.z();
}

__host__ __device__ __forceinline__
vec3 cross(const vec3& v, const vec3& u) {
    return vec3(v.y() * u.z() - v.z() * u.y(),
                v.z() * u.x() - v.x() * u.z(),
                v.x() * u.y() - v.y() * u.x());
}

__host__ __device__ __forceinline__
double norm2(const vec3& u, const vec3& v) {
    return (u.x() - v.x()) * (u.x() - v.x()) + (u.y() - v.y()) * (u.y() - v.y()) + (u.z() - v.z()) * (u.z() - v.z());
}

__host__ __device__ __forceinline__
double norm(const vec3& u, const vec3& v) {
    return sqrt(norm2(u, v));
}

__host__  __device__ __forceinline__
vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}

__host__ __device__ __forceinline__
vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.x() - v.x(), u.y() - v.y(), u.z() - v.z());
}

__host__ __device__ __forceinline__
vec3 operator*(double t, const vec3& u) {
    return vec3(t * u.x(), t * u.y(), t * u.z());
}

__host__ __device__ __forceinline__
vec3 operator*(const vec3& u, double t) {
    return t * u;
}

__host__ __device__ __forceinline__
vec3 operator/(const vec3& u, double t) {
    return (1/t) * u;
}

__host__ __device__ __forceinline__
vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
}

__host__ __device__ __forceinline__
quat4 operator*(const quat4& q1, const quat4& q2) {
    vec3 q1_v(q1.x(), q1.y(), q1.z());
    vec3 q2_v(q2.x(), q2.y(), q2.z());

    vec3 prod_v;
    quat4 prod;

    prod_v = cross(q1_v, q2_v) + q1.w() * q2_v + q2.w() * q1_v;
    prod.w() = q1.w() * q2.w() - dot(q1_v, q2_v);
    prod.x() = prod_v.x();
    prod.y() = prod_v.y();
    prod.z() = prod_v.z();

    return prod;
}

__host__ __device__ __forceinline__
vec3 rotate(const vec3& v, const quat4& q) {
    vec3 qvec(q.x(), q.y(), q.z());
    double qw = q.w();
    
    vec3 uv = cross(qvec, v);
    vec3 uuv = cross(qvec, uv);
    
    return v + ((uv * qw) + uuv) * 2.0;
}

__host__ __device__ __forceinline__
quat4 euler_to_quat(double roll, double pitch, double yaw) {
    float cy = cos(yaw * 0.5);
    float sy = sin(yaw * 0.5);
    float cp = cos(pitch * 0.5);
    float sp = sin(pitch * 0.5);
    float cr = cos(roll * 0.5);
    float sr = sin(roll * 0.5);
    
    return quat4(
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    );
}

__host__ __device__ __forceinline__
void quat_to_mat(quat4 quat, float* mat) {
    mat[0]  = 1 - 2 * quat.y() * quat.y() - 2 * quat.z() * quat.z();
    mat[1]  = 2 * quat.x() * quat.y() + 2 * quat.w() * quat.z();
    mat[2]  = 2 * quat.x() * quat.z() - 2 * quat.w() * quat.y();
    mat[3]  = 0;
    mat[4]  = 2 * quat.x() * quat.y() - 2 * quat.w() * quat.z();
    mat[5]  = 1 - 2 * quat.x() * quat.x() - 2 * quat.z() * quat.z();
    mat[6]  = 2 * quat.y() * quat.z() + 2 * quat.w() * quat.x();
    mat[7]  = 0;
    mat[8]  = 2 * quat.x() * quat.z() + 2 * quat.w() * quat.y();
    mat[9]  = 2 * quat.y() * quat.z() - 2 * quat.w() * quat.x();
    mat[10] = 1 - 2 * quat.x() * quat.x() - 2 * quat.y() * quat.y();
    mat[11] = 0;
    mat[12] = 0;
    mat[13] = 0;
    mat[14] = 0;
    mat[15] = 1;
}

__host__ __device__ __forceinline__
void mat_to_quat(double* mat, quat4& quat) {
    quat.w() = sqrt(1 + mat[0] + mat[5] + mat[10]) / 2;
    quat.x() = (mat[6] - mat[9]) / (4 * quat.w());
    quat.y() = (mat[8] - mat[2]) / (4 * quat.w());
    quat.z() = (mat[1] - mat[4]) / (4 * quat.w());
}

__host__ __device__ __forceinline__
quat4 quat_from_axis_angle(const vec3& axis, double angle) {
    vec3 unit_axis = axis / axis.norm();
    return quat4(cos(angle/2), sin(angle/2) * unit_axis.x(), sin(angle/2) * unit_axis.y(), sin(angle/2) * unit_axis.z());
}
