#include "sdf.cuh"

__global__
void eval_sdf_kernel(
    RobotSDF sdf,
    const double* q,
    double* out
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = sdf(q);
    }
}

// __global__ void ray_intersection_kernel(
//     RobotSDF sdf,
//     const double* d_start,
//     const double* d_goal,
//     const double* d_main_dir,
//     double max_dist,
//     int num_rays,
//     int samples,
//     int max_bisection_iters,
//     double tol,
//     double* d_out_intersections,
//     int* d_out_count,
//     int max_outputs,
//     int dof
// ) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= num_rays * 2) return;

//     int origin_type = tid % 2; 
//     const double* origin = (origin_type == 0) ? d_start : d_goal;

//     curandState localState;
//     curand_init(1337, tid, 0, &localState);

//     double ray_dir[6];
//     double projection = 0;
//     while (projection <= 0.01) {
//         double norm = 0;
//         for (int i = 0; i < dof; i++) {
//             ray_dir[i] = curand_normal_double(&localState);
//             norm += ray_dir[i] * ray_dir[i];
//         }
//         norm = sqrt(norm);
//         projection = 0;
//         for (int i = 0; i < dof; i++) {
//             ray_dir[i] /= norm;
//             projection += ray_dir[i] * d_main_dir[i];
//         }
//     }

//     auto is_outside_bounds = [&](const double* q) {
//         for (int i = 0; i < dof; i++) {
//             if (q[i] > 3.14159265358979323846 || q[i] < -3.14159265358979323846) 
//                 return true;
//         }
//         return false;
//     };

//     auto point_at = [&](double t, double* q_out) {
//         for (int i = 0; i < dof; i++) q_out[i] = origin[i] + t * ray_dir[i];
//     };

//     double q_prev[6], q_curr[6], q_m[6];
//     point_at(0, q_prev);
    
//     if (is_outside_bounds(q_prev)) return;
//     double f_prev = sdf(q_prev);

//     for (int s = 1; s <= samples; s++) {
//         double t_curr = (double(s) / samples) * max_dist;
//         point_at(t_curr, q_curr);

//         if (is_outside_bounds(q_curr)) break;

//         double f_curr = sdf(q_curr);

//         if (f_prev * f_curr <= 0.0) {
//             double ta = (double(s - 1) / samples) * max_dist;
//             double tb = t_curr;
//             double fa = f_prev;

//             for (int k = 0; k < max_bisection_iters; k++) {
//                 double tm = 0.5 * (ta + tb);
//                 point_at(tm, q_m);
//                 double fm = sdf(q_m);

//                 if (abs(fm) < tol || abs(tb - ta) < 1e-8) {
//                     int idx = atomicAdd(d_out_count, 1);
//                     if (idx < max_outputs) {
//                         for (int d = 0; d < dof; d++) 
//                             d_out_intersections[idx * dof + d] = q_m[d];
//                     }
//                     break;
//                 }
//                 if (fa * fm > 0.0) { ta = tm; fa = fm; }
//                 else { tb = tm; }
//             }
//             f_curr = sdf(q_curr); 
//         }
//         f_prev = f_curr;
//     }
// }


__global__ void ray_intersection_kernel(
    RobotSDF sdf,
    const double* d_start,
    const double* d_goal,
    const double* d_main_dir,
    double max_dist,
    int num_rays,
    int samples,
    int max_bisection_iters,
    double tol,
    double* d_out_intersections,
    int* d_out_count,
    int max_outputs,
    int dof
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rays * 2) return;
    int origin_type = tid % 2;
    const double* origin = (origin_type == 0) ? d_start : d_goal;

    curandState localState;
    curand_init(1337, tid, 0, &localState);

    double sg[6];       // g - s
    double sg_norm_sq = 0.0;
    for (int i = 0; i < dof; i++) {
        sg[i] = d_goal[i] - d_start[i];
        sg_norm_sq += sg[i] * sg[i];
    }

    double ray_dir[6];
    double projection = 0;
    while (projection <= 0.01) {
        double norm = 0;
        for (int i = 0; i < dof; i++) {
            ray_dir[i] = curand_normal_double(&localState);
            norm += ray_dir[i] * ray_dir[i];
        }
        norm = sqrt(norm);
        projection = 0;
        for (int i = 0; i < dof; i++) {
            ray_dir[i] /= norm;
            projection += ray_dir[i] * d_main_dir[i];
        }
    }

    double d_dot_sg = 0.0;
    for (int i = 0; i < dof; i++)
        d_dot_sg += ray_dir[i] * sg[i];

    double t_plane_min = 0.0;
    double t_plane_max = max_dist;

    const double parallel_eps = 1e-10;
    if (fabs(d_dot_sg) > parallel_eps) {
        double num_s = 0.0;
        for (int i = 0; i < dof; i++)
            num_s += (d_start[i] - origin[i]) * sg[i];
        double t_plane_s = num_s / d_dot_sg;

        double num_g = 0.0;
        for (int i = 0; i < dof; i++)
            num_g += (d_goal[i] - origin[i]) * sg[i];
        double t_plane_g = num_g / d_dot_sg;

        double t_lo = fmin(t_plane_s, t_plane_g);
        double t_hi = fmax(t_plane_s, t_plane_g);

        t_plane_min = fmax(0.0, t_lo);
        t_plane_max = fmin(max_dist, t_hi);

        if (t_plane_min >= t_plane_max) return;
    }

    auto is_outside_bounds = [&](const double* q) {
        for (int i = 0; i < dof; i++) {
            if (q[i] > 3.14159265358979323846 || q[i] < -3.14159265358979323846)
                return true;
        }
        return false;
    };

    auto point_at = [&](double t, double* q_out) {
        for (int i = 0; i < dof; i++) q_out[i] = origin[i] + t * ray_dir[i];
    };

    double q_prev[6], q_curr[6], q_m[6];
    point_at(t_plane_min, q_prev);
    if (is_outside_bounds(q_prev)) return;
    double f_prev = sdf(q_prev);

    for (int s = 1; s <= samples; s++) {
        // Sample within [t_plane_min, t_plane_max] instead of [0, max_dist]
        double t_curr = t_plane_min + (double(s) / samples) * (t_plane_max - t_plane_min);
        point_at(t_curr, q_curr);
        if (is_outside_bounds(q_curr)) break;
        double f_curr = sdf(q_curr);

        if (f_prev * f_curr <= 0.0) {
            double ta = t_plane_min + (double(s - 1) / samples) * (t_plane_max - t_plane_min);
            double tb = t_curr;
            double fa = f_prev;

            for (int k = 0; k < max_bisection_iters; k++) {
                double tm = 0.5 * (ta + tb);
                point_at(tm, q_m);
                double fm = sdf(q_m);

                if (abs(fm) < tol || abs(tb - ta) < 1e-8) {
                    int idx = atomicAdd(d_out_count, 1);
                    if (idx < max_outputs) {
                        for (int d = 0; d < dof; d++)
                            d_out_intersections[idx * dof + d] = q_m[d];
                    }
                    break;
                }

                if (fa * fm > 0.0) { ta = tm; fa = fm; }
                else { tb = tm; }
            }
            f_curr = sdf(q_curr);
        }
        f_prev = f_curr;
    }
}

bool RobotSDF::get_line_intersections(
    const double* start_point,
    const double* goal_point,
    std::vector<double>& intersections,
    int samples,
    int max_bisection_iters,
    double tol
) const
{
    intersections.clear();

    double* d_q;
    double* d_val;
    cudaMalloc(&d_q, sizeof(double) * dof);
    cudaMalloc(&d_val, sizeof(double));

    auto eval = [&](const std::vector<double>& q) {
        cudaMemcpy(d_q, q.data(), sizeof(double) * dof, cudaMemcpyHostToDevice);
        eval_sdf_kernel<<<1,1>>>(*this, d_q, d_val);
        cudaDeviceSynchronize();
        double v;
        cudaMemcpy(&v, d_val, sizeof(double), cudaMemcpyDeviceToHost);
        return v;
    };

    std::vector<double> dir(dof);
    for (int i = 0; i < dof; i++)
        dir[i] = goal_point[i] - start_point[i];

    auto point_at = [&](double t) {
        std::vector<double> q(dof);
        for (int i = 0; i < dof; i++)
            q[i] = start_point[i] + t * dir[i];
        return q;
    };

    double t_prev = 0.0;
    std::vector<double> q_prev = point_at(t_prev);
    double f_prev = eval(q_prev);

    for (int s = 1; s <= samples; s++) {
        double t_curr = double(s) / samples;
        std::vector<double> q_curr = point_at(t_curr);
        double f_curr = eval(q_curr);

        if (f_prev * f_curr <= 0.0) {
            double ta = t_prev;
            double tb = t_curr;

            for (int k = 0; k < max_bisection_iters; k++) {
                double tm = 0.5 * (ta + tb);
                std::vector<double> qm = point_at(tm);
                double fm = eval(qm);

                if (std::abs(fm) < tol) {
                    for (auto& q : qm) intersections.push_back(q);
                    break;
                }

                if (f_prev * fm > 0.0) {
                    ta = tm;
                    f_prev = fm;
                } else {
                    tb = tm;
                }

                if (std::abs(tb - ta) < 1e-8) {
                    for (auto& p: point_at(0.5 * (ta + tb))) intersections.push_back(p);
                    break;
                }
            }
        }

        t_prev = t_curr;
        q_prev = q_curr;
        f_prev = f_curr;
    }

    cudaFree(d_q);
    cudaFree(d_val);

    return !intersections.empty();
}

bool RobotSDF::get_line_rays(
    const double* start_point,
    const double* goal_point,
    std::vector<double>& intersections,
    int num_rays,
    int samples,
    int max_bisection_iters,
    double tol
) const {
    get_line_intersections(start_point, goal_point, intersections, samples, max_bisection_iters, tol);

    std::vector<double> main_dir(dof);
    double main_len = 0;
    for (int i = 0; i < dof; i++) {
        main_dir[i] = goal_point[i] - start_point[i];
        main_len += main_dir[i] * main_dir[i];
    }
    main_len = sqrt(main_len);
    for (int i = 0; i < dof; i++) main_dir[i] /= (main_len + 1e-12);

    double *d_start, *d_goal, *d_main_dir, *d_out_intersections;
    int *d_out_count;
    int max_outputs = num_rays * 20; 

    cudaMalloc(&d_start, dof * sizeof(double));
    cudaMalloc(&d_goal, dof * sizeof(double));
    cudaMalloc(&d_main_dir, dof * sizeof(double));
    cudaMalloc(&d_out_intersections, max_outputs * dof * sizeof(double));
    cudaMalloc(&d_out_count, sizeof(int));

    cudaMemcpy(d_start, start_point, dof * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_goal, goal_point, dof * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_main_dir, main_dir.data(), dof * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_out_count, 0, sizeof(int));

    int total_threads = num_rays * 2;
    int blocks = (total_threads + 255) / 256;
    
    double max_dist = 10.0; 

    ray_intersection_kernel<<<blocks, 256>>>(
        *this, d_start, d_goal, d_main_dir, max_dist, num_rays,
        samples, max_bisection_iters, tol, d_out_intersections, d_out_count, max_outputs, dof
    );

    int h_count;
    cudaMemcpy(&h_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost);
    h_count = std::min(h_count, max_outputs);

    std::vector<double> h_out(h_count * dof);
    cudaMemcpy(h_out.data(), d_out_intersections, h_count * dof * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < h_count; i++) {
        for (int d = 0; d < dof; d++) intersections.push_back(h_out[i * dof + d]);
    }

    cudaFree(d_start); cudaFree(d_goal); cudaFree(d_main_dir);
    cudaFree(d_out_intersections); cudaFree(d_out_count);

    std::cout << "Number of intersections " << intersections.size() / dof << std::endl;

    return !intersections.empty();
}

__global__
void eval_sdf_kernel(
    SphereSDF sdf,
    const double* q,
    double* out
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = sdf(q);
    }
}

__global__ void ray_intersection_kernel(
    SphereSDF sdf,
    const double* d_start,
    const double* d_goal,
    const double* d_main_dir,
    double max_dist,
    int num_rays,
    int samples,
    int max_bisection_iters,
    double tol,
    double* d_out_intersections,
    int* d_out_count,
    int max_outputs,
    int dof
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rays * 2) return;
    int origin_type = tid % 2;
    const double* origin = (origin_type == 0) ? d_start : d_goal;

    curandState localState;
    curand_init(1337, tid, 0, &localState);

    double sg[6];       // g - s
    double sg_norm_sq = 0.0;
    for (int i = 0; i < dof; i++) {
        sg[i] = d_goal[i] - d_start[i];
        sg_norm_sq += sg[i] * sg[i];
    }

    double ray_dir[6];
    double projection = 0;
    while (projection <= 0.01) {
        double norm = 0;
        for (int i = 0; i < dof; i++) {
            ray_dir[i] = curand_normal_double(&localState);
            norm += ray_dir[i] * ray_dir[i];
        }
        norm = sqrt(norm);
        projection = 0;
        for (int i = 0; i < dof; i++) {
            ray_dir[i] /= norm;
            projection += ray_dir[i] * d_main_dir[i];
        }
    }

    double d_dot_sg = 0.0;
    for (int i = 0; i < dof; i++)
        d_dot_sg += ray_dir[i] * sg[i];

    double t_plane_min = 0.0;
    double t_plane_max = max_dist;

    const double parallel_eps = 1e-10;
    if (fabs(d_dot_sg) > parallel_eps) {
        double num_s = 0.0;
        for (int i = 0; i < dof; i++)
            num_s += (d_start[i] - origin[i]) * sg[i];
        double t_plane_s = num_s / d_dot_sg;

        double num_g = 0.0;
        for (int i = 0; i < dof; i++)
            num_g += (d_goal[i] - origin[i]) * sg[i];
        double t_plane_g = num_g / d_dot_sg;

        double t_lo = fmin(t_plane_s, t_plane_g);
        double t_hi = fmax(t_plane_s, t_plane_g);

        t_plane_min = fmax(0.0, t_lo);
        t_plane_max = fmin(max_dist, t_hi);

        if (t_plane_min >= t_plane_max) return;
    }

    auto is_outside_bounds = [&](const double* q) {
        for (int i = 0; i < dof; i++) {
            if (q[i] > 3.14159265358979323846 || q[i] < -3.14159265358979323846)
                return true;
        }
        return false;
    };

    auto point_at = [&](double t, double* q_out) {
        for (int i = 0; i < dof; i++) q_out[i] = origin[i] + t * ray_dir[i];
    };

    double q_prev[6], q_curr[6], q_m[6];
    point_at(t_plane_min, q_prev);
    if (is_outside_bounds(q_prev)) return;
    double f_prev = sdf(q_prev);

    for (int s = 1; s <= samples; s++) {
        // Sample within [t_plane_min, t_plane_max] instead of [0, max_dist]
        double t_curr = t_plane_min + (double(s) / samples) * (t_plane_max - t_plane_min);
        point_at(t_curr, q_curr);
        if (is_outside_bounds(q_curr)) break;
        double f_curr = sdf(q_curr);

        if (f_prev * f_curr <= 0.0) {
            double ta = t_plane_min + (double(s - 1) / samples) * (t_plane_max - t_plane_min);
            double tb = t_curr;
            double fa = f_prev;

            for (int k = 0; k < max_bisection_iters; k++) {
                double tm = 0.5 * (ta + tb);
                point_at(tm, q_m);
                double fm = sdf(q_m);

                if (abs(fm) < tol || abs(tb - ta) < 1e-8) {
                    int idx = atomicAdd(d_out_count, 1);
                    if (idx < max_outputs) {
                        for (int d = 0; d < dof; d++)
                            d_out_intersections[idx * dof + d] = q_m[d];
                    }
                    break;
                }

                if (fa * fm > 0.0) { ta = tm; fa = fm; }
                else { tb = tm; }
            }
            f_curr = sdf(q_curr);
        }
        f_prev = f_curr;
    }
}

bool SphereSDF::get_line_intersections(
    const double* start_point,
    const double* goal_point,
    std::vector<double>& intersections,
    int samples,
    int max_bisection_iters,
    double tol
) const
{
    intersections.clear();

    double* d_q;
    double* d_val;
    cudaMalloc(&d_q, sizeof(double) * dof);
    cudaMalloc(&d_val, sizeof(double));

    auto eval = [&](const std::vector<double>& q) {
        cudaMemcpy(d_q, q.data(), sizeof(double) * dof, cudaMemcpyHostToDevice);
        eval_sdf_kernel<<<1,1>>>(*this, d_q, d_val);
        cudaDeviceSynchronize();
        double v;
        cudaMemcpy(&v, d_val, sizeof(double), cudaMemcpyDeviceToHost);
        return v;
    };

    std::vector<double> dir(dof);
    for (int i = 0; i < dof; i++)
        dir[i] = goal_point[i] - start_point[i];

    auto point_at = [&](double t) {
        std::vector<double> q(dof);
        for (int i = 0; i < dof; i++)
            q[i] = start_point[i] + t * dir[i];
        return q;
    };

    double t_prev = 0.0;
    std::vector<double> q_prev = point_at(t_prev);
    double f_prev = eval(q_prev);

    for (int s = 1; s <= samples; s++) {
        double t_curr = double(s) / samples;
        std::vector<double> q_curr = point_at(t_curr);
        double f_curr = eval(q_curr);

        if (f_prev * f_curr <= 0.0) {
            double ta = t_prev;
            double tb = t_curr;

            for (int k = 0; k < max_bisection_iters; k++) {
                double tm = 0.5 * (ta + tb);
                std::vector<double> qm = point_at(tm);
                double fm = eval(qm);

                if (std::abs(fm) < tol) {
                    for (auto& q : qm) intersections.push_back(q);
                    break;
                }

                if (f_prev * fm > 0.0) {
                    ta = tm;
                    f_prev = fm;
                } else {
                    tb = tm;
                }

                if (std::abs(tb - ta) < 1e-8) {
                    for (auto& p: point_at(0.5 * (ta + tb))) intersections.push_back(p);
                    break;
                }
            }
        }

        t_prev = t_curr;
        q_prev = q_curr;
        f_prev = f_curr;
    }

    cudaFree(d_q);
    cudaFree(d_val);

    return !intersections.empty();
}

bool SphereSDF::get_line_rays(
    const double* start_point,
    const double* goal_point,
    std::vector<double>& intersections,
    int num_rays,
    int samples,
    int max_bisection_iters,
    double tol
) const {
    get_line_intersections(start_point, goal_point, intersections, samples, max_bisection_iters, tol);

    std::vector<double> main_dir(dof);
    double main_len = 0;
    for (int i = 0; i < dof; i++) {
        main_dir[i] = goal_point[i] - start_point[i];
        main_len += main_dir[i] * main_dir[i];
    }
    main_len = sqrt(main_len);
    for (int i = 0; i < dof; i++) main_dir[i] /= (main_len + 1e-12);

    double *d_start, *d_goal, *d_main_dir, *d_out_intersections;
    int *d_out_count;
    int max_outputs = num_rays * 20; 

    cudaMalloc(&d_start, dof * sizeof(double));
    cudaMalloc(&d_goal, dof * sizeof(double));
    cudaMalloc(&d_main_dir, dof * sizeof(double));
    cudaMalloc(&d_out_intersections, max_outputs * dof * sizeof(double));
    cudaMalloc(&d_out_count, sizeof(int));

    cudaMemcpy(d_start, start_point, dof * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_goal, goal_point, dof * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_main_dir, main_dir.data(), dof * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_out_count, 0, sizeof(int));

    int total_threads = num_rays * 2;
    int blocks = (total_threads + 255) / 256;
    
    double max_dist = 10.0; 

    ray_intersection_kernel<<<blocks, 256>>>(
        *this, d_start, d_goal, d_main_dir, max_dist, num_rays,
        samples, max_bisection_iters, tol, d_out_intersections, d_out_count, max_outputs, dof
    );

    int h_count;
    cudaMemcpy(&h_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost);
    h_count = std::min(h_count, max_outputs);

    std::vector<double> h_out(h_count * dof);
    cudaMemcpy(h_out.data(), d_out_intersections, h_count * dof * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < h_count; i++) {
        for (int d = 0; d < dof; d++) intersections.push_back(h_out[i * dof + d]);
    }

    cudaFree(d_start); cudaFree(d_goal); cudaFree(d_main_dir);
    cudaFree(d_out_intersections); cudaFree(d_out_count);

    std::cout << "Number of intersections " << intersections.size() / dof << std::endl;

    return !intersections.empty();
}