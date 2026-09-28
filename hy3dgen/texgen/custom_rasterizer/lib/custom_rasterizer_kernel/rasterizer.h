#ifndef RASTERIZER_H_
#define RASTERIZER_H_

#ifdef __CUDACC__
#include <torch/types.h>
#else
#include <torch/extension.h>
#endif
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h> // For CUDA context

#define INT64 unsigned long long
#define MAXINT 2147483647

__host__ __device__ inline float calculateSignedArea2(float* a, float* b, float* c) {
    return ((c[0] - a[0]) * (b[1] - a[1]) - (b[0] - a[0]) * (c[1] - a[1]));
}

__host__ __device__  inline void calculateBarycentricCoordinate(float* a, float* b, float* c, float* p,
    float* barycentric)
{
    float beta_tri = calculateSignedArea2(a, p, c);
    float gamma_tri = calculateSignedArea2(a, b, p);
    float area = calculateSignedArea2(a, b, c);
    if (area == 0) {
        barycentric[0] = -1.0;
        barycentric[1] = -1.0;
        barycentric[2] = -1.0;
        return;
    }
    float tri_inv = 1.0 / area;
    float beta = beta_tri * tri_inv;
    float gamma = gamma_tri * tri_inv;
    float alpha = 1.0 - beta - gamma;
    barycentric[0] = alpha;
    barycentric[1] = beta;
    barycentric[2] = gamma;
}

__host__ __device__  inline bool isBarycentricCoordInBounds(float* barycentricCoord) {
    return barycentricCoord[0] >= 0.0 && barycentricCoord[0] <= 1.0 &&
           barycentricCoord[1] >= 0.0 && barycentricCoord[1] <= 1.0 &&
           barycentricCoord[2] >= 0.0 && barycentricCoord[2] <= 1.0;
}

std::vector<torch::Tensor> rasterize_image_gpu(torch::Tensor V, torch::Tensor F, torch::Tensor D,
    int width, int height, float occlusion_truncation, int use_depth_prior);

std::vector<std::vector<torch::Tensor>> build_hierarchy(std::vector<torch::Tensor> view_layer_positions, std::vector<torch::Tensor> view_layer_normals, int num_level, int resolution);

std::vector<std::vector<torch::Tensor>> build_hierarchy_with_feat(
    std::vector<torch::Tensor> view_layer_positions,
    std::vector<torch::Tensor> view_layer_normals,
    std::vector<torch::Tensor> view_layer_feats,
    int num_level, int resolution);

#endif