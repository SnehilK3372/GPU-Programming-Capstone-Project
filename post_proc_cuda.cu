#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// GPU kernel to compute IOU matrix for boxes (N x 4)
// boxes layout: [x1, y1, x2, y2], float32
__global__ void iou_kernel(const float* boxes, float* iou_mat, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;

    const float* bi = boxes + i*4;
    const float* bj = boxes + j*4;

    float x1 = max(bi[0], bj[0]);
    float y1 = max(bi[1], bj[1]);
    float x2 = min(bi[2], bj[2]);
    float y2 = min(bi[3], bj[3]);

    float w = x2 - x1;
    if (w < 0.0f) w = 0.0f;
    float h = y2 - y1;
    if (h < 0.0f) h = 0.0f;
    float inter = w * h;

    float area_i = max(0.0f, bi[2] - bi[0]) * max(0.0f, bi[3] - bi[1]);
    float area_j = max(0.0f, bj[2] - bj[0]) * max(0.0f, bj[3] - bj[1]);
    float uni = area_i + area_j - inter + 1e-6f;
    float iou = inter / uni;

    iou_mat[i * N + j] = iou;
}

// C++ interface callable from postproc_cuda.cpp
std::vector<at::Tensor> compute_iou_matrix(at::Tensor boxes) {
    // Expect boxes: (N,4) float32 CUDA tensor
    TORCH_CHECK(boxes.device().is_cuda(), "boxes must be a CUDA tensor");
    TORCH_CHECK(boxes.dim() == 2 && boxes.size(1) == 4, "boxes must be (N,4)");

    int N = boxes.size(0);
    auto options = boxes.options();
    at::Tensor iou_mat = at::zeros({N, N}, options);

    const float* boxes_ptr = boxes.data_ptr<float>();
    float* iou_ptr = iou_mat.data_ptr<float>();

    const int THREAD = 16;
    dim3 threads(THREAD, THREAD);
    dim3 blocks((N + THREAD - 1) / THREAD, (N + THREAD - 1) / THREAD);

    iou_kernel<<<blocks, threads>>>(boxes_ptr, iou_ptr, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    return {iou_mat};
}
