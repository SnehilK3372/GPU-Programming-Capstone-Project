#include <torch/extension.h>
#include <vector>

// forward declare (implemented in .cu)
std::vector<at::Tensor> compute_iou_matrix(at::Tensor boxes);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_iou_matrix", &compute_iou_matrix, "Compute IOU matrix (GPU)");
}
