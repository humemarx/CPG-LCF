#include <torch/extension.h>
#include <vector>

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void voxel_maxpooling_cuda_forward(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor voxel_out, at::Tensor voxel_max_idx,
at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size, at::Tensor scale_rate);
void voxel_maxpooling_cuda_backward(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor voxel_out, at::Tensor voxel_max_idx,
at::Tensor grad_pcds_feat, at::Tensor grad_voxel_out, at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size, at::Tensor scale_rate);


void voxel_maxpooling_forward(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor voxel_out, at::Tensor voxel_max_idx,
at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size, at::Tensor scale_rate)
{
    CHECK_INPUT(pcds_feat);
    CHECK_INPUT(pcds_ind);
    CHECK_INPUT(voxel_out);
    CHECK_INPUT(voxel_max_idx);

    CHECK_INPUT(voxel_out_size);
    CHECK_INPUT(voxel_out_stride);
    CHECK_INPUT(output_size);
    CHECK_INPUT(scale_rate);

    voxel_maxpooling_cuda_forward(pcds_feat, pcds_ind, voxel_out, voxel_max_idx,
    voxel_out_size, voxel_out_stride, output_size, scale_rate);
}

void voxel_maxpooling_backward(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor voxel_out, at::Tensor voxel_max_idx,
at::Tensor grad_pcds_feat, at::Tensor grad_voxel_out, at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size, at::Tensor scale_rate)
{
    CHECK_INPUT(pcds_feat);
    CHECK_INPUT(pcds_ind);
    CHECK_INPUT(voxel_out);
    CHECK_INPUT(voxel_max_idx);

    CHECK_INPUT(grad_pcds_feat);
    CHECK_INPUT(grad_voxel_out);
    CHECK_INPUT(voxel_out_size);
    CHECK_INPUT(voxel_out_stride);
    CHECK_INPUT(output_size);
    CHECK_INPUT(scale_rate);

    voxel_maxpooling_cuda_backward(pcds_feat, pcds_ind, voxel_out, voxel_max_idx,
    grad_pcds_feat, grad_voxel_out, voxel_out_size, voxel_out_stride, output_size, scale_rate);
}


void grid2point_cuda_forward(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor grid_in,
at::Tensor grid_in_size, at::Tensor grid_in_stride, at::Tensor scale_rate);
void grid2point_cuda_backward(at::Tensor pcds_ind, at::Tensor grad_pcds_feat, at::Tensor grad_grid_in,
at::Tensor grid_in_size, at::Tensor grid_in_stride, at::Tensor scale_rate);


void grid2point_forward(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor grid_in,
at::Tensor grid_in_size, at::Tensor grid_in_stride, at::Tensor scale_rate)
{
    CHECK_INPUT(pcds_feat);
    CHECK_INPUT(pcds_ind);
    CHECK_INPUT(grid_in);

    CHECK_INPUT(grid_in_size);
    CHECK_INPUT(grid_in_stride);
    CHECK_INPUT(scale_rate);

    grid2point_cuda_forward(pcds_feat, pcds_ind, grid_in, grid_in_size, grid_in_stride, scale_rate);
}

void grid2point_backward(at::Tensor pcds_ind, at::Tensor grad_pcds_feat, at::Tensor grad_grid_in,
at::Tensor grid_in_size, at::Tensor grid_in_stride, at::Tensor scale_rate)
{
    CHECK_INPUT(pcds_ind);
    CHECK_INPUT(grad_pcds_feat);
    CHECK_INPUT(grad_grid_in);

    CHECK_INPUT(grid_in_size);
    CHECK_INPUT(grid_in_stride);
    CHECK_INPUT(scale_rate);

    grid2point_cuda_backward(pcds_ind, grad_pcds_feat, grad_grid_in, grid_in_size, grid_in_stride, scale_rate);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("voxel_maxpooling_forward", &voxel_maxpooling_forward, "maxpooling forward (CUDA)");
  m.def("voxel_maxpooling_backward", &voxel_maxpooling_backward, "maxpooling backward (CUDA)");

  m.def("grid2point_forward", &grid2point_forward, "grid2point bilinear sample forward (CUDA)");
  m.def("grid2point_backward", &grid2point_backward, "grid2point bilinear sample backward (CUDA)");
}