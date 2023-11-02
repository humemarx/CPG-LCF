#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include "atomics.cuh"

template<typename scalar_t>
inline scalar_t* DATA_PTR(at::Tensor mat){
    return at::cuda::detail::getTensorInfo<scalar_t, int32_t>(mat).data;
}

// maxpool
namespace maxpool{
    // voxel max pooling forward
    // pcds_feat, (BS, C, N, 1)
    // pcds_ind,(BS, N, D, 1), D -> d1, d2, ..., dn
    // voxel_out, (BS, C, D1, D2, ..., Dn)
    // voxel_max_idx, (BS, N)
    __global__ void VoxelMaxPoolUpdateOutputComputeIdx(float* pcds_ind_data, int32_t* voxel_max_idx_data,
    int32_t BS, int32_t C, int32_t N, int32_t D, int32_t loop,
    int32_t* voxel_out_size, int32_t* voxel_out_stride, int32_t* output_size, float* scale_rate)
    {
        int32_t bs, n;
        int32_t index_ind, index_voxel;
        bool flag;
        for(int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < loop; i = i + blockDim.x * gridDim.x){
            bs = i / N;
            n = i - bs * N;

            index_ind = i * D;
            index_voxel = bs * voxel_out_stride[0]; // bs and c=0

            flag = true;
            for(int32_t d=0; d < D; d++){
                int32_t ind_tmp = static_cast<int32_t>(pcds_ind_data[index_ind + d] * scale_rate[d]);
                if((ind_tmp >=0) && (ind_tmp < output_size[d])){
                    index_voxel = index_voxel + ind_tmp * voxel_out_stride[2 + d];
                }
                else{
                    flag = false;
                }
            }
            if(flag){
                voxel_max_idx_data[i] = index_voxel;
            }
            else{
                voxel_max_idx_data[i] = -1;
            }
        }
    }

    template<typename real>
    __global__ void VoxelMaxPoolUpdateOutputInit(real* pcds_feat_data, real* voxel_out_data, int32_t* voxel_max_idx_data,
    int32_t BS, int32_t C, int32_t N, int32_t D, int32_t loop,
    int32_t* voxel_out_size, int32_t* voxel_out_stride, int32_t* output_size)
    {
        int32_t bs, c, n;
        int32_t index_pcds, index_voxel0;
        int32_t index_res;
        for(int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < loop; i = i + blockDim.x * gridDim.x){
            bs = i / (C * N);
            index_res = i - bs * C * N;
            c = index_res / N;
            n = index_res - c * N;

            index_pcds = i;
            index_voxel0 = voxel_max_idx_data[bs * N + n];
            if(index_voxel0 >= 0){
                voxel_out_data[index_voxel0 + c * voxel_out_stride[1]] = pcds_feat_data[index_pcds];
            }
        }
    }

    template<typename real>
    __global__ void VoxelMaxPoolUpdateOutputKernel(real* pcds_feat_data, real* voxel_out_data, int32_t* voxel_max_idx_data,
    int32_t BS, int32_t C, int32_t N, int32_t D, int32_t loop,
    int32_t* voxel_out_size, int32_t* voxel_out_stride, int32_t* output_size)
    {
        int32_t bs, c, n;
        int32_t index_pcds, index_voxel0;
        int32_t index_res;
        for(int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < loop; i = i + blockDim.x * gridDim.x){
            bs = i / (C * N);
            index_res = i - bs * C * N;
            c = index_res / N;
            n = index_res - c * N;

            index_pcds = i;
            index_voxel0 = voxel_max_idx_data[bs * N + n];
            if(index_voxel0 >= 0){
                atomMax(&voxel_out_data[index_voxel0 + c * voxel_out_stride[1]], pcds_feat_data[index_pcds]);
            }
        }
    }

    // backward
    // pcds_feat, (BS, C, N, 1)
    // pcds_ind,(BS, N, D, 1), D -> d1, d2, ..., dn
    // voxel_out, (BS, C, D1, D2, ..., Dn)
    // voxel_max_idx, (BS, N)
    // grad_pcds_feat, (BS, C, N, 1)
    // grad_voxel_out, (BS, C, D1, D2, ..., Dn)
    template<typename real>
    __global__ void VoxelMaxPoolUpdateBackwardKernel(real* pcds_feat_data, real* voxel_out_data, real* grad_pcds_feat_data, real* grad_voxel_out_data, int32_t* voxel_max_idx_data,
    int32_t BS, int32_t C, int32_t N, int32_t D, int32_t loop,
    int32_t* voxel_out_size, int32_t* voxel_out_stride, int32_t* output_size)
    {
        int32_t bs, c, n;
        int32_t index_pcds, index_voxel0;
        int32_t index_res;
        for(int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < loop; i = i + blockDim.x * gridDim.x){
            bs = i / (C * N);
            index_res = i - bs * C * N;
            c = index_res / N;
            n = index_res - c * N;

            index_pcds = i;
            index_voxel0 = voxel_max_idx_data[bs * N + n];
            if(index_voxel0 >= 0){
                int32_t index_voxel = index_voxel0 + c * voxel_out_stride[1];
                if(voxel_out_data[index_voxel] == pcds_feat_data[index_pcds]){
                    grad_pcds_feat_data[index_pcds] = grad_voxel_out_data[index_voxel];
                }
            }
        }
    }
}

void voxel_maxpooling_cuda_forward(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor voxel_out, at::Tensor voxel_max_idx,
at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size, at::Tensor scale_rate)
{
    cudaSetDevice(pcds_feat.get_device());
    int32_t BS = pcds_feat.size(0);
    int32_t C = pcds_feat.size(1);
    int32_t N = pcds_feat.size(2);
    int32_t D = pcds_ind.size(2);

    int32_t loop1 = BS * N;
    int32_t loop2 = BS * C * N;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(pcds_feat.scalar_type(), "VoxelMaxPoolUpdateOutputComputeIdx", [&] {
        scalar_t* pcds_feat_data = DATA_PTR<scalar_t>(pcds_feat);
        float* pcds_ind_data = DATA_PTR<float>(pcds_ind);
        scalar_t* voxel_out_data = DATA_PTR<scalar_t>(voxel_out);
        int32_t* voxel_max_idx_data = DATA_PTR<int32_t>(voxel_max_idx);

        int32_t* voxel_out_size_data = DATA_PTR<int32_t>(voxel_out_size);
        int32_t* voxel_out_stride_data = DATA_PTR<int32_t>(voxel_out_stride);
        int32_t* output_size_data = DATA_PTR<int32_t>(output_size);
        float* scale_rate_data = DATA_PTR<float>(scale_rate);

        maxpool::VoxelMaxPoolUpdateOutputComputeIdx<<<BLOCKS(loop1), THREADS, 0, stream>>>(pcds_ind_data, voxel_max_idx_data, BS, C, N, D, loop1,
        voxel_out_size_data, voxel_out_stride_data, output_size_data, scale_rate_data);

        maxpool::VoxelMaxPoolUpdateOutputInit<scalar_t><<<BLOCKS(loop2), THREADS, 0, stream>>>(pcds_feat_data, voxel_out_data, voxel_max_idx_data, BS, C, N, D, loop2,
        voxel_out_size_data, voxel_out_stride_data, output_size_data);

        maxpool::VoxelMaxPoolUpdateOutputKernel<scalar_t><<<BLOCKS(loop2), THREADS, 0, stream>>>(pcds_feat_data, voxel_out_data, voxel_max_idx_data, BS, C, N, D, loop2,
        voxel_out_size_data, voxel_out_stride_data, output_size_data);
    });
}

void voxel_maxpooling_cuda_backward(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor voxel_out, at::Tensor voxel_max_idx,
at::Tensor grad_pcds_feat, at::Tensor grad_voxel_out, at::Tensor voxel_out_size, at::Tensor voxel_out_stride, at::Tensor output_size, at::Tensor scale_rate)
{
    cudaSetDevice(pcds_feat.get_device());
    int32_t BS = pcds_feat.size(0);
    int32_t C = pcds_feat.size(1);
    int32_t N = pcds_feat.size(2);
    int32_t D = pcds_ind.size(2);

    int32_t loop = BS * C * N;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_voxel_out.scalar_type(), "VoxelMaxPoolUpdateBackwardKernel", [&] {
        scalar_t* pcds_feat_data = DATA_PTR<scalar_t>(pcds_feat);
        scalar_t* voxel_out_data = DATA_PTR<scalar_t>(voxel_out);
        int32_t* voxel_max_idx_data = DATA_PTR<int32_t>(voxel_max_idx);

        scalar_t* grad_pcds_feat_data = DATA_PTR<scalar_t>(grad_pcds_feat);
        scalar_t* grad_voxel_out_data = DATA_PTR<scalar_t>(grad_voxel_out);

        maxpool::VoxelMaxPoolUpdateBackwardKernel<scalar_t><<<BLOCKS(loop), THREADS, 0, stream>>>(pcds_feat_data, voxel_out_data, grad_pcds_feat_data, grad_voxel_out_data, voxel_max_idx_data,
        BS, C, N, D, loop, DATA_PTR<int32_t>(voxel_out_size), DATA_PTR<int32_t>(voxel_out_stride), DATA_PTR<int32_t>(output_size));
    });
}


//grid2point
namespace grid2point{
    // forward
    // grid_in, (BS, C, H, W)
    // pcds_ind,(BS, N, 2, 1), 2 -> h, w
    // pcds_feat, (BS, C, N, 1)
    template<typename real>
    __global__ void Grid2PointUpdateOutputCudaKernel(real* pcds_feat_data, float* pcds_ind_data, real* grid_in_data,
    int32_t BS, int32_t C, int32_t N, int32_t D, int32_t loop,
    int32_t* grid_in_size, int32_t* grid_in_stride, float* scale_rate)
    {
        int32_t bs, n;
        int32_t index_pcds, index_ind, index_grid;
        float h_tmp, w_tmp;
        for(int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < loop; i = i + blockDim.x * gridDim.x){
            bs = i / N;
            n = i - bs * N;

            index_ind = i * D;
            index_grid = bs * grid_in_stride[0]; // bs and c=0
            index_pcds = bs * C * N + n; // bs, c=0, n

            h_tmp = pcds_ind_data[index_ind] * scale_rate[0];
            w_tmp = pcds_ind_data[index_ind + 1] * scale_rate[1];
            if((h_tmp >= 0) && (h_tmp < grid_in_size[2]) && (w_tmp >= 0) && (w_tmp < grid_in_size[3])){
                for(float hi=floor(h_tmp); hi <= (floor(h_tmp) + 1); hi++){
                    for(float wi=floor(w_tmp); wi <= (floor(w_tmp) + 1); wi++){
                        if((hi >= 0) && (hi < grid_in_size[2]) && (wi >= 0) && (wi < grid_in_size[3])){
                            int32_t index_tmp = index_grid + static_cast<int32_t>(hi) * grid_in_stride[2] + static_cast<int32_t>(wi);

                            float weight = (1 - abs(h_tmp - hi)) * (1 - abs(w_tmp - wi));
                            for(int32_t c = 0; c < C; c++){
                                pcds_feat_data[index_pcds + c * N] += grid_in_data[index_tmp + c * grid_in_stride[1]] * static_cast<real>(weight);
                            }
                        }
                    }
                }
            }
        }
    }

    // backward
    // pcds_ind,(BS, N, 2, 1), 2 -> h, w
    // grad_pcds_feat, (BS, C, N, 1)
    // grad_grid_in, (BS, C, H, W)
    template<typename real>
    __global__ void Grid2PointUpdateBackwardCudaKernel(float* pcds_ind_data, real* grad_pcds_feat_data, real* grad_grid_in_data,
    int32_t BS, int32_t C, int32_t N, int32_t D, int32_t loop,
    int32_t* grid_in_size, int32_t* grid_in_stride, float* scale_rate)
    {
        int32_t bs, n;
        int32_t index_pcds, index_ind, index_grid;
        float h_tmp, w_tmp;
        for(int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < loop; i = i + blockDim.x * gridDim.x){
            bs = i / N;
            n = i - bs * N;

            index_ind = i * D;
            index_grid = bs * grid_in_stride[0]; // bs and c=0
            index_pcds = bs * C * N + n; // bs, c=0, n

            h_tmp = pcds_ind_data[index_ind] * scale_rate[0];
            w_tmp = pcds_ind_data[index_ind + 1] * scale_rate[1];
            if((h_tmp >= 0) && (h_tmp < grid_in_size[2]) && (w_tmp >= 0) && (w_tmp < grid_in_size[3])){
                for(float hi=floor(h_tmp); hi <= (floor(h_tmp) + 1); hi++){
                    for(float wi=floor(w_tmp); wi <= (floor(w_tmp) + 1); wi++){
                        if((hi >= 0) && (hi < grid_in_size[2]) && (wi >= 0) && (wi < grid_in_size[3])){
                            int32_t index_tmp = index_grid + static_cast<int32_t>(hi) * grid_in_stride[2] + static_cast<int32_t>(wi);

                            float weight = (1 - abs(h_tmp - hi)) * (1 - abs(w_tmp - wi));
                            for(int32_t c = 0; c < C; c++){
                                atomAdd(&grad_grid_in_data[index_tmp + c * grid_in_stride[1]], grad_pcds_feat_data[index_pcds + c * N] * static_cast<real>(weight));
                            }
                        }
                    }
                }
            }
        }
    }
}

void grid2point_cuda_forward(at::Tensor pcds_feat, at::Tensor pcds_ind, at::Tensor grid_in,
at::Tensor grid_in_size, at::Tensor grid_in_stride, at::Tensor scale_rate)
{
    cudaSetDevice(pcds_feat.get_device());
    int32_t BS = pcds_feat.size(0);
    int32_t C = pcds_feat.size(1);
    int32_t N = pcds_feat.size(2);

    int32_t D = pcds_ind.size(2);
    int32_t loop = BS * N;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grid_in.scalar_type(), "Grid2PointUpdateOutputCudaKernel", [&] {
        scalar_t* pcds_feat_data = DATA_PTR<scalar_t>(pcds_feat);
        float* pcds_ind_data = DATA_PTR<float>(pcds_ind);
        scalar_t* grid_in_data = DATA_PTR<scalar_t>(grid_in);

        grid2point::Grid2PointUpdateOutputCudaKernel<scalar_t><<<BLOCKS(loop), THREADS, 0, stream>>>(pcds_feat_data, pcds_ind_data, grid_in_data, BS, C, N, D, loop,
        DATA_PTR<int32_t>(grid_in_size), DATA_PTR<int32_t>(grid_in_stride), DATA_PTR<float>(scale_rate));
    });
}

void grid2point_cuda_backward(at::Tensor pcds_ind, at::Tensor grad_pcds_feat, at::Tensor grad_grid_in,
at::Tensor grid_in_size, at::Tensor grid_in_stride, at::Tensor scale_rate)
{
    cudaSetDevice(grad_pcds_feat.get_device());
    int32_t BS = grad_pcds_feat.size(0);
    int32_t C = grad_pcds_feat.size(1);
    int32_t N = grad_pcds_feat.size(2);

    int32_t D = pcds_ind.size(2);
    int32_t loop = BS * N;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_pcds_feat.scalar_type(), "Grid2PointUpdateBackwardCudaKernel", [&] {
        float* pcds_ind_data = DATA_PTR<float>(pcds_ind);
        scalar_t* grad_pcds_feat_data = DATA_PTR<scalar_t>(grad_pcds_feat);
        scalar_t* grad_grid_in_data = DATA_PTR<scalar_t>(grad_grid_in);

        grid2point::Grid2PointUpdateBackwardCudaKernel<scalar_t><<<BLOCKS(loop), THREADS, 0, stream>>>(pcds_ind_data, grad_pcds_feat_data, grad_grid_in_data, BS, C, N, D, loop,
        DATA_PTR<int32_t>(grid_in_size), DATA_PTR<int32_t>(grid_in_stride), DATA_PTR<float>(scale_rate));
    });
}