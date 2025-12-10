// BiFPN_Concat2 CUDA Kernel
// 实现加权融合 + concat操作

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// PyTorch实现:
// weight = w / (torch.sum(w, dim=0) + epsilon)
// x = [weight[0] * x[0], weight[1] * x[1]]
// return torch.cat(x, dim)

__global__ void bifpn_concat_kernel(
    const float* input1,  // [B, C, H, W]
    const float* input2,  // [B, C, H, W]
    float* output,        // [B, 2C, H, W]
    float weight1,
    float weight2,
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    
    if (idx >= total) return;
    
    // 计算索引
    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int b = idx / (W * H * C);
    
    // 读取输入值
    float val1 = input1[idx] * weight1;
    float val2 = input2[idx] * weight2;
    
    // 写入输出（concat在channel维度）
    // 前C个通道放weight1 * input1
    int out_idx1 = ((b * (2 * C) + c) * H + h) * W + w;
    output[out_idx1] = val1;
    
    // 后C个通道放weight2 * input2
    int out_idx2 = ((b * (2 * C) + (C + c)) * H + h) * W + w;
    output[out_idx2] = val2;
}

extern "C" void bifpn_concat_forward(
    const float* input1,
    const float* input2,
    float* output,
    float weight1,
    float weight2,
    int B, int C, int H, int W,
    cudaStream_t stream
) {
    int total = B * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    bifpn_concat_kernel<<<blocks, threads, 0, stream>>>(
        input1, input2, output,
        weight1, weight2,
        B, C, H, W
    );
}


