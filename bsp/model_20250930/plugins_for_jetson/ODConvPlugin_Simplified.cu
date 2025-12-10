// ODConv CUDA Kernel - 简化版
// 策略：标准分组卷积 + 简化的SE注意力

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ==================== 简化版：标准分组卷积 + SE注意力 ====================

// 1. 全局平均池化
__global__ void global_avg_pool_kernel(
    const float* input,   // [B, C, H, W]
    float* output,        // [B, C]
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C) return;
    
    int c = idx % C;
    int b = idx / C;
    
    float sum = 0.0f;
    int base = (b * C + c) * H * W;
    for (int i = 0; i < H * W; i++) {
        sum += input[base + i];
    }
    
    output[idx] = sum / (H * W);
}

// 2. SE注意力（简化版）：FC -> ReLU -> FC -> Sigmoid
__global__ void se_attention_kernel(
    const float* pool_out,    // [B, C]
    float* attention,         // [B, C]
    int B, int C,
    int reduction_ratio
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C) return;
    
    // 简化版：直接使用sigmoid(pool_out)作为注意力
    // 完整版需要FC层，这里为了简化省略
    float val = pool_out[idx];
    attention[idx] = 1.0f / (1.0f + expf(-val));  // sigmoid
}

// 3. 应用通道注意力
__global__ void apply_channel_attention_kernel(
    const float* input,       // [B, C, H, W]
    const float* attention,   // [B, C]
    float* output,            // [B, C, H, W]
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) return;
    
    int c = (idx / (H * W)) % C;
    int b = idx / (C * H * W);
    
    float attn = attention[b * C + c];
    output[idx] = input[idx] * attn;
}

// 4. 标准分组卷积（im2col + GEMM的简化版）
__global__ void grouped_conv2d_simple_kernel(
    const float* input,       // [B, C_in, H, W]
    const float* weight,      // [C_out, C_in/groups, K, K]
    const float* bias,        // [C_out] or nullptr
    float* output,            // [B, C_out, H_out, W_out]
    int B, int C_in, int C_out, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding, int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C_out * H_out * W_out;
    if (idx >= total) return;
    
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int b = idx / (W_out * H_out * C_out);
    
    int group = c_out / (C_out / groups);
    int c_in_start = group * (C_in / groups);
    int c_in_end = c_in_start + (C_in / groups);
    
    float sum = 0.0f;
    
    // 卷积计算
    for (int c_in = c_in_start; c_in < c_in_end; c_in++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;
                
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    int in_idx = ((b * C_in + c_in) * H + h_in) * W + w_in;
                    int w_idx = ((c_out * (C_in / groups) + (c_in - c_in_start)) * kernel_size + kh) * kernel_size + kw;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    
    // 加bias
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    
    output[idx] = sum;
}

// ==================== 简化版ODConv前向传播 ====================

extern "C" void odconv_forward_simplified(
    const float* input,       // [B, C_in, H, W]
    const float* weight,      // [C_out, C_in/groups, K, K]
    const float* bias,        // [C_out] or nullptr
    float* output,            // [B, C_out, H_out, W_out]
    int B, int C_in, int C_out, int H, int W,
    int kernel_size, int stride, int padding, int groups,
    cudaStream_t stream
) {
    // 计算输出尺寸
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    
    int threads = 256;
    
    // 分配临时内存
    float *d_pool, *d_attention, *d_attended;
    cudaMalloc(&d_pool, B * C_in * sizeof(float));
    cudaMalloc(&d_attention, B * C_in * sizeof(float));
    cudaMalloc(&d_attended, B * C_in * H * W * sizeof(float));
    
    // 步骤1: 全局平均池化
    int pool_total = B * C_in;
    int pool_blocks = (pool_total + threads - 1) / threads;
    global_avg_pool_kernel<<<pool_blocks, threads, 0, stream>>>(
        input, d_pool, B, C_in, H, W
    );
    
    // 步骤2: SE注意力
    se_attention_kernel<<<pool_blocks, threads, 0, stream>>>(
        d_pool, d_attention, B, C_in, 4  // reduction_ratio=4
    );
    
    // 步骤3: 应用注意力
    int attn_total = B * C_in * H * W;
    int attn_blocks = (attn_total + threads - 1) / threads;
    apply_channel_attention_kernel<<<attn_blocks, threads, 0, stream>>>(
        input, d_attention, d_attended, B, C_in, H, W
    );
    
    // 步骤4: 分组卷积
    int conv_total = B * C_out * H_out * W_out;
    int conv_blocks = (conv_total + threads - 1) / threads;
    grouped_conv2d_simple_kernel<<<conv_blocks, threads, 0, stream>>>(
        d_attended, weight, bias, output,
        B, C_in, C_out, H, W, H_out, W_out,
        kernel_size, stride, padding, groups
    );
    
    // 释放临时内存
    cudaFree(d_pool);
    cudaFree(d_attention);
    cudaFree(d_attended);
}

// ==================== 最简化版本：直接使用标准卷积 ====================

extern "C" void odconv_forward_basic(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int C_in, int C_out, int H, int W,
    int kernel_size, int stride, int padding, int groups,
    cudaStream_t stream
) {
    // 直接调用分组卷积，不使用注意力机制
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    
    int total = B * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    grouped_conv2d_simple_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        B, C_in, C_out, H, W, H_out, W_out,
        kernel_size, stride, padding, groups
    );
}


