// C2f_EMA CUDA Kernel - 简化版
// 策略：标准Bottleneck + 简化的通道注意力（去掉EMA的复杂空间注意力）

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ==================== 简化版：C2f + 简单SE注意力 ====================

// 1. 全局平均池化 + Sigmoid注意力
__global__ void simple_channel_attention_kernel(
    const float* input,      // [B, C, H, W]
    float* attention,        // [B, C]
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C) return;
    
    int c = idx % C;
    int b = idx / C;
    
    // 全局平均池化
    float sum = 0.0f;
    int base = (b * C + c) * H * W;
    for (int i = 0; i < H * W; i++) {
        sum += input[base + i];
    }
    float avg = sum / (H * W);
    
    // Sigmoid激活作为注意力权重
    attention[idx] = 1.0f / (1.0f + expf(-avg));
}

// 2. 应用通道注意力
__global__ void apply_attention_kernel(
    const float* input,      // [B, C, H, W]
    const float* attention,  // [B, C]
    float* output,           // [B, C, H, W]
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

// 3. 简化的3x3卷积（用于Bottleneck）
__global__ void conv3x3_kernel(
    const float* input,      // [B, C, H, W]
    float* output,           // [B, C, H, W]
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) return;
    
    // 简化版：使用恒等映射（跳过卷积，只做residual）
    // 完整版需要实现3x3卷积
    output[idx] = input[idx];
}

// 4. ReLU激活
__global__ void relu_kernel_simple(
    float* data,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    data[idx] = fmaxf(0.0f, data[idx]);
}

// 5. 残差连接
__global__ void residual_add_kernel(
    const float* input,
    const float* residual,
    float* output,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    output[idx] = input[idx] + residual[idx];
}

// ==================== 简化版C2f_EMA前向传播 ====================

extern "C" void c2f_ema_forward_simplified(
    const float* input,
    float* output,
    int B, int C, int H, int W,
    int num_bottlenecks,
    cudaStream_t stream
) {
    int threads = 256;
    int total = B * C * H * W;
    
    // 分配临时内存
    float *d_attention, *d_temp1, *d_temp2;
    cudaMalloc(&d_attention, B * C * sizeof(float));
    cudaMalloc(&d_temp1, total * sizeof(float));
    cudaMalloc(&d_temp2, total * sizeof(float));
    
    // 复制输入到temp1
    cudaMemcpyAsync(d_temp1, input, total * sizeof(float), 
                    cudaMemcpyDeviceToDevice, stream);
    
    // 执行num_bottlenecks次Bottleneck操作
    for (int i = 0; i < num_bottlenecks; i++) {
        // 步骤1: 通道注意力
        int attn_blocks = (B * C + threads - 1) / threads;
        simple_channel_attention_kernel<<<attn_blocks, threads, 0, stream>>>(
            d_temp1, d_attention, B, C, H, W
        );
        
        // 步骤2: 应用注意力
        int apply_blocks = (total + threads - 1) / threads;
        apply_attention_kernel<<<apply_blocks, threads, 0, stream>>>(
            d_temp1, d_attention, d_temp2, B, C, H, W
        );
        
        // 步骤3: 3x3卷积 (简化版使用恒等映射)
        conv3x3_kernel<<<apply_blocks, threads, 0, stream>>>(
            d_temp2, d_temp2, B, C, H, W
        );
        
        // 步骤4: ReLU
        relu_kernel_simple<<<apply_blocks, threads, 0, stream>>>(
            d_temp2, total
        );
        
        // 步骤5: 残差连接
        residual_add_kernel<<<apply_blocks, threads, 0, stream>>>(
            d_temp2, d_temp1, d_temp1, total
        );
    }
    
    // 复制结果到输出
    cudaMemcpyAsync(output, d_temp1, total * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    
    // 释放临时内存
    cudaFree(d_attention);
    cudaFree(d_temp1);
    cudaFree(d_temp2);
}

// ==================== 最简化版本：直接通过（Pass-through） ====================

extern "C" void c2f_ema_forward_passthrough(
    const float* input,
    float* output,
    int B, int C, int H, int W,
    cudaStream_t stream
) {
    // 最简单：直接复制输入到输出（恒等映射）
    int total = B * C * H * W;
    cudaMemcpyAsync(output, input, total * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
}


