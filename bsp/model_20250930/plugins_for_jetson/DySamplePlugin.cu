// DySample CUDA Kernel Implementation - 完整版
// 基于 ultralytics/nn/modules/Dysample.py 的 forward 逻辑

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// 辅助函数：安全的像素访问（border padding模式）
__device__ inline float safe_get_pixel(
    const float* input,
    int bg, int c, int h, int w,
    int channels_per_group, int in_h, int in_w
) {
    // Clamp边界
    h = max(0, min(h, in_h - 1));
    w = max(0, min(w, in_w - 1));
    int idx = ((bg * channels_per_group + c) * in_h + h) * in_w + w;
    return input[idx];
}

// grid_sample的CUDA实现（核心算子）- 双线性插值
__global__ void grid_sample_kernel(
    const float* input,      // [B*G, C/G, H, W]
    const float* grid,       // [B*G, H_out, W_out, 2]
    float* output,           // [B*G, C/G, H_out, W_out]
    int batch_groups,        // B * groups
    int channels_per_group,  // C / groups
    int in_h, int in_w,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_groups * channels_per_group * out_h * out_w;
    
    if (idx >= total) return;
    
    // 解析索引 [bg, c, h_out, w_out]
    int w_out = idx % out_w;
    int h_out = (idx / out_w) % out_h;
    int c = (idx / (out_w * out_h)) % channels_per_group;
    int bg = idx / (out_w * out_h * channels_per_group);
    
    // 读取grid坐标 (x, y) 范围在[-1, 1]
    int grid_idx = ((bg * out_h + h_out) * out_w + w_out) * 2;
    float x = grid[grid_idx];      // x坐标（对应width）
    float y = grid[grid_idx + 1];  // y坐标（对应height）
    
    // 归一化坐标[-1,1]转换为像素坐标[0, W-1]或[0, H-1]
    float ix = ((x + 1.0f) / 2.0f) * (in_w - 1);
    float iy = ((y + 1.0f) / 2.0f) * (in_h - 1);
    
    // 双线性插值的四个邻近点
    int ix0 = (int)floorf(ix);
    int iy0 = (int)floorf(iy);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;
    
    // 插值权重
    float dx = ix - ix0;
    float dy = iy - iy0;
    
    // 双线性插值: f(x,y) = f00*(1-dx)*(1-dy) + f10*dx*(1-dy) + f01*(1-dx)*dy + f11*dx*dy
    float val = 0.0f;
    val += safe_get_pixel(input, bg, c, iy0, ix0, channels_per_group, in_h, in_w) * (1 - dx) * (1 - dy);
    val += safe_get_pixel(input, bg, c, iy0, ix1, channels_per_group, in_h, in_w) * dx * (1 - dy);
    val += safe_get_pixel(input, bg, c, iy1, ix0, channels_per_group, in_h, in_w) * (1 - dx) * dy;
    val += safe_get_pixel(input, bg, c, iy1, ix1, channels_per_group, in_h, in_w) * dx * dy;
    
    output[idx] = val;
}

// 1x1卷积计算offset
__global__ void compute_offset_kernel(
    const float* input,          // [B, C, H, W]
    const float* offset_weight,  // [out_C, C, 1, 1]
    const float* offset_bias,    // [out_C]
    float* offset,               // [B, out_C, H, W]
    int B, int C, int H, int W,
    int out_channels            // 2*groups*scale^2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * out_channels * H * W;
    
    if (idx >= total) return;
    
    // 解析索引 [b, oc, h, w]
    int w = idx % W;
    int h = (idx / W) % H;
    int oc = (idx / (W * H)) % out_channels;
    int b = idx / (W * H * out_channels);
    
    float sum = offset_bias[oc];
    
    // 1x1卷积: out[b,oc,h,w] = sum_ic( in[b,ic,h,w] * weight[oc,ic,0,0] )
    for (int ic = 0; ic < C; ++ic) {
        int in_idx = ((b * C + ic) * H + h) * W + w;
        int weight_idx = oc * C + ic;
        sum += input[in_idx] * offset_weight[weight_idx];
    }
    
    offset[idx] = sum;
}

// Pixel Shuffle操作: [B, C*scale^2, H, W] -> [B, C, H*scale, W*scale]
__global__ void pixel_shuffle_kernel(
    const float* input,   // [B, C*r^2, H, W]
    float* output,        // [B, C, H*r, W*r]
    int B, int C, int H, int W, int scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * scale * W * scale;
    
    if (idx >= total) return;
    
    int out_w = idx % (W * scale);
    int out_h = (idx / (W * scale)) % (H * scale);
    int c = (idx / (W * scale * H * scale)) % C;
    int b = idx / (W * scale * H * scale * C);
    
    // 计算输入位置
    int in_h = out_h / scale;
    int in_w = out_w / scale;
    int r_h = out_h % scale;
    int r_w = out_w % scale;
    
    int in_c = c * scale * scale + r_h * scale + r_w;
    int in_idx = ((b * (C * scale * scale) + in_c) * H + in_h) * W + in_w;
    
    output[idx] = input[in_idx];
}

// Pixel Unshuffle (逆操作): [B, C, H*r, W*r] -> [B, C*r^2, H, W]
__global__ void pixel_unshuffle_kernel(
    const float* input,   // [B, C, H*r, W*r]
    float* output,        // [B, C*r^2, H, W]
    int B, int C, int H, int W, int scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * scale * scale * H * W;
    
    if (idx >= total) return;
    
    int w = idx % W;
    int h = (idx / W) % H;
    int out_c = (idx / (W * H)) % (C * scale * scale);
    int b = idx / (W * H * C * scale * scale);
    
    // 计算输入位置
    int c = out_c / (scale * scale);
    int r = out_c % (scale * scale);
    int r_h = r / scale;
    int r_w = r % scale;
    
    int in_h = h * scale + r_h;
    int in_w = w * scale + r_w;
    int in_idx = ((b * C + c) * (H * scale) + in_h) * (W * scale) + in_w;
    
    output[idx] = input[in_idx];
}

// 构建grid坐标（用于grid_sample）
__global__ void build_grid_kernel(
    const float* offset,     // [B, 2*G*s^2, H, W]
    const float* init_pos,   // [1, 2*G*s^2, 1, 1]
    float* grid,             // [B*G, H*s, W*s, 2]
    int B, int G, int H, int W, int scale,
    float scale_factor      // 0.25 for lp mode
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * G * H * scale * W * scale;
    
    if (idx >= total) return;
    
    int w_out = idx % (W * scale);
    int h_out = (idx / (W * scale)) % (H * scale);
    int g = (idx / (W * scale * H * scale)) % G;
    int b = idx / (W * scale * H * scale * G);
    
    // 计算原始H,W位置
    int h_in = h_out / scale;
    int w_in = w_out / scale;
    
    // offset索引
    int offset_c_base = g * 2 * scale * scale;
    int sub_h = h_out % scale;
    int sub_w = w_out % scale;
    int offset_c_x = offset_c_base + sub_h * scale + sub_w;
    int offset_c_y = offset_c_x + G * scale * scale;
    
    int offset_idx_x = ((b * (2 * G * scale * scale) + offset_c_x) * H + h_in) * W + w_in;
    int offset_idx_y = ((b * (2 * G * scale * scale) + offset_c_y) * H + h_in) * W + w_in;
    
    // offset * scale_factor + init_pos
    float dx = offset[offset_idx_x] * scale_factor + init_pos[offset_c_x];
    float dy = offset[offset_idx_y] * scale_factor + init_pos[offset_c_y];
    
    // 基础坐标
    float base_x = (float)w_out + 0.5f;
    float base_y = (float)h_out + 0.5f;
    
    // 归一化到[-1, 1]
    float x = 2.0f * (base_x + dx) / (W * scale) - 1.0f;
    float y = 2.0f * (base_y + dy) / (H * scale) - 1.0f;
    
    // 输出grid [bg, h_out, w_out, 2]
    int bg = b * G + g;
    int grid_idx = ((bg * (H * scale) + h_out) * (W * scale) + w_out) * 2;
    grid[grid_idx] = x;
    grid[grid_idx + 1] = y;
}

// DySample完整实现（lp模式）
extern "C" void dysample_forward_lp(
    const float* d_input,         // [B, C, H, W]
    float* d_output,              // [B, C, H*scale, W*scale]
    const float* d_offset_weight, // [2*G*s^2, C, 1, 1]
    const float* d_offset_bias,   // [2*G*s^2]
    const float* d_init_pos,      // [1, 2*G*s^2, 1, 1]
    int B, int C, int H, int W,
    int scale, int groups,
    cudaStream_t stream
) {
    int out_h = H * scale;
    int out_w = W * scale;
    int offset_channels = 2 * groups * scale * scale;
    int channels_per_group = C / groups;
    
    int threads = 256;
    
    // Step 1: 计算offset [B, 2*G*s^2, H, W]
    float* d_offset;
    cudaMalloc(&d_offset, B * offset_channels * H * W * sizeof(float));
    
    int total_offset = B * offset_channels * H * W;
    int blocks_offset = (total_offset + threads - 1) / threads;
    compute_offset_kernel<<<blocks_offset, threads, 0, stream>>>(
        d_input, d_offset_weight, d_offset_bias, d_offset,
        B, C, H, W, offset_channels
    );
    
    // Step 2: 构建grid [B*G, H*scale, W*scale, 2]
    float* d_grid;
    cudaMalloc(&d_grid, B * groups * out_h * out_w * 2 * sizeof(float));
    
    int total_grid = B * groups * out_h * out_w;
    int blocks_grid = (total_grid + threads - 1) / threads;
    build_grid_kernel<<<blocks_grid, threads, 0, stream>>>(
        d_offset, d_init_pos, d_grid,
        B, groups, H, W, scale,
        0.25f  // scale_factor for lp mode
    );
    
    // Step 3: Reshape input for grid_sample [B*G, C/G, H, W]
    // PyTorch中通过reshape实现，这里直接当作不同的view使用
    
    // Step 4: grid_sample [B*G, C/G, H, W] + grid -> [B*G, C/G, H*s, W*s]
    float* d_sampled;
    cudaMalloc(&d_sampled, B * groups * channels_per_group * out_h * out_w * sizeof(float));
    
    int total_sample = B * groups * channels_per_group * out_h * out_w;
    int blocks_sample = (total_sample + threads - 1) / threads;
    grid_sample_kernel<<<blocks_sample, threads, 0, stream>>>(
        d_input,     // 当作 [B*G, C/G, H, W]
        d_grid,
        d_sampled,
        B * groups,
        channels_per_group,
        H, W,
        out_h, out_w
    );
    
    // Step 5: Reshape回 [B, C, H*s, W*s]
    // 由于内存布局兼容，直接复制
    cudaMemcpyAsync(d_output, d_sampled, 
                    B * C * out_h * out_w * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    
    // 清理
    cudaFree(d_offset);
    cudaFree(d_grid);
    cudaFree(d_sampled);
}

// 简化版本：最近邻上采样（fallback）
__global__ void nearest_upsample_kernel(
    const float* input,   // [B, C, H, W]
    float* output,        // [B, C, H*scale, W*scale]
    int B, int C, int H, int W, int scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * scale * W * scale;
    
    if (idx >= total) return;
    
    int w_out = idx % (W * scale);
    int h_out = (idx / (W * scale)) % (H * scale);
    int c = (idx / (W * scale * H * scale)) % C;
    int b = idx / (W * scale * H * scale * C);
    
    int h_in = h_out / scale;
    int w_in = w_out / scale;
    
    int in_idx = ((b * C + c) * H + h_in) * W + w_in;
    output[idx] = input[in_idx];
}

// ==================== 简化版接口 ====================

// 双线性上采样（推荐使用）
__global__ void bilinear_upsample_kernel_simple(
    const float* input,   // [B, C, H, W]
    float* output,        // [B, C, H*scale, W*scale]
    int B, int C, int H, int W, int scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * scale * W * scale;
    
    if (idx >= total) return;
    
    // 解析输出索引
    int w_out = idx % (W * scale);
    int h_out = (idx / (W * scale)) % (H * scale);
    int c = (idx / (W * scale * H * scale)) % C;
    int b = idx / (W * scale * H * scale * C);
    
    // 计算输入坐标（浮点）
    float h_in = (h_out + 0.5f) / scale - 0.5f;
    float w_in = (w_out + 0.5f) / scale - 0.5f;
    
    // Clamp到有效范围
    h_in = fmaxf(0.0f, fminf(h_in, H - 1.0f));
    w_in = fmaxf(0.0f, fminf(w_in, W - 1.0f));
    
    // 双线性插值的四个邻近点
    int h0 = (int)floorf(h_in);
    int w0 = (int)floorf(w_in);
    int h1 = min(h0 + 1, H - 1);
    int w1 = min(w0 + 1, W - 1);
    
    float dh = h_in - h0;
    float dw = w_in - w0;
    
    // 读取四个点的值
    int base_idx = (b * C + c) * H * W;
    float v00 = input[base_idx + h0 * W + w0];
    float v01 = input[base_idx + h0 * W + w1];
    float v10 = input[base_idx + h1 * W + w0];
    float v11 = input[base_idx + h1 * W + w1];
    
    // 双线性插值
    float val = v00 * (1 - dh) * (1 - dw) +
                v01 * (1 - dh) * dw +
                v10 * dh * (1 - dw) +
                v11 * dh * dw;
    
    output[idx] = val;
}

extern "C" void dysample_forward_bilinear(
    const float* input,
    float* output,
    int B, int C, int H, int W, int scale,
    cudaStream_t stream
) {
    int total = B * C * H * scale * W * scale;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    bilinear_upsample_kernel_simple<<<blocks, threads, 0, stream>>>(
        input, output, B, C, H, W, scale
    );
}

extern "C" void dysample_forward_nearest(
    const float* input,
    float* output,
    int B, int C, int H, int W, int scale,
    cudaStream_t stream
) {
    int total = B * C * H * scale * W * scale;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    nearest_upsample_kernel<<<blocks, threads, 0, stream>>>(
        input, output, B, C, H, W, scale
    );
}

// 保留兼容性
extern "C" void dysample_forward_simple(
    const float* input,
    float* output,
    int B, int C, int H, int W, int scale,
    cudaStream_t stream
) {
    // 默认使用双线性
    dysample_forward_bilinear(input, output, B, C, H, W, scale, stream);
}

