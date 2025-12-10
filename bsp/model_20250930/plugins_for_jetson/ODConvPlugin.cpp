// ODConv TensorRT Plugin Implementation
// 简化版：使用标准分组卷积 + 简化的通道注意力

#include "ODConvPlugin.h"
#include <cstring>
#include <iostream>
#include <vector>

// CUDA前向声明
extern "C" void odconv_forward_simplified(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int C_in, int C_out, int H, int W,
    int kernel_size, int stride, int padding,
    int groups,
    cudaStream_t stream
);

namespace nvinfer1 {
namespace plugin {

// ==================== ODConvPlugin Implementation ====================

ODConvPlugin::ODConvPlugin(
    int in_channels, int out_channels, int kernel_size,
    int stride, int padding, int groups, int kernel_num
) : mInChannels(in_channels),
    mOutChannels(out_channels),
    mKernelSize(kernel_size),
    mStride(stride),
    mPadding(padding),
    mGroups(groups),
    mKernelNum(kernel_num)
{
}

ODConvPlugin::ODConvPlugin(const void* data, size_t length)
{
    const char* d = static_cast<const char*>(data);
    mInChannels = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mOutChannels = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mKernelSize = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mStride = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mPadding = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mGroups = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mKernelNum = *reinterpret_cast<const int*>(d); d += sizeof(int);
}

ODConvPlugin::~ODConvPlugin()
{
    terminate();
}

int ODConvPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DimsExprs ODConvPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs,
    IExprBuilder& exprBuilder) noexcept
{
    DimsExprs output;
    output.nbDims = 4;
    output.d[0] = inputs[0].d[0];  // B
    output.d[1] = exprBuilder.constant(mOutChannels);  // C_out
    
    // H_out = (H + 2*padding - kernel_size) / stride + 1
    auto* two = exprBuilder.constant(2);
    auto* padded_h = exprBuilder.operation(
        DimensionOperation::kSUM,
        *inputs[0].d[2],
        *exprBuilder.operation(DimensionOperation::kPROD, *two, *exprBuilder.constant(mPadding))
    );
    auto* h_minus_k = exprBuilder.operation(
        DimensionOperation::kSUB,
        *padded_h,
        *exprBuilder.constant(mKernelSize)
    );
    auto* h_out_minus1 = exprBuilder.operation(
        DimensionOperation::kFLOOR_DIV,
        *h_minus_k,
        *exprBuilder.constant(mStride)
    );
    output.d[2] = exprBuilder.operation(
        DimensionOperation::kSUM,
        *h_out_minus1,
        *exprBuilder.constant(1)
    );
    
    // W_out = (W + 2*padding - kernel_size) / stride + 1
    auto* padded_w = exprBuilder.operation(
        DimensionOperation::kSUM,
        *inputs[0].d[3],
        *exprBuilder.operation(DimensionOperation::kPROD, *two, *exprBuilder.constant(mPadding))
    );
    auto* w_minus_k = exprBuilder.operation(
        DimensionOperation::kSUB,
        *padded_w,
        *exprBuilder.constant(mKernelSize)
    );
    auto* w_out_minus1 = exprBuilder.operation(
        DimensionOperation::kFLOOR_DIV,
        *w_minus_k,
        *exprBuilder.constant(mStride)
    );
    output.d[3] = exprBuilder.operation(
        DimensionOperation::kSUM,
        *w_out_minus1,
        *exprBuilder.constant(1)
    );
    
    return output;
}

bool ODConvPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // 支持 FLOAT 和 HALF，LINEAR 格式
    const PluginTensorDesc& desc = inOut[pos];
    return (desc.type == DataType::kFLOAT || desc.type == DataType::kHALF) &&
           desc.format == TensorFormat::kLINEAR;
}

int ODConvPlugin::initialize() noexcept
{
    return 0;
}

void ODConvPlugin::terminate() noexcept
{
}

size_t ODConvPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs,
    const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    // 需要workspace存储attention权重
    int B = inputs[0].dims.d[0];
    int attention_size = B * mInChannels;  // 简化版只需要通道注意力
    return attention_size * sizeof(float);
}

int ODConvPlugin::enqueue(
    const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{
    int B = inputDesc[0].dims.d[0];
    int C_in = inputDesc[0].dims.d[1];
    int H = inputDesc[0].dims.d[2];
    int W = inputDesc[0].dims.d[3];
    
    const float* input = static_cast<const float*>(inputs[0]);
    const float* weight = static_cast<const float*>(inputs[1]);  // 权重从输入传入
    // 简化版暂不支持bias
    const float* bias = nullptr;
    float* output = static_cast<float*>(outputs[0]);
    
    // 调用简化版CUDA kernel
    odconv_forward_simplified(
        input, weight, bias, output,
        B, C_in, mOutChannels, H, W,
        mKernelSize, mStride, mPadding, mGroups,
        stream
    );
    
    return 0;
}

DataType ODConvPlugin::getOutputDataType(
    int index, const DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

void ODConvPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs,
    const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t ODConvPlugin::getSerializationSize() const noexcept
{
    return 7 * sizeof(int);
}

void ODConvPlugin::serialize(void* buffer) const noexcept
{
    char* d = static_cast<char*>(buffer);
    *reinterpret_cast<int*>(d) = mInChannels; d += sizeof(int);
    *reinterpret_cast<int*>(d) = mOutChannels; d += sizeof(int);
    *reinterpret_cast<int*>(d) = mKernelSize; d += sizeof(int);
    *reinterpret_cast<int*>(d) = mStride; d += sizeof(int);
    *reinterpret_cast<int*>(d) = mPadding; d += sizeof(int);
    *reinterpret_cast<int*>(d) = mGroups; d += sizeof(int);
    *reinterpret_cast<int*>(d) = mKernelNum; d += sizeof(int);
}

const char* ODConvPlugin::getPluginType() const noexcept
{
    return "ODConv2d";
}

const char* ODConvPlugin::getPluginVersion() const noexcept
{
    return "1";
}

void ODConvPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2DynamicExt* ODConvPlugin::clone() const noexcept
{
    auto* plugin = new ODConvPlugin(mInChannels, mOutChannels, mKernelSize,
                                     mStride, mPadding, mGroups, mKernelNum);
    return plugin;
}

void ODConvPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const char* ODConvPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// ==================== ODConvPluginCreator Implementation ====================

// 定义 static 成员变量
PluginFieldCollection ODConvPluginCreator::mFC;
std::vector<PluginField> ODConvPluginCreator::mPluginAttributes;

ODConvPluginCreator::ODConvPluginCreator()
{
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ODConvPluginCreator::getPluginName() const noexcept
{
    return "ODConv2d";
}

const char* ODConvPluginCreator::getPluginVersion() const noexcept
{
    return "1";
}

const PluginFieldCollection* ODConvPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* ODConvPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept
{
    int in_channels = 256, out_channels = 256, kernel_size = 3;
    int stride = 1, padding = 1, groups = 1, kernel_num = 4;
    
    for (int i = 0; i < fc->nbFields; i++) {
        std::string field_name(fc->fields[i].name);
        if (field_name == "in_channels") {
            in_channels = *static_cast<const int*>(fc->fields[i].data);
        } else if (field_name == "out_channels") {
            out_channels = *static_cast<const int*>(fc->fields[i].data);
        } else if (field_name == "kernel_size") {
            kernel_size = *static_cast<const int*>(fc->fields[i].data);
        } else if (field_name == "stride") {
            stride = *static_cast<const int*>(fc->fields[i].data);
        } else if (field_name == "padding") {
            padding = *static_cast<const int*>(fc->fields[i].data);
        } else if (field_name == "groups") {
            groups = *static_cast<const int*>(fc->fields[i].data);
        } else if (field_name == "kernel_num") {
            kernel_num = *static_cast<const int*>(fc->fields[i].data);
        }
    }
    
    return new ODConvPlugin(in_channels, out_channels, kernel_size,
                            stride, padding, groups, kernel_num);
}

IPluginV2DynamicExt* ODConvPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new ODConvPlugin(serialData, serialLength);
}

void ODConvPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const char* ODConvPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// 注册插件
REGISTER_TENSORRT_PLUGIN(ODConvPluginCreator);

} // namespace plugin
} // namespace nvinfer1


