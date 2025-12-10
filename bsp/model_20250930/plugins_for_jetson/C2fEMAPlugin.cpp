// C2f_EMA TensorRT Plugin Implementation
// 简化版：标准C2f + 简化的通道注意力（去掉复杂的EMA）

#include "C2fEMAPlugin.h"
#include <cstring>
#include <iostream>

// CUDA前向声明
extern "C" void c2f_ema_forward_simplified(
    const float* input,
    float* output,
    int B, int C, int H, int W,
    int num_bottlenecks,
    cudaStream_t stream
);

namespace nvinfer1 {
namespace plugin {

// ==================== C2fEMAPlugin Implementation ====================

C2fEMAPlugin::C2fEMAPlugin(int channels, int num_bottlenecks)
    : mChannels(channels), mNumBottlenecks(num_bottlenecks)
{
}

C2fEMAPlugin::C2fEMAPlugin(const void* data, size_t length)
{
    const char* d = static_cast<const char*>(data);
    mChannels = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mNumBottlenecks = *reinterpret_cast<const int*>(d); d += sizeof(int);
}

C2fEMAPlugin::~C2fEMAPlugin()
{
    terminate();
}

int C2fEMAPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DimsExprs C2fEMAPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs,
    IExprBuilder& exprBuilder) noexcept
{
    // C2f_EMA输出维度与输入相同
    return inputs[0];
}

bool C2fEMAPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // 支持 FLOAT 和 HALF，LINEAR 格式
    const PluginTensorDesc& desc = inOut[pos];
    return (desc.type == DataType::kFLOAT || desc.type == DataType::kHALF) &&
           desc.format == TensorFormat::kLINEAR;
}

int C2fEMAPlugin::initialize() noexcept
{
    return 0;
}

void C2fEMAPlugin::terminate() noexcept
{
}

size_t C2fEMAPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs,
    const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    int B = inputs[0].dims.d[0];
    int C = inputs[0].dims.d[1];
    int H = inputs[0].dims.d[2];
    int W = inputs[0].dims.d[3];
    
    // 需要workspace存储中间特征和注意力
    size_t workspace = 0;
    workspace += B * C * H * W * sizeof(float) * 2;  // 中间特征
    workspace += B * C * sizeof(float);               // 注意力权重
    
    return workspace;
}

int C2fEMAPlugin::enqueue(
    const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{
    int B = inputDesc[0].dims.d[0];
    int C = inputDesc[0].dims.d[1];
    int H = inputDesc[0].dims.d[2];
    int W = inputDesc[0].dims.d[3];
    
    const float* input = static_cast<const float*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    
    // 调用简化版CUDA kernel
    c2f_ema_forward_simplified(
        input, output,
        B, C, H, W,
        mNumBottlenecks,
        stream
    );
    
    return 0;
}

DataType C2fEMAPlugin::getOutputDataType(
    int index, const DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

void C2fEMAPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs,
    const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t C2fEMAPlugin::getSerializationSize() const noexcept
{
    return 2 * sizeof(int);
}

void C2fEMAPlugin::serialize(void* buffer) const noexcept
{
    char* d = static_cast<char*>(buffer);
    *reinterpret_cast<int*>(d) = mChannels; d += sizeof(int);
    *reinterpret_cast<int*>(d) = mNumBottlenecks; d += sizeof(int);
}

const char* C2fEMAPlugin::getPluginType() const noexcept
{
    return "C2f_EMA";
}

const char* C2fEMAPlugin::getPluginVersion() const noexcept
{
    return "1";
}

void C2fEMAPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2DynamicExt* C2fEMAPlugin::clone() const noexcept
{
    auto* plugin = new C2fEMAPlugin(mChannels, mNumBottlenecks);
    return plugin;
}

void C2fEMAPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const char* C2fEMAPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// ==================== C2fEMAPluginCreator Implementation ====================

// 定义 static 成员变量
PluginFieldCollection C2fEMAPluginCreator::mFC;
std::vector<PluginField> C2fEMAPluginCreator::mPluginAttributes;

C2fEMAPluginCreator::C2fEMAPluginCreator()
{
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* C2fEMAPluginCreator::getPluginName() const noexcept
{
    return "C2f_EMA";
}

const char* C2fEMAPluginCreator::getPluginVersion() const noexcept
{
    return "1";
}

const PluginFieldCollection* C2fEMAPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* C2fEMAPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept
{
    int channels = 256;
    int num_bottlenecks = 3;
    
    for (int i = 0; i < fc->nbFields; i++) {
        std::string field_name(fc->fields[i].name);
        if (field_name == "channels") {
            channels = *static_cast<const int*>(fc->fields[i].data);
        } else if (field_name == "num_bottlenecks") {
            num_bottlenecks = *static_cast<const int*>(fc->fields[i].data);
        }
    }
    
    return new C2fEMAPlugin(channels, num_bottlenecks);
}

IPluginV2DynamicExt* C2fEMAPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new C2fEMAPlugin(serialData, serialLength);
}

void C2fEMAPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const char* C2fEMAPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// 注册插件
REGISTER_TENSORRT_PLUGIN(C2fEMAPluginCreator);

} // namespace plugin
} // namespace nvinfer1


