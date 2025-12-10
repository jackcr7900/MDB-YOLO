// DySample TensorRT Plugin Implementation
// 基于 DySample PyTorch模块的C++封装

#include "DySamplePlugin.h"
#include <cassert>
#include <cstring>
#include <vector>
#include <iostream>

using namespace nvinfer1;
using namespace plugin;

namespace {
    const char* DYSAMPLE_PLUGIN_VERSION{"1"};
    const char* DYSAMPLE_PLUGIN_NAME{"DySample"};
}

// 声明CUDA函数
extern "C" void dysample_forward_simple(
    const float* input, float* output,
    int B, int C, int H, int W, int scale,
    cudaStream_t stream
);

// ===================== DySamplePlugin 实现 =====================

DySamplePlugin::DySamplePlugin(int scale, const std::string& style, int groups)
    : mScale(scale), mStyle(style), mGroups(groups)
{
}

DySamplePlugin::DySamplePlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    const char* a = d;
    
    mScale = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    
    int styleLen = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    
    mStyle = std::string(d, styleLen);
    d += styleLen;
    
    mGroups = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    
    assert(d == a + length);
}

DySamplePlugin::~DySamplePlugin()
{
    terminate();
}

IPluginV2DynamicExt* DySamplePlugin::clone() const noexcept
{
    auto* plugin = new DySamplePlugin(mScale, mStyle, mGroups);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

DimsExprs DySamplePlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs,
    IExprBuilder& exprBuilder) noexcept
{
    // 输入: [B, C, H, W]
    // 输出: [B, C, H*scale, W*scale]
    DimsExprs output;
    output.nbDims = 4;
    output.d[0] = inputs[0].d[0];  // B
    output.d[1] = inputs[0].d[1];  // C
    
    auto* scale_expr = exprBuilder.constant(mScale);
    output.d[2] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *scale_expr);  // H*scale
    output.d[3] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[3], *scale_expr);  // W*scale
    
    return output;
}

bool DySamplePlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // 只支持FP32/FP16 + NCHW
    assert(nbInputs == 1 && nbOutputs == 1);
    const PluginTensorDesc& desc = inOut[pos];
    
    return (desc.type == DataType::kFLOAT || desc.type == DataType::kHALF) &&
           desc.format == TensorFormat::kLINEAR;
}

void DySamplePlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs,
    const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    // 配置插件（可以在这里做一些预处理）
}

size_t DySamplePlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs,
    const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    // 需要临时存储offset等中间结果
    int B = inputs[0].dims.d[0];
    int C = inputs[0].dims.d[1];
    int H = inputs[0].dims.d[2];
    int W = inputs[0].dims.d[3];
    
    int offset_size = B * (2 * mGroups * mScale * mScale) * H * W;
    return offset_size * sizeof(float) * 2;  // offset + grid
}

int DySamplePlugin::enqueue(
    const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{
    // 获取输入输出维度
    int B = inputDesc[0].dims.d[0];
    int C = inputDesc[0].dims.d[1];
    int H = inputDesc[0].dims.d[2];
    int W = inputDesc[0].dims.d[3];
    
    const float* input = static_cast<const float*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    
    // 使用简化版本：双线性上采样
    // 如果需要更快速度，可以改用 dysample_forward_nearest
    dysample_forward_simple(input, output, B, C, H, W, mScale, stream);
    
    return 0;
}

DataType DySamplePlugin::getOutputDataType(
    int index, const DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

const char* DySamplePlugin::getPluginType() const noexcept
{
    return DYSAMPLE_PLUGIN_NAME;
}

const char* DySamplePlugin::getPluginVersion() const noexcept
{
    return DYSAMPLE_PLUGIN_VERSION;
}

int DySamplePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int DySamplePlugin::initialize() noexcept
{
    return 0;
}

void DySamplePlugin::terminate() noexcept
{
}

size_t DySamplePlugin::getSerializationSize() const noexcept
{
    return sizeof(int) * 3 + mStyle.size();
}

void DySamplePlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    
    *reinterpret_cast<int*>(d) = mScale;
    d += sizeof(int);
    
    int styleLen = mStyle.size();
    *reinterpret_cast<int*>(d) = styleLen;
    d += sizeof(int);
    
    std::memcpy(d, mStyle.data(), styleLen);
    d += styleLen;
    
    *reinterpret_cast<int*>(d) = mGroups;
    d += sizeof(int);
}

void DySamplePlugin::destroy() noexcept
{
    delete this;
}

void DySamplePlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const char* DySamplePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// ===================== DySamplePluginCreator 实现 =====================

PluginFieldCollection DySamplePluginCreator::mFC{};
std::vector<PluginField> DySamplePluginCreator::mPluginAttributes;

DySamplePluginCreator::DySamplePluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("scale", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("style", nullptr, PluginFieldType::kCHAR, 1));
    mPluginAttributes.emplace_back(PluginField("groups", nullptr, PluginFieldType::kINT32, 1));
    
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* DySamplePluginCreator::getPluginName() const noexcept
{
    return DYSAMPLE_PLUGIN_NAME;
}

const char* DySamplePluginCreator::getPluginVersion() const noexcept
{
    return DYSAMPLE_PLUGIN_VERSION;
}

const PluginFieldCollection* DySamplePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* DySamplePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    int scale = 2;
    std::string style = "lp";
    int groups = 4;
    
    for (int i = 0; i < fc->nbFields; ++i) {
        std::string field_name(fc->fields[i].name);
        if (field_name == "scale") {
            scale = *static_cast<const int*>(fc->fields[i].data);
        } else if (field_name == "style") {
            style = static_cast<const char*>(fc->fields[i].data);
        } else if (field_name == "groups") {
            groups = *static_cast<const int*>(fc->fields[i].data);
        }
    }
    
    DySamplePlugin* plugin = new DySamplePlugin(scale, style, groups);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2DynamicExt* DySamplePluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    DySamplePlugin* plugin = new DySamplePlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void DySamplePluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const char* DySamplePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// 注册插件
REGISTER_TENSORRT_PLUGIN(DySamplePluginCreator);

