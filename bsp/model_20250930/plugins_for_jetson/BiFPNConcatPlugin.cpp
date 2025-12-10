// BiFPN_Concat2 Plugin Implementation
// 最简单的插件实现，用于入门

#include "BiFPNConcatPlugin.h"
#include <cassert>
#include <cstring>
#include <iostream>

using namespace nvinfer1;
using namespace plugin;

namespace {
    const char* BIFPN_CONCAT_PLUGIN_VERSION{"1"};
    const char* BIFPN_CONCAT_PLUGIN_NAME{"BiFPNConcat2"};
}

// CUDA kernel声明
extern "C" void bifpn_concat_forward(
    const float* input1, const float* input2,
    float* output,
    float weight1, float weight2,
    int B, int C, int H, int W,
    cudaStream_t stream
);

// ===================== BiFPNConcatPlugin 实现 =====================

BiFPNConcatPlugin::BiFPNConcatPlugin(int dimension)
    : mDimension(dimension)
{
    // 初始化权重为1.0（PyTorch中是可学习的参数）
    mWeights[0] = 1.0f;
    mWeights[1] = 1.0f;
}

BiFPNConcatPlugin::BiFPNConcatPlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    const char* a = d;
    
    mDimension = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    
    mWeights[0] = *reinterpret_cast<const float*>(d);
    d += sizeof(float);
    
    mWeights[1] = *reinterpret_cast<const float*>(d);
    d += sizeof(float);
    
    mEpsilon = *reinterpret_cast<const float*>(d);
    d += sizeof(float);
    
    assert(d == a + length);
}

BiFPNConcatPlugin::~BiFPNConcatPlugin()
{
    terminate();
}

IPluginV2DynamicExt* BiFPNConcatPlugin::clone() const noexcept
{
    auto* plugin = new BiFPNConcatPlugin(mDimension);
    plugin->mWeights[0] = mWeights[0];
    plugin->mWeights[1] = mWeights[1];
    plugin->mEpsilon = mEpsilon;
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

DimsExprs BiFPNConcatPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs,
    IExprBuilder& exprBuilder) noexcept
{
    // 输入: 两个tensor [B, C, H, W]
    // 输出: [B, 2*C, H, W]  (在channel维度concat)
    assert(nbInputs == 2);
    
    DimsExprs output;
    output.nbDims = 4;
    output.d[0] = inputs[0].d[0];  // B
    
    // channel维度翻倍
    auto* two = exprBuilder.constant(2);
    output.d[1] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[1], *two);
    
    output.d[2] = inputs[0].d[2];  // H
    output.d[3] = inputs[0].d[3];  // W
    
    return output;
}

bool BiFPNConcatPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    assert(nbInputs == 2 && nbOutputs == 1);
    const PluginTensorDesc& desc = inOut[pos];
    
    return (desc.type == DataType::kFLOAT || desc.type == DataType::kHALF) &&
           desc.format == TensorFormat::kLINEAR;
}

void BiFPNConcatPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs,
    const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    // 无需特殊配置
}

size_t BiFPNConcatPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs,
    const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;  // 不需要额外工作空间
}

int BiFPNConcatPlugin::enqueue(
    const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{
    int B = inputDesc[0].dims.d[0];
    int C = inputDesc[0].dims.d[1];
    int H = inputDesc[0].dims.d[2];
    int W = inputDesc[0].dims.d[3];
    
    const float* input1 = static_cast<const float*>(inputs[0]);
    const float* input2 = static_cast<const float*>(inputs[1]);
    float* output = static_cast<float*>(outputs[0]);
    
    // 计算归一化权重
    float sum_w = mWeights[0] + mWeights[1] + mEpsilon;
    float w1 = mWeights[0] / sum_w;
    float w2 = mWeights[1] / sum_w;
    
    // 调用CUDA kernel
    bifpn_concat_forward(input1, input2, output, w1, w2, B, C, H, W, stream);
    
    return 0;
}

DataType BiFPNConcatPlugin::getOutputDataType(
    int index, const DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

const char* BiFPNConcatPlugin::getPluginType() const noexcept
{
    return BIFPN_CONCAT_PLUGIN_NAME;
}

const char* BiFPNConcatPlugin::getPluginVersion() const noexcept
{
    return BIFPN_CONCAT_PLUGIN_VERSION;
}

int BiFPNConcatPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int BiFPNConcatPlugin::initialize() noexcept
{
    return 0;
}

void BiFPNConcatPlugin::terminate() noexcept
{
}

size_t BiFPNConcatPlugin::getSerializationSize() const noexcept
{
    return sizeof(int) + sizeof(float) * 3;
}

void BiFPNConcatPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    
    *reinterpret_cast<int*>(d) = mDimension;
    d += sizeof(int);
    
    *reinterpret_cast<float*>(d) = mWeights[0];
    d += sizeof(float);
    
    *reinterpret_cast<float*>(d) = mWeights[1];
    d += sizeof(float);
    
    *reinterpret_cast<float*>(d) = mEpsilon;
    d += sizeof(float);
}

void BiFPNConcatPlugin::destroy() noexcept
{
    delete this;
}

void BiFPNConcatPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const char* BiFPNConcatPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// ===================== BiFPNConcatPluginCreator 实现 =====================

PluginFieldCollection BiFPNConcatPluginCreator::mFC{};
std::vector<PluginField> BiFPNConcatPluginCreator::mPluginAttributes;

BiFPNConcatPluginCreator::BiFPNConcatPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("dimension", nullptr, PluginFieldType::kINT32, 1));
    
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* BiFPNConcatPluginCreator::getPluginName() const noexcept
{
    return BIFPN_CONCAT_PLUGIN_NAME;
}

const char* BiFPNConcatPluginCreator::getPluginVersion() const noexcept
{
    return BIFPN_CONCAT_PLUGIN_VERSION;
}

const PluginFieldCollection* BiFPNConcatPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* BiFPNConcatPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    int dimension = 1;
    
    for (int i = 0; i < fc->nbFields; ++i) {
        std::string field_name(fc->fields[i].name);
        if (field_name == "dimension") {
            dimension = *static_cast<const int*>(fc->fields[i].data);
        }
    }
    
    BiFPNConcatPlugin* plugin = new BiFPNConcatPlugin(dimension);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2DynamicExt* BiFPNConcatPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    BiFPNConcatPlugin* plugin = new BiFPNConcatPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void BiFPNConcatPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const char* BiFPNConcatPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// 注册插件
REGISTER_TENSORRT_PLUGIN(BiFPNConcatPluginCreator);


