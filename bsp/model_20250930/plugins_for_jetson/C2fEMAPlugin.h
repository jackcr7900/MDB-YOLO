// C2f_EMA TensorRT Plugin Header
// 基于 ultralytics/nn/EMA.py

#ifndef C2F_EMA_PLUGIN_H
#define C2F_EMA_PLUGIN_H

#include "NvInfer.h"
#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace nvinfer1 {
namespace plugin {

class C2fEMAPlugin : public IPluginV2DynamicExt {
public:
    C2fEMAPlugin(int channels, int num_bottlenecks);
    C2fEMAPlugin(const void* data, size_t length);
    ~C2fEMAPlugin() override;

    // IPluginV2DynamicExt Methods
    IPluginV2DynamicExt* clone() const noexcept override;
    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs,
                                 int nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut,
                                  int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs,
                        const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs,
                           const PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
               const void* const* inputs, void* const* outputs,
               void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    int mChannels;
    int mNumBottlenecks;
    std::string mNamespace;
};

class C2fEMAPluginCreator : public IPluginCreator {
public:
    C2fEMAPluginCreator();
    ~C2fEMAPluginCreator() override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;
    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // C2F_EMA_PLUGIN_H


