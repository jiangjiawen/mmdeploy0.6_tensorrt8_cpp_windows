//
// Created by Administrator on 9/15/2022.
//

#ifndef MMDEPLOY_PLUGINS_MASKRCNN_H
#define MMDEPLOY_PLUGINS_MASKRCNN_H

#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include <opencv2/opencv.hpp>

#include <cuda_runtime_api.h>
#include "NvInferPlugin.h"
#include "NvInfer.h"
#include "logger.h"
#include "util.h"

#include <windows.h>

#include <chrono>

using namespace std::literals;

using sample::gLogError;
using sample::gLogInfo;

struct offset{
    int top;
    int left;
    float scala_factor;
};

class SampleMaskRCNN
{

public:
    explicit SampleMaskRCNN(const std::string& engineFilename);
    ~SampleMaskRCNN();

    bool InitInfo();
    bool infer(const std::string& input_filename, std::unique_ptr<int>& output_buffer_label, std::unique_ptr<float>& output_buffer_dets,
               std::unique_ptr<float>& output_buffer_masks);
    cv::Mat preprocessedImage(const cv::Mat& imageBGR);

    int32_t get_outputsize_labels();
    int32_t get_outputsize_dets();
    int32_t get_outputsize_masks();

    offset GetOffsetInfo();

private:

    int W = 1280;
    int H = 800;

    int O_W = 1280;
    int O_H = 1280;
    float scale_factor = 1.6;
    int top = 0;
    int left = 0;

    //convert get 100 result
    int res_count = 100;

    std::string mEngineFilename;                    //!< Filename of the serialized engine.

    //nvinfer1::Dims mInputDims;                      //!< The dimensions of the input to the network.
    //nvinfer1::Dims mOutputDims;                     //!< The dimensions of the output to the network.

    util::UniquePtr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    util::UniquePtr<nvinfer1::IExecutionContext> context;
    int32_t input_idx;
    nvinfer1::Dims4 input_dims;
    size_t input_size;

    int32_t output_idx_labels;
    nvinfer1::Dims output_dims_labels;
    size_t output_size_labels;

    int32_t output_idx_dets;
    nvinfer1::Dims output_dims_dets;
    size_t output_size_dets;

    int32_t output_idx_masks;
    nvinfer1::Dims output_dims_masks;
    size_t output_size_masks;

    void* input_mem{ nullptr };
    void* output_mem_labels{ nullptr };
    void* output_mem_dets{ nullptr };
    void* output_mem_masks{ nullptr };

    cudaStream_t stream;
};

inline offset SampleMaskRCNN::GetOffsetInfo() {
    offset off{};
    off.top = top;
    off.left = left;
    off.scala_factor = scale_factor;
    return off;
}

inline SampleMaskRCNN::SampleMaskRCNN(const std::string& engineFilename)
        : mEngineFilename(engineFilename)
        , mEngine(nullptr)
{
    //load plugins
    bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
    void* handle_nms = LoadLibrary("trtbatchednms.dll");
    void* handle_roi = LoadLibrary("mmcvmultilevelroialign.dll");
    if(handle_nms!=nullptr && handle_roi!= nullptr){
        std::cout << "load plugins success " <<  didInitPlugins << std::endl;
    }

    // De-serialize engine from file
    std::ifstream engineFile(engineFilename, std::ios::binary);
    if (engineFile.fail())
    {
        return;
    }

    engineFile.seekg(0, std::ifstream::end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    util::UniquePtr<nvinfer1::IRuntime> runtime{ nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()) };
    mEngine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    assert(mEngine.get() != nullptr);
}

inline SampleMaskRCNN::~SampleMaskRCNN()
{
    // Free CUDA resources
    cudaFree(input_mem);
    cudaFree(output_mem_labels);
    cudaFree(output_mem_dets);
    cudaFree(output_mem_masks);
}

inline int32_t SampleMaskRCNN::get_outputsize_labels()
{
    return this->output_size_labels;
}

inline int32_t SampleMaskRCNN::get_outputsize_dets()
{
    return this->output_size_dets;
}

inline int32_t SampleMaskRCNN::get_outputsize_masks()
{
    return this->output_size_masks;
}

inline cv::Mat SampleMaskRCNN::preprocessedImage(const cv::Mat& imageBGR) {
    this->O_W = imageBGR.cols;
    this->O_H = imageBGR.rows;

    float ratio_h = static_cast<float>(H) / static_cast<float>(O_H);
    float ratio_w = static_cast<float>(W) / static_cast<float>(O_W);
    scale_factor = std::min(ratio_h, ratio_w);

    int new_shape_w = std::round(O_W * scale_factor);
    int new_shape_h = std::round(O_H * scale_factor);

    float padw = (W - new_shape_w) / 2.;
    float padh = (H - new_shape_h) / 2.;

    top = std::round(padh - 0.1);
    int bottom = std::round(padh + 0.1);
    left = std::round(padw - 0.1);
    int right = std::round(padw + 0.1);

    cv::Mat middelImg;
    cv::resize(
            imageBGR, middelImg, cv::Size(new_shape_w, new_shape_h), 0, 0, cv::INTER_AREA);

    cv::Mat resized(cv::Size(W, H), imageBGR.type());
    cv::copyMakeBorder(middelImg,
                       resized,
                       top,
                       bottom,
                       left,
                       right,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(0));

    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32F, 1.0);

    cv::Mat channels[3];
    cv::split(floatImg, channels);
    channels[0] = (channels[0] - 103.53) / 1.0;
    channels[1] = (channels[1] - 116.28) / 1.0;
    channels[2] = (channels[2] - 123.675) / 1.0;
    cv::merge(channels, 3, floatImg);

    cv::Mat blob;
    cv::dnn::blobFromImage(floatImg, blob);

//    cv::Mat blob;
//    cv::dnn::blobFromImage(resized, blob, 1.0, cv::Size(W, H), cv::Scalar(103.53, 116.28, 123.675), false, false);

    return blob;
}

inline bool SampleMaskRCNN::InitInfo()
{

    context = util::UniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    input_idx = mEngine->getBindingIndex("input");
    if (input_idx == -1)
    {
        return false;
    }
    assert(mEngine->getBindingDataType(input_idx) == nvinfer1::DataType::kFLOAT);
    input_dims = nvinfer1::Dims4{ 1, 3, H, W };
    context->setBindingDimensions(input_idx, input_dims);
    input_size = util::getMemorySize(input_dims, sizeof(float));
    std::cout << "input size" << input_size << std::endl;

    //labels
    output_idx_labels = mEngine->getBindingIndex("labels");
    std::cout << "output_idx_labels is " << output_idx_labels << std::endl;
    //output_idx = mEngine->getBindingIndex("prob");
    if (output_idx_labels == -1)
    {
        return false;
    }
    assert(mEngine->getBindingDataType(output_idx_labels) == nvinfer1::DataType::kINT32);
    output_dims_labels = context->getBindingDimensions(output_idx_labels);
    output_size_labels = util::getMemorySize(output_dims_labels, sizeof(int32_t));
    std::cout << "output_size_labels size" <<output_size_labels << std::endl;

    //dets
    output_idx_dets = mEngine->getBindingIndex("dets");
    std::cout << "output_idx_dets is " << output_idx_dets << std::endl;
    //output_idx = mEngine->getBindingIndex("prob");
    if (output_idx_dets == -1)
    {
        return false;
    }
    assert(mEngine->getBindingDataType(output_idx_dets) == nvinfer1::DataType::kFLOAT);
    output_dims_dets = context->getBindingDimensions(output_idx_dets);
    output_size_dets = util::getMemorySize(output_dims_dets, sizeof(float));
    std::cout << "output_size_dets size" <<output_size_dets << std::endl;

    //masks
    output_idx_masks = mEngine->getBindingIndex("masks");
    std::cout << "output_idx_masks is " << output_idx_masks << std::endl;
    //output_idx = mEngine->getBindingIndex("prob");
    if (output_idx_masks == -1)
    {
        return false;
    }
    assert(mEngine->getBindingDataType(output_idx_masks) == nvinfer1::DataType::kFLOAT);
    output_dims_masks = context->getBindingDimensions(output_idx_masks);
    output_size_masks = util::getMemorySize(output_dims_masks, sizeof(int32_t));
    std::cout << "output_size_masks size" <<output_size_masks << std::endl;

    // Allocate CUDA memory for input and output bindings
    if (cudaMalloc(&input_mem, input_size) != cudaSuccess)
    {
        gLogError << "ERROR: input cuda memory allocation failed, size = " << input_size << " bytes" << std::endl;
        return false;
    }

    if (cudaMalloc(&output_mem_labels, output_size_labels) != cudaSuccess)
    {
        gLogError << "ERROR: output labels cuda memory allocation failed, size = " << output_size_labels << " bytes" << std::endl;
        return false;
    }

    if (cudaMalloc(&output_mem_dets, output_size_dets) != cudaSuccess)
    {
        gLogError << "ERROR: output dets cuda memory allocation failed, size = " << output_size_dets << " bytes" << std::endl;
        return false;
    }

    if (cudaMalloc(&output_mem_masks,output_size_masks) != cudaSuccess)
    {
        gLogError << "ERROR: output masks cuda memory allocation failed, size = " << output_size_masks << " bytes" << std::endl;
        return false;
    }

    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return false;
    }

}

inline bool SampleMaskRCNN::infer(const std::string& input_filename, std::unique_ptr<int>& output_buffer_label, std::unique_ptr<float>& output_buffer_dets, std::unique_ptr<float>& output_buffer_masks)
{

    cv::Mat img = cv::imread(input_filename);
    cv::Mat input_img_mat = preprocessedImage(img);

    if (cudaMemcpyAsync(input_mem, input_img_mat.data, input_size, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        gLogError << "ERROR: CUDA memory copy of input failed, size = " << input_size << " bytes" << std::endl;
        return false;
    }

    // Run TensorRT inference
    //attention the sort of the result
    void* bindings[] = { input_mem, output_mem_dets,output_mem_labels, output_mem_masks };
    bool status = context->enqueueV2(bindings, stream, nullptr);
    if (!status)
    {
        gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return false;
    }

    //auto output_buffer = std::unique_ptr<float>{ new float[output_size] };
    if (cudaMemcpyAsync(output_buffer_label.get(), output_mem_labels, output_size_labels, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        gLogError << "ERROR: CUDA memory copy of output failed, size = " << output_size_labels << " bytes" << std::endl;
        return false;
    }
    if (cudaMemcpyAsync(output_buffer_dets.get(), output_mem_dets, output_size_dets, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        gLogError << "ERROR: CUDA memory copy of output failed, size = " << output_size_dets << " bytes" << std::endl;
        return false;
    }
    if (cudaMemcpyAsync(output_buffer_masks.get(), output_mem_masks, output_size_masks, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        gLogError << "ERROR: CUDA memory copy of output failed, size = " << output_size_masks << " bytes" << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream);

    return true;
}


#endif //MMDEPLOY_PLUGINS_MASKRCNN_H
