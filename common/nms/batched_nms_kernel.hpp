//
// Created by Administrator on 9/14/2022.
//

#ifndef TRTBATCHEDNMS_BATCHED_NMS_KERNEL_HPP
#define TRTBATCHEDNMS_BATCHED_NMS_KERNEL_HPP

#include "cuda_runtime_api.h"
#include "kernel.h"

pluginStatus_t nmsInference(cudaStream_t stream, const int N, const int perBatchBoxesSize,
                            const int perBatchScoresSize, const bool shareLocation,
                            const int backgroundLabelId, const int numPredsPerClass,
                            const int numClasses, const int topK, const int keepTopK,
                            const float scoreThreshold, const float iouThreshold,
                            const DataType DT_BBOX, const void* locData, const DataType DT_SCORE,
                            const void* confData, void* nmsedDets, void* nmsedLabels,
                            void* nmsedIndex, void* workspace, bool isNormalized, bool confSigmoid,
                            bool clipBoxes, bool rotated = false);

#endif //TRTBATCHEDNMS_BATCHED_NMS_KERNEL_HPP
