//
// Created by Administrator on 9/15/2022.
//

#ifndef MMDEPLOY_PLUGINS_TRT_MULTI_LEVEL_ROI_ALIGN_KERNEL_HPP
#define MMDEPLOY_PLUGINS_TRT_MULTI_LEVEL_ROI_ALIGN_KERNEL_HPP

#include <cuda_runtime.h>

template <typename T>
void multi_level_roi_align(T *output, const T *rois, int num_rois, const void *const *feats,
                           int num_feats, int n, int c, int *h, int *w, float *strides,
                           int aligned_height, int aligned_width, int pool_mode, int sample_num,
                           float roi_scale_factor, int finest_scale, bool aligned,
                           cudaStream_t stream);


#endif //MMDEPLOY_PLUGINS_TRT_MULTI_LEVEL_ROI_ALIGN_KERNEL_HPP