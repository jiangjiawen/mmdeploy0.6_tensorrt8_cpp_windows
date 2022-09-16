# mmdeploy, only tensorrt env

A windows tensorrt env uses mmdeploy trained models, normal mask-rcnn as the example test.

## Usage

The following are the steps to deploy a mmdeploy mask-rcnn models. The most important feature of this project is that I copy out mmdeploy plugins and the codes are all tensorrt-based. The origin mmdeploy env is packaged too much, So hard to learn.

### Prerequisites

windows env, vs2019 and clion2022.2, cuda10.2 and cudnn8.4.1.50_cuda10.2

* TensorRT-8.4.3.1
* Opencv460
* mmdeploy 0.6.0
* mmdetection 2.25.0
* mmcv 1.5.3

### ONNX 

The onnx file is generated by the tensorrt static config. And need fold constants

```
$ python tools/onnx2tensorrt.py configs/mmdet/instance-seg/instance-seg_tensorrt_static.....

$ polygraphy surgeon sanitize end2end.onnx --fold-constants -o end2end_folded.onnx
```

### mmdeploy tensorrt plugins

Normal mask-rcnn model only uses "TRTBatchedNMS" and "MMCVMultiLevelRoiAlign". So I create these two plugins.

## tensorrt engine

And now we need to convert the onnx file to binary tensorrt engine.

```
$ ..\TensorRT-8.4.3.1\bin\trtexec.exe --onnx=end2end_folded.onnx --saveEngine=res18_maskrcnn.engine --plugins=..\TensorRT-8.4.3.1\bin\trtbatchednms.dll --plugins=..\TensorRT-8.4.3.1\bin\mmcvmultilevelroialign.dll --explicitBatch --tacticSources=-cublasLt,+cublas
```
## mask rcnn test
I load the plugin dlls in the code.
```
void* handle_nms = LoadLibrary("trtbatchednms.dll");
void* handle_roi = LoadLibrary("mmcvmultilevelroialign.dll");
```

## Additional Attentions

* cmakelist is copy from tensorrtx yolov5, https://github.com/wang-xinyu/tensorrtx
* cmakelist should add NOMINMAX to avoid some mistakes on windows 10.
* static and shared library.
* cuh is copy from mmdeploy third party dir.

