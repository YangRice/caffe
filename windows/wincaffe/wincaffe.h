#pragma once

// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the FACEAGEDLL_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// WINCAFFE_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef WINCAFFE_EXPORTS
#define WINCAFFE_API __declspec(dllexport)
#else
#define WINCAFFE_API __declspec(dllimport)
#endif

#include <opencv2\opencv.hpp>

typedef void *CaffeNet;
typedef void *FasterRCNNNet;
typedef void *FCNNet;

struct CaffeBlobSize
{
	int channels, width, height;
};

struct FasterRCNNBox
{
	float x1, y1, x2, y2;
};

WINCAFFE_API CaffeNet NewClassificationNet(const char *netParam, const char *trainedModel, const char *meanFile);
WINCAFFE_API void GetClassificationNetInputSize(CaffeNet net, CaffeBlobSize *inputSize);
WINCAFFE_API void GetClassificationNetOutputSize(CaffeNet net, CaffeBlobSize *outputSize);
WINCAFFE_API void RunClassificationNet(CaffeNet net, unsigned char *img, float *prob);
WINCAFFE_API void DeleteClassificationNet(CaffeNet net);

WINCAFFE_API FasterRCNNNet NewFasterRCNNNet(const char *netParam, const char *trainedModel);
WINCAFFE_API void GetFasterRCNNNetBlobSize(FasterRCNNNet net, CaffeBlobSize *blobSize, const char *blobName);
WINCAFFE_API int RunFasterRCNNNet(FasterRCNNNet net, unsigned char *img, int width, int height, float nms_threshold);
WINCAFFE_API bool GetFasterRCNNScoreAndBox(FasterRCNNNet net, int *labels, float *scores, FasterRCNNBox *boxes, int count);
WINCAFFE_API void DeleteFasterRCNNNet(FasterRCNNNet net);

WINCAFFE_API FCNNet NewFCNNet(const char *netParam, const char *trainedModel, const char *meanFile);
WINCAFFE_API void GetFCNNetBlobSize(FCNNet net, CaffeBlobSize *blobSize, const char *blobName);
WINCAFFE_API void RunFCNNet(FCNNet net, unsigned char *img, int width, int height);
WINCAFFE_API bool GetFCNNetProbabilities(FCNNet net, int label, float *heatmap, int width, int height);
WINCAFFE_API bool GetFCNNetSegmentation(FCNNet net, int *map, int width, int height);
WINCAFFE_API void DeleteFCNNet(FCNNet net);

WINCAFFE_API void SetCPUMode();
WINCAFFE_API void SetGPUMode(int deviceid);
