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

#
#include <opencv2\opencv.hpp>

typedef void *CaffeNet;
typedef void *FasterRCNNNet;

struct CaffeBlobSize
{
	int channels, width, height;
};

WINCAFFE_API CaffeNet NewClassificationNet(const char *netParam, const char *trainedModel);
WINCAFFE_API void GetInputSize(CaffeNet net, CaffeBlobSize *inputSize);
WINCAFFE_API void GetOutputSize(CaffeNet net, CaffeBlobSize *outputSize);
WINCAFFE_API void RunClassificationNet(CaffeNet net, unsigned char *img, float *prob);
WINCAFFE_API void DeleteClassificationNet(CaffeNet net);

WINCAFFE_API void SetCPUMode();
WINCAFFE_API void SetGPUMode(int deviceid);
