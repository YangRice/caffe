// wincaffe.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "wincaffe.h"
#include <iostream>
#include <memory>

using namespace caffe;
using namespace cv;

class CaffeInit
{
public:
	CaffeInit()
	{
		google::InitGoogleLogging({ "wincaffe.dll" });
	}
};

class MeanNet
{
public:
	Net<float> *net;
	Blob<float> mean;
	MeanNet(const std::string &param_file) : net(new Net<float>(param_file, TEST)) {}
	~MeanNet()
	{
		delete net;
	}
};

inline void Mat2Blob(const Mat& img, Blob<float> &blob, const float scale = 1.0)
{
	vector<Mat> splitted;
	int channelSize = blob.width() * blob.height();
	float *data = blob.mutable_cpu_data();

	split(img, splitted);
	for (auto i = 0; i < blob.channels(); i++)
	{
		Mat channelImg(blob.height(), blob.width(), CV_32FC1, data + channelSize * i);
		splitted[i].convertTo(channelImg, CV_32FC1, scale);
	}
}

// This is an example of an exported function.
CaffeNet NewClassificationNet(const char *netParam, const char *trainedModel, const char *meanFile)
{
	static std::unique_ptr<CaffeInit> caffeInit(new CaffeInit());

	MeanNet *meanNet = new MeanNet(netParam);
	meanNet->net->CopyTrainedLayersFrom(trainedModel);

	if (meanFile != nullptr)
	{
		BlobProto blob_proto;
		ReadProtoFromBinaryFile(meanFile, &blob_proto);
		meanNet->mean.FromProto(blob_proto);
	}

	return meanNet;
}

void SetCPUMode()
{
	Caffe::set_mode(Caffe::CPU);
}

void SetGPUMode(int deviceid)
{
	Caffe::SetDevice(deviceid);
	Caffe::set_mode(Caffe::GPU);
}

void GetInputSize(CaffeNet net, CaffeBlobSize *inputSize)
{
	Net<float> *caffenet = static_cast<MeanNet *>(net)->net;

	auto *blob = caffenet->input_blobs()[0];
	inputSize->channels = blob->channels();
	inputSize->height = blob->height();
	inputSize->width = blob->width();
}

void GetOutputSize(CaffeNet net, CaffeBlobSize *outputSize)
{
	Net<float> *caffenet = static_cast<MeanNet *>(net)->net;

	auto *blob = caffenet->output_blobs()[0];
	outputSize->channels = blob->channels();
	outputSize->height = blob->height();
	outputSize->width = blob->width();
}

void RunClassificationNet(CaffeNet net, BYTE *bmp, float *prob)
{
	Net<float> *caffenet = static_cast<MeanNet *>(net)->net;
	auto *inputBlob = caffenet->input_blobs()[0];
	auto *outputBlob = caffenet->output_blobs()[0];

	Mat img(inputBlob->height(), inputBlob->width(), CV_8UC3, bmp);
	//resize(img, img, Size(inputBlob->width(), inputBlob->height()));

	Mat2Blob(img, *inputBlob);
	Blob<float> &meanBlob = static_cast<MeanNet *>(net)->mean;
	if (meanBlob.count() > 0)
	{
		float *inputData = inputBlob->mutable_cpu_data();
		float *meanData = meanBlob.mutable_cpu_data();
		for (auto i = 0; i < inputBlob->count(); i++)
		{
			inputData[0] -= meanData[0];
		}
	}

	caffenet->Forward();
	const auto results = caffenet->output_blobs();
	memcpy(prob, results[0]->mutable_cpu_data(),
		outputBlob->channels() * outputBlob->height() * outputBlob->width() * sizeof(float));
}

void DeleteClassificationNet(CaffeNet net)
{
	delete net;
}
