// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"
#include "caffe/caffe.hpp"
#include <sys/stat.h>
#include <iostream>

#include "faster r-cnn\misc.hpp"
#include "faster r-cnn\proposal_layer.hpp"
#include "faster r-cnn\nms.hpp"

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files:
#include <windows.h>

#define DEBUG_FLAG	"wincaffe.debug"
#define DEBUG_IMAGE	"wincaffe.jpg"

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
	std::shared_ptr<caffe::Net<float>> net;
	caffe::Blob<float> mean;
	MeanNet(const std::string &param_file) : net(new caffe::Net<float>(param_file, caffe::TEST)) {}
};

class _FasterRCNNNet
{
public:
	std::shared_ptr<caffe::Net<float>> net;
	std::vector<int> labels;
	std::vector<float> scores;
	std::vector<caffe::Box<float>> boxes;
};



inline void Mat2Blob(const cv::Mat& img, caffe::Blob<float> &blob, const float scale = 1.0)
{
	std::vector<cv::Mat> splitted;
	int channelSize = blob.width() * blob.height();
	float *data = blob.mutable_cpu_data();

	split(img, splitted);
	for (auto i = 0; i < blob.channels(); i++)
	{
		cv::Mat channelImg(blob.height(), blob.width(), CV_32FC1, data + channelSize * i);
		splitted[i].convertTo(channelImg, CV_32FC1, scale);
	}
}

inline bool FileExist(const std::string& name)
{
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

inline bool DebugFlagEnabled()
{
	static bool debugFlag = FileExist(DEBUG_FLAG);
	return debugFlag;
}