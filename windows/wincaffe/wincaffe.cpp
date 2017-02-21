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
static std::unique_ptr<CaffeInit> caffeInit(new CaffeInit());

class MeanNet
{
public:
	std::shared_ptr<Net<float>> net;
	Blob<float> mean;
	MeanNet(const std::string &param_file) : net(new Net<float>(param_file, TEST)) {}
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
	MeanNet *meanNet = new MeanNet(netParam);
	meanNet->net->CopyTrainedLayersFrom(trainedModel);

	if (meanFile != nullptr)
	{
		BlobProto blob_proto;
		if (ReadProtoFromBinaryFile(meanFile, &blob_proto))
		{
			meanNet->mean.FromProto(blob_proto);
		}
		else
		{
			std::cerr << "[wincaffe.dll] Can't read meanfile: " << string(meanFile) << std::endl;
		}
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

void GetClassificationNetInputSize(CaffeNet net, CaffeBlobSize *inputSize)
{
	auto caffenet = static_cast<MeanNet *>(net)->net;

	auto *blob = caffenet->input_blobs()[0];
	inputSize->channels = blob->channels();
	inputSize->height = blob->height();
	inputSize->width = blob->width();
}

void GetClassificationNetOutputSize(CaffeNet net, CaffeBlobSize *outputSize)
{
	auto caffenet = static_cast<MeanNet *>(net)->net;

	auto *blob = caffenet->output_blobs()[0];
	outputSize->channels = blob->channels();
	outputSize->height = blob->height();
	outputSize->width = blob->width();
}

void RunClassificationNet(CaffeNet net, BYTE *bmp, float *prob)
{
	auto caffenet = static_cast<MeanNet *>(net)->net;
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
		for (auto i = 0; i < inputBlob->count(); i++, inputData++, meanData++)
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

class _FasterRCNNNet
{
public:
	std::shared_ptr<Net<float>> net;
	vector<int> labels;
	vector<float> scores;
	vector<Box<float>> boxes;
};

FasterRCNNNet NewFasterRCNNNet(const char *netParam, const char *trainedModel)
{
	_FasterRCNNNet *frcn = new _FasterRCNNNet();
	frcn->net = std::shared_ptr<Net<float>>(new Net<float>(netParam, TEST));
	frcn->net->CopyTrainedLayersFrom(trainedModel);
	return frcn;
}

void GetFasterRCNNNetBlobSize(FasterRCNNNet net, CaffeBlobSize *blobSize, const char *blobName)
{
	_FasterRCNNNet *frcn = static_cast<_FasterRCNNNet *>(net);
	auto caffenet = frcn->net;

	const auto blob = caffenet->blob_by_name(blobName);
	blobSize->channels = blob->channels();
	blobSize->height = blob->height();
	blobSize->width = blob->width();
}

int RunFasterRCNNNet(FasterRCNNNet net, uchar *bmp, int width, int height, float nms_threshold)
{
	_FasterRCNNNet *frcn = static_cast<_FasterRCNNNet *>(net);
	auto frcnnet = frcn->net;
	auto data = frcnnet->blob_by_name("data");
	auto info = frcnnet->blob_by_name("im_info");

	Mat im(height, width, CV_8UC3, bmp);
	data->Reshape(1, 3, height, width);
	Mat2Blob(im, *data);

	info->mutable_cpu_data()[0] = static_cast<float>(height);
	info->mutable_cpu_data()[1] = static_cast<float>(width);
	info->mutable_cpu_data()[2] = 1.0f;

	frcnnet->Forward();

	auto score = frcnnet->blob_by_name("cls_prob");
	auto rois = frcnnet->blob_by_name("rois");
	auto box_deltas = frcnnet->blob_by_name("bbox_pred");

	frcn->labels.reserve(score->count());
	frcn->boxes.reserve(score->count());
	frcn->scores.reserve(score->count());
	frcn->labels.resize(0);
	frcn->boxes.resize(0);
	frcn->scores.resize(0);
	static vector<Box<float>> boxes;
	boxes.reserve(rois->num());
	boxes.resize(0);

	// unscale back to raw image space
	for (int i = 0; i<rois->num(); i++)
	{
		const auto &x1 = rois->cpu_data()[rois->offset(i, 1, 0, 0)];
		const auto &y1 = rois->cpu_data()[rois->offset(i, 2, 0, 0)];
		const auto &x2 = rois->cpu_data()[rois->offset(i, 3, 0, 0)];
		const auto &y2 = rois->cpu_data()[rois->offset(i, 4, 0, 0)];

		Box<float> box(x1, y1, x2, y2);
		boxes.push_back(box);
	}

	// Apply bounding-box regression deltas
	boxes = bbox_transform_inv1(boxes, box_deltas);
	std::for_each(boxes.begin(), boxes.end(), [&](Box<float> &b)
	{
		b.x1 = max(min(b.x1, (float)width - 1), (float)0);
		b.y1 = max(min(b.y1, (float)height - 1), (float)0);
		b.x2 = max(min(b.x2, (float)width - 1), (float)0);
		b.y2 = max(min(b.y2, (float)height - 1), (float)0);
	});

	for (unsigned int i = 1; i < score->shape()[1]; i++)	// # of class
	{
		vector<pair<float, Box<float> *>> score_and_box;
		for (unsigned int j = 0; j < score->shape()[0]; j++)	// # of boxes per class
		{
			auto index = score->offset(j, i);
			float s = score->cpu_data()[index];
			Box<float> *b = &boxes[index];
			score_and_box.push_back(pair<float, Box<float>*>(s, b));
		}
		sort(score_and_box, [](const pair<float, Box<float> *> &i, const pair<float, Box<float> *> &j){ return i.first > j.first; });
		nms(score_and_box, nms_threshold);

		for (auto &sb : score_and_box)
		{
			frcn->labels.push_back(i);
			frcn->scores.push_back(sb.first);
			frcn->boxes.push_back(*sb.second);
		}
	}
	return frcn->scores.size();
}

bool GetFasterRCNNScoreAndBox(FasterRCNNNet net, int *labels, float *scores, FasterRCNNBox *boxes, int count)
{
	using namespace std;
	_FasterRCNNNet *frcn = static_cast<_FasterRCNNNet *>(net);
	if (frcn->labels.size() != count)
	{
		cerr << "frcn->labels.size(" << frcn->labels.size() << ") != " << count << endl;
		return false;
	}
	if (frcn->scores.size() != count)
	{
		cerr << "frcn->scores.size(" << frcn->scores.size() << ") != " << count << endl;
		return false;
	}
	if (frcn->boxes.size() != count)
	{
		cerr << "frcn->boxes.size(" << frcn->boxes.size() << ") != " << count << endl;
		return false;
	}

	memcpy(labels, &frcn->labels[0], count * sizeof(int));
	memcpy(scores, &frcn->scores[0], count * sizeof(float));
	for (int i = 0; i < count; i++)
	{
		Box<float> &box = frcn->boxes[i];
		boxes[i].x1 = box.x1;
		boxes[i].y1 = box.y1;
		boxes[i].x2 = box.x2;
		boxes[i].y2 = box.y2;
	}
	return true;
}

void DeleteFasterRCNNNet(FasterRCNNNet net)
{
	delete net;
}

FCNNet NewFCNNet(const char *netParam, const char *trainedModel, const char *meanFile)
{
	return NewClassificationNet(netParam, trainedModel, meanFile);
}

void GetFCNNetBlobSize(FCNNet net, CaffeBlobSize *blobSize, const char *blobName)
{
	auto caffenet = static_cast<MeanNet *>(net)->net;

	const auto blob = caffenet->blob_by_name(blobName);
	blobSize->channels = blob->channels();
	blobSize->height = blob->height();
	blobSize->width = blob->width();
}

void RunFCNNet(FCNNet net, unsigned char *img, int width, int height)
{
	auto caffenet = static_cast<MeanNet *>(net)->net;
	auto *inputBlob = caffenet->input_blobs()[0];
	auto *outputBlob = caffenet->output_blobs()[0];

	Mat im(height, width, CV_8UC3, img);
	inputBlob->Reshape(1, 3, height, width);
	Mat2Blob(im, *inputBlob);

	Blob<float> &meanBlob = static_cast<MeanNet *>(net)->mean;
	if (meanBlob.count() > 0)
	{
		float *inputData = inputBlob->mutable_cpu_data();
		float *meanData = meanBlob.mutable_cpu_data();
		for (auto i = 0; i < inputBlob->count(); i++, inputData++, meanData++)
		{
			inputData[0] -= meanData[0];
		}
	}

	caffenet->Forward();
}

bool GetFCNNetOutput(FCNNet net, int label, float *heatmap, int width, int height)
{
	auto caffenet = static_cast<MeanNet *>(net)->net;
	const auto blob = caffenet->output_blobs()[0];
	float *data = blob->mutable_cpu_data();
	int channelSize = blob->width() * blob->height();

	Mat result(blob->height(), blob->width(), CV_32FC1, data + channelSize * label);
	Mat output(height, width, CV_32FC1, heatmap);
	if (output.rows >= result.rows && output.cols >= result.cols)
	{
		for (int i = 0; i < result.rows; i++)
		{
			memcpy(output.data + i * output.step.p[0],
				result.data + i * result.step.p[0],
				result.step.p[0]);
		}
		return true;
	}
	else
	{
		std::cerr << "input size(" << output.size() << ") != output size(" << result.size() << ")" << std::endl;
		return false;
	}
}

void DeleteFCNNet(FCNNet net)
{
	delete net;
}