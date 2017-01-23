#pragma once

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "misc.hpp"

namespace caffe {
/* ProposalLayer -  Outputs object detection proposals by applying
* estimated bounding-box transformations to a set of regular boxes
* (called "anchors").
*/
template <typename Dtype>
class ProposalLayer : public Layer<Dtype> {
public:
	explicit ProposalLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "Proposal"; }

	virtual inline int MinBottomBlobs() const { return 3; }
	virtual inline int MaxBottomBlobs() const { return 3; }
	virtual inline int MinTopBlobs() const { return 1; }
	virtual inline int MaxTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	//  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	//      const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	//  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	//      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int _feat_stride;
	std::vector<Box<Dtype> > _anchors;
	int _num_anchors;
};

}  // namespace caffe