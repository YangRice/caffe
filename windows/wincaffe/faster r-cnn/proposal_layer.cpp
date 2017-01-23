// ------------------------------------------------------------------
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaohua Wan
// ------------------------------------------------------------------

#include <cfloat>
#include <chrono>

#include "proposal_layer.hpp"
#include "nms.hpp"

#define RPN_PRE_NMS_TOP_N  6000
#define RPN_POST_NMS_TOP_N 300
#define RPN_NMS_THRESH     0.7
#define RPN_MIN_SIZE       16

namespace caffe {

	template <typename Dtype>
	void ProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		ProposalParameter proposal_param = this->layer_param_.proposal_param();
		CHECK_GT(proposal_param.feat_stride(), 0)
			<< "feat_stride must be > 0";
		_feat_stride = proposal_param.feat_stride();

		int base_size = 16;
		std::vector<Dtype> ratios(3);
		ratios[0] = 0.5;
		ratios[1] = 1.0;
		ratios[2] = 2.0;
		std::vector<int> scales(3);
		scales[0] = 8;
		scales[1] = 16;
		scales[2] = 32;
		_anchors = generate_anchors<Dtype>(base_size, ratios, scales);
		_num_anchors = _anchors.size();

		std::vector<int> shape(2);
		shape[0] = 1;
		shape[1] = 5;
		top[0]->Reshape(shape);
		if (top.size() > 1)
			top[1]->Reshape(1, 1, 1, 1);
	}

	template <typename Dtype>
	void ProposalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	}

	template <typename Dtype>
	void ProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		auto start = chrono::steady_clock::now();

		int pre_nms_topN = RPN_PRE_NMS_TOP_N;
		int post_nms_topN = RPN_POST_NMS_TOP_N;
		Dtype nms_thresh = RPN_NMS_THRESH;
		int min_size = RPN_MIN_SIZE;

		// the first set of _num_anchors channels are bg probs
		// the second set are the fg probs, which we want
		Blob<Dtype>* scores = bottom[0];
		Blob<Dtype>* bbox_deltas = bottom[1];
		Blob<Dtype>* im_info = bottom[2];

		// 1. Generate proposals from bbox deltas and shifted anchors
		int height = scores->height();
		int width = scores->width();

		// Enumerate all shifts
		std::vector<int> shift_x(width), shift_y(height);
		for (int i = 0; i < width; i++)
			shift_x[i] = i*_feat_stride;
		for (int j = 0; j < height; j++)
			shift_y[j] = j*_feat_stride;

		// Enumerate all shifted anchors:
		//
		// add A anchors (1, A, 4) to
		// cell K shifts (K, 1, 4) to get
		// shift anchors (K, A, 4)
		// reshape to (K*A, 4) shifted anchors
		int A = _num_anchors;
		int K = width*height;
		std::vector<Box<Dtype> > anchors;
		for (int j = 0; j < height; j++)
			for (int k = 0; k < width; k++)
				for (int i = 0; i < A; i++)
				{
					Box<Dtype> box(_anchors[i], shift_x[k], shift_y[j]);
					anchors.push_back(box);
				}

		typedef std::pair<Dtype, Box<Dtype> *> ScoreAndBox;
		std::vector<ScoreAndBox> score_and_box;

		// Convert anchors into proposals via bbox transformations
		std::vector<Box<Dtype> > proposals = bbox_transform_inv(anchors, bbox_deltas);
		std::vector<Dtype> scores_;
		int n = scores->num();
		int c = scores->channels();
		int h = scores->height();
		int w = scores->width();
		int iproposals = 0;
		auto pscores = scores->cpu_data();
		for (int i = 0; i < n; i++)
			for (int k = 0; k < h; k++)
				for (int l = 0; l < w; l++)
					for (int j = c / 2; j < c; j++)
					{
						int idx = scores->offset(i, j, k, l);
						score_and_box.push_back(ScoreAndBox{ scores->cpu_data()[idx], &proposals[iproposals++] });
					}

		// 2. clip predicted boxes to image
		Dtype im_rows = im_info->cpu_data()[im_info->offset(0, 0)];
		Dtype im_cols = im_info->cpu_data()[im_info->offset(0, 1)];
		Dtype im_scale = im_info->cpu_data()[im_info->offset(0, 2)];
		std::for_each(score_and_box.begin(), score_and_box.end(), [&](ScoreAndBox &sb)
		{
			sb.second->x1 = max(min(sb.second->x1, (Dtype)im_cols - 1), (Dtype)0);
			sb.second->y1 = max(min(sb.second->y1, (Dtype)im_rows - 1), (Dtype)0);
			sb.second->x2 = max(min(sb.second->x2, (Dtype)im_cols - 1), (Dtype)0);
			sb.second->y2 = max(min(sb.second->y2, (Dtype)im_rows - 1), (Dtype)0);
		});

		// 3. remove predicted boxes with either height or width < threshold
		// (NOTE: convert min_size to input image scale stored in im_info[2])
		min_size *= im_scale;
		auto removed = std::remove_if(score_and_box.begin(), score_and_box.end(), [&](ScoreAndBox &sb)
		{
			Dtype w = sb.second->x2 - sb.second->x1 + 1;
			Dtype h = sb.second->y2 - sb.second->y1 + 1;
			return (w < (Dtype)min_size && h < (Dtype)min_size);
		});
		score_and_box.erase(removed, score_and_box.end());

		// 4. sort all (proposal, score) pairs by score from highest to lowest
		std::sort(score_and_box.begin(), score_and_box.end(), [](const ScoreAndBox &i, const ScoreAndBox &j)
		{
			return i.first > j.first;
		});

		// 5. take top pre_nms_topN (e.g. 6000)
		// 6. apply nms (e.g. threshold = 0.7)
		// 7. take after_nms_topN (e.g. 300)
		// 8. return the top proposals (-> RoIs top)
		nms(score_and_box, (Dtype)RPN_NMS_THRESH, pre_nms_topN, post_nms_topN);

		// Output rois blob
		// Our RPN implementation only supports a single input image, so all
		// batch inds are 0
		std::vector<int> top_shape(2);
		top_shape[0] = score_and_box.size();
		top_shape[1] = 4 + 1;
		top[0]->Reshape(top_shape);
		auto ptop = top[0]->mutable_cpu_data();
		for (int i = 0; i < top_shape[0]; i++)
		{
			(*ptop++) = 0;
			(*ptop++) = score_and_box[i].second->x1;
			(*ptop++) = score_and_box[i].second->y1;
			(*ptop++) = score_and_box[i].second->x2;
			(*ptop++) = score_and_box[i].second->y2;
		}

		auto duration = chrono::steady_clock::now() - start;
		//LOG(INFO) << "ProposalLayer " << chrono::duration_cast<chrono::milliseconds>(duration).count() << "ms" << endl;
	}

	template <typename Dtype>
	void ProposalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		NOT_IMPLEMENTED;
	}


#ifdef CPU_ONLY
	STUB_GPU(ProposalLayer);
#endif

	INSTANTIATE_CLASS(ProposalLayer);
	REGISTER_LAYER_CLASS(Proposal);

}  // namespace caffe
