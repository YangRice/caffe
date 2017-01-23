#pragma once

#include <vector>
#include "caffe/blob.hpp"

namespace caffe
{
	template <typename Dtype>
	void nms(std::vector<std::pair<Dtype, Box<Dtype> *>>& score_and_box, const Dtype& nms_thresh,
		const int max_size, const int min_size)
	{
		for (int i = 0; i < score_and_box.size() && i < min_size; i++)
		{
			Dtype score = score_and_box[i].first;
			Box<Dtype> &box1 = *score_and_box[i].second;
			Dtype area1 = (box1.x2 - box1.x1 + 1) * (box1.y2 - box1.y1 + 1);

			auto end = score_and_box.size() > max_size ? score_and_box.begin() + max_size : score_and_box.end();
			auto removed = std::remove_if(score_and_box.begin() + i + 1, end,
				[&](const std::pair<Dtype, Box<Dtype> *> &sb)
			{
				Box<Dtype> box2 = *sb.second;
				Dtype inter_w = max(Dtype(0.0), min(box1.x2, box2.x2) - max(box1.x1, box2.x1) + 1);
				Dtype inter_h = max(Dtype(0.0), min(box1.y2, box2.y2) - max(box1.y1, box2.y1) + 1);
				Dtype inter = inter_w * inter_h;
				Dtype area2 = (box2.x2 - box2.x1 + 1) * (box2.y2 - box2.y1 + 1);
				Dtype ovr = inter / (area1 + area2 - inter);
				return ovr >= nms_thresh;
			});
			score_and_box.erase(removed, score_and_box.end());
		}
	}

	template
		void nms(std::vector<std::pair<double, Box<double> *>>& score_and_box, const double& nms_thresh,
		const int max_size, const int min_size);

	template
		void nms(std::vector<std::pair<float, Box<float> *>>& score_and_box, const float& nms_thresh,
		const int max_size, const int min_size);

	template <typename Dtype>
	void nms(std::vector<std::pair<Dtype, Box<Dtype> *>>& score_and_box, const Dtype& nms_thresh)
	{
		nms(score_and_box, nms_thresh, score_and_box.size(), score_and_box.size());
	}

	template
		void nms(std::vector<std::pair<double, Box<double> *>>& score_and_box, const double& nms_thresh);

	template
		void nms(std::vector<std::pair<float, Box<float> *>>& score_and_box, const float& nms_thresh);
}