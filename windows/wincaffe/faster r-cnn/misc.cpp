#include "misc.hpp"
#include <vector>

#include <iostream>

namespace caffe {

using std::max;
using std::min;
using std::floor;
using std::ceil;

std::vector<int> TESTSCALES(1, 600);
bool TEST_HAS_RPN = true;
bool TEST_BBOX_REG = true;
int TEST_MAX_SIZE = 1000;

template <typename Dtype>
std::vector<Box<Dtype> > _mkanchors(const std::vector<Dtype>& ws, 
                                    const std::vector<Dtype>& hs, 
                                    Dtype x_ctr, Dtype y_ctr)
{
  int len = ws.size();
  std::vector<Box<Dtype> > boxes;
  for(int i=0; i<len; i++)
  {
    Box<Dtype> box( x_ctr - 0.5 * (ws[i] - 1),
                    y_ctr - 0.5 * (hs[i] - 1),
                    x_ctr + 0.5 * (ws[i] - 1),
                    y_ctr + 0.5 * (hs[i] - 1)); 
    
    boxes.push_back(box);
  }
  return boxes;
}

template 
std::vector<Box<double> > _mkanchors<double>(const std::vector<double>& ws, 
                                             const std::vector<double>& hs, 
                                             double x_ctr, double y_ctr);
template 
std::vector<Box<float> > _mkanchors<float>(const std::vector<float>& ws, 
                                           const std::vector<float>& hs, 
                                           float x_ctr, float y_ctr);

template <typename Dtype>
std::vector<Box<Dtype> > _ratio_enum(const Box<Dtype> & box,
                                     const std::vector<Dtype>& ratios)
{
  Dtype w     = box.x2 - box.x1 + 1;
  Dtype h     = box.y2 - box.y1 + 1;
  Dtype x_ctr = box.x1 + 0.5 * (w - 1);
  Dtype y_ctr = box.y1 + 0.5 * (h - 1);
  Dtype size  = w * h;

  int len = ratios.size();
  std::vector<Dtype> size_ratios( len );
  std::vector<Dtype> ws( len );
  std::vector<Dtype> hs( len );
  for(int i=0; i<len; i++)
  {
    size_ratios[i] = size / ratios[i];
    ws[i] = round( sqrt(size_ratios[i]) );
    hs[i] = round( ws[i] * ratios[i] );
  }

  std::vector<Box<Dtype> > anchors = _mkanchors(ws, hs, x_ctr, y_ctr);
  return anchors; 
}

template 
std::vector<Box<double> > _ratio_enum<double>(const Box<double> & box,
                                              const std::vector<double>& ratios);
template 
std::vector<Box<float> > _ratio_enum<float>(const Box<float> & box,
                                            const std::vector<float>& ratios);

template <typename Dtype>
std::vector<Box<Dtype> > _scale_enum(const Box<Dtype> & box,
                                     const std::vector<int>& scales)
{
  Dtype w     = box.x2 - box.x1 + 1;
  Dtype h     = box.y2 - box.y1 + 1;
  Dtype x_ctr = box.x1 + 0.5 * (w - 1);
  Dtype y_ctr = box.y1 + 0.5 * (h - 1);

  int len = scales.size();
  std::vector<Dtype> ws( len );
  std::vector<Dtype> hs( len );
  for(int i=0; i<len; i++)
  {
    ws[i] = w * scales[i];
    hs[i] = h * scales[i];
  }

  std::vector<Box<Dtype> > anchors = _mkanchors(ws, hs, x_ctr, y_ctr);
  return anchors; 
}

template
std::vector<Box<double> > _scale_enum<double>(const Box<double> & box,
                                              const std::vector<int>& scales);
template
std::vector<Box<float> > _scale_enum<float>(const Box<float> & box,
                                            const std::vector<int>& scales);

template <typename Dtype>
std::vector<Box<Dtype> > generate_anchors(const int& base_size,
                                          const std::vector<Dtype>& ratios, 
                                          const std::vector<int>& scales )
{
  Box<Dtype> base_anchor( 0, 0, base_size-1, base_size-1 );
  std::vector<Box<Dtype> > ratio_anchors = _ratio_enum<Dtype>(base_anchor, ratios);

  std::vector<Box<Dtype> > anchors;
  for(int i=0; i<ratio_anchors.size(); i++)
  {
    std::vector<Box<Dtype> > anchors_ = _scale_enum<Dtype>( ratio_anchors[i], scales );
    anchors.insert( anchors.end(), anchors_.begin(), anchors_.end() );
  }
  return anchors;
}

template 
std::vector<Box<double> > generate_anchors<double>(const int& base_size,
                                                   const std::vector<double>& ratios, 
                                                   const std::vector<int>& scales );
template 
std::vector<Box<float> > generate_anchors<float>(const int& base_size,
                                                 const std::vector<float>& ratios, 
                                                 const std::vector<int>& scales );

template <typename Dtype>
std::vector<int> _filter_boxes( const std::vector<Box<Dtype> >& boxes, Dtype min_size )
{
  std::vector<int> keep_ind;
  for(int i=0; i<boxes.size(); i++)
  {
     Dtype w  = boxes[i].x2 - boxes[i].x1 +1;
     Dtype h  = boxes[i].y2 - boxes[i].y1 +1;
     if(w>=(Dtype)min_size && h>=(Dtype)min_size)
       keep_ind.push_back( i );
  }
  return keep_ind;
}

template
std::vector<int> _filter_boxes<double>( const std::vector<Box<double> >& boxes, double min_size );
template
std::vector<int> _filter_boxes<float>( const std::vector<Box<float> >& boxes, float min_size );

template <typename Dtype>
std::vector<Box<Dtype> > bbox_transform_inv(const std::vector<Box<Dtype> >& boxes, Blob<Dtype>* deltas)
{
  int deltas_w = deltas->width();
  int deltas_h = deltas->height();
  int deltas_c = deltas->channels();
  auto *pdeltas = deltas->mutable_cpu_data();
  auto deltas_step = deltas->offset(0, 1, 0, 0) - deltas->offset(0, 0, 0, 0);

  std::vector<Box<Dtype> > pred_boxes;
  pred_boxes.reserve(deltas_w * deltas_h * deltas_c >> 2);

  if(deltas->channels()%4 != 0)
    LOG(ERROR)<<"deltas channels should be the multiple of 4";

  for(int k=0; k<deltas_h; k++)
    for(int l=0; l<deltas_w; l++)
      for(int j=0; j<deltas_c; j=j+4)
      {
        int i     = ( k*deltas_w + l) * deltas_c/4 + j/4;
        Dtype wid   = boxes[i].x2 - boxes[i].x1 + 1.0;
        Dtype hei   = boxes[i].y2 - boxes[i].y1 + 1.0;
        Dtype ctr_x = boxes[i].x1 + 0.5 * wid;
        Dtype ctr_y = boxes[i].y1 + 0.5 * hei;

		auto ideltas = deltas->offset(0, j, k, l);
		Dtype dx = pdeltas[ideltas]; ideltas += deltas_step;
		Dtype dy = pdeltas[ideltas]; ideltas += deltas_step;
		Dtype dw = pdeltas[ideltas]; ideltas += deltas_step;
		Dtype dh = pdeltas[ideltas];

        Dtype pred_ctr_x = dx*wid + ctr_x;
        Dtype pred_ctr_y = dy*hei + ctr_y;
        Dtype pred_w  = exp(dw)*wid;
		Dtype pred_h = exp(dh)*hei;

        Dtype pred_x1 = pred_ctr_x - 0.5*pred_w;
        Dtype pred_y1 = pred_ctr_y - 0.5*pred_h;
        Dtype pred_x2 = pred_ctr_x + 0.5*pred_w;
		Dtype pred_y2 = pred_ctr_y + 0.5*pred_h;

        pred_boxes.push_back(Box<Dtype>( pred_x1, pred_y1, pred_x2, pred_y2) );
	  }
  return pred_boxes;
}

template 
std::vector<Box<double> > bbox_transform_inv<double>(const std::vector<Box<double> >& boxes, Blob<double>* deltas);
template 
std::vector<Box<float> > bbox_transform_inv<float>(const std::vector<Box<float> >& boxes, Blob<float>* deltas);


cv::Scalar_<float> PIXEL_MEANS(102.9801, 115.9465, 122.7717);

template<typename Dtype>
shared_ptr<Blob<Dtype> > _get_image_blob(const cv::Mat& im, 
                                         std::vector<Dtype>& im_scale_factors,
										 shared_ptr<Blob<Dtype>> &blob)
{
    cv::Mat im_orig;
    im.convertTo(im_orig, CV_32F); 
    im_orig = im_orig - PIXEL_MEANS;

    int im_size_min = min( im_orig.rows, im_orig.cols );
    int im_size_max = max( im_orig.rows, im_orig.cols );

    std::vector<cv::Mat> processed_ims;

    for(int i=0; i<TESTSCALES.size(); i++)
    {
        int target_size = TESTSCALES[i];
        Dtype im_scale = Dtype(target_size) / Dtype(im_size_min);
        // Prevent the biggest axis from being more than MAX_SIZE
        if ( round(im_scale * im_size_max) > TEST_MAX_SIZE )
            im_scale = Dtype(TEST_MAX_SIZE) / Dtype(im_size_max);

        cv::Mat im_resized;
        cv::resize(im_orig, im_resized, cv::Size(), im_scale, im_scale, cv::INTER_LINEAR);
        im_scale_factors.push_back(im_scale);
        processed_ims.push_back(im_resized);
    }

    // Create a blob to hold the input images
	return im_list_to_blob<Dtype>(processed_ims, blob);
}

template
shared_ptr<Blob<double> > _get_image_blob<double>(const cv::Mat& im, 
                                                  std::vector<double>& im_scale_factors,
												  shared_ptr<Blob<double>> &blob);
template
shared_ptr<Blob<float> > _get_image_blob<float>(const cv::Mat& im, 
                                                std::vector<float>& im_scale_factors,
												shared_ptr<Blob<float>> &blob);


template<typename Dtype>
shared_ptr<Blob<Dtype> > im_detect(  shared_ptr<Net<Dtype> > net, 
                   const cv::Mat& im, std::vector<Box<Dtype> >& boxes,
                   std::vector<Box<Dtype> >& pred_boxes)
{
	shared_ptr<Blob<Dtype> > blob_data = net->blob_by_name("data");
	std::vector<Dtype> im_scales;
	blob_data = _get_image_blob<Dtype>(im, im_scales, blob_data);

    if(TEST_HAS_RPN)
    {
        std::vector<int> im_info_shape(2);
        im_info_shape[0] = 1;
        im_info_shape[1] = 3;
        shared_ptr<Blob<Dtype> > blob_im_info = net->blob_by_name("im_info");
        blob_im_info->Reshape( im_info_shape ) ;
        blob_im_info->mutable_cpu_data()[blob_im_info->offset(0,0)] = blob_data->height();
        blob_im_info->mutable_cpu_data()[blob_im_info->offset(0,1)] = blob_data->width();
        blob_im_info->mutable_cpu_data()[blob_im_info->offset(0,2)] = im_scales[0];
    }

	//net_forward<Dtype>(net, std::vector<Blob<Dtype> *>(), "", "", blob_data);
	net->Forward();
    
    shared_ptr<Blob<Dtype> > rois;
    
    if( TEST_HAS_RPN )
	{
        rois = net->blob_by_name("rois"); 
        // unscale back to raw image space
        for(int i=0; i<rois->num(); i++)
        {
            Dtype x1 = rois->cpu_data()[ rois->offset(i,1,0,0) ] / im_scales[0]; 
            Dtype y1 = rois->cpu_data()[ rois->offset(i,2,0,0) ] / im_scales[0]; 
            Dtype x2 = rois->cpu_data()[ rois->offset(i,3,0,0) ] / im_scales[0]; 
            Dtype y2 = rois->cpu_data()[ rois->offset(i,4,0,0) ] / im_scales[0]; 

            Box<Dtype> box(x1, y1, x2, y2);
            boxes.push_back(box);
		}
    }
    shared_ptr<Blob<Dtype> > scores = net->blob_by_name("cls_prob");

    if(TEST_BBOX_REG)
	{
        // Apply bounding-box regression deltas
        shared_ptr<Blob<Dtype> > box_deltas = net->blob_by_name("bbox_pred");
        pred_boxes = bbox_transform_inv1<Dtype>(boxes, box_deltas);
		std::for_each(pred_boxes.begin(), pred_boxes.end(), [&](Box<Dtype> &b)
		{
			b.x1 = max(min(b.x1, (Dtype)im.cols - 1), (Dtype)0);
			b.y1 = max(min(b.y1, (Dtype)im.rows - 1), (Dtype)0);
			b.x2 = max(min(b.x2, (Dtype)im.cols - 1), (Dtype)0);
			b.y2 = max(min(b.y2, (Dtype)im.rows - 1), (Dtype)0);
		});
    }

    return scores;
}

template
shared_ptr<Blob<double> > im_detect(  shared_ptr<Net<double> > net, 
                   const cv::Mat& im, std::vector<Box<double> >& boxes,
                   std::vector<Box<double> >& pred_boxes);
template
shared_ptr<Blob<float> > im_detect(  shared_ptr<Net<float> > net, 
                   const cv::Mat& im, std::vector<Box<float> >& boxes,
                   std::vector<Box<float> >& pred_boxes);

template<typename Dtype>
shared_ptr<Blob<Dtype> > im_list_to_blob(const std::vector<cv::Mat>& ims, shared_ptr<Blob<Dtype>> &blob)
{
    int width_max = 0, height_max = 0;
    int num_images = ims.size();

    for(int i=0; i<num_images; i++)
    {
        width_max  = max( ims[i].cols, width_max  );
        height_max = max( ims[i].rows, height_max );
    }

    //shared_ptr<Blob<Dtype> > blob( new Blob<Dtype>( num_images, 3, height_max, width_max) ); 
	std::vector<int> data_shape;
	data_shape.push_back(num_images);
	data_shape.push_back(3);
	data_shape.push_back(height_max);
	data_shape.push_back(width_max);
	blob->Reshape(data_shape);

	for (int i = 0; i < num_images; i++)
	{
		vector<cv::Mat> bgr;
		cv::split(ims[i], bgr); 
		for (int j = 0; j < 3; j++)
		{
			if (ims[i].cols == width_max)
				memcpy(blob->mutable_cpu_data() + blob->offset(i, j, 0, 0), bgr[j].ptr<Dtype>(), ims[i].rows * ims[i].cols * sizeof(Dtype));
			else
			{
				auto *p = bgr[j].ptr<Dtype>();
				for (int k = 0; k < ims[i].rows; k++, p += ims[i].cols)
				{
					memcpy(blob->mutable_cpu_data() + blob->offset(i, j, k, 0), p, ims[i].cols * sizeof(Dtype));
				}
			}
		}
	}
    return blob;
}

template
shared_ptr<Blob<double> > im_list_to_blob<double>(const std::vector<cv::Mat>& ims, shared_ptr<Blob<double>> &blob);
template
shared_ptr<Blob<float> > im_list_to_blob<float>(const std::vector<cv::Mat>& ims, shared_ptr<Blob<float>> &blob);

template <typename Dtype>
std::vector<Box<Dtype> > bbox_transform_inv1(const std::vector<Box<Dtype> >& boxes, 
                                             shared_ptr<Blob<Dtype> > deltas)
{
  int deltas_n = deltas->num();
  int deltas_c = deltas->channels();
  int deltas_w = deltas->width();
  int deltas_h = deltas->height();
  auto *pdeltas = deltas->mutable_cpu_data();
  auto deltas_step = deltas->offset(0, 1, 0, 0) - deltas->offset(0, 0, 0, 0);

  std::vector<Box<Dtype> > pred_boxes;

  if(deltas->channels()%4 != 0)
    LOG(ERROR)<<"deltas channels should be the multiple of 4";

  pred_boxes.reserve(deltas_n * deltas_c >> 2);
  for(int i=0; i<deltas_n; i++)
  {
    Dtype wid   = boxes[i].x2 - boxes[i].x1 + 1.0;
    Dtype hei   = boxes[i].y2 - boxes[i].y1 + 1.0;
    Dtype ctr_x = boxes[i].x1 + 0.5 * wid;
    Dtype ctr_y = boxes[i].y1 + 0.5 * hei;

    for(int j=0; j<deltas_c; j=j+4)
	{
		auto ideltas = deltas->offset(i, j, 0, 0);
		Dtype dx = pdeltas[ideltas]; ideltas += deltas_step;
		Dtype dy = pdeltas[ideltas]; ideltas += deltas_step;
		Dtype dw = pdeltas[ideltas]; ideltas += deltas_step;
		Dtype dh = pdeltas[ideltas];

        Dtype pred_ctr_x = dx*wid + ctr_x;
        Dtype pred_ctr_y = dy*hei + ctr_y;
        Dtype pred_w  = exp(dw)*wid;
        Dtype pred_h  = exp(dh)*hei;

        Dtype pred_x1 = pred_ctr_x - 0.5*pred_w;
        Dtype pred_y1 = pred_ctr_y - 0.5*pred_h;
        Dtype pred_x2 = pred_ctr_x + 0.5*pred_w;
        Dtype pred_y2 = pred_ctr_y + 0.5*pred_h;
        pred_boxes.push_back(Box<Dtype>( pred_x1, pred_y1, pred_x2, pred_y2) );
    }
  }
  return pred_boxes;
}

template 
std::vector<Box<double> > bbox_transform_inv1(const std::vector<Box<double> >& boxes, 
                                              shared_ptr<Blob<double> > deltas);
template
std::vector<Box<float> > bbox_transform_inv1(const std::vector<Box<float> >& boxes, 
                                             shared_ptr<Blob<float> > deltas);

}  // namespace caffe
