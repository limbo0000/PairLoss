#ifndef CAFFE_PAIRWISE_LOSS_LAYER_HPP_
#define CAFFE_PAIRWISE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe{

/** 
 * @brief Computes the pairwise loss 
 */  
template <typename Dtype>  
class PairwiseLossLayer : public LossLayer<Dtype> {  
 public:  
  explicit PairwiseLossLayer(const LayerParameter& param)  
      : LossLayer<Dtype>(param){}  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
  
  virtual inline int ExactNumBottomBlobs() const { return 5; }  
  virtual inline const char* type() const { return "PairwiseLoss"; }  

  virtual inline bool AllowForceBackward(const int bottom_index) const {  
    return bottom_index != 4;  
  }  
  
 protected:  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,  
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);  
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,  
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);  
  
  Blob<Dtype> diff_ap_;  
  Blob<Dtype> diff_wn_;    
  Blob<Dtype> diff_pn_;    
  
  Blob<Dtype> diff_sq_ap_;   
  Blob<Dtype> diff_sq_wn_;   
  
  Blob<Dtype> dist_sq_ap_;   
  Blob<Dtype> dist_sq_wn_;    
  
  Blob<Dtype> summer_vec_;   
  Blob<Dtype> dist_binary_;    
};  

}

#endif // CAFFE_PAIRWISE_LOSS_LAYER_HPP_
