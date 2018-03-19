/* 
 * pairwise_loss_layer.cu 
 * 
 *  Created on: Jan 3, 2017 
 *      Author: Limbo 
 */  
  
#include <algorithm>
#include <cfloat>  
#include <vector>  
  #include "caffe/util/io.hpp"
#include "caffe/layers/pairwise_loss_layer.hpp"  
#include "caffe/util/math_functions.hpp"  
  
namespace caffe {  
  
template <typename Dtype>  
void PairwiseLossLayer<Dtype>::Forward_gpu(  
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {  
  const int count = bottom[0]->count();  
  caffe_gpu_sub(  
      count,  
      bottom[0]->gpu_data(),  
      bottom[1]->gpu_data(),  
      diff_ap_.mutable_gpu_data()); 
  caffe_gpu_sub(  
      count,  
      bottom[2]->gpu_data(),  
      bottom[3]->gpu_data(),  
      diff_wn_.mutable_gpu_data()); 
  caffe_gpu_sub(  
      count,  
      bottom[1]->gpu_data(),  
      bottom[2]->gpu_data(),  
      diff_pn_.mutable_gpu_data());  
  
  caffe_gpu_powx(  
      count,  
      diff_ap_.mutable_gpu_data(),  
      Dtype(2),  
      diff_sq_ap_.mutable_gpu_data());  
  caffe_gpu_gemv(  
      CblasNoTrans,  
      bottom[0]->num(),  
      bottom[0]->channels(),  
      Dtype(1.0),                                         
      diff_sq_ap_.gpu_data(),              
      summer_vec_.gpu_data(),                             
      Dtype(0.0),                                         
      dist_sq_ap_.mutable_gpu_data());  
  
  caffe_gpu_powx(  
        count,  
        diff_wn_.mutable_gpu_data(),  
        Dtype(2),  
        diff_sq_wn_.mutable_gpu_data());  
  caffe_gpu_gemv(  
        CblasNoTrans,  
        bottom[0]->num(),  
        bottom[0]->channels(),  
        Dtype(1.0),                                         
        diff_sq_wn_.gpu_data(),  
        summer_vec_.gpu_data(),                             
        Dtype(0.0),                                         
        dist_sq_wn_.mutable_gpu_data());  
  
  Dtype margin = this->layer_param_.triplet_loss_param().margin();  
  Dtype loss(0.0); 
  Dtype loss1(0.0);
  Dtype loss2(0.0); 
  Dtype Sam(0.0);
  Dtype unfaml(0.0);
  const Dtype* sampleW = bottom[4]->gpu_data();												//1111
  for (int i = 0; i < bottom[0]->num(); ++i) { 
	loss1 +=  std::max(Dtype(0.05) - margin +dist_sq_ap_.cpu_data()[i], Dtype(0.0));
	Sam +=dist_sq_ap_.cpu_data()[i];
  }
   for (int i = 0; i < bottom[0]->num(); ++i) { 
	unfaml +=dist_sq_wn_.cpu_data()[i];
	loss2 +=  std::max(Dtype(0.05) + margin -dist_sq_wn_.cpu_data()[i], Dtype(0.0));
  }
	loss = loss1 + loss2 ;
 
  loss = loss / static_cast<Dtype>(bottom[0]->num());
  top[0]->mutable_cpu_data()[0] = loss /Dtype(2) ;  
}  

template <typename Dtype>  
__global__ void CLLBackward(const int count, const int channels,  
    const Dtype margin, const Dtype alpha, const Dtype* sampleW,  
    const Dtype* diff, const Dtype* dist_sq_ap_, const Dtype* dist_sq_wn_,  
    Dtype *bottom_diff,const Dtype type) {  
	if (type == 1){									
		  CUDA_KERNEL_LOOP(i, count) {  
    		  int n = i / channels;  
   		  Dtype mdist(0.0);  
    		  mdist = Dtype(0.05) - margin + dist_sq_ap_[n] ;  
              	  if (mdist > 0.0) {  
        		  bottom_diff[i] =  alpha*diff[i];  
				
                   } else {  
        		   bottom_diff[i] = 0;  
     			}  
  		  } 
	} 
	if (type == 0){									
		  CUDA_KERNEL_LOOP(i, count) {  
    		  int n = i / channels;  
   		  Dtype mdist(0.0);  
    		  mdist = Dtype(0.05) + margin - dist_sq_wn_[n];  
              	  if (mdist > 0.0) {  
        		  bottom_diff[i] =  alpha*diff[i];  
                   } else {  
        		   bottom_diff[i] = 0;  
     			}  
  		  } 
	} 
	
}    
  
template <typename Dtype>  
void PairwiseLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,  
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {  
  Dtype margin = this->layer_param_.triplet_loss_param().margin();  
  const int count = bottom[0]->count();  
  const int channels = bottom[0]->channels();  
  
  for (int i = 0; i < 4; ++i) {  
    if (propagate_down[i]) { 
      const Dtype type = (i<2) ? 1 : 0; 
      const Dtype sign = ((i<=2)&&(i>=1)) ? -1 : 1;				
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /  	
          static_cast<Dtype>(bottom[0]->num());  
      if(i==0){  
          // NOLINT_NEXT_LINE(whitespace/operators)  
          CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(  
              count, channels, margin, alpha,  
              bottom[4]->gpu_data(),  
              diff_ap_.gpu_data(),  
              dist_sq_ap_.gpu_data(),  
              dist_sq_wn_.gpu_data(),  
              bottom[i]->mutable_gpu_diff(),
		type);  
          CUDA_POST_KERNEL_CHECK;  
      }else if(i==1){  
          // NOLINT_NEXT_LINE(whitespace/operators)  
          CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(  
              count, channels, margin, alpha,  
              bottom[4]->gpu_data(),  
              diff_ap_.gpu_data(),  
              dist_sq_ap_.gpu_data(),  
              dist_sq_wn_.gpu_data(),  
              bottom[i]->mutable_gpu_diff(),
		type);  
          CUDA_POST_KERNEL_CHECK;  
      }else if(i==2){  
          // NOLINT_NEXT_LINE(whitespace/operators)  
          CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(  
              count, channels, margin, alpha,  
              bottom[4]->gpu_data(),  
              diff_wn_.gpu_data(),  
              dist_sq_ap_.gpu_data(),  
              dist_sq_wn_.gpu_data(), 
              bottom[i]->mutable_gpu_diff(),
		type);  
          CUDA_POST_KERNEL_CHECK;  
  
      } else if(i==3){  
          // NOLINT_NEXT_LINE(whitespace/operators)  
          CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(  
              count, channels, margin, alpha,  
              bottom[4]->gpu_data(),  
              diff_wn_.gpu_data(),  
              dist_sq_ap_.gpu_data(),  
              dist_sq_wn_.gpu_data(),  
              bottom[i]->mutable_gpu_diff(),
		type);  
          CUDA_POST_KERNEL_CHECK;  

  
      }  
    }  
  }  
}  
  
INSTANTIATE_LAYER_GPU_FUNCS(PairwiseLossLayer);  
  
}  // namespace caffe 
