# PairLoss

The implementation of PairLoss in [When Deep Learning Meets Metric Learning: Remote Sensing Image Scene Classification via Learning Discriminative CNNs](http://ieeexplore.ieee.org/document/8252784/).



## Compiling
You can follow these tutorial [Caffe](http://caffe.berkeleyvision.org/) and [link](https://zhuanlan.zhihu.com/p/25484850) to compile your own caffe.

As is define in our paper,  our PairLoss function:

$$J_{2}(X, W, B) = \sum_{i,j} max(0, (0.05 - y_{i,j} (\tau - ||O_{L}(x_i)-O_{L}(x_j)||_{2}^2)))$$

where  $y_{i,j}=\begin{cases} +1 & y_{i} = y_{j} \\-1 & y_{i} \not = y_{j} \\\end{cases}$.

Therefore, our loss function needs **four** inputs which are divided into 2 groups.

One group calculates the loss of the inter-class, another calculates that of the intra-class.

If you utilize our loss function on your task, you can adopt a 4-stream siamese structure or *slice* operation.

## Citation

```
@article{cheng2018deep,
  title={When Deep Learning Meets Metric Learning: Remote Sensing Image Scene Classification via Learning Discriminative CNNs},
  author={Cheng, Gong and Yang, Ceyuan and Yao, Xiwen and Guo, Lei and Han, Junwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2018},
  publisher={IEEE}
}
```



