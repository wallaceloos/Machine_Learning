### People Segmentation  

<p align="justify">
This report presents how a well-known segmentation architecture, called U-net, can be used for a people segmentation task. U-net is an architecture built upon a fully convolutional network (FCN), that consists of an encoder (extracting the feature map)  and a decoder (upsampling the feature map). Two different U-net architectures are presented. Both are an adaption from the  U-net original architecture   <a href="https://arxiv.org/pdf/1505.04597.pdf" target="_blank">[1]</a>. The first one has a smaller number of filters in the bottleneck part, and the second one uses an encoder as a pre-trained network VGG16 architecture  <a href="https://arxiv.org/pdf/1505.04597.pdf" target="_blank">[2]</a>. The training data is a set of 376 images (256x256) from PASCAL VOC. The results presented that the second architecture had a better performance than the first architecture with dice coef = 0.7 and 0.5 respectively.  It shows just how important the encoder phase is. For future work, there will be experiments conducted with larger images and data augmentation. 

<p align="center">
<img src="https://github.com/wallaceloos/Machine_Learning/blob/master/images/resultado_seg_peoplev1.png" width="90%" height="90%">
</p>


