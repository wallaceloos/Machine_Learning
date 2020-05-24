### Gradient-weighted Class Activation Mapping (Grad-CAM)

<p align="center">
<img src="https://github.com/wallaceloos/Machine_Learning/blob/master/images/git_grad_cam.png" width="100%" height="100%">
</p>

<p align="justify">
Visualizing the features your model is learning can be very useful for understanding  it.  Gradient-weighted Class Activation Mapping (Grad-CAM) is a technique that helps to visualize the features learned by your model. It computes the gradient  of the score for a class c,  with respect to the last feature maps. It is expected that  the last convolutional layers have the best trade off between high-level semantics and  spatial information. First, the gradient is computed and then the global--average--pooled is performed to obtain the neuron importance weights. After that a weighted combination of the neuron importance and the feature maps are made followed by the rectified linear unit (ReLU).
    
You can read about Grad-CAM [here](https://arxiv.org/abs/1610.02391).
 
</p>

### Deep Inside Convolutional Networks: VisualisingImage Classification Models and Saliency Maps

You can read about Saliency Maps [here](https://arxiv.org/pdf/1312.6034.pdf).   

### Layer-wise Relevance Propagation for DeepNeural Network Architectures

You can read about Layer-wise Relevance Propagation [here](https://arxiv.org/pdf/1312.6034.pdf). 
