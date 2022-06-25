# CoreSegNet

## Abstract
Deep fully-convolutional neural networks-based approaches have shown remarkable results for the task of pixel-wise semantic segmentation. Domain adaptation techniques allow for training the model with highly realistic synthetic data performing adversarial unsupervised domain adaptation. However, performances are heavily affected by the domain shift. The lack of semantic consistency leads to undesired outcomes that can be easily detected by the human eye.
    In this paper, we introduce prior knowledge within the model to induce domain-specific common sense and improve the results yielded by the segmentation network. This is achieved by leveraging the spatial distribution of the classes in a specialised setting and penalising nonsensical predictions. Furthermore, we introduce class weights that allow us to measure the relative frequency of each class to mitigate the imbalance of the datasets. Extensive experiments and ablation studies are conducted with various domain adaptation settings and different fashions to encode prior knowledge within the model.
    
 ![](figures/coresegnet.pdf)
 
 
## Previous repository
### Domain Adaptation Network
The repository of the adversarial domain adaptation network is available [here](http://www.github.com/gio99c/AdaptSegNet)
### Segmentation Network
The repository of the real-time segmentation network is available [here](http://www.github.com/gio99c/BiSeNet)
