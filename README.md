# ResNet50-From-Scratch

This Repo implements the basic building blocks of Deep Residual Networks which is trained for Image Classification using Dogs-vs-Cats Dataset

![resnet50](https://user-images.githubusercontent.com/99212200/190844300-6d18332b-00cd-4c4a-ae39-e126a4e3080a.png)

# ResNet Intuition

As a Neural Network gets very deep, vanishing / exploding gradients become a huge problem. ResNet solves this by using “Skip Connections” where layer 1 output goes directly to layer N input.

The concept of “Residual Block”:


![residualblock](https://user-images.githubusercontent.com/99212200/190844348-091fd65a-27b7-4d45-94e4-a36b77ce2f61.png)

# References

ResNet algorithm due to He et al. (2015). The implementation here also took significant inspiration and follows the structure given in the GitHub repository of Francois Chollet:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun - Deep Residual Learning for Image Recognition (2015)
Francois Chollet's GitHub repository: https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
