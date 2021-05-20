## TEXT to IMAGE Generation
This repository hosts the course project done in course CSE631(deep learning) @ IIITD in winter semestor 2021. The project aims to employ Generative Adversarial Neural Networks (GANs) trained on Oxford Flower Dataset to generate real-life images of flowers from textual description. Code can be used for training model from scratch/ contact for pretrained models. 
Baseline models are implemented along the works of 
 - [Radford et al.](https://arxiv.org/abs/1511.06434) DCGAN 
 - [Reed et al.](https://arxiv.org/abs/1605.05396) DCGAN-CLS-INT
 
Text Embeddings are referenced from [reedscot/icml2016](https://github.com/reedscot/icml2016) and they are used for further train the model.
Models were trained for 100 epochs. To prevent the overfitting of multi
### Results
#### DCGAN
<img src = "Baseline/results/1.png">

#### DCGAN-CLS-INT
<img src = "Baseline/results/2.png">

Report: [HERE](../main/Baseline/Report.pdf)

**StackGAN** directory hosts the custom implementation of Stack GAN referenced from [Paper](https://arxiv.org/abs/1612.03242) & [Repo](https://github.com/hanzhanggit/StackGAN). However, we didn't trained the model enough for it to produce pleasing results.
