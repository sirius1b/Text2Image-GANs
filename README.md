## TEXT to IMAGE Generation
The project aims to employ Generative Adversarial Neural Networks (GANs) trained on Oxford Flower Dataset to generate real-life images of flowers from textual description. Code can be used for training model from scratch/ contact for pretrained models. Models are implemented in PyTorch and they have theoritical outline [Radford et al.](https://arxiv.org/abs/1511.06434) and [Reed et al.](https://arxiv.org/abs/1605.05396) work.
These models utilize pretrained text embeddings for contextual encoder from the work of [reedscot/icml2016](https://github.com/reedscot/icml2016)
 
Models were trained for 100 epochs. Generated image is expected to be 64x64 Px based on embedding dim of 1024. To prevent the overfitting of L1 & L2 regualarization were applied during training loop.
### Results
#### DCGAN
<img src = "Baseline/results/1.png">

#### DCGAN-CLS-INT
<img src = "Baseline/results/2.png">

Report: [HERE](../main/Baseline/Report.pdf)

