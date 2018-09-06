The code for SSGAN.

There are three folders, one for each dataset - MNIST, CelebA, and CIFAR 10. Each folder has its own set of code for models and evaluation. 

Each folder has five different train_* scripts - one for each of the GAN models. In addition, in the evaluate_code folder, there are 4 different evaluaton scripts - one for each evaluation metric considered in the paper. 

(a) train_C_GAN implements supervised conditional GAN.

(b) train_AC_GAN implements supervised AC GAN.

(c) train_SS_C_GAN implements semi-supervised C GAN.

(d) train_SS_AC_GAN implements semi-supervised AC GAN.

(e) train_SS_S_GAN implements proposed semi-supervised GAN.

Dependencies: Theano, and for MNIST and CIFAR10: Keras. For CelebA, the dataset needs to be downloaded into the celebAsmall folder first using the download.py script. 

