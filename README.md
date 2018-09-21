The code for SSGAN.

There are three folders, one for each dataset - MNIST, CelebA, and CIFAR 10. Each folder has its own set of code for models and evaluation. 

Each folder has five different train_* scripts - one for each of the GAN models. In addition, in the evaluate_code folder, there are 4 different evaluaton scripts - one for each evaluation metric considered in the paper. 

(a) train_C_GAN implements supervised conditional GAN.

(b) train_AC_GAN implements supervised AC GAN.

(c) train_SS_C_GAN implements semi-supervised C GAN.

(d) train_SS_AC_GAN implements semi-supervised AC GAN.

(e) train_SS_S_GAN implements proposed semi-supervised GAN.

Dependencies: Theano, and for MNIST and CIFAR10: Keras. For CelebA, the dataset needs to be downloaded into the celebAsmall folder first using the download.py script. 

License
Copyright (C) 2018 Palo Alto Research Center, Inc.

This program is free software for non-commercial uses: you can redistribute it and/or modify it under the terms of the Aladdin Free Public License - see the LICENSE.md file for details.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

If this license does not meet your needs, please submit an Issue on github with your contact information and institution, or email engage@parc.com, so we can discuss how to meet your needs.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
