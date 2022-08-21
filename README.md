# MNIST-GAN-JAX
An implementation of InfoGAN using [JAX](https://github.com/google/jax), a highly performant NumPy replacement from Google Research with added capabilities for automatic differentiation and just-in-time compilation of Python code.  [Flax](https://github.com/google/flax) is used for machiene learning on top of JAX and is also provided by Google Research.

## InfoGAN
[InfoGAN](https://arxiv.org/pdf/1606.03657.pdf) is an extension on top of a normal GAN that adjusts the loss function to maximize the mutual information between a small subset of inputs to the Generateor and a new output added to the Discriminator.  The idea is that if the Discriminator can predict an aspect of the Generator's input then that part of the Generator's input will correspond to some semantically meaningful part of the image.

In this implementation, only one of the Generator's input dimensions is analyzed, which ends up correlating to which number 0-9 is generated.

## Loading Pre-trained Model
Example output is provided in the notebook at [MNIST_GAN_Jax.ipynb](MNIST_GAN_Jax.ipynb).  Running all cells up until the "Run Model Below" section will create a directory `./saved_models/mnist_gan`.  Copying [checkpoint_130](checkpoint_130) into that directory and then running the first cell in "Run Model Below" will result in the pretrained model being run.  To train the model more, increase `num_epochs` above 130

## Example Output
The first row is data from the MNIST dataset while each of the other rows has a different value 0-9 for the categorical input to the Generator.  It's definitely not perfect, but there is clearly a strong relationship between that input and which number is displayed.  It's also important to note how the 10 different possible categorical inputs end up including all 10 possible numbers from MNIST.

![index](https://user-images.githubusercontent.com/42822986/185795577-4e66ace5-d71d-4ebe-bfa8-30eedfdd5c23.png)
