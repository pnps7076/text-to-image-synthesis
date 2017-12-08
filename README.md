# Generative Adversarial Text to Image Synthesis

In this project, I have designed and trained a model to take natural language captions in english and generate images relevant to the image captions. 

We have used skip-thoughts to encode the input caption before feeding to the generator and to the discriminator.  

The method used is inspired by [Generative Adversarial Text to Image Synthesis](https://arxiv.org/abs/1605.05396).

### Prerequisites

In order to clone and reproduce results from this repository, you will need to install:
* Python 2.7
* Numpy
* Tensorflow
* Theano (For using skip-thoughts)
* OpenCV

## Contributing

Feel free to clone and add extensions. Pull Requests are welcome.

## Authors

* **Vijay Veerabadran** - [Vijay V](https://vijayvee.github.io)

## Acknowledgments

* [Generative Adversarial Text to Image Synthesis](https://arxiv.org/abs/1605.05396)
* [Skip-Thought Vectors](https://arxiv.org/abs/1506.06726)

# Sample output
Included below are a few samples from our experiments:

Input caption | Generated image
------------ | -------------
**these flowers have an open face with many pale pink petals.** | ![Image 1](Output_Satisf/19_3700_60_Gen.jpg)
**this flower is red in color, and has petals that are closely wrapped around the center.** | ![Image 1](Output_Satisf/23_3700_60_Gen.jpg)
**this is a very cool flower with very bold white on the petals and a very unique middle.** | ![Image 1](Output_Satisf/17_3700_60_Gen.jpg)
**multiple layers of reddish-yellow petals that decrease in size as the are closer to the top of the flower.** | ![Image 1](Output_Satisf/7_3700_60_Gen.jpg)
**petals are light purple in color with longer stamens and a prominent green pistil.** | ![Image 1](Output_Satisf/12_3700_60_Gen.jpg)
