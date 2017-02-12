# Generative Adversarial Text to Image Synthesis

In this project, we have designed and trained a model to take natural language captions in english and generate images relevant to the image captions. The core of our model is a conditional GAN, the generation being conditioned on the input text. We have used skip-thoughts to encode the input caption before feeding to the generator and to the discriminator. 
We have used [this paper](https://arxiv.org/abs/1605.05396) to implement a huge part of our model. 