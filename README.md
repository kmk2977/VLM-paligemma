# VLM-paligemma

## Contrastive Vision Learning
The Contrastive Vision Encoder takes in a image and converts into a series of embeddings, so there will be one for each of the block of pixels[grid] in the given image. This embedding is a vector of a fixed size which will be concatinated with the tokens embeddings.

The Transformer Decoder receieves the collection of the image embeddings and the tokens embeddings. The transformer decoder will attend to the image tokens as a condition to generate the text.

This is visual representation of the above text.

![paligemma](images/architecture.webp)

**What is Contrastive Learning?**

Here we have 2 Encoders, one for texts and one for the images. Both of these makes embeddings for the given individual data respectively. 

**Embeddings** is a vector that captures most of the information for the given data(image and text).

> [!NOTE]
> With Contrastive Learning we want the dot product of the image embedding with the corresponding text embedding to be the highest and it will be the lower when u do the dot 
> product with not the correspoinding text embedding.