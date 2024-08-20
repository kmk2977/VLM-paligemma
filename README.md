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

So in **Contrastive Learning** what we do is we take a list of images and list of text which is corresponding text for each of the images, we encode them into list of embeddings and then train this model the **Text Encoder** and the **Image Encoder** to produce embeddings in such a way that when the dot product of the image with its corresponding text produces a higher value and when done with other non corresponding text it produces a lower value.

We then want to find a **Loss Function** that forces the dot products to be high forming a diagonal and all others to be low. For this we use the **Cross Entropy Loss**.

## How Language Models are trained?

Language Models are trained using the **Next token Prediciton task**.

We give the language model a prompt and it produces a series of embeddings which are then converted into **logits**. **Logits** is a distribution, vector which tells us what score the language model has assigned to what the next token should be among all the tokens in the vocabulary.

The **Cross Entropy Loss** takes in the vector, converts it into a distribution with the **softmax** function and then we compare it with the labels and we force the output to be equal to the label.

![next-token-prediction](images/llm-training.png)

We use the same in contrastive learning.

![contrastive](images/contrastivelearning.png)

Here is the code for how to implement the CLIP training with contrastive loss.

```
# image_encoder - ResNet or Vision Transformer
# text_encoder - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, 1] - minibatch of aligned text
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t - learned temperature parameter

# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i] #convert list of images into a list of embeddings
T_f = text_encoder(T) #[n, d_t] #convert list of prompts into a list of embeddings

# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1) #make sure that both image and text embeddings have the same number of dimensions and normalize the vectors

# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t) #compute all the possible dot products

# symmetric loss function
labels = np.arrange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t) / 2 # teach the model which item in each row/column needs to be maximized
```

## Problem with the CLIP model

How do we tell the model we want one item in each row/column to be maximized while minimizinf all other?

We use the Cross-Entropy-Loss!

## Numerical Stability of Softmax

The softmax function takes in vector as input and converts into a distribution, to be a distribution all the values must be non negative (>=0) i.e they are in range [0,1] and the sumation of all the values must be equal to 1.

![softmax](images/softmax.png) 

We take the outputs of the model and then we exponentiate each item in the output vector and then we also divide it with the sum of all the other items which is also known as the **normalization constant**.

> [!WARNING]
> The exponentail is a function that grows very fast, so if the argument of the exponential grows then the exponential will be huge. This is a problem for the computers as we 
> store numbers using a fixed representation 16bit ot 32bit.

Solution :-> do not make the exp grow to infinity

![softmax-fix](images/softmax-fix.png)

What we do is we multiply both the numerator and denominator with a constant **c**. We use log here as exp and log are inverse functions.

We normally choose ![softmax-fix1](images/softmax-fix1.png) This will push the arguments of the exp towards negative numbers and the exp itself towards zero.

### The Normalization Factor in softmax

To calculate the normalization factor, we must fo through all the elements of each row and each column.

> [!NOTE]
> Due to the asymmetry of the softmax loss, the normalization is independently performed twot times: across images and across texts.
> ![normalization-factor](images/normalization-factor.png)

**Problem with CLIP model is that its computationally expensive to calculate this contranstive loss**

So as a solution in the SigLIP paper they mentioned to replace the **Cross-Entropy-Loss** with **Sigmoid-Loss**.

So in Sigmoid-Loss, we again have an Imgae encoder that converts a list of images into embeddings and Text encoder that converts a list of text into embeddings.

We then calculate all the possible dot products, and then instead of treating the loss as a distribution over row or column, we use it as a binary classification task. Each of this dot products is treated as independently from each other.

**This allows us to grow the batch size to millions of items and to parallelize this task**

![sigmoid](images/sigmoid.png) Here z is the dot product of the vectors.

## Why Contrastive Vision Encoder though and not any ordinary Vision Encoder?

We want the embeddings to capture the information about the image and be good representations that can be then contrasted and can be used along with text embeddings.

# Vision Transformer

A transformer model is a sequence to sequnce model, we feed it a sequence of embeddings and then it outputs a sequence of contextualize embeddings.

What we do is we take in an image, split it into patches or group of pixels. We then extract the information form the group of pixels with the help of Convolution and then we Flatten them meaning we loose the positional information and we just concatenate them with each other[we loose the 2dimentionality]. We then add this positional information which is **learned** and not calculated without any sinusoidal functions. We then feed it into the Transformer model which later on contextualises the embeddings through the attention mechanism to a series of embeddings, meaning it will not only capture the information about itself but also of the other patches/embeddings.

> [!NOTE]
> Text has a Auto regressive relationship meaning we write text from left to right, each word we write depends on what we have written previously and on the other hand Images 
> dont have a auto regressive relationship, in images the patches depends on all other patches to make the meaning of the particular image.

We use the contextualized embeddings to capture the information about the each patch and how its present in the image, meaning we want each patch to include the information about its position[positional encoding] and also whats surrounding this patch in image by contextualizing them.

This is how sequence to sequence model works with transformer models.

![language-model](images/language-model.PNG)

# Normalization

There are a list of linear layers and they are defined by two parameters **input-features** and **output-features** and one another term named **bias** which is a constant which is added to the product of features and **weigths** which are the values which determine the strength of the connections and they influence one neurons output on one neurons input. Usually we have a batch of items and each item is made up of features.

What we do is say we have one batch here, we multiply the input vector x with the weight and add some bias to it if its present and then it generates one output feature.

![normalization](images/normalization.png)

But there is one big problem here **Covariate Shift**

## Covariate Shift

When you have a input vector that changes from one batch to another in magnitude then the output of the layer will also change in magnitude depending on what is the incomming .

**Why is it bad?**

Big change in input of a layer -> Big change in outout of layer -> Big change in loss -> Big change in gradient -> Big change in the weights of the network -> therefore slow learning.

Solution-1 :-> Use **Batch Normalization**

We have a batch of items as input. for ex. we have a image classification task, we have as input a list of images and then features[dimensions that represent the particular image].

In Batch Normalization we calculate the statistics for each dimensions of each item[we calculate mean and varaince] and then we normalize each item by substracting mean and then dividing it by standard deviation. This will make each dimension of each item be distributed according to a gausian with mean 0 and a variance of 1.

![batchnorm](images/batchnorm.png)

**Problem with Batch Normalization**

The problem with batch norm is that each statistics [mean and the varaince] are calculated along the batch dimension. The statistics depends on what other items are in the batch and therefore to get good results we must use a **big batch size**.

Solution 2 :-> **Layer Normalization**

We calculate the statistics along the item dimension instead of the batch dimension, but insted of the mean and standard deviation coming from the first dimention of each item it comes from the average of all the dimensions of the each item independently thus making the training more stable and even we are not forced to use the large batch size.

![layernorm](images/layernorm-1.png) ![layernorm](images/layernorm-2.png) Here H is the no. of hidden units in a layer.

## Multi-Head Attention

It is a way of contextualizing stuff, you start with a seqeuence of patches. Say we have 4 patches, each of these patches is represented by a single vector of 1024 dimensions. Each of these patch is being extracted from a group of pixels from the image and it represents information about the image.

With multi head attention we contextualize these patches meaning the output of this is a tensor of the same size as of the input. Each of the patches in the output does not just catches information about itself but also about the other patches aswell.

![vision](images/vision.png)

>[!NOTE]
> The above explainantion is for Vision transformer, for Language model we have am input sequence which is a sequence of tokens each representing one single word. Each token is representated as an embedding. We want to contextualize each token with all the tokens that come before it.
>
>![text](images/text.png)
>
> In language models we basically train on the next token prediction, we start with some tokens which are the prompts and ask the model what would be the next token till the predicition task is finished and we receive the complete output.
>
> Predict next tokens given the past tokens and the transformer allows us to do that in parallel with training . We have uncontextualized inputs and a series of contextualized output, in such a way that each tokens captures information about itself and also about the previous token.
>
> The self attention model will take the prompt as input and generate the output in parallel using the multi head attention.
>
> We also need some labels to train the model.


**Step-1** Convert the input sequence X into Query, Key, Values

We convert them using 3 transformations W <sub>q </sup> , W <sub>k </sub> , W <sub>v </sub> , which is a matrix multiplication. 

![step-1](images/step1.png)

here sequence is the no. of tokens or patches, hidden_size represents the size of this embedding vector

the transformations are of size (1024,8,128) -> the second 1024 is divided into 8 groups of 128. The second matrix has 1024 rows with 8 invidual vectors and its made up of 128 dimensions. Therefore the resulting output is also split into multiple sub groups 4 rows, 8 vectors and 128 dimensions

>[!NOTE]
> Multi head attntion is a way to relate tokens with each other, we dont want the tokens to relate by watching the full embedding of each token, we want to do it with the 8 individual heads such as each heads works with a smaller part of embedding of each token. 

01:40:23