---
categories: 
   - Projects
tags:
   - 
---
> The real-time pedestrian detection using trained model from CNN.


- - - 


## Dataset
For this project, I used around 1600 images consists of either with pedestrian or no pedestrian on the road. For the labels, I used 

Source: https://www.kaggle.com/tejasvdante/pedestrian-no-pedestrian

![structure](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/pedestrian.PNG)

## CNN Model


```python
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 198, 198, 32)      320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 99, 99, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 97, 97, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 48, 48, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 46, 46, 64)        36928     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 23, 23, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 33856)             0         
_________________________________________________________________
dense (Dense)                (None, 128)               4333696   
_________________________________________________________________
dropout (Dropout)            (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 258       
=================================================================

```


## Evaluation
We can see that our accuracy goes up with each epoch and loss comes down.

![structure](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/pedestrian.PNG)

## Real-time Demo

<iframe width="560"
        height="315"
        src="https://www.youtube.com/embed/mlv56ChJPcY"
        frameborder="0"
        allow="autoplay; encrypted-media"
        allowfullscreen></iframe>



