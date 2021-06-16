---
categories: 
   - Projects
tags:
   - 
---
> Extract 2D trajectory information by performing object tracking on the drone using on-line AdaBoost algorithm. Apply the Kalman filter to improve trajectory estimation. Then feed 2D trajectory into ad-hoc camera network to reconstruct 3D trajectory.

- - - 


## Dataset
For this project, I used around 1600 images consists of either with pedestrian or no pedestrian on the road. For the labels, I used "0" to indicate pedestrian and "1" to indicate no-pedestrian as you will see in the implemented codes.

Dateset Source: https://www.kaggle.com/tejasvdante/pedestrian-no-pedestrian

![pedestrian](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/pedestrian.PNG)




## Drone Tracking Demo

<iframe width="560"
        height="315"
        src="https://www.youtube.com/embed/JjipDVfWmAE"
        frameborder="0"
        allow="autoplay; encrypted-media"
        allowfullscreen></iframe>



