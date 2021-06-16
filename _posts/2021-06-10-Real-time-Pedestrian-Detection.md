---
categories: 
   - Projects
tags:
   - 
---
> The real-time pedestrian detection using trained model from CNN.


- - - 


## Overall Module Structure
For this project, there will the main module that will connects to all various modules. The reason for having different modules for each different functionalities is that adding or modifying different modules becomes very convenient in this structure for the future upgrade.

![structure](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/structure.PNG)

## Step 1 - Finding Lane

Since I used a regular A4 white paper as path, simply applying the color detection will find the path. Thus, the ```thresholding``` function will be defined as below:

```python
def thresholding(img):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert to HSV
    lower_white = np.array([80, 0, 0])            # Define Range of White color in HSV
    upper_white = np.array([255, 160, 255])
    mask_white = cv2.inRange(imgHsv, lower_white, upper_white)
    return mask_white 
```
The above code simply converts the image to HSV color space and then apply a range of color to find the white paper path.


## Real-time Demo

<iframe width="560"
        height="315"
        src="https://www.youtube.com/embed/9CDj9lb-8HU"
        frameborder="0"
        allow="autoplay; encrypted-media"
        allowfullscreen></iframe>



