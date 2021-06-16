---
categories: 
   - Projects
tags:
   - 
---
> The real-time lane keep assistant using opencv.


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

![imgThres](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/imgThres.PNG)

## Step 2 - Bird Eye View (Warping)
Since calculating the curve of the path would be only done by the path right in front and not a few seconds ahead, we need to crop the image. This is also known as a bird eye view and it is important to find the curve value. We need to set up initial points to warp the image and the values determined manually.


```python
def warpImg(img, points, w, h, inv=False):
    pts1 = np.float32(points) # Setting manually
    pts2 = np.float32([[0,0], [w,0], [0,h], [w,h]]) # four points for warp

    # Inverse just for displaying
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1) # Create Transform Matrix
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    return imgWarp
```
We receives the transformation matrix based on the input points and then warp the image using the ```warpPerspective``` function.

![warpIMG](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/warpIMG.PNG)

## Step 3 - Curve Value Computation
Now it's time to find curve value through the summation of pixels. Since the warped image is not binary image that has either black or white pixels, summation in the y-direction will give us how many columns have higher threshold values than other portion of the image.

![pixel](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/pixel.png)

In the above image, if we have the sum of the pixels in every column. Let's set our threshold value as 1000. From the red line in the center, we can count the number of columns that pass the threshold is 8 on the left side. On the other hand, there are only 3 columns on the right side. Thus, this tells us the curve is towards left. However, we need to resolve one problem on this concept.

![sumerr](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/sumerr.PNG)

The above image demonstrates cases where this methods would not properly work. There is no gaurantee that the camera will always align with the straight line when driving in real environment. Thus it can detects the path as either left or right curve although actual path is the straight line. Thus we need center line adjustment.

![sumcor](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/sumcor.PNG)

Now we only look at the 1/4 portion at the bottom of the image to find proper center line. Suppose the index of the column starts from 0. If we calculate the center point based on the entire image, the index of center line becomes 5, which is not what we want to obtain. Thus, we only consider the bottom part of the image and calcuate the center line. Then we get 8.5, which is resonable enough to calculate the curve value from the current position.


```python
def getHistogram(img, minPer = 0.1, display = False, region = 1):
    # Sum all the Columns
    if region == 1: # For Entire Image
        histVals = np.sum(img, axis=0) # y-axis
    else:           # 1/4 of Image
        histVals = np.sum(img[img.shape[0]//region: :], axis=0)

    # Find max value (to define correct curve regardless of noise)
    maxVal = np.max(histVals)

    # Set Threshold
    minVal = minPer * maxVal

    # Save columns that passes threshold and average them to find center
    indexArray = np.where(histVals >= minVal)
    centerPoint = int(np.average(indexArray))

    # Plot histogram and Center point
    if display:
        imgHist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8) # Create Empty Image

        for x, intensity in enumerate(histVals):
            cv2.line(imgHist, (x, img.shape[0]), (x, img.shape[0] - intensity // 255 // region), (0, 255, 0), 1) # Histogram
            cv2.circle(imgHist, (centerPoint, img.shape[0]), 20, (0, 255,255), cv2.FILLED) # Center Point
        return centerPoint, imgHist

    return centerPoint
```
Sumation of pixels on each column is same as finding the histogram, so inside ```getHistogram()``` function, it adds all the number of pixels on each side and find left, right, or straight direciton. Since we also need to figure out how much to the direction, we also find index of columns that have value higher than our threshold value and average them to find true center point.

Now when we visualize our base point, it correctly finds the center point although the path is toward left side.
![histIMG](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/histIMG.PNG)

## Step 4 - Optimizing Curve
Now we want to know the intensity of curve. Suppose the center point is at 240 from the entire image. We can also find out the center point on the 4th portion at the bottom in the previous step. We get the average value of 278. This means the actual center of the image is 278 instead. Thus, we subtract our average value from the center value we got before from the entire image, which is 240 - 278 = -38. The negative sign indicates the curve is towards the left side and the intensity of curve is 51. Then we can append curve value to a list to average them to allow smooth motion.

```python
# Find Center Point Using Histogram
middlePoint, imgHist = utlis.getHistogram(imgWarp, display=True, minPer=0.5, region=4) # 1/4 of the image
curveAvgPoint, imgHist1 = utlis.getHistogram(imgWarp, display=True, minPer=0.9)        # Total Image
curveRaw = curveAvgPoint - middlePoint # This will be averaged

# Averaging for smooth motion instead of dramatic movements
curveList.append(curveRaw)
if len(curveList) > avgVal:
    curveList.pop(0)

# Find Curve
curve = int(sum(curveList)/len(curveList))
```
Now we can visualize the curve values with respect to the current path of our self-driving car in real-time.
![curveval](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/curveval.PNG)

## Hardware Implementation
* OS: Raspberry Pi™ 3
* Camera: Raspberry Pi Camera 8MP V2
* Motor: iRobot Create® 2 Programmable Robot

### Webcam Module


## Real-time Demo

<iframe width="560"
        height="315"
        src="https://www.youtube.com/embed/9CDj9lb-8HU"
        frameborder="0"
        allow="autoplay; encrypted-media"
        allowfullscreen></iframe>



