---
categories: 
   - Projects
tags:
   - 
---
> The self-driving car with lane keep assistant using opencv.


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
```python
def getImg(display=False, size=[480, 240]):
    stream = io.BytesIO()

    with picamera.PiCamera() as camera:
            camera.start_preview()
            #time.sleep(0.1)
            with picamera.array.PiRGBArray(camera) as stream:
                camera.capture(stream, format='bgr')
                # At this point the image is available as stream.array
                img = stream.array

    #_, img = cap.read()
    img = cv2.resize(img, (size[0], size[1]))

    if display:
        cv2.imshow('IMG', img)
    return img


if __name__ == '__main__':
    while True:
        img = getImg(True)
```

### Motor Module
```python
PORT = "/dev/ttyUSB0" # Don't remove the /dev/ portion

class Motor():
    def __init__(self):
        # Create a Create2
        self.bot = Create2(PORT)

        # Start the Create 2
        self.bot.start()

        # The Create has several modes, Off, Passive, Safe, and Full.
        # Safe: Roomba stops when it detects a cliff, detects a wheel drop, or if on the charger
        # Full: Roomba does not stop when it encounters an event above
        # Passive: Roomba sends sensor data but does not accept changes to sensors or wheels

        self.bot.safe()

    def move(self, speed=0.5, turn=0, t=0):
        speed_val = 80
        speed *= 80
        turn *= 40
        leftSpeed = speed - turn
        rightSpeed = speed + turn

        if leftSpeed > 80:
            leftSpeed = 80
        elif leftSpeed < -80:
            leftSpeed = -80
        if rightSpeed > 80:
            rightSpeed = 80
        elif rightSpeed < -80:
            rightSpeed = -80

        self.bot.drive_direct(int(rightSpeed), int(leftSpeed))  # inputs for motors are +/- 500 max

    def stop(self):
        # Stop the bot
        self.bot.drive_stop()

def main():
    motor.move(1, 0, 2) # speed, turn, seconds to run
    time.sleep(2)
    motor.move(-1, 0, 2)
    time.sleep(2)
    motor.move(1, 0.5, 2)
    time.sleep(2)
    motor.move(1, -0.5, 2)
    time.sleep(2)

if __name__ == "__main__":
    motor = Motor()
    main()
```

### Main Module
```python
motor = Motor()

def main():
    img = WebcamModule.getImg() # Get image from webcam
    curveVal = getLaneCurve(img, 1) # Get the curve value
    cv2.imshow("img", img)
    #print(curveVal)

    sen = 1.3 # SENSITIVITY (How much impact on curve)
    maxVAl = 0.3  # MAX SPEED

    # Define Maximum Speed (Range from 0 to 1)
    if curveVal > maxVAl: curveVal = maxVAl
    if curveVal < -maxVAl: curveVal = -maxVAl

    # Define Two Dead Zones (Straight Line)
    if curveVal > 0:
        sen = 1.7
        if curveVal < 0.05: curveVal = 0 # Keep Straight Line
    else:
        if curveVal > -0.08: curveVal = 0

    res, objInfo = getObjects(img, False, ['person'])
    if len(objInfo) == 1:
        motor.stop()
    else:
        motor.move(1, -curveVal * sen, 0.05) # Send to robot
    cv2.waitKey(1)

if __name__ == '__main__':
    while True:
        main()
```

## Real-time Demo

<iframe width="560"
        height="315"
        src="https://www.youtube.com/embed/9CDj9lb-8HU"
        frameborder="0"
        allow="autoplay; encrypted-media"
        allowfullscreen></iframe>

\
If you are interested in pedestrian detection, also check this post! 



