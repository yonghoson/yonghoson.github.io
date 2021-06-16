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

![pixel](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/pixel.PNG)

In the above image, if we have the sum of the pixels in every column. Let's set our threshold value as 1000. From the red line in the center, we can count the number of columns that pass the threshold is 8 on the left side. On the other hand, there are only 3 columns on the right side. Thus, this tells us the curve is towards left. However, we need to resolve one problem on this concept.

![sumerr](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/sumerr.PNG)

The above image demonstrates cases where this methods would not properly work. There is no gaurantee that the camera will always align with the straight line when driving in real environment. Thus it can detects the path as either left or right curve although actual path is the straight line. Thus we need center line adjustment.

![sumcol](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/sumcol.PNG)



## Step 4 - Optimizing Curve



## Application



## Conclusion

* 빠른 속도, 안정성.
* 쉽게 설치하고, 사용할 수 있다.
* 어노테이션을 사용하여 직관적이고, 가독성이 뛰어나다.
* 플러그인 형태를 취하고 있어, 유지보수에 편하다.


## 4. 단점
* 타 라이브러리에 비해 추가기능이 많은 편은 아니다.
* OkHttp의 상위에서 구현된 라이브러리로, 기본적으로 OkHttp에 의존적이다. <sub>(사실 OkHttp도 매우 좋은 라이브러리로, 그다지 단점은 아님)</sub>

