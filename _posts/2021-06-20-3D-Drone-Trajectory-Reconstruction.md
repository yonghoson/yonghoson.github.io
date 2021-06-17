---
categories: 
   - Projects
tags:
   - 
---
> Extract 2D trajectory information by performing object tracking on the drone using on-line AdaBoost algorithm. Apply the Kalman filter to improve trajectory estimation. Then feed 2D trajectory into ad-hoc camera network to reconstruct 3D trajectory.

- - - 


## On-line Adaboost
In this section, we briefly introduce the details of the on-line boosting algorithm with the following definition terms:
* Weak classifier: A classifier that can perform slightly better than random guessing. In
case of binary classification, the error rate must be less than 50 percent

* Selector: Given a set of N weak classifiers, a selector selects exactly one of those. The
choice decision is made by the estimated error of each weak classifier.

* Strong classifier: Given a set of M weak classifiers, a strong classifier is computed by a
linear combination of selectors

The main idea of the algorithm is done by ```selectors```. Each of them holds a separate feature
of weak classifiers and the weak classifier with the lowest error is selected by the selector. When
a new frame of the video arrives, the weak classifiers of each selector are updated. The selectors
consecutively change the best weak classifier with respect to the weight passed on to the next
selector. Finally, the strong classifier is available at each time step of the video for
object tracking.

![adaprinciple](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/adaprinciple.PNG)


### Implementation
Now it's time to actually implement this tracker. The OpenCV's legacy library provides this on-line Adaboost based tracking. Below is the code implementation:

```python
# Tracker Object Function
trackers = [cv2.legacy.TrackerBoosting_create]
trackerIdx = 0
tracker = None
isFirst = True
measuredTrack = []

video_src = 0           # For Connected Camera
video_src = "cam0.mp4"  # Video
cap = cv2.VideoCapture(video_src)
fps = cap.get(cv2.CAP_PROP_FPS) # Get Frame Number
delay = int(1000/fps)
win_name = 'Tracking'

# Full Size Setting
width = cap.get(3)  # float `width`
height = cap.get(4)  # float `height`
cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('Cannot read video file')
        break

    img_draw = frame.copy()
    if tracker is None: # No Tracker
        cv2.putText(img_draw, "Press Space to set ROI!", \
            (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)
    else:
        ok, bbox = tracker.update(frame)   # Track on new Frame
        (x,y,w,h) = bbox
        if ok: # Tracking Successful
            points = [w/2+x, h/2+y, cap.get(cv2.CAP_PROP_POS_FRAMES)]
            measuredTrack.append(points) # Store x, y, frame_id
            cv2.rectangle(img_draw, (int(x), int(y)), (int(x + w), int(y + h)), \
                          (0,255,0), 2, 1)
        else : # Tracking Failure
            cv2.putText(img_draw, "Tracking fail.", (100,80), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)

    trackerName = tracker.__class__.__name__
    cv2.putText(img_draw, str(trackerIdx) + ":"+trackerName , (100,20), \
                 cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0),2,cv2.LINE_AA)
    cv2.imshow(win_name, img_draw)
    key = cv2.waitKey(delay) & 0xff

    # Space bar or start of the video
    if key == ord(' ') or (video_src != 0 and isFirst):
        isFirst = False
        roi = cv2.selectROI(win_name, frame, False)  # Set initial ROI
        if roi[2] and roi[3]:                        # If there are position values
            tracker = trackers[trackerIdx]()
            isInit = tracker.init(frame, roi)
    elif key in range(48, 56):
        trackerIdx = key-48
        if bbox is not None:
            tracker = trackers[trackerIdx]()
            isInit = tracker.init(frame, bbox)
    elif key == 8: # Backspace
        tracker = None
        isFirst = True
    elif key == 27:
        break
else:
    print("Cannot open video")

cap.release()
np.save("Drone_Trajectory", measuredTrack) # Save NP File
cv2.destroyAllWindows()
```
As you can see at the end, we stored all the 2D trajectory information into ```.npy``` format for the later use, which is to improve trajectory estimation using the Kalman filter.

### Adaptivity


![adaptivity](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/adaptivity.PNG)


### Robustness


![robustness](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/robustness.PNG)


## Drone Tracking Demo
It tracks the drone successfully although the size is extremely small in the video. However, when the drone disappears or goes befind the building, the trackers loses its target object, which produces some noises in the trajectory.

<iframe width="560"
        height="315"
        src="https://www.youtube.com/embed/JjipDVfWmAE"
        frameborder="0"
        allow="autoplay; encrypted-media"
        allowfullscreen></iframe>


## Kalman Filter



## Ad-hoc Camera Network



## Conclusion


To read more about this project: [Paper](/res/327report.pdf)





