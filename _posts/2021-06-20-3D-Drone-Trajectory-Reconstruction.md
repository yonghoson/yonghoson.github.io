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
The adaptivity of the tracker is defined by selecting the best features depending on the background.
Now we demonstrate how the on-line feature selection method can adapt proficiently to the
current tracking problem. We changed the background to the same texture, which is the same as
the patch. The image below shows that the target object was still successfully tracked by the algorithm
although the background was the same texture as the object

![adaptivity](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/adaptivity.PNG)


### Robustness
A successful tracker should manage various appearance changes of the target object, including
occlusions, rotations, and movement. The sequence shows a clock keeps tracked by the tracker although
a significant portion of the object has been occluded. It implies that the tracker is continuously
adapting to the background. Also, the tracker keeps track of the object with the movement in
an upward direction and rotation by the hand. It shows that the tracker has been updated the
object’s feature and confidence score in real-time.

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
We applied the Kalman filter on the trajectory output from
AdaBoost to produce a smoother trajectory for better trajectory estimation. The basic idea of
the Kalman filter is using the prior knowledge of the state, which is the output from Adaboost in
our project.
\
The code below implemented by using the ```pykalman``` library. As you can see, we set initial settings, such as transition matrix, observation matrix and corresponding covariances.

```python
Measured = np.load("Drone_Trajectory0.npy")
MarkedMeasure = []

for m in Measured:
    m = m[:-1]
    MarkedMeasure.append(m)

MarkedMeasure = np.array(MarkedMeasure)

Transition_Matrix = [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]
Observation_Matrix = [[1,0,0,0],[0,1,0,0]]
xinit = MarkedMeasure[0,0]
yinit = MarkedMeasure[0,1]
vxinit = MarkedMeasure[1,0]-MarkedMeasure[0,0]
vyinit = MarkedMeasure[1,1]-MarkedMeasure[0,1]
initstate = [xinit,yinit,vxinit,vyinit]
initcovariance = 1.0e-3*np.eye(4)
transistionCov = 1.0e-4*np.eye(4)
observationCov = 1.0e-1*np.eye(2)

kf = KalmanFilter(transition_matrices=Transition_Matrix,
                  observation_matrices =Observation_Matrix,
                  initial_state_mean=initstate,
                  initial_state_covariance=initcovariance,
                  transition_covariance=transistionCov,
                  observation_covariance=observationCov)


(filtered_state_means, filtered_state_covariances) = kf.smooth(MarkedMeasure)

# Produce Kalman Output
plt.figure(figsize=(12,8))
plt.subplot(1, 2, 1)
plt.plot(MarkedMeasure[:,0], MarkedMeasure[:,1],'xr',label='measured')
plt.title("Original")
plt.subplot(1, 2, 2)
plt.plot(filtered_state_means[:,0], filtered_state_means[:,1], 'ob', label='kalman output')
plt.legend(loc=2)
plt.title("Kalman Filter")
plt.show()
```
Kalman filter output from the original trajectory. Cut-offs that are recovered from the
filter are specified by a black circle.
![kalman](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/kalman.png)

## Ad-hoc Camera Network
Until this section, we only produced the 2D trajectory of the drone in each of the seven videos from
the dataset that were filmed with cheap and easy-to-deploy types of equipment. Now it’s time
to reconstruct 3D trajectories from unsynchronized consumer cameras. We only assume intrinsic
parameters for calibration for each camera, such as focal length and radial distortion. All other
parameters of the seven camera setup are recovered during the operation, which is synchronization
between different cameras and camera poses.

This is the pipeline code for the Ad-hoc Camera Network: [Pipeline Code](https://github.com/CenekAlbl/mvus)
This is the dataset which contains videos of the drone: [Dataset](https://github.com/CenekAlbl/drone-tracking-datasets)


The image below is the reconstructed 3D trajectory from the unsynchronized seven videos with out 2D trajectory detection mechanism.
![kmlearth](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/kmlearth.PNG)


## Conclusion
In this project, we have demonstrated a robust real-time tracking technique that formulates the
tracking as a binary classification between object and background. Since coping with the variations in appearance during tracking was the key, we brought on-line AdaBoost algorithm which
updates features of the tracker during tracking the object. Then we presented a two-step based
implementation of the Kalman filter, which is a very powerful tool when our 2D trajectory detections have some noises like cut-offs. After improving trajectory estimation, we reconstruct the
trajectory of the drone using seven different 2D paths obtained from the external cameras. The
network takes care of calibration, rolling shutter effects, and geometry computation to produce
cm-accurate trajectories.

To read more about this project: [Paper](/res/327report.pdf)





