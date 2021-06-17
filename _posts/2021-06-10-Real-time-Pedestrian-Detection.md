---
categories: 
   - Projects
tags:
   - 
---
> The real-time pedestrian detection using trained model from CNN.

- - - 


## Overview
CNNs have been gaining popularity in the past couple of years due to their ability to generalize and classify the data with high accuracy. In this project, I will train around 1600 images of two different classes with the help of tensorflow and keras. Classes are classified as either ```pedestrian``` or ```no pedestrian``` on road images. For the labels, I used "0" to indicate pedestrian and "1" to indicate no-pedestrian as you will see in the implemented codes.

Dataset Source: [Dataset](https://www.kaggle.com/tejasvdante/pedestrian-no-pedestrian)

![pedestrian](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/pedestrian.PNG)

## Importing Images
First we import images from the directory path where images are stored using ```os``` library. Then we split the data into train and test sets as you can see in the ```parameters``` section, the ratio has been set to 0.2, which means if there are 1000 images, then we will use 200 imgages for testing.

```python
# Parameters
path = "myData"  # folder with all the class folders
labelFile = 'labels.csv'  # file with all names of classes
batch_size_val = 32  # how many to process together
steps_per_epoch_val = 2000
epochs_val = 100
imageDimesions = (200, 200, 3)
testRatio = 0.2        # if 1000 images, 200 will be used for testing
validationRatio = 0.2  # if 1000 images 20% of remaining 800 will be 160 for validation


# Importing Images
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes.....")

for x in range(0, len(myList)):
    myPicList = os.listdir(path + "/" + str(count))

    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)
        if curImg is not None:
            images.append(curImg)
            classNo.append(count)
    count += 1

print(" ")
images = np.array(images)
classNo = np.array(classNo)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio, shuffle=True)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio, shuffle=True)
```

# Pre-process Images
First we convert the image into grayscale, then we equalizing the image to standardize the lighting in the image. Lastly we normalizing the values in range between 0 to 1 instead to having values 0 to 255.

```python
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img) # To GrayScale
    img = equalize(img)  # Standardize the lighting in the image
    img = img / 255      # To normalize in range between 0 to 1
    return img

X_train = np.array(list(map(preprocessing, X_train)))  # To iterate and pre-process all images
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))
#cv2.imshow("GrayScale Images", X_train[random.randint(0, len(X_train) - 1)])  # Display one example
```

## Augmentation
Image augmentation is required if we want to make the model to be more generic. It includes shifting the image into left, right, and zooming-in. Eventually it creates different dataset, which is more generalized.

```python
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,    # 0.2 -> Can go from 0.8 to 1.2
                             shear_range=0.1,   # Magnitude of share angle
                             rotation_range=10) # Degrees

dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size=20)  # Generate Images with corresponding batch size
X_batch, y_batch = next(batches)
```

## Convolutional Neural Network Model
Now it's time to create our CNN model. It has some convolutional layers, pooling layers, and drop out layers. Lastly we have a dense layer as an output layer to obtain probability of each labels of our dataset. 


```python
def myModel():
    model = Sequential()
    model.add((Conv2D(filters=32, kernel_size=(3,3), input_shape=(imageDimesions[0], imageDimesions[1], 1), activation='relu')))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add((Conv2D(filters=64, kernel_size=(3,3), activation='relu')))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add((Conv2D(filters=64, kernel_size=(3, 3), activation='relu')))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    # Compile Model
    model.compile(Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = myModel()
```

## Training
Now we send our data directly into the model with the parameters we already defined for training. When the training is done, it will show us the plot with the loss and accuracy scores. At the end, we will store the model into ```h5``` format, so that we can use for testing later on.

```python
history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                              epochs=epochs_val,
                              validation_data=(X_validation, y_validation),
                              shuffle=1)

# Plot Eval
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()

losses = pd.DataFrame(model.history.history)
losses.plot()

# Store the Model
model.save('trained_model.h5')
```

We can see that our accuracy goes up with each epoch and loss comes down. And now we have about 85% accuracy model to test with.

![eval](https://raw.githubusercontent.com/yonghoson/yonghoson.github.io/master/images/eval.png)

## Testing
Now it's time to test our trained model in real-time. Again we start from defining parameters for the testing. For the threshold value, this is the probability percentage that is required to be higher to accept as the class is correctly detected. 

```python
frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 150
threshold = 0.75  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
```

Now we need to import our CNN model as below:
```python
model = load_model('trained_model.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

We have to pre-process the images coming from our camera in real-time before feed into our model. Same pre-processing procedure applies to the captured images from the webcam. And we have class names so that we can display on the console. Inside the while loop we continously receive the image and call all the defined functions to detect pedestrian detection in real-time.

```python
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getClassName(classNo):
    if classNo == 0:
        return 'No Pedestrian'
    elif (classNo == 1):
        return 'Pedestrian'


while True:
    # READ IMAGE
    success, imgOrignal = cap.read()

    # Pre-process
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (200, 200))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 200, 200, 1)
    cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    # Predict Image
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue = np.amax(predictions)

    if probabilityValue > threshold:
        #print(getCalssName(classIndex))
        cv2.putText(imgOrignal, str(classIndex) + " " + str(getClassName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Result", imgOrignal)

    if cv2.waitKey(1) == 27:
        break
```

## Real-time Demo

<iframe width="560"
        height="315"
        src="https://www.youtube.com/embed/mlv56ChJPcY"
        frameborder="0"
        allow="autoplay; encrypted-media"
        allowfullscreen></iframe>



