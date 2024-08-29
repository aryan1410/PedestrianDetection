# NNDL_MLOps
## Semester Project

### Introduction
**Motive:**

The primary motivation behind this project was to develop an efficient pedestrian detection system using deep learning techniques. Pedestrian detection plays a crucial role in various applications, such as autonomous vehicles, surveillance, and safety systems. The objective was to leverage transfer learning and fine-tuning of pre-trained models to create an accurate and reliable pedestrian detection system.

**Process in Brief:**

1. **Data Preprocessing:** The project began with the collection of a pedestrian dataset from Kaggle. The dataset contained images of pedestrians, including variations of poses and backgrounds. Data preprocessing involved resizing images to a common size (200x200 pixels) and categorizing them into two classes: "person" and "person-like."

2. **Model Selection:** Three pre-trained models - VGG19, ResNet101, and InceptionV3 - were chosen for transfer learning. The last 15 layers of these models were frozen to retain their feature extraction capabilities.

3. **Model Customization:** A custom classification head was added to each model, consisting of Global Average Pooling, Dense layers with ReLU activation, Dropout, and a final Dense layer with softmax activation to classify pedestrians into the two defined classes.

4. **Model Training:** The models were compiled using the Adam optimizer and Sparse Categorical Cross-Entropy loss. Training was conducted for 10 epochs with appropriate callbacks, such as model checkpointing, learning rate reduction, and early stopping, to save the best-performing model.

5. **Model Storage:** After training, the model weights were saved in separate JSON files for each of the three models.

6. **Pedestrian Detection in Videos:** The trained models were applied to custom videos downloaded from the internet. For pedestrian detection within video frames, OpenCV and Haar cascades were employed. Regions of interest (ROIs) within each frame were identified, marked, and returned as the output.

**End Use:**

The pedestrian detection system developed in this project has broad applications. It can enhance the safety and efficiency of autonomous vehicles by identifying pedestrians and potential collision risks. In surveillance systems, it can be used to monitor public spaces and alert security personnel to suspicious behavior. Furthermore, the technology can aid in smart city initiatives, helping authorities improve traffic management and pedestrian safety. Overall, the system contributes to enhancing public safety and optimizing various urban processes.

**Data:**

The dataset used for this project was obtained from Kaggle and contained images of pedestrians in diverse settings. The data was preprocessed by resizing images to a uniform size (200x200 pixels) and categorizing them into two classes: "person" and "person-like." This dataset was essential for training and evaluating the pedestrian detection models. While a more comprehensive dataset with a wider range of pedestrian images could further improve model accuracy, the chosen dataset served as a strong foundation for this project.


**Models:**

Three pre-trained deep learning models were selected for transfer learning: VGG19, ResNet101, and InceptionV3. These models were chosen due to their well-documented success in image classification tasks. The models were customized with a classification head and fine-tuned on the pedestrian dataset. This transfer learning approach significantly reduced the training time and data requirements while maintaining high performance. The combination of these models with the custom classification head yielded accurate and efficient pedestrian detection models.

**How Haar Cascades Work:**

Haar cascades are a machine learning object detection method used to identify objects or features within images or video frames. They work by training on positive and negative images. Positive images contain the target object (e.g., pedestrians), while negative images do not.

The training process involves creating a cascade of classifiers, where each classifier is a binary decision-making unit. These classifiers are organized in a cascade, with the easy-to-decide features evaluated first. Features are rectangular patterns, often black and white, which are shifted and scaled over the image to extract relevant information.

During detection, the Haar cascades slide over the image and apply the learned features to identify regions of interest that may contain the target object. If a region passes through all cascade stages, it is marked as a positive detection. The process is fast and efficient, making it suitable for real-time object detection in videos and images.

In this project, Haar cascades were used in conjunction with OpenCV to detect pedestrians in video frames, producing regions of interest for further analysis. This combination of deep learning models and Haar cascades allowed for robust and accurate pedestrian detection in a real-world video scenario.

**Inference:**
```python
import cv2
import numpy as np
from keras.models import model_from_json


pedestrian_dict = {0: "None", 1: "Pedestrian"}

#The model weights and the haarcascade can be downloaded from the link provided in the ReadME file, after which they can be loaded using the code below
ask = input("Which model do you wanna use?")

if ask == "vgg":
    
# load json and create model
    json_file = open('VGG19/VGG19_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    pedestrian_model = model_from_json(loaded_model_json)

# load weights into new model
    pedestrian_model.load_weights("VGG19\VGG19_model.h5")
    print("Loaded model from disk")

if ask == "resnet":
    
# load json and create model
    json_file = open('Resnet101/ResNet101_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    pedestrian_model = model_from_json(loaded_model_json)

# load weights into new model
    pedestrian_model.load_weights("Resnet101\ResNet101_model.h5")
    print("Loaded model from disk")

if ask == "inception":
    
# load json and create model
    json_file = open('Inceptionv3/inceptionv3_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    pedestrian_model = model_from_json(loaded_model_json)

# load weights into new model
    pedestrian_model.load_weights("Inceptionv3\inceptionv3_model.h5")
    print("Loaded model from disk")
# start the webcam feed

cap = cv2.VideoCapture('testvideo.mp4') # Add your video path here

while True:
    # Find haar cascade to draw bounding box around pedestrian
    ret, frame = cap.read()
    if not ret:
        # End of the video, break out of the loop
        break
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    pedestrian_detector = cv2.CascadeClassifier('haarcascade/haarcascade_fullbody.xml')# add your haarcascade path here
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect pedestrians available on camera
    num_peds = pedestrian_detector.detectMultiScale(rgb_frame, scaleFactor=6.0, minNeighbors=1)
    print(f"Number of pedestrians detected: {len(num_peds)}")

    # take each pedestrian available on the camera and Preprocess it
    for (x, y, w, h) in num_peds:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_rgb_frame = rgb_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_rgb_frame, (200, 200)), -1), 0)

        pedestrian_prediction = pedestrian_model.predict(cropped_img)
        maxindex = int(np.argmax(pedestrian_prediction))
        cv2.putText(frame, pedestrian_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
    cv2.imshow('Pedestrian Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```

**Link for model weights**:
<https://drive.google.com/drive/folders/1w-4v_4ZLjaxT6XlAiIQkOUpQnKni6aeW?usp=sharing>