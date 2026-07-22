# Glossary

### **AI Model**
A trained set of values that can be input into an AI model Architecture to classify a provided image into the correct class. It results from training an AI and is unique to the values provided during the training phase.

### **AI Model Architecture**
The structural design of an artificial intelligence model. It is the blueprint that defines how data flows through the model. For most cases, this will be YOLOv5.

### **Dataset**
The images and annotations that will be used to train the AI model.

### **Epoch**
One complete pass through the entire training dataset during the train AI process. When training, it is important to have a good-sized epoch to ensure the AI learns effectively from the provided training data.

### **Box Loss**
A measure of how accurately a model predicts the bounding box around an object. It is calculated from the difference between the predicted box and the ground truth (the box provided in the training data). A lower box loss means the predicted boxes closely match the true boxes.

### **CLS Loss**
Classification loss. The loss is calculated based on how well the model classifies an object into the correct category. As the system only features one type of flaw detection, this figure is not relevant, but is still output by the AI software.

### **IoU**
Intersection over Union. A measure of how well the predicted bounding box overlaps with the ground truth (the box provided during training.) An IoU of 1 is a perfect overlap, and zero means there is no overlap.

### **Precision**
Of the predicted boxes, how many are correct? Precision measures how well a model can make accurate positive predictions. It measures how many positive instances are correct. Precision is a valuable tool when the cost of false positives is also high.

### **Recall**
Of all the true boxes (correctly identified as flaws or not flaws), how many did the model find? The recall statistic focuses on identifying results that contain a defect but are labelled correctly and those that contain a defect but were incorrectly labelled as not containing one. It emphasises the model’s ability to identify positive cases and penalises the model for failing to detect positives.

### **Mean Average Precision IoU**
Precision and recall are calculated for a range of IoU thresholds. These values are then plotted on a graph with precision on the Y-axis and recall on the X-axis. This allows comparison between having a high precision and a high recall. Average precision is then derived as the area under the precision-recall curve.
This is normally at a IoU threshold of 0.5. Meaning a detection is correct if the overlap is bigger than 50%.

### **Mean Average Precision IoU (0.5-0.95)**
The same as mean average precision IoU, but with different threshold values that are stricter and more comprehensive.

### **Overfitting**
Overfitting is a concern in developing AI models. It occurs when a model performs well with its training and test data but fails to generalise unseen data. This can happen when the model learns incorrect features, such as background noise or image rotation, instead of actual relevant features.

### **Metamorphic Test**
Metamorphic testing has become a common approach to testing ML systems. It utilises metamorphic relations (MRs) based on domain knowledge. These are the relationship between input and output, and essentially substitute test oracles for metamorphic relations.

A test is deemed successful if the metamorphic property has been satisfied. The most common use is to identify a system’s ability to handle variations of the same input data with the same metamorphic properties. They should, therefore, produce the same output that doesn’t violate the MRs. For example, if an image that has previously been classified as flawed is rotated, the rotated image should still be classified as a flawed image.

In this system, the test image is analysed, rotated by 90 degrees, and then analysed again. The predicted bounding boxes are then compared to see if the same areas have been identified as defective. If they have, the test passes; otherwise, it fails.

### **Differential Test**
Differential testing compares the performance of test cases on similar versions of the software implementations. It ensures that the previous iteration’s characteristics are not lost when the model is retained.

In this system, the recall for the current model is calculated using part of the training, testing and whole dataset. It is also calculated for the previous model based on the date, or if the current model has been trained from another model, it is selected instead. The difference in their recall is returned.

### **Fuzzing Test**
Fuzzing testing involves generating random and unexpected inputs for the system. The primary goal is identifying issues such as program crashes, memory corruption and other vulnerabilities. Overall, it helps to evaluate the system's robustness and uncover weaknesses that might go undetected during regular operation.

In this system, the test images are corrupted by channel swapping, adding gaussian noise and occlusion. If YOLO doesn't throw an exception on the image, it is treated as a pass.