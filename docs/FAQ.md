# FAQs

## Training FAQs
### **Q: What AI models can be used to train an AI?**
**A:** Currently, only YOLOv5 can be used to train an AI.

### **Q: How many epochs should I use to train an AI?**
**A:** The recommended number of epochs usually is around 50. A smaller number of epochs may generalise too much and identify flaws that are not flaws because they have a similar pattern. However, too many epochs increase the risk of overfitting, where the model can only detect flaws that the system has examples of and may identify unique or novel flaws.

### **Q: How many images should I have before I create an AI?**
**A:** As many as possible! The minimum would be around 100 to 200, but the more images, the better the system's performance!

### **Q: The AI is taking too long to train.**
**A:** Depending on the performance of your system, the AI may take longer to train than expected. The best thing to do would be to start training the AI and leave the system to complete the task.

### **Q: Should I always choose a model from which to continue training?**
**A:** The continue training from option has been implemented so that small changes and improvements can be added to existing models. However, if this feature is used more than once to create new AIs, e.g. creating a new AI -> Using AI1 to train AI2 -> using AI 2 to create AI3 -> etc, the risk of overfitting increases. It is recommended only to use this feature to add small changes to a model, but if a large number of new images have been added, a new model should be trained from scratch.


## Config and Stored Images FAQs

### **Q: How do I delete an Image?**
**A:** To delete an image from the program, navigate to the training data tab. Then, select the image you would like to remove. Then, press the delete image button.

### **Q: Will my deleted images be removed from all the configs?**
**A:** Yes, all deleted images are removed from all existing configs to prevent errors when training a new AI.

### **Q: What image formats are supported?**
**A:** Only .JPG and .PNG are currently supported.

### **Q: Will my images be deleted when I delete a config?**
**A:** No, only the config will be deleted; the images contained in the config will remain in the system until they are deleted individually.

## Analysis FAQs

### **Q: I want to add an image to the dataset even though the AI identified it correctly.**
**A:** Good idea; it is essential to create a balanced dataset to potentially improve the model's performance on existing data. Just follow the instructions as if the model misidentified the image. It is also important to bear in mind that adding too many correct images may, once again, increase the chances of overfitting.