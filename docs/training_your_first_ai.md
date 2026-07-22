# Training Your First AI

Once the program has been installed, there are several steps to creating your first AI.

## 1. Adding new Images

Images are needed to train an AI. When training begins, the images stored in the system will be loaded into the training area. The more images, the better. See the FAQ section for more information on what types of images and how many.

**Adding new images instructions:**

- Navigate to the "Training Data" Tab.
- Click the add new image button.
- Select your desired image. It must either be a "png" or a "jpg".
- If your image contains flaws and needs annotations, see below:
  - Select the add annotation button. 
  - Click and drag over the area where the flaw is located.
  - If you decide you don't want to add an annotation, click cancel.
  - If you decide an annotation is incorrect, select it from the list and delete it.
- Once you are finished, press the save changes button.
- Repeat this for every new image until you have a sufficient number of images.

## 2. (Optional) Creating a Dataset Config

When starting to train an AI, you may only want to use a select number of images. If this is the case, a dataset config is required. This tells the program which images to load and which to ignore.

**Creating a dataset config**

- You can create a new configuration by selecting the "Create New Config" button at the top right of the "Training Data" tab.

- Name it appropriately with a unique name and select "Create".

**Editing a dataset config**

- Select the config you would like to edit from the dropdown in the top left corner.

- Click the edit config button on the top row.

- Select the images you would like to add or remove by selecting the button below the desired image.

- Once you are finished, press the save button.

- Press the return button if you want to return without saving.

**Deleting a dataset config**

- Select the config you would like to delete from the dropdown in the top left corner.

- Click the delete config button.

- Press yes on the popup.

- The images contained in the config will still exist. Only the config itself will be deleted.

## 3. Training the AI

Once you have added all the images you need to train the AI, you can start training it.

**Steps to Start Training an AI**

- Navigate to the Train AI tab.

- Give your AI a name. Duplicate names are allowed, so ensure that the name you give it is unique.

- If desired, select the dataset config that will be used to load the images.

- Currently, the program only supports YOLOv5, but more may be added in the future.

- If desired, you can select a model to continue training from. This is intended to improve upon the previous model's values. However, it can increase the risk of "Overfitting" (see glossary for definition), so be careful when using this option. It may be better to use fewer Epochs to reduce this risk when training.

- Enter the desired number of Epoch (see glossary for definition.)

- Press the train button to start training.

**AI Training in Progress**

Once the training process has begun, an overview screen will be displayed. The system goes through a number of steps when starting to train the AI:

1. All the previous data and artefacts are cleaned up, ready for another execution.

2. The images from the selected config are loaded into the training area.

3. A YOLO instance is then created to train the AI.

4. While training, YOLO sometimes take a while to progress. The program is not frozen and training is still processing.

5. When the AI model has been created, YOLO performs evaluation metrics on the created instance.

**Cancelling the AI Training**

While it is not recommended that you cancel the model training, it can be completed.

1. Select the switch view button at the top of the page.

2. Press the cancel button at the bottom of the page.

3. It may take a little while to finish cancelling as it cleans up behind the scenes.

4. Wait until the button is replaced by the train button once again.

## 4. Viewing Your Created Model

Once you have created a model, you can view it and its performance statistics by navigating to the View Models tab.

1. Select the model you would like to view the details of.

2. For details on what each figure means, visit the glossary.