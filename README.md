# X-Ray_Image_Analysis

This program has been created in collaboration with the University of Sheffield and AMRC Castings.
It contributes to the dissertation project created by @TomPearson38

The system has been coded in Python and allows the user to provide an image of an X-rayed alloy to be analysed. The system then
detects faults in the provided image and provides feedback and a confidence score on the areas it believes contain faults. The user is
then able to judge the performance of the system and either correct it or confirm that it is correct. The image has the option to be added to
the dataset to retrain the AI at a later date and improve its functionality

AIs can be trained using a all images or a limited set provided through dataset configuration files,
and can even build upon previous iterations by using their final weights as starting values.

This project was developed using the [GDX-ray dataset](https://domingomery.ing.uc.cl/material/gdxray/). This project has been developed for research and educational purposes. Many thanks to GDX-ray for providing high quality testing images. 

# Requirements

To install the project, please ensure you have an Nvidia GPU. 
For full details on how to install from scratch, please visit the [install_instructions.md](install_instructions.md)

# License Information

Please see the [license document](LICENSE.md)

# Video Demonstration

Please see the following [YouTube Video](https://youtu.be/26JdtoJ3ZDw) for a demonstration of the system.
