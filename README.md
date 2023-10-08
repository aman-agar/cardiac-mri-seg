# cardiac-mri-seg
This repository contains the code for training a Block Attention enhanced U-Net model for Cardiac MRI Segmentation. The ACDC dataset was used for original training with input image size set as (224,224,1). 
The ACDC dataset contains 3D images for Cardiac MRI which were converted to 2D using the **create_dataset.py** script 
Dice Coef is used as the segmentation metric and dice coef loss is used as the loss function. 
The repository contains code both in form of Jupyter Notebook and Python Script. 
The **model.py** contains the model and **train.py** contains the code for training the model. 

Framework used - TensorFlow
The dice coef achieved is **0.94**
