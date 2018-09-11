# A Real Time Pixel Segmentation Using TensorFlow
This program is inspired by [this article](https://github.com/priya-dwivedi/Deep-Learning/blob/master/Mask_RCNN/Mask_RCNN_Videos.ipynb) (Pixel Segementation in a batch mode)
## Installation
### Install Object_Detection for Tensorflow
Follow official instructions [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
### Choose your favorite Object_Detection model
Go to the [model_zoo page](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and choose a *mask_rcnn_...* model
Download it and save it to *exported_model/* folder
Set the model path here *PATH_TO_CKPT = 'exported_model/YOUR_MODEL_NAME/frozen_inference_graph.pb'*
## Run the program
Install all dependencies and run with the command
*python3 Main.py*
