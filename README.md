# ErgonomicRecognition

Implementation of a pre-trained pose estimator network for 
real-time sitting posture inspection and feedback.

### Introduction

The keras implementation of the model was adopted from 
[a pure Python / Keras version](https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation/)
of the [Realtime Multi-Person Pose Estimation project](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation), 
where also the network weights (converted from Caffe) have been downloaded.

The paper describing the initial work can be found 
[here](https://arxiv.org/abs/1611.08050).

Besides using the pre-trained network, a convolutional network has been implemented and trained to detect whether each eye is opened or closed. 
This is utilized for an enhanced controlling of the program flow. 
Therefore the application includes the possibility of collecting eye-data for retraining the classifier.

The application is intended to test the possibilities of pose estimation and key-point analysis in a real-life scenario.

## Demo

### Calibration

![Calibration Demo](https://j.gifs.com/nxRREE.gif)


### Posture Analysis

Four different false postures can be recognized: 

#### Head twisted

![Posture Analysis Demo 1](https://j.gifs.com/8133km.gif)


#### Head leaned sidewards

![Posture Analysis Demo 2](https://j.gifs.com/p8ZZG2.gif)


#### Head leaned forward

![Posture Analysis Demo 3](https://j.gifs.com/Jy88Lo.gif)


#### Head ducked

![Posture Analysis Demo 4](https://j.gifs.com/ZY88l6.gif)


### Eye dataset acquisition
For the training of the eyes opened/closed classifier a data acquisition mode exists.
Pictures can be saved by clicking 'o' (opened) or 'c' (closed).

![Eye data acquisition](https://j.gifs.com/lxRRj6.gif)


## Requirements

A webcam is required.

The application is tested using Python 3.5 on a machine running Ubuntu 16.04.

Besides OpenCV, Keras (2.2.2) with Tensorflow-GPU (1.4.0) backend is used.

The GPU support is highly recommended since the fps are probably too low for the configuration otherwise (not tested).  

