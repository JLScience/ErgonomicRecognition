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

### Demo

Todo ... 


![Test](https://j.gifs.com/E8qqZY.gif)

### Requirements

A webcam is required.

The application is tested using Python 3.5 on a machine running Ubuntu 16.04.

Besides OpenCV, Keras (2.2.2) with Tensorflow-GPU (1.4.0) backend is used.

The GPU support is highly recommended since the fps are probably too low for the configuration otherwise.  

