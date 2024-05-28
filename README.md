# DDSTFDN
DDSTFDN for cross-vieew gait recognition
Overview

This repository is the official implementation of:
DDSTFDN: Dynamic Densely Connected Spatial-Temporal Feature Decoupling Network for Cross-view Gait Recognition

Shuo Qiao , Chao Tang , Huosheng Hu , Wenjian Wang , Anyang Tong , Fang Ren

In this paper, we propose a Dynamic Densely Connected Spatial-Temporal Feature Decoupling Network (DDSTFDN) for Cross-view Gait Recognition. It is based on the concept of feature reuse so that shallow features can be passed to deep networks to improve the performance of gait recognition. Then the dense spatiotemporal decoupling feature extraction module can obtain gait spatiotemporal association features and spatiotemporal decoupling features, thus solving the problem of insufficient feature representation. Finally, the enhanced convolutional block attention mechanism can help the network focus on valuable information. We hope that this work can provide some inspiration and help for subsequent researchers studying gait recognition.

version:
Torch veresion == 2.0.1
Python version == 3.9
Operating system == Windows11

clarification:
The config folder is the configuration file required for the network.
The data folder refers to the files related to the pre-processing of the data.
The model folder refers to the files associated with the model component.
The lossees folder is the files related to the loss function.
The utils folder refers to some of the tool files needed to run the project.
The datapre file is used for the processing of the rawest data.
The evaluation file is used to evaluate the correctness of gait recognition.
The main file is used for model training, parameter saving and performance evaluation.
