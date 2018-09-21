# Deep Part Induction from Articulated Object Pairs
### Introduction
This work is based on our SIGGRAPH Asia 2018 paper. You can find the arXiv version of the paper [here](http://arxiv.org/abs/1809.07417). In this repository, we release the training and evaluation code, data as well as the pre-trained models.

### To Get Start

Download training and validation data from
  
    https://shapenet.cs.stanford.edu/ericyi/data_partmob.zip
  
Compile the PointNet++ code in "pointnet2"

Train the corrspondence proposal and the flow module through

    python train.py --stage 1

Train the hypothesis generation and the verification submodule through

    python train.py --stage 2
    
Train the hypothesis selection submodule through

    python train.py --stage 3
    
Evaluate the model through

    python evaluation.py
    
You can also download the pretrained model from the following link:

    https://shapenet.cs.stanford.edu/ericyi/pretrained_model_partmob.zip

### License

Our code and data are released under MIT License (see LICENSE file for details).
