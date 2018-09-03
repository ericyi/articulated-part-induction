# Deep Part Induction from Articulated Object Pairs
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
