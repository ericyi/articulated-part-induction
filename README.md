# Deep Part Induction from Articulated Object Pairs
### To Get Start

Download training and validation data from
  
    https://shapenet.cs.stanford.edu/ericyi/data_partmob.zip
  
Train the corrspondence proposal and the flow module through

    python train.py --stage 1

Train the hypothesis generation and the verification submodule through

    python train.py --stage 2
    
Train the hypothesis selection submodule through

    python train.py --stage 3
