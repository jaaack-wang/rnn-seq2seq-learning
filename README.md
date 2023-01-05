## Update

The current research is under re-design and re-experimentation. Previously, I experimented with string reversal and string copying (or identity function). In new experiments, I am studying the capabilities of RNN seq2seq models in learning a  hierarchy of reduplicative functions (i.e., string copying, total reduplication, input-specified reduplication, corresponding to rational, regular, polyregular functions respectively). I will show that these tasks are discriminative of the generative capacity of RNNs, RNN seq2seq models, and RNN seq2seq models with attention and the architectural biases that come with them. (For example, pure RNN seq2seq models cannot learn identity function and the attentional counterparts can only learn this function for in-distrbution examples, namely, strings of unseen or close lengths. However, RNNs can learn such a task perfectly, depiste their inability to model non-rational functions) It is also found that RNN seq2seq models with attention are strong in-distribution learners with certain out-of-distribution generalization abilities. 

You can also check this [rnn-transduction](https://github.com/jaaack-wang/rnn-transduction) repository for using RNNs to model string transducation tasks. 

## Original Description

This repository contains materials related to my paper on RNN seq2seq models learning non-regular transduction tasks. Source code, data, experimental results (including saved models), training logs, can also found in [this project folder](https://drive.google.com/drive/folders/1BO2LdbCFId3rY4BOX6Rcwn_m1WA8o912?usp=share_link) on Google Drive. 

Here is a brief description of the meaning of each sub-folders inside [the project folder](https://drive.google.com/drive/folders/1BO2LdbCFId3rY4BOX6Rcwn_m1WA8o912?usp=share_link) on Google Drive:

- ``data``: three versions of train/dev/test/generalization sets for reversed string and reduplicated string transduction tasks. Used for formal model training and evaluation. Results are reported. 
- ``tuning_data``: two sets of train/dev sets for the learning tasks. Used for manually searching for optimal hyperparameters. Results are not reported. 
- ``scripts``: code created for this project. 
- ``notebooks``: training and hyperparameter searching logs for eight experiments.
- ``Experiments_Logs``: experimental results of training and testing models on ``data``.



``notebooks`` and ``scripts`` are placed in this repository for temporary viewing. To be updated.