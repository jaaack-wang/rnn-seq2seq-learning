In this project, I examine the capabilities of Recurrent-Neural-Network sequence to sequence (RNN seq2seq) models in learning four transduction tasks of varying complexity and that can be described as alignment learning. Here is the [current draft](https://drive.google.com/file/d/1z2wjo2E2n9VVdZ_5kjlRPexTb4yw0OcW/view?usp=share_link).



The repository contains source code and the experimental notebooks for a quick overview. Please check [this project folder](https://drive.google.com/drive/u/0/folders/1R47r-YGgU02H3DOW43A-JENmwZJCSVQj) On Google Drive to access the experimental results, including the raw results, those summarized results that are used in the paper, as well as the saved models for all the experiments. 



## Transduction Tasks

For a given string, the four transduction tasks can be defined as follows:

- Identity: the given string itself. Ex: <i>abc ===> abc</i> 
- Reversal: the reverse of the given string. Ex: <i>abc ===> cba</i> 
- Total reduplication: two copies of the given string. Ex: <i>abc ===> abcabc</i>
- Input specified reduplication: needs a specific number **N** of instruction symbols @ and produces as many extra copies of the given string as **N**. Ex: <i>abc@ ===> abc<ins>abc</ins></i>; <i>abc@@ ===> abc<ins>abcabc</ins></i>; <i>abc@@@ ===> abc<ins>abcabcabc</ins></i>

These four functions have been traditionally studied under the viewpoint of Finite State Tranducers (FSTs) and characterized accordingly. The FST-theoretic characterizations propose the following complexity hierarchy for these tasks: Input specified reduplication (polyregular function) > Reversal, Total reduplication (both regular) > Identity (rational).



## Alignment Learning

<ins>RNN seq2seq models take an encoder-decoder structure, where the decoder only “writes” after the encoder “reads” all the input symbols, unlike the read-and-write operation seen in FSTs</ins>. 

The following figure shows the conjectured mechanism for RNN seq2seq models learning identity and reversal functions. Other two functions can be learnt in a similar manner, and input specified reduplication additionally requires counting. I term such a learning process as **alignment learning**.



 <p align='center'>
 <img align="center" src="./imgs/alignments_learning.png">
</p>



## Basic Findings

- **Generalization abilities**: RNN seq2seq models, attentional or not, are prone to learning a plausible function that fits the training or in-distribution. Their out-of-distribution generalization abilities are highly limited. <ins>In other words, they are not learning the underlying data generation functions</ins>. 
- **Attention** makes learning alignment between input and target sequences significantly more efficient and robust, but does not solve the out-of-distribution generalization limitation. 
- **RNN variants**: we found the attentional SRNN models have better out-of-distribution generalization abilities, but attention-less GRU and LSTM are generally more expressive. LSTM models, regardless of attention, show a stronger ability of counting. 

- **Task complexity**: for attention-less RNN seq2seq models, it is empirically found that, total reduplication > identity > reversal, where > is a “more complex than” relation. We hypothesized that input specified reduplication is more complex than total reduplication, which, however, is not attested due to lack of computational resources to set out the experiments at a more proper scale. 

  

The following figure shows full-sequence accuracy (on unseen examples) per input length across the three types of RNN seq2seq models for the four tasks, where only input lengths 6-15 are seen during training.

 <p align='center'>
 <img align="center" src="./imgs/models_perf_across_tasks_per_len_full_sequence_accuracy.png">
</p>



## Reproduce the results

To reproduce the results, simply download everything in [the project folder](https://drive.google.com/drive/u/0/folders/1R47r-YGgU02H3DOW43A-JENmwZJCSVQj),  upload the folder to your Google Drive, and re-run the notebook in `notebooks`in a GPU runtime. Alternatively, you can also save the project folder to your Google Drive by using the "Add shortcut to Drive" function. In doing so, you should be able to run any of these notebooks successfully, but the results will not be saved on your Google Drive. 

It is recommened that you subscribe to Google Pro+ in order to reproduce the results.



## Reuse the code

- [The project folder](https://drive.google.com/drive/u/0/folders/1R47r-YGgU02H3DOW43A-JENmwZJCSVQj) provides a very neat but customizable pipeline to conduct experiments using RNN seq2seq models. Just make sure that your data is also saved in an identical format as the one provided inside the `data` folder. 

- If you prefer highly customized codebase to run experiments on command line, consider the following two repositories of mine:
  - [RNN Seq2seq transduction](https://github.com/jaaack-wang/rnn-seq2seq-transduction): nearly identical core codebase as used in my project!
  - [RNN transduction](https://github.com/jaaack-wang/rnn-transduction):  model language transduction tasks using RNNs with a detailed tutorial. 
