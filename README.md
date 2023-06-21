The repository contains the source code, data, and some of the experimental notebooks for [the paper](https://arxiv.org/abs/2303.06841) accepted at [the 16 ICGI](http://www.fsr.ac.ma/icgi2023/index.html), titled <i>Learning Transductions and Alignments with RNN Seq2seq Models</i>. Through unified training/evaluation conditions and comprehensive experiments, I compared the learning results across tasks, different model configurations, and generalization performance etc., and highlighted factors that influence the learning capabilities and generalization capacity of RNN seq2seq models.

For more details or everything related to the experiments, including data, training logs, trained models, and experimental results, please  check [this project folder](https://drive.google.com/drive/u/0/folders/1R47r-YGgU02H3DOW43A-JENmwZJCSVQj) On Google Drive.



## Transduction Tasks

For a given string, the four transduction tasks can be defined as follows:

- Identity: the given string itself. Ex: <i>abc ===> abc</i> 
- Reversal: the reverse of the given string. Ex: <i>abc ===> cba</i> 
- Total reduplication: two copies of the given string. Ex: <i>abc ===> abcabc</i>
- Quadratic copying: make `n` copies of the given string of length `n`. Ex: <i>abc ===> abcabcabc</i>   

These four functions have been traditionally studied under the viewpoint of Finite State Tranducers (FSTs) and characterized accordingly. The FST-theoretic characterizations propose the following complexity hierarchy for these tasks: quadratic copying (polyregular function) > total reduplication (regular) > reversal (regular) > identity (rational).



## Learning Input-target Alignments

<ins>RNN seq2seq models take an encoder-decoder structure, where the decoder only “writes” after the encoder “reads” all the input symbols, unlike the read-and-write operation seen in FSTs</ins>. 

The following figure shows the conjectured mechanism for RNN seq2seq models learning identity and reversal functions. Other two functions can be learnt in a similar manner, and input specified reduplication additionally requires counting. 



 <p align='center'>
 <img align="center" src="./imgs/alignments_learning.png">
</p>



## Basic Findings

- **Generalization abilities**: RNN seq2seq models, attentional or not, are prone to learning a function that fits the training or in-distribution data. Their out-of-distribution generalization abilities are highly limited. <ins>In other words, they are not learning the underlying data generation functions</ins>, probably because of the inherent limitation of auto-regressive models. 

  - Please note that, task complexity is strongly tied to the structure of the learner. For example, RNNs of few hundred parameters can easily learn the function of identity whereas RNNs seq2seq models cannot. See: [RNNs-learn-identity](https://github.com/jaaack-wang/RNNs-learn-identity).

- **Attention**: makes learning alignment between input and target sequences significantly more efficient and robust, but does not overcome the out-of-distribution generalization limitation. 

- **Task complexity**: for attention-less RNN seq2seq models, it is empirically found that quadratic copying > total reduplication > identity > reversal. The relative complexity between identity and reversal is due to the long-term dependency learning issue that comes with RNNs trained with gradient descent and backpropagation. 

- **RNN variants**: The effect of RNN variants on the seq2seq models is a complicated one and interacts with other factors, e.g., attention and the task to learn. Generally, GRU and LSTM are generally more expressive than SRNN. SRNN cannot count. 

  

The following figure shows full-sequence accuracy (on unseen examples) per input length across the four tasks for the three types of RNN seq2seq models, where only input lengths 6-15 are seen during training.

 <p align='center'>
 <img align="center" src="./imgs/tasks_perf_across_models_per_len_full_sequence_accuracy.png">
</p>




## Reproduce the results

To reproduce the results, simply download everything in [the project folder](https://drive.google.com/drive/u/0/folders/1R47r-YGgU02H3DOW43A-JENmwZJCSVQj),  upload the folder to your Google Drive, and re-run the notebook in `notebooks` in a GPU runtime. Alternatively, you can also save the project folder to your Google Drive by clicking the "Add shortcut to Drive" button. In doing so, you should be able to run any of these notebooks successfully, but the results will not be saved on your Google Drive (unless you download the entire folder and upload it to your Google Drive). 

It is recommened that you subscribe to Google Pro+ in order to reproduce the results.



## Citation 

```bibtex
@misc{zw2023rnn-seq2seq,
  doi = {10.48550/ARXIV.2303.06841},
  url = {https://arxiv.org/abs/2303.06841},
  author = {Wang, Zhengxiang},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Learning Transductions and Alignments with RNN Seq2seq Models},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}
```



## Reuse the code

- [The project folder](https://drive.google.com/drive/u/0/folders/1R47r-YGgU02H3DOW43A-JENmwZJCSVQj) provides a very neat but customizable pipeline to conduct experiments using RNN seq2seq models. Just make sure that your data is also saved in an identical format as the one provided inside the `data` folder. 

- If you prefer highly customized codebase to run experiments on command line, consider the following two repositories of mine:
  - [RNN Seq2seq transduction](https://github.com/jaaack-wang/rnn-seq2seq-transduction): nearly identical core codebase as used in my project!
  - [RNN transduction](https://github.com/jaaack-wang/rnn-transduction):  model language transduction tasks using RNNs with a detailed tutorial. 
