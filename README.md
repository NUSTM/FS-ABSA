# FS-ABSA: A Simple yet Effective Framework for Few-Shot Aspect-Based Sentiment Analisys 

## Quick links

  - [Overview](#overview)
  - [Requirements](#requirements)
  - [Run FS-ABSA](#run-fs-absa)
    - [Fully-supervised Setting](#fully-supervised-setting)
    - [Non-English Low-resource Setting](#non-english-low-resource-setting)
  - [Main Results](#main-results)
  - [Ablation Study](#ablation-study)
  - [Any Question?](#any-questions)
  - [Citation](#citation) 
  
******************Updates******************

- 2023/4/24: We upload the domain-adaptive pre-training models ([restaurant-t5-base](https://huggingface.co/NUSTM/restaurant-t5-base), [laptop-t5-base](https://huggingface.co/NUSTM/laptop-t5-base), [dutch-restaurant-mt5-small](https://huggingface.co/NUSTM/dutch-restaurant-mt5-small) and [french-restaurant-mt5-small](https://huggingface.co/NUSTM/french-restaurant-mt5-small)) to huggingface.



  

## Overview

<div align=center><img alt="image" src="https://user-images.githubusercontent.com/84706021/231090090-1cd20863-467b-44cd-8717-e6585d3c24b5.png" width="50%" height="40%"></div>

In this work, we introduce a simple yet effective framework called **FS-ABSA**, which involves domain-adaptive pre-training and textinfilling fine-tuning.
Specifically, 
- we approach the End-to-End ABSA task as a text-infilling problem. 
- we perform domain-adaptive pre-training with the text-infilling objective, narrowing the two gaps, i.e., domain gap and objective gap, and consequently facilitating the knowledge transfer.

## Requirements

To run the code, please install all the dependency packages by using the following command:

```
pip install -r requirements.txt
```

**NOTE**: All experiments are conducted on NVIDIA RTX 3090 (and Linux OS). Different versions of packages and GPU may lead to different results.

## Run FS-ABSA

**NOTE**: All experiment scripts are with multiple runs (three seeds).



### Few-Shot Setting

```
## English Dataset: 14lap
$ bash script/run_aspe_fewshot_14lap.sh

## English Dataset: 14res
$ bash script/run_aspe_fewshot_14res.sh

## Dutch Dataset: 16res
$ bash script/run_aspe_fewshot_dutch.sh

## French Dataset: 16res
$ bash script/run_aspe_fewshot_french.sh
```

### Fully-supervised Setting

```
## English Dataset: 14lap
$ bash script/run_aspe_14lap.sh

## English Dataset: 14res
$ bash script/run_aspe_14res.sh

## Dutch Dataset: 16res
$ bash script/run_aspe_dutch.sh

## French Dataset: 16res
$ bash script/run_aspe_french.sh
```


## Main Results

Results on 14-Lap and 14-Res under different training data size scenarios

<div align=center><img alt="image" src="https://user-images.githubusercontent.com/84706021/231090977-dfca504b-0524-4801-9841-4e894edc3649.png" width="50%" height="50%"></div>

Comparison with SOTA under the full data setting

<div align=center><img  alt="image" src="https://user-images.githubusercontent.com/84706021/231092372-17714139-7337-4392-b823-9f02face4abf.png" width="40%"></div>

  
Results in two low-resource languages under different training data sizes

<div align=center><img alt="image" src="https://user-images.githubusercontent.com/84706021/231091364-43d93815-ca84-485e-adf6-4ba67f662d96.png" width="50%" height="50%"></div>



## Ablation Study

<div align=center><img alt="image" src="https://user-images.githubusercontent.com/84706021/231097628-81d6f869-f9f8-4ba9-8f96-8c63108a3237.png" width="50%" height="50%"></div>


## Citation

If you find this work helpful, please cite our paper as follows:

```
@inproceedings{wang2023fs-absa,
    author = {Wang, Zengzhi and Xie, Qiming and Xia, Rui},
    title = {A Simple yet Effective Framework for Few-Shot Aspect-Based Sentiment Analysis},
    year = {2023},
    isbn = {9781450394086},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3539618.3591940},
    doi = {10.1145/3539618.3591940},
    booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
    numpages = {6},
    location = {Taipei, Taiwan},
    series = {SIGIR '23}
}
```

Note that the complete citation format will be announced once our paper is published in the SIGIR 2023 conference proceedings.

## Any Questions?

If you have any questions related to this work, you can open an issue with details or feel free to email Zengzhi(`zzwang@njust.edu.cn`), Qiming(`qmxie@njust.edu.cn`).


## Acknowledgements

Our code is based on [ABSA-QUAD](https://github.com/IsakZhang/ABSA-QUAD). Thanks for their work.

