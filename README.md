# Zero-Cost Operation Scoring in Differentiable Architecture Search (Zero-Cost-PT)
Official impementation for AAAI 2023 submission: 
"**Zero-Cost Operation Scoring in Differentiable Architecture Search**".


## Installation 
```
Python >= 3.6
PyTorch >= 1.7.1
torchvision == 0.8.2
tensorboard == 2.4.1
scipy == 1.5.2
gpustat
```
    
## Usage/Examples

### Experiments on NAS-Bench-201
Scripts for reproducing our experiments can be found under the ```exp_scripts/``` folder.

#### 1. Prepare NAS-Bench-201 Data
1. Download NAS-Bench-201 checkpoint from [NAS-Bench-201-v1_0-e61699.pth](https://drive.google.com/file/d/1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs/view), and place it under ```./data``` folder.

#### 2. Prepare NAS-Bench-201 and Zero-Cost-NAS API
i. Install NAS-Bench-201 api via `pip`
```
pip install nas-bench-201
```
ii. Install Zero-Cost-NAS API 

Clone the code repository from [Zero-Cost-NAS](https://github.com/SamsungLabs/zero-cost-nas). Go to the root directory of the cloned repo and run:
```
pip install .
```

#### 3. Run Zero-Cost-PT on NAS-Bench-201

You can run our Zero-Cost-PT with the following script:
```
bash zerocostpt_nb201_pipeline.sh --seed [SEED]
```
You can specify random seeds with ``` --seed ``` for reproducibility. In our experiments we use random seeds 0, 1, 2, 3. 

You could also run with different zero-cost proxies by specifying ```--metrics```, and different edge discretization order with ```--edge_decision```. The number of searching interations (N in our paper) is controlled by parameter ```--pool_size```, while the number of validation iterations (V in our paper) can be specified by ```--validate_rounds```. Please see Section 4.2 in our paper for more information on those parameters.

For example, a typical experiement setting could be: 

```--pool_size 10 --edge_decision random --validate_rounds 100 --metrics jacob --seed 0```

### Experiments on NAS-Bench-1shot1
Scripts for reproducing our experiments on NAS-Bench-1shot1 can be found under the ```nasbench1shot1``` folder.

#### 1. Prepare NAS-Bench-101 Data
1. Download NAS-Bench-1shot1 dataset from [nasbench_full.tfrecord](https://storage.googleapis.com/nasbench/nasbench_full.tfrecord), and place it under ```./data``` folder.

#### 2. Prepare NAS-Bench-101 API
Please refer orginal [NAS-Bench-101](https://github.com/google-research/nasbench) for details of API installation

#### 3. Run Zero-Cost-PT on NAS-Bench-1shot1

You can reproduce our Zero-Cost-PT with the following script:
```
  cd nasbench1shot1/optimizers/darts/
```
i. Run the following script for search architectures with Zero-Cost-PT from different sub-search-space on NAS-Bench-1shot1
```
   python network_proposal.py --seed [SEED] --search_space [SEARCH_SPACE]
```
In NAS-Bench-1shot1, it contains 3 sub-search-space which you can select from [1, 2, 3]

ii. Evaluated final searched model 
```
   python post_validate.py --seed [SEED] --search_exp_path [PATH_to_LAST_STEP_LOG_FOLDER]
```

### Experiments on NAS-Bench-Macro
Scripts for reproducing our experiments on NAS-Bench-1shot1 can be found under the ```nasbenchmacro``` folder.

#### 1. Prepare NAS-Bench-Macro Data
1. Download NAS-Bench-Macro dataset from [nas-bench-macro_cifar10.jsonnas-bench-macro_cifar10.json](https://github.com/xiusu/NAS-Bench-Macro/tree/master/data/nas-bench-macro_cifar10.json), and place it under ```./data``` folder.

#### 2. Run Zero-Cost-PT on NAS-Bench-Macro

You can reproduce our Zero-Cost-PT with the following script:
```
  cd nasbenchmacro/
```
i. Run the following script for search architectures with Zero-Cost-PT on NAS-Bench-Macro
```
   python network_proposal.py --seed [SEED] 
```

### Experiments on DARTS-like Spaces
Scripts for reproducing our experiments can be found under the ```exp_scripts/``` folder, and Zero-Cost-NAS API is also needed.

#### 1. For DARTS CNN space

Run the following script to search architectures with Zero-Cost-PT and train the searched architecture directly (with the same random seed): 
```
bash zerocostpt_darts_pipeline.sh --seed [SEED]
```
Our default parameter settings are: 

```--pool_size 10 --edge_decision random --validate_rounds 100 --metrics jacob```


#### 2、For DARTS subspaces S1-S4

On CIFAR-10 use the following script:

```
bash zerocostpt_darts_pipeline.sh --seed [SEED] --space [s1-s4] 
```

On CIFAR-100 and SVHN, use the following scripts:

```
bash zerocostpt_darts_pipeline_svhn.sh --seed [SEED] --space [s1-s4]
bash zerocostpt_darts_pipeline_c100.sh --seed [SEED] --space [s1-s4]
```

#### 3、Directly train the searched architectures reported in our paper

For reproducibility we also provide training scripts for evaluation of all the reported architectures in our paper. For an architecture specified by ```[genotype_name]```, run the following scrips to train:

```
bash eval.sh --arch [genotype_name] # for DARTS C10 
bash eval-c100.sh --arch [genotype_name] # for DARTS C100
bash eval-svhn.sh --arch [genotype_name] # for DARTS SVHN
```

The model genotypes are provided in ```sota/cnn/genotypes.py```. For instance, genotype `init_pt_s5_C10_0_100_N10` specifies the architecture searched by Zero-Cost-PT (with default settings as explaind above) on DARTS CNN space (S5), using 10 search iterations (N=10), 100 validation iterations (V=100), and random seed 0. 

#


### Other Experiments Reported in Appendix
We also provide code to reproduce experiment results reported in appendix, e.g. genotypes for maximum-param and random-sampling baselines, and Zero-Cost-PT for MobileNet-like spaces.



## Reference
Our code (Zero-Cost-PT) is based on [dart-pt](https://github.com/ruocwang/darts-pt) and [Zero-Cost-NAS](https://github.com/SamsungLabs/zero-cost-nas). For experiments on Nasbench1shot1 is based on [nasbench-1shot1](https://github.com/automl/nasbench-1shot1)
