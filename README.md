# Fighting CopyCat Agents Behavioral Cloning from Observation Histories

This is the code of the paper [Fighting CopyCat Agents Behavioral Cloning from Observation Histories](https://papers.nips.cc/paper/2020/file/1b113258af3968aaf3969ca67e744ff8-Paper.pdf).
This code has implemented the most important part of our contribution — Target-Conditioned Adversary module.
And it also supports two important baselines: BC-SO and BC-OH. 

## Dependencies

```shell script
conda create -n fight-copycat python=3.6
conda activate fight-copycat
pip install -r requirements.txt
```

## Usage

First, generate the dataset (can be skipped because the data has been generated and saved in ./data/trajectories):

```python
python -m imitation_learning.gen_data Hopper 50000
```

Then normalize the dataset (can be skipped because it has been normalized):

```python
CUDA_VISIBLE_DEVICES=0 python main.py normalize env=Hopper config/hopper.yaml
```

Behavioral cloning with single observation:

```python
CUDA_VISIBLE_DEVICES=0 python main.py main env=Hopper config/hopper.yaml name=Hopper-bcso policy.policy_mode=bc-so
```

Behavioral cloning with observation histories:

```python
CUDA_VISIBLE_DEVICES=0 python main.py main env=Hopper config/hopper.yaml name=Hopper-bcoh policy.policy_mode=bc-oh
```

Our algorithm (FCA):
```python
CUDA_VISIBLE_DEVICES=0 python main.py main env=Hopper config/hopper.yaml name=Hopper-fca policy.policy_mode=fca optim.discriminator_lr=4e-4 policy.gan_loss_weight=2.0
```

## Citations
Please consider citing our paper in your publications if it helps. Here is the bibtex:

```
@inproceedings{NEURIPS2020_1b113258,
 author = {Wen, Chuan and Lin, Jierui and Darrell, Trevor and Jayaraman, Dinesh and Gao, Yang},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {2564--2575},
 publisher = {Curran Associates, Inc.},
 title = {Fighting Copycat Agents in Behavioral Cloning from Observation Histories},
 url = {https://proceedings.neurips.cc/paper/2020/file/1b113258af3968aaf3969ca67e744ff8-Paper.pdf},
 volume = {33},
 year = {2020}
}
```
