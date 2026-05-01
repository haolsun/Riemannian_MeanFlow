# Riemannian_MeanFlow
Accepted ICML 2026 Paper: **Riemannian MeanFlow for One-Step Generation on Manifolds**

[Paper](https://arxiv.org/abs/2603.10718)



1. Install the dependencies:
```sh
conda env create -f environment.yaml -n rmf
```

2. Activate the environment:
```sh
conda activate rmf
```

3. For the experimental data, please follow the instructions given in the [Riemannian Flow Matching repository](https://github.com/facebookresearch/riemannian-fm) and [Riemannian Consistency Model](https://github.com/ccr-cheng/riemannian-consistency-model). The structure should be:
```sh
rmf/
├── configs
├── data
    ├──raw
    ├──earth
    ├──top500
    ├──rna
    
├── rmf
```

4. Run the experiment you want! :boom: For example,
```sh
# earth fire
python -m rmf.train experiment=meanflow/meanflow_earth_fire trainer=gpu seed=0
# torus rna
python -m rmf.train experiment=meanflow/meanflow_amino_rna trainer=gpu seed=0
#   SO(3)
python -m rmf.train experiment=meanflow/meanflow_so3 trainer=cpu seed=0
```
