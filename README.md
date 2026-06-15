# Riemannian_MeanFlow
Accepted ICML 2026 Paper: **Riemannian MeanFlow for One-Step Generation on Manifolds**.

[Paper](https://arxiv.org/abs/2603.10718)


## 1. Sphere, Torus and SO(3)

1. Install the dependencies:
```sh
cd rmf
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

4. Run the experiment. For example,
```sh
# earth fire
python -m rmf.train experiment=meanflow/meanflow_earth_fire trainer=gpu seed=0
# torus rna
python -m rmf.train experiment=meanflow/meanflow_amino_rna trainer=gpu seed=0
# SO(3)
python -m rmf.train experiment=meanflow/meanflow_so3 trainer=cpu seed=0
```

## 2. SE(3)-Grasp

1. Get data:

There are 8836 files in the [acronym grasp](https://github.com/NVlabs/acronym) dataset.  
There are 7897 files in the [ShapeNetSem dataset](https://huggingface.co/datasets/ShapeNet/ShapeNetSem-archive/tree/main) that follows the same format.  
Both parts of the data are necessary.

2. Project Structure

```sh
├── data/
│   ├── grasp
│   └── meshes
├── src/
│   ├── core/
│   ├── data/
│   └── models/
├── scripts/  
│   ├── train_mlp.py
│   └── train_mlp_1000.py
```

3. Install the dependencies:
```sh
cd grasp
uv venv
uv pip install -e .
```

4. Run the experiment. For example,

```sh
# 200 sample small experiment
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mlp

# 2000 sample experiment
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mlp_1000
```

## 3. DNA

The experiment can be conducted by following the steps outlined in this project: [DNA Frame RMF](https://github.com/dongyeop3813/Riemannian-MeanFlow).

**Note**: Evaluating MSE in this project requires a pretrained Sei model. While the project is implemented in JAX, the Sei model is built in PyTorch, so the model parameters need to be converted from PyTorch to JAX before evaluation.


## 4. Protein

The experiment can be conducted by following the steps outlined in this project: [Protein Frame RMF](https://github.com/dongyeop3813/Protein-RMF)

**Note:** Evaluating the scRMSD score in this project requires loading the pretrained ProteinMPNN and ESMFold models. Since these models require a different environment from the main project, their dependencies need to be configured separately.

The evaluation procedure is as follows:

```sh
# Generate 100 samples
python -m experiments.inference_se3_flowmaps

# Navigate to the directory containing the scRMSD evaluation code
cd protein_rewards/Scaffold-Lab

# Run the evaluation script 
# Before running, please modify the configuration file: 
# Protein-RMF-main/protein_rewards/Scaffold-Lab/config/unconditional.yaml
python -m scaffold_lab.unconditional.refolding
```

## 5. Acknowledgements

This project builds upon several excellent open-source projects.   
We sincerely thank the authors and contributors of these projects for making their code publicly available.

1.[Riemannian Flow Matching repository](https://github.com/facebookresearch/riemannian-fm)  
2.[Generalised Flow Maps](https://github.com/olsdavis/gfm)  
3.[Riemannian Consistency Model](https://github.com/ccr-cheng/riemannian-consistency-model)  
4.[RFM-Grasp](https://github.com/Ferdydh/RFM-Grasp)  
5.[DNA Frame RMF](https://github.com/dongyeop3813/Riemannian-MeanFlow)  
6.[Protein Frame RMF](https://github.com/dongyeop3813/Protein-RMF)  