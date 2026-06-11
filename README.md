<h1 align="center">CAPruner</h1>

<p align="center">
    <a href='https://arxiv.org/abs/2606.07529'>
      <img src='https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=arXiv&logoColor=red' alt='Paper arXiv'>
    </a>
    <a href='https://fz-zsl.github.io/capruner'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
    <a href='https://huggingface.co/fzzsl/CAPruner/tree/main'>
      <img src='https://img.shields.io/badge/Checkpoints-HF-yellow?style=plastic&logo=huggingface&logoColor=yellow' alt='Checkpoints'>
    </a>
</p>


This repository contains the official PyTorch implementation for the paper:

> Shengli Zhou, Xiangchen Wang, Guanhua Chen, and Feng Zheng. 2026. CAPruner: Conceptual-Adjacent Scene Graph Pruner for Enhancing 3D Spatial Reasoning of Large Language Models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics.

## Overview

Large language models (LLMs) have recently been applied to 3D vision-language (3D-VL) tasks, in which spatial reasoning is required to identify target objects based on their positions relative to others (i.e., anchors). To facilitate effective scene layout understanding, scene graphs are commonly used to represent such spatial relations. However, reasoning over full graphs incurs high token costs and computational inefficiencies, motivating the use of scene graph pruning. Existing pruning methods predominantly rely on spatial proximity and often remove task-relevant relations, thereby undermining reliable spatial reasoning. To address these limitations, we derive a key requirement for scene graph pruning: preserving the spatial relations that are most relevant to the specific 3D-VL task. Guided by this insight, we propose the Conceptual-Adjacent Scene Graph Pruner (CAPruner). CAPruner integrates fuzzy semantic relevance with spatial proximity to estimate relation importance, enabling the selection of critical relations in a task-specific context. Moreover, to avoid costly relation-level annotations, CAPruner is trained by supervising the aggregated scores of each node's incident edges. Extensive experiments demonstrate that CAPruner effectively preserves relations essential for spatial reasoning, leading to substantial performance improvements of LLMs on 3D-VL tasks.

## Environment Installation

### CAPruner

Prepare the environment for CAPruner:

```sh
conda create -n capruner python=3.9.17
conda activate capruner
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Data Preparation

1. Download annotations from [Hugging Face](https://huggingface.co/datasets/ZzZZCHS/Chat-Scene/tree/main/annotations) AND [Yandex Disk](https://disk.yandex.ru/d/LpPJgHg8Qg6BpA) and place them in the `annotations/` directory. More details for data preparation can be found [here](https://github.com/CognitiveAISystems/3DGraphLLM/tree/main/preprocess).
2. Extract `output_vlsat.zip` under `annotations/` (i.e., `annotations/output_vlsat/`).
3. Prepare bounding boxes for ground-truth (GT) and Mask3D (M3D) segmentations:

```sh
python preprocess/prepare_gt_boxes.py
python preprocess/prepare_m3d_boxes.py
```

## Model Usage

1. Pretrain CAPruner on ScanRefer, ScanQA, and Multi3DRefer datasets.

```sh
which_python=$(which python)
export PYTHONPATH=${PYTHONPATH}:${which_python}:.
python model/train.py --config config/capruner_m3d_pretrain.yml
```

2. Validate the model: Modify the `config/capruner_m3d_pretrain.yml` to set `pretrained` to the path to the checkpoint.

```sh
python model/train.py --config config/capruner_m3d_pretrain.yml --val
```

3. Prune scene graphs using CAPruner (pruning method, dataset, and split are specified in the config file):

```sh
python model/inference.py --config config/capruner_m3d_pretrain.yml
```

**To skip previous steps**, you can download the inference result based on the [official checkpoint](https://huggingface.co/fzzsl/CAPruner/blob/main/capruner.pth) at [Hugging Face](https://huggingface.co/fzzsl/CAPruner/tree/main/preds). Each file is a `(num_questions, max_num_obj, num_edge_per_obj)` tensor `G`, representing that there is an edge between `j` and `G[i][j][k]` in the scene graph corresponding to the `i`-th 3D-VL task.

4. Generate GNN features for the pruned scene graphs (the `base_path` is the directory where the checkpoint is saved):

```sh
python preprocess/batch_gen_gnn_feat.py --base_path outputs/path_to_model_dir
```

## Acknowledgement

We would like to thank the anonymous reviewers for their constructive feedback.

## Citation

If you find this project useful in your research, please consider citing:

```bibtex
@misc{zhou2026caprunerconceptualadjacentscenegraph,
      title={CAPruner: Conceptual-Adjacent Scene Graph Pruner for Enhancing 3D Spatial Reasoning of Large Language Models}, 
      author={Shengli Zhou and Xiangchen Wang and Guanhua Chen and Feng Zheng},
      year={2026},
      eprint={2606.07529},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.07529}, 
}
```
