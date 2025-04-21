# Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding


[Akash Kumar](https://akash2907.github.io/), [Zsolt Kira](https://faculty.cc.gatech.edu/~zk15/), [Yogesh S Rawat](https://www.crcv.ucf.edu/person/rawat/)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2501.17053)


Official code for our paper "Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding"

## :rocket: News
* **(Jan 21, 2025)**
  * This paper has been accepted for publication in the [ICLR 2025](https://iclr.cc/) conference.
* **(Feb 20, 2025)**
  * Project website is now live at [cospal-webpage](https://akash2907.github.io/cospal_webpage/)
* **(Apr 20, 2025)**
  * Code for our method on weakly supervised spatio-temporal video grounding has been released.

<hr>


![method-diagram](https://akash2907.github.io/cospal_webpage/static/images/cospal.png)
> **Abstract:** *In this work, we focus on Weakly Supervised Spatio-Temporal Video Grounding (WSTVG). It is a multimodal task aimed at localizing specific subjects spatio-temporally based on textual queries without bounding box supervision. Motivated by recent advancements in multi-modal foundation models for grounding tasks, we first explore the potential of state-of-the-art object detection models for WSTVG. Despite their robust zero-shot capabilities, our adaptation reveals significant limitations, including inconsistent temporal predictions, inadequate understanding of complex queries, and challenges in adapting to difficult scenarios. We propose CoSPaL (Contextual Self-Paced Learning), a novel approach which is designed to overcome these limitations. CoSPaL integrates three core components: (1) Tubelet Phrase Grounding (TPG), which introduces spatio-temporal prediction by linking textual queries to tubelets; (2) Contextual Referral Grounding (CRG), which improves comprehension of complex queries by extracting contextual information to refine object identification over time; and (3) Self-Paced Scene Understanding (SPS), a training paradigm that progressively increases task difficulty, enabling the model to adapt to complex scenarios by transitioning from coarse to fine-grained understanding. Code and models are publicly available.*
>

## :trophy: Achievements and Features

- We establish **state-of-the-art results (SOTA)** in weakly-supervised video action detection on HCSTVG-v1, HCSTVG-v2 and VidSTG.
- We propose a context-aware progressive learning approach to solve the problem.


## :hammer_and_wrench: Setup and Installation
We have used `python=3.8.16`, and `torch=1.10.0` for all the code in this repository. It is recommended to follow the below steps and setup your conda environment in the same way to replicate the results mentioned in this paper and repository.

1. Clone this repository into your local machine as follows:
```bash
git clone https://github.com/AKASH2907/copsal-weakly-stvg.git
```
2. Change the current directory to the main project folder :
```bash
cd copsal-weakly-stvg
```
3. To install the project dependencies and libraries, use [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and install the defined environment from the .yml file by running:
```bash
conda env create -f environment.yml
```
4. Activate the newly created conda environment:
```bash
conda activate wstvg 
```
### Datasets
To download and setup the required datasets used in this work, please follow these steps:
1. Download the HCSTVG-v1 dataset and annotations from their official [website](https://github.com/tzhhhh123/HC-STVG). 
2. Download the HCSTVG-v2 dataset and annotations from their official [website](https://github.com/tzhhhh123/HC-STVG).
3. Download the VidSTG dataset and annotations from their official [website](https://github.com/Guaranteer/VidSTG-Dataset). 
4. Use preproc files from this [location](https://github.com/antoyang/TubeDETR/tree/main/preproc) to arrange the annotation.

<!-- 
- `coco/`
  - `annotations/`
    - `instances_train2017.json`
    - `instances_val2017.json`
    - `ovd_instances_train2017_base.json`
    - `ovd_instances_val2017_basetarget.json`
    - `..other coco annotation json files (optional)..`
  - `train2017/`
  - `val2017/`
  - `test2017/`
- `lvis/`
  - `lvis_v1_val.json`
  - `lvis_v1_train.json`
  - `lvis_v1_val_subset.json` 

 ### Model Weights
 All the pre-trained model weights can be downloaded from this link: [model weights](https://huggingface.co/akashkumar29/stable-mean-teacher/tree/main). 

 - **I3D_weights.pth**: I3D model weights used for Video Action detection task for intitalization is found at this [link](https://github.com/piergiaj/pytorch-i3d/tree/master/models).
 
### Training
Run the following script from the main project directory as follows:
1. JHMDB-21
   ```bash
   python semi_loc_feat_const_pa_stn_aug_add_aux_dop_jhmdb.py --epochs 50 --bs 8 --lr 1e-4 --pkl_file_label train_annots_10_labeled_random.pkl --pkl_file_unlabel train_annots_90_unlabeled_random.pkl --wt_loc 1 --wt_cls 1 --wt_cons 0.1 --const_loss l2 --thresh_epoch 11 -at 2 -ema 0.99 --opt3 --opt4 --ramp_thresh 0 --scheduler --exp_id 10_per/semi_loc_const_l2_thresh_10_eps_per10_aug_st_aux_op_raw_ramp_const_thresh_0_l2_dop_temporal_ramp_up_both_dop_l2_till_0_w_scheduler 
   ```
2. UCF101-24
    ```bash
   python semi_loc_feat_const_pa_stn_aug_add_aux_dop_ucf.py --epochs 50 --bs 8 --lr 1e-4 --txt_file_label jhmdb_classes_list_per_20_labeled.txt --txt_file_unlabel jhmdb_classes_list_per_80_unlabeled.txt --wt_loc 1 --wt_cls 1 --wt_cons 0.1 --const_loss l2 --thresh_epoch 11 -at 2 -ema 0.99 --opt3 --opt4 --ramp_thresh 16 --exp_id semi_loc_const_l2_thresh_10_eps_per20_aug_st_aux_op_raw_ramp_const_thresh_15_l2_dop_temporal_ramp_up_both_dop_l2_till_15 
   ```

* `--epochs`: Number of epochs.
* `--bs`: batch size.
* `--lr`: Learning rate.
* `--pkl_file_label/txt_file_label`: Labeled set video list.
* `--pkl_file_unlabel/txt_file_unlabel`: Unlabeled set video list.
* `--wt_loc/wt_cls/wt_cons`: Localization/Classification/Consistency loss weights.
* `--thresh_epoch/ramp_thresh`: Threshold epoch to model reach it's confidence (ignore labeled set consistency loss till then)/Ramping up loss epoch.
* `--opt3/opt4`: Ramp up DoP + L2 both, based on ramp thresh epochs/Both main+aux loss added same weight w/ any rampup.
* `--at`: Aug type: 0-spatial, 1- temporal, 2 - both.

### Evaluation

Run the following script from the main project directory as follows:
1. JHMDB-21
   ```bash
   ython multi_model_evalCaps_jhmdb.py --ckpt EXP-FOLDER-NAME 
   ```
2. UCF101-24
    ```bash
   python multi_model_evalCaps_ucf.py --ckpt EXP-FOLDER-NAME 
   ```
* `--ckpt`: Checkpoint Path.

## :medal_military: Semi-Supervised Action Detection Results on UCF101-24 and JHMDB21

This table presents the performance of various semi-supervised action detection approaches on the **UCF101-24** and **JHMDB21** datasets using the I3D backbone. The table reports results for different annotation percentages, comparing f@0.5, v@0.2, and v@0.5 metrics.

### Results Table
| **Semi-Supervised Approaches** | **Backbone** | **Annot.** | **UCF101-24** f@0.5 | v@0.2 | v@0.5 | **JHMDB21** Annot. | f@0.5 | v@0.2 | v@0.5 |
|--------------------------------|-------------|------------|----------------|------|------|------------|------|------|------|
| MixMatch (Berthelot et al. 2019) | I3D | 10% | 10.3 | 54.7 | 4.9 | 30% | 7.5 | 46.2 | 5.8 |
| Pseudo-label (Lee et al. 2013) | I3D | 10% | 59.3 | 89.9 | 58.3 | 20% | 55.3 | 87.6 | 52.0 |
| ISD (Jeong et al. 2021) | I3D | 10% | 60.2 | 91.3 | 64.0 | 20% | 57.8 | 90.2 | 57.0 |
| E2E-SSL (Kumar and Rawat 2022) | I3D | 10% | 65.2 | 91.8 | 66.7 | 20% | 59.1 | 93.2 | 58.7 |
| Mean Teacher (Tarvainen and Valpola 2017) | I3D | 10% | 67.3 | 92.7 | 70.5 | 20% | 56.3 | 88.8 | 52.8 |
| **Stable Mean Teacher (Ours)** | I3D | 10% | **73.9** | **95.8** | **76.3** | 20% | **69.8** | **98.8** | **70.7** |
| | | | _(↑ 6.6)_ | _(↑ 3.1)_ | _(↑ 5.8)_ | | _(↑ 13.5)_ | _(↑ 10.0)_ | _(↑ 17.9)_ |

### Notes:
- `f@0.5`: Frame-level detection accuracy at IoU 0.5.
- `v@0.2`: Video-level detection accuracy at IoU 0.2.
- `v@0.5`: Video-level detection accuracy at IoU 0.5.

-->
## :framed_picture: Qualitative Visualization
![qual-analysis-1](https://akash2907.github.io/cospal_webpage/static/images/cospal_qual_analysis.png)

## :email: Contact
Should you have any questions, please create an issue in this repository or contact at akash.kumar@ucf.edu

## :black_nib: Citation
If you found our work helpful, please consider starring the repository ⭐⭐⭐ and citing our work as follows:


```bibtex
@article{kumar2025contextual,
      title={Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding},
      author={Kumar, Akash and Kira, Zsolt and Rawat, Yogesh Singh},
      journal={arXiv preprint arXiv:2501.17053},
      year={2025}
    }
```

