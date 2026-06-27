# CRSITracker: Multi-Object Tracking for Satellite Videos via Contrastive Reweighting and Sparse Cross-Frame Injection

This repository provides the official implementation of CRSITracker for satellite-video multi-object tracking.

## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

Tested environment:

```text
PyTorch 1.8.1 + CUDA 11.1
```

## Dataset Preparation

The experiments in the paper are conducted on AIR-MOT and VISO. Please download the datasets from their original sources and organize them under paths such as:

```text
/workspace/AIR-MOT/
  annotations/
  train/
  test/

/workspace/VISO/
  annotations/
  train/
  test/
```

Detailed preprocessing scripts, annotation conversion instructions, and split settings used in the paper are being organized and will be released soon.

## Model Checkpoints

Model checkpoints and pretrained initialization files used for reproducing the reported results are being organized and will be released soon.

## Training and Testing

First, enter the source directory:

```bash
cd src/
```

### AIR-MOT

#### Train

```bash
CUDA_VISIBLE_DEVICES=0 python main.py tracking --gpus 0 --dataset custom --custom_dataset_ann_path /workspace/AIR-MOT/annotations --custom_dataset_img_path /workspace/AIR-MOT --num_classes 2 --input_h 608 --input_w 992 --batch_size 10 --lr 1.25e-4 --lr_step '25,40' --save_point '30,32' --num_epochs 35 --val_intervals 25 --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --load_model /models/crowdhuman.pth --atten_method reweight --exp_id JL_Att --mask_enable --mask_adaptive --upm --gat
```

#### Test

```bash
python demo.py tracking --video_h 1080 --video_w 1920 --input_h 1088 --input_w 1920 --num_classes 2 --demo /workspace/AIR-MOT/test --dataset custom --demo_videos --track_thresh 0.4 --pre_thresh 0.5 --pre_hm --ltrb_amodal --exp_id JL_Att --load_model /exp/tracking/JL_Att/model_32.pth --atten_method reweight --max_age 30 --mode test
```

### VISO

#### Train

```bash
CUDA_VISIBLE_DEVICES=0 python main_viso.py tracking --gpus 0 --dataset viso --custom_dataset_ann_path /workspace/VISO/annotations --custom_dataset_img_path /workspace/VISO --num_classes 4 --input_h 1024 --input_w 1024 --batch_size 4 --lr 1.25e-4 --lr_step '20,30' --save_point '30' --num_epochs 35 --val_intervals 20 --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --load_model /models/crowdhuman.pth --atten_method reweight --exp_id JL_Att-VS --mask_enable --mask_adaptive --upm --gat --lowfeat --down_ratio 2
```

#### Test

```bash
python demo_viso.py tracking --video_h 1024 --video_w 1024 --input_h 1024 --input_w 1024 --num_classes 4 --demo /workspace/VISO/test --dataset viso --demo_videos --track_thresh 0.4 --pre_thresh 0.5 --pre_hm --ltrb_amodal --exp_id JL_Att-VS --load_model /exp/tracking/JL_Att-VS/model_30.pth --atten_method reweight --max_age 30 --mode test --lowfeat --down_ratio 2
```

## Evaluation

### VISO

First, enter the evaluation directory:

```bash
cd TrackEval_sat/
```

Then run:

```bash
python scripts/run_sat_challenge.py --TRACKERS_TO_EVAL CFTracker
```

To evaluate your own results:

1. Put tracking results in `TrackEval_sat/data/trackers/mot_challenge/MOT16-val/Your`.
2. Modify `tracker_name` in `scripts/run_sat_challenge.py`.
3. Run:

```bash
python scripts/run_sat_challenge.py --TRACKERS_TO_EVAL Your
```

Note: Because the test ground truth provided by VISO is not fully consistent with the ground truth used by DSFNet, the paper reports VISO results using the DSFNet ground truth for fair comparison. The reported results can be reproduced by replacing the files in `TrackEval_sat/data/gt/mot_challenge/MOT16-val/` with those in `TrackEval_sat/viso_gt_bak/` and setting the threshold to 0.4.


## Reproducibility Status

The core implementation and training/testing commands are provided in this repository. Detailed dataset preprocessing scripts, paper checkpoints, and additional evaluation files are being organized and will be released soon.

## License

The original implementation and modifications introduced in CRSITracker are released under the MIT License. Third-party components adapted from upstream open-source projects retain their original licenses and copyright notices. Please refer to [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.

## Acknowledgements

This repository is developed based on the open-source CFTracker and CenterTrack codebases. We thank the authors of these projects and the related third-party components.
