# CRSITracker: Multi-Object Tracking for Satellite Videos via Contrastive Reweighting and Sparse Cross-7 Frame Injection

## Installation
Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

My env: torch1.8.1 + cudnn 11.1


## Use CenterTrack
First, `cd src/`

**For AIR-MOT dataset：**
### 1. train
```
CUDA_VISIBLE_DEVICES=0 python main.py tracking --gpus 0 --dataset custom --custom_dataset_ann_path /workspace/AIR-MOT/annotations --custom_dataset_img_path /workspace/AIR-MOT --num_classes 2 --input_h 608 --input_w 992 --batch_size 10 --lr 1.25e-4 --lr_step '25,40' --save_point '30,32' --num_epochs 35 --val_intervals 25 --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --load_model /models/crowdhuman.pth --atten_method reweight --exp_id JL_Att --mask_enable --mask_adaptive --upm --gat
```
### 2. test
```
python demo.py tracking --video_h 1080 --video_w 1920 --input_h 1088 --input_w 1920 --num_class 2 --demo /workspace/AIR-MOT/test --dataset custom --demo_videos --track_thresh 0.4 --pre_thresh 0.5 --pre_hm --ltrb_amodal --exp_id JL_Att --load_model /exp/tracking/JL_Att/model_32.pth --atten_method reweight --max_age 30 --mode test
```

**For VISO dataset：**
### 1. train
```
CUDA_VISIBLE_DEVICES=0 python main_viso.py tracking --gpus 0 --dataset viso --custom_dataset_ann_path /workspace/VISO/annotations --custom_dataset_img_path /workspace/VISO --num_classes 4 --input_h 1024 --input_w 1024 --batch_size 4 --lr 1.25e-4 --lr_step '20,30' --save_point '30' --num_epochs 35 --val_intervals 20 --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --load_model /models/crowdhuman.pth --atten_method reweight --exp_id JL_Att-VS --mask_enable --mask_adaptive --upm --gat --lowfeat --down_ratio 2
```
### 2. test
```
python demo_viso.py tracking --video_h 1024 --video_w 1024 --input_h 1024 --input_w 1024 --num_class 4 --demo /workspace/VISO/test --dataset viso --demo_videos --track_thresh 0.4 --pre_thresh 0.5 --pre_hm --ltrb_amodal --exp_id JL_Att-VS --load_model /exp/tracking/JL_Att-VS/model_30.pth --atten_method reweight --max_age 30 --mode test --lowfeat --down_ratio 2
```
## Evalution for VISO
First, `cd TrackEval_sat/`, then run `python scripts/run_sat_challenge.py --TRACKERS_TO_EVAL CFTracker`

To evaluate your results, 
- put your results in `TrackEval_sat/data/trackers/mot_challenge/MOT16-val/Your`, 
- modify `tracker_name` in `scripts/run_sat_challenge.py` 
- run `python scripts/run_sat_challenge.py --TRACKERS_TO_EVAL Your`.

Notice: Due to my discovery that the test-gt provided by VISO is not completely consistent with the gt provided by DSFNet, for a fair comparison, I used the DSFNet gt to obtain the following evaluation results:


(The results in the paper can be obtained by replacing the files in `TrackEval_sat/data/gt/mot_challenge/MOT16-val/` with `TrackEval_sat/viso_gt_bak/`, threshold is 0.4)


## Thanks
This code is heavily borrowed from [CenterTrack](https://github.com/xingyizhou/CenterTrack), thanks the authors.

```
