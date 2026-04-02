from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import copy
import numpy as np
import time
import torch
import torch.nn.functional as F
import math
import os

from model.model import create_model, load_model
from model.decode import generic_decode
from model.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform, affine_transform
from utils.image import draw_umich_gaussian, gaussian_radius
from utils.post_process import generic_post_process
from utils.debugger import Debugger
from dataset.dataset_factory import get_dataset
from model.nms_wrapper import nms

class InfLoss(torch.nn.Module):
    def __init__(self):
        super(InfLoss, self).__init__()
        self.eps = 1e-9
        self.pool = torch.nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        # self.pool = torch.nn.MaxPool2d(3, stride=1, padding=1)
        self.mseloss = torch.nn.MSELoss(reduction='none')
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, pre_out, out):
        pre_hm = pre_out["hm"]
        hm = out["hm"]
        offset = out['tracking']

        p_0_y, p_0_x = torch.meshgrid(torch.arange(0, hm.shape[2]), torch.arange(0, hm.shape[3]))
        p_0_y, p_0_x = p_0_y.contiguous(), p_0_x.contiguous()
        p_0 = torch.stack((p_0_x, p_0_y), dim=0).unsqueeze(0).repeat(hm.shape[0], 1, 1, 1).to(hm.device)
        p_0 = p_0.clone() + offset
        # p_0[:, 0, :, :] = torch.clamp(p_0[:, 0, :, :], 0, hm.shape[3] - 1)  # 这个操作会丢失梯度
        # p_0[:, 1, :, :] = torch.clamp(p_0[:, 1, :, :], 0, hm.shape[2] - 1)
        p_0 = p_0.permute(0, 2, 3, 1)
        p_0[:, :, :, 0] = p_0[:, :, :, 0] / ((p_0.shape[2] - 1) / 2) - 1
        p_0[:, :, :, 1] = p_0[:, :, :, 1] / ((p_0.shape[1] - 1) / 2) - 1
        hmf = torch.nn.functional.grid_sample(hm, p_0, mode='bilinear',
                                                        padding_mode='border', align_corners=False)

        mask = (pre_hm == pre_hm.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
        mask = torch.mul(mask, pre_hm)

        # zero = torch.zeros_like(mask)
        # mask = torch.where(mask < 0.3, zero, mask)

        mask = torch.sum(mask, dim=1, keepdim=True)
        pre_hm2 = torch.softmax(pre_out["hm"]/2.0, dim=1)
        # loss_hm = self.l2_loss(hm[:,0,:,:], out["hm"][:,0,:,:], mask)
        loss_hm1 = self.l2_loss(pre_hm[:,0,:,:], hmf[:,0,:,:], mask)
        # loss_hm2 = self.l2_loss(pre_hm2[:, 1, :, :], hmf[:, 1, :, :], mask)


        loss_wh = self.l2_loss(pre_out["wh"], out["wh"], mask)
        # loss_ltrb = 0.5 * self.l2_loss(pre_out["ltrb_amodal"], out["ltrb_amodal"], mask)
        loss_off = self.cos_loss(pre_out["tracking"], out["tracking"], mask)

        dist_pre = torch.norm(pre_out["tracking"], dim=1)
        dist = torch.norm(out["tracking"], dim=1)

        loss_off2 = self.l2_loss(dist_pre, dist, mask)
        loss = loss_hm1 + loss_wh  + loss_off + loss_off2
        return loss


    def l2_loss(self, pre_out, out, mask):
        loss = self.mseloss(pre_out, out)
        loss = torch.sum(loss * mask)
        # loss = ((pre_out - out) ** 2 * mask) + self.eps
        # loss = torch.sum(loss) ** 0.5
        return loss

    def cos_loss(self, pre_out, out, mask):
        loss = mask * (1-self.cos(pre_out, out))
        return 0.5 * torch.sum(loss)

class Detector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        if opt.track_method == "byte":
            from utils.byteTracker import Tracker
        elif opt.track_method == "sort":
            from utils.sort import MCSortT as Tracker
        else:
            from utils.tracker import Tracker

        print('Creating model...')
        self.model = create_model(
            opt.arch, opt.heads, opt.head_conv, opt=opt)
        self.model = load_model(self.model, opt.load_model, opt)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.opt = opt
        self.trained_dataset = get_dataset(opt.dataset)
        self.mean = np.array(
            self.trained_dataset.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(
            self.trained_dataset.std, dtype=np.float32).reshape(1, 1, 3)
        self.pause = not opt.no_pause
        self.rest_focal_length = self.trained_dataset.rest_focal_length \
            if self.opt.test_focal_length < 0 else self.opt.test_focal_length
        self.flip_idx = self.trained_dataset.flip_idx
        self.cnt = 0
        self.pre_images = None
        self.pre_image_ori = None
        self.tracker = Tracker(opt)
        self.leiji_htmap = None
        self.debugger = Debugger(opt=opt, dataset=self.trained_dataset)

        if self.opt.inference_train:
            self.optimizer = torch.optim.Adam(self.model.parameters(), opt.lr)
            self.loss = InfLoss()
            self.pre_output = None

    def reset(self):
        self.cnt = 0
        self.pre_images = None
        self.pre_image_ori = None
        self.leiji_htmap = None
        self.tracker.reset()

    def reset_tracking(self):
        self.tracker.reset()
        self.pre_images = None
        self.pre_image_ori = None
        self.leiji_htmap = None

    def run(self, image_or_path_or_tensor, meta={}, f_rst=[], other_bbox=[], cnt=0):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, track_time, tot_time, display_time = 0, 0, 0, 0
        self.debugger.clear()
        start_time = time.time()

        # read image
        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            image = cv2.imread(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []

        # for multi-scale testing
        for scale in self.opt.test_scales:
            scale_start_time = time.time()
            if not pre_processed:
                # not prefetch testing or demo
                images, meta = self.pre_process(image, scale, meta)
            else:
                # prefetch testing
                images = pre_processed_images['images'][scale][0]

                # 先拿 scale 对应的 meta
                meta_scale = pre_processed_images['meta'][scale]
                meta = {}
                for k, v in meta_scale.items():
                    # 只对 tensor / ndarray 做 numpy()[0]，其余原样保留
                    if hasattr(v, 'numpy'):
                        meta[k] = v.numpy()[0]
                    else:
                        meta[k] = v

                # 根级的 meta 里可能还有 pre_dets / cur_dets / gt_det 等信息
                root_meta = pre_processed_images.get('meta', {})
                if isinstance(root_meta, dict):
                    if 'pre_dets' in root_meta:
                        meta['pre_dets'] = root_meta['pre_dets']
                    if 'cur_dets' in root_meta:
                        meta['cur_dets'] = root_meta['cur_dets']

            images = images.to(self.opt.device, non_blocking=self.opt.non_block_test)

            # initializing tracker
            pre_hms, pre_inds, boxes_prev_xywh = None, None, None
            if self.opt.tracking:
                # initialize the first frame
                if self.pre_images is None:
                    print('Initialize tracking!')
                    self.pre_images = images
                    self.tracker.init_track(
                            meta['pre_dets'] if 'pre_dets' in meta else [])

                if self.opt.pre_hm or getattr(self.opt, 'upm', False):
                    if self.opt.leiji_test:
                        pre_detm, pre_hms, pre_inds, boxes_prev_xywh = self._get_additional_inputs(
                            self.tracker.tracks, meta,
                            with_hm=not self.opt.zero_pre_hm,
                            other_bbox=other_bbox)
                        pre_dets = pre_detm
                    else:
                        pre_hms, pre_inds, boxes_prev_xywh = self._get_additional_inputs(
                            self.tracker.tracks, meta,
                            with_hm=not self.opt.zero_pre_hm,
                            other_bbox=other_bbox)

            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            # run the network
            # output: the output feature maps, only used for visualizing
            # dets: output tensors after extracting peaks
            if self.opt.leiji_test:
                output, dets, forward_time = self.process(
                    images, self.pre_images, pre_hms, pre_inds, return_time=True, pre_dets=pre_dets,boxes_prev_xywh=boxes_prev_xywh)
            else:
                output, dets, forward_time = self.process(
                    images, self.pre_images, pre_hms, pre_inds, return_time=True,boxes_prev_xywh=boxes_prev_xywh)
            if self.opt.inference_train and self.pre_output is None:
                for key in output:
                    output[key] = output[key].detach()
                self.pre_output = output
            elif self.opt.inference_train:
                loss = self.loss(self.pre_output, output)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                for key in output:
                    output[key] = output[key].detach()
                self.pre_output = output
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            # convert the cropped and 4x downsampled output coordinate system
            # back to the input image coordinate system
            result = self.post_process(dets, meta, scale)
            post_process_time = time.time()
            post_time += post_process_time - decode_time


            detections.append(result)
        # merge multi-scale testing results
        results = self.merge_outputs(detections)
        if self.opt.give_first_gt and len(f_rst) != 0:
            results = f_rst
        if self.opt.give_sot_gt and cnt == 1:
            for i in range(len(results)):
                if results[i]['score'] > self.opt.track_thresh:
                    results[i]['score'] = 1

        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time

        if self.opt.tracking:
            # public detection mode in MOT challenge
            public_det = meta['cur_dets'] if self.opt.public_det else None
            # add tracking id to results
            results = self.tracker.step(results, public_det)
            self.pre_images = images

        tracking_time = time.time()
        track_time += tracking_time - end_time
        tot_time += tracking_time - start_time

        # return results and run time
        ret = {'results': results, 'tot': tot_time, 'load': load_time,
               'pre': pre_time, 'net': net_time, 'dec': dec_time,
               'post': post_time, 'merge': merge_time, 'track': track_time,
               'display': display_time}
        if self.opt.save_video or self.opt.mysave_imgs:
            try:
                # return debug image for saving video
                ret.update({'generic': self.debugger.imgs['generic']})
            except:
                pass
        return ret

    def _transform_scale(self, image, scale=1):
        '''
          Prepare input image in different testing modes.
            Currently support: fix short size/ center crop to a fixed size/
            keep original resolution but pad to a multiplication of 32
        '''
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        if self.opt.fix_short > 0:
            if height < width:
                inp_height = self.opt.fix_short
                inp_width = (int(width / height * self.opt.fix_short) + 63) // 64 * 64
            else:
                inp_height = (int(height / width * self.opt.fix_short) + 63) // 64 * 64
                inp_width = self.opt.fix_short
            c = np.array([width / 2, height / 2], dtype=np.float32)
            s = np.array([width, height], dtype=np.float32)
        elif self.opt.fix_res:
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
            # s = np.array([inp_width, inp_height], dtype=np.float32)
        else:
            inp_height = (new_height | self.opt.pad) + 1
            inp_width = (new_width | self.opt.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image, c, s, inp_width, inp_height, height, width

    def pre_process(self, image, scale, input_meta={}):
        '''
        Crop, resize, and normalize image. Gather meta data for post processing
          and tracking.
        '''
        resized_image, c, s, inp_width, inp_height, height, width = \
            self._transform_scale(image)
        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        out_height = inp_height // self.opt.down_ratio
        out_width = inp_width // self.opt.down_ratio
        trans_output = get_affine_transform(c, s, 0, [out_width, out_height])

        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        if self.opt.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta = {'calib': np.array(input_meta['calib'], dtype=np.float32) \
            if 'calib' in input_meta else \
            self._get_default_calib(width, height)}
        meta.update({'c': c, 's': s, 'height': height, 'width': width,
                     'out_height': out_height, 'out_width': out_width,
                     'inp_height': inp_height, 'inp_width': inp_width,
                     'trans_input': trans_input, 'trans_output': trans_output})
        if 'pre_dets' in input_meta:
            meta['pre_dets'] = input_meta['pre_dets']
        if 'cur_dets' in input_meta:
            meta['cur_dets'] = input_meta['cur_dets']
        # ====== 新增：把数据集 meta 里的 GT 信息也带进来 ======
        for k in ['gt_det', 'img_id', 'img_path']:
            if k in input_meta:
                meta[k] = input_meta[k]
        # ====== 新增结束 ======
        return images, meta

    def _trans_bbox(self, bbox, trans, width, height):
        '''
        Transform bounding boxes according to image crop.
        '''
        bbox = np.array(copy.deepcopy(bbox), dtype=np.float32)
        bbox[:2] = affine_transform(bbox[:2], trans)
        bbox[2:] = affine_transform(bbox[2:], trans)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
        return bbox

    def _get_additional_inputs(self, dets, meta, with_hm=True, other_bbox=None):
        '''
        Render input heatmap from previous trackings.
        '''
        trans_input, trans_output = meta['trans_input'], meta['trans_output']
        inp_width, inp_height = meta['inp_width'], meta['inp_height']
        out_width, out_height = meta['out_width'], meta['out_height']
        input_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32)
        pre_detm = np.zeros((2, inp_height, inp_width), dtype=np.float32)

        output_inds = []
        boxes_prev_xywh = []
        for det in dets:
            if det['score'] < self.opt.pre_thresh or det['active'] == 0:
                continue
            # bbox = self._trans_bbox(det['bbox'], trans_input, inp_width, inp_height)
            # x1, y1, x2, y2 = bbox
            # w = x2 - x1
            # h = y2 - y1
            # boxes_prev_xywh.append([x1, y1, w, h])
            # bbox_out = self._trans_bbox(
            #     det['bbox'], trans_output, out_width, out_height)
            # h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            # if (h > 0 and w > 0):
            #     radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            #     if self.opt.big_radius:
            #         radius = max(radius, min(math.ceil(h), math.ceil(w)) / 2.0)
            #     radius = max(0, int(radius))
            #     ct = np.array(
            #         [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            bbox = self._trans_bbox(det['bbox'], trans_input, inp_width, inp_height)
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            # if w <= 0 or h <= 0:
            #     continue

            # 使用“中心 + 宽高”的语义，和训练保持一致
            cx = x1 + 0.5 * w
            cy = y1 + 0.5 * h
            #boxes_prev_xywh.append([x1, y1, w, h])
            boxes_prev_xywh.append([cx, cy, w, h])

            # 下面画热力图仍然可以用 bbox 本身，不需要改
            bbox_out = self._trans_bbox(det['bbox'], trans_output, out_width, out_height)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if (h > 0 and w > 0):
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                if self.opt.big_radius:
                    radius = max(radius, min(math.ceil(h), math.ceil(w)) / 2.0)
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)

                ct_int = ct.astype(np.int32)
                if with_hm:
                    draw_umich_gaussian(input_hm[0], ct_int, radius)
                    if self.opt.leiji_test:
                        draw_umich_gaussian(pre_detm[det['class']-1], ct_int, radius)
                ct_out = np.array(
                    [(bbox_out[0] + bbox_out[2]) / 2,
                     (bbox_out[1] + bbox_out[3]) / 2], dtype=np.int32)
                output_inds.append(ct_out[1] * out_width + ct_out[0])
        for det in other_bbox:
            bbox = self._trans_bbox(det['bbox'], trans_input, inp_width, inp_height)
            bbox_out = self._trans_bbox(
                det['bbox'], trans_output, out_width, out_height)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if (h > 0 and w > 0):
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                if self.opt.big_radius:
                    radius = max(radius, min(math.ceil(h), math.ceil(w)) / 2.0)
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                if with_hm:
                    draw_umich_gaussian(input_hm[0], ct_int, radius)
                    # if self.opt.leiji_test:
                    #     draw_umich_gaussian(pre_detm[det['class']-1], ct_int, radius)
                ct_out = np.array(
                    [(bbox_out[0] + bbox_out[2]) / 2,
                     (bbox_out[1] + bbox_out[3]) / 2], dtype=np.int32)
                output_inds.append(ct_out[1] * out_width + ct_out[0])
        if with_hm:
            input_hm = input_hm[np.newaxis]
            pre_detm = pre_detm[np.newaxis]
            if self.opt.flip_test:
                input_hm = np.concatenate((input_hm, input_hm[:, :, :, ::-1]), axis=0)
                pre_detm = np.concatenate((pre_detm, pre_detm[:, :, :, ::-1]), axis=0)
            input_hm = torch.from_numpy(input_hm).to(self.opt.device)
            pre_detm = torch.from_numpy(pre_detm).to(self.opt.device)
            pre_detm = F.interpolate(pre_detm, scale_factor=0.25)
        if len(boxes_prev_xywh) > 0:
            boxes_prev_xywh = torch.from_numpy(
                np.array(boxes_prev_xywh, dtype=np.float32)
            ).to(self.opt.device)
        else:
            boxes_prev_xywh = None

        output_inds = np.array(output_inds, np.int64).reshape(1, -1)
        output_inds = torch.from_numpy(output_inds).to(self.opt.device)

        if self.opt.leiji_test:
            return pre_detm, input_hm, output_inds, boxes_prev_xywh
        return input_hm, output_inds, boxes_prev_xywh

    def _get_default_calib(self, width, height):
        calib = np.array([[self.rest_focal_length, 0, width / 2, 0],
                          [0, self.rest_focal_length, height / 2, 0],
                          [0, 0, 1, 0]])
        return calib

    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = output['hm'].sigmoid_()
        if 'hm_hp' in output:
            output['hm_hp'] = output['hm_hp'].sigmoid_()
        if 'dep' in output:
            output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
            output['dep'] *= self.opt.depth_scale
        return output

    def _flip_output(self, output):
        average_flips = ['hm', 'wh', 'dep', 'dim']
        neg_average_flips = ['amodel_offset']
        single_flips = ['ltrb', 'nuscenes_att', 'velocity', 'ltrb_amodal', 'reg',
                        'hp_offset', 'rot', 'tracking', 'pre_hm']
        for head in output:
            if head in average_flips:
                output[head] = (output[head][0:1] + flip_tensor(output[head][1:2])) / 2
            if head in neg_average_flips:
                flipped_tensor = flip_tensor(output[head][1:2])
                flipped_tensor[:, 0::2] *= -1
                output[head] = (output[head][0:1] + flipped_tensor) / 2
            if head in single_flips:
                output[head] = output[head][0:1]
            if head == 'hps':
                output['hps'] = (output['hps'][0:1] +
                                 flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
            if head == 'hm_hp':
                output['hm_hp'] = (output['hm_hp'][0:1] + \
                                   flip_lr(output['hm_hp'][1:2], self.flip_idx)) / 2

        return output

    def process(self, images, pre_images=None, pre_hms=None,
                pre_inds=None, return_time=False, pre_dets=None, boxes_prev_xywh=None):
        if self.opt.inference_train:
            torch.cuda.synchronize()

            output = self.model(images, pre_images, pre_hms)[-1]
            output = self._sigmoid_output(output)
            output.update({'pre_inds': pre_inds})
            if self.opt.flip_test:
                output = self._flip_output(output)
            torch.cuda.synchronize()
            forward_time = time.time()

            if self.opt.atten_method != "none":
                if self.opt.leiji_test:
                    if self.leiji_htmap is not None:
                        self.leiji_htmap = 0.5 * pre_dets + 0.5 * self.leiji_htmap
                    dets, self.model.heat_att, self.leiji_htmap = generic_decode(
                        output, K=self.opt.K, opt=self.opt, leiji_htmap=self.leiji_htmap)
                elif self.opt.auto_thresh:
                    dets, self.model.heat_att, self.opt.track_thresh = generic_decode(
                        output, K=self.opt.K, opt=self.opt)
                    self.opt.out_thresh = self.opt.track_thresh
                    self.opt.new_thresh = self.opt.track_thresh
                else:
                    dets, self.model.heat_att, self.model.offset = generic_decode(
                        output, K=self.opt.K, opt=self.opt)
            else:
                dets = generic_decode(output, K=self.opt.K, opt=self.opt)

            torch.cuda.synchronize()

            for k in dets:
                dets[k] = dets[k].detach().cpu().numpy()

            if return_time:
                return output, dets, forward_time
            else:
                return output, dets

        else:
            with torch.no_grad():
                torch.cuda.synchronize()

                outputs_all = self.model(images, pre_images, pre_hms)
                output = outputs_all[-1]
                model_core = self.model.module if hasattr(self.model, 'module') else self.model

                batch_ids = None
                if boxes_prev_xywh is not None and boxes_prev_xywh.numel() > 0 \
                and getattr(self.opt, 'upm', False):
                    batch_ids = torch.zeros(
                        boxes_prev_xywh.shape[0],
                        dtype=torch.long,
                        device=boxes_prev_xywh.device
                    )

                if getattr(self.opt, 'upm', False) and getattr(self.opt, 'gat', False) \
                and hasattr(model_core, 'upm') and hasattr(model_core, 'gat') \
                and batch_ids is not None:

                    upm_out = model_core.upm.infer(
                        feat_prev=model_core.last_feat_prev,
                        feat_curr=model_core.last_feat_curr,
                        boxes_prev_xywh=boxes_prev_xywh,
                        batch_ids=batch_ids,
                    )
                    p_hat = upm_out['p_hat']
                    peak_ratio_vec = upm_out.get('peak_ratio_vec', None)
                    disp_conf_vec = upm_out.get('disp_conf_vec', None)

                    F_prev = model_core.last_feat_prev
                    F_curr = model_core.last_feat_curr

                    F_enh = model_core.gat.fuse(
                        F_prev,
                        F_curr,
                        boxes_prev_xywh,
                        p_hat,
                        disp_conf=disp_conf_vec,
                        epoch=None,
                    )
                    heads_all = model_core.heads_from_single(F_enh)
                    output = heads_all[-1]

                output = self._sigmoid_output(output)
                output.update({'pre_inds': pre_inds})
                if self.opt.flip_test:
                    output = self._flip_output(output)

                torch.cuda.synchronize()
                forward_time = time.time()

                if self.opt.atten_method != "none":
                    if self.opt.leiji_test:
                        if self.leiji_htmap is not None:
                            self.leiji_htmap = 0.5 * pre_dets + 0.5 * self.leiji_htmap
                        dets, self.model.heat_att, self.leiji_htmap = generic_decode(
                            output, K=self.opt.K, opt=self.opt, leiji_htmap=self.leiji_htmap)
                    elif self.opt.auto_thresh:
                        dets, self.model.heat_att, self.opt.track_thresh = generic_decode(
                            output, K=self.opt.K, opt=self.opt)
                        self.opt.out_thresh = self.opt.track_thresh
                        self.opt.new_thresh = self.opt.track_thresh
                    else:
                        dets, self.model.heat_att, self.model.offset = generic_decode(
                            output, K=self.opt.K, opt=self.opt)
                else:
                    dets = generic_decode(output, K=self.opt.K, opt=self.opt)

                torch.cuda.synchronize()

                for k in dets:
                    dets[k] = dets[k].detach().cpu().numpy()

            if return_time:
                return output, dets, forward_time
            else:
                return output, dets
    def apply_nms(self, all_boxes, thresh):
        """Apply non-maximum suppression to all predicted boxes output by the
        test_net method.
        """
        num_classes = len(all_boxes)
        num_images = 1  # TODO: demo only support 1 batchsize now
        nms_boxes = []
        for cls_ind in range(num_classes):
            temp_boxes = np.array(all_boxes[cls_ind])
            for im_ind in range(num_images):
                boxes = []
                for box in all_boxes[cls_ind]:
                    box_list = np.append(box["bbox"], box["score"])
                    boxes.append(box_list)
                dets = np.array(boxes, dtype=np.float32)
                if len(dets) == 0:
                    continue
                # print('dets', dets)
                x1 = dets[:, 0]
                y1 = dets[:, 1]
                x2 = dets[:, 2]
                y2 = dets[:, 3]
                scores = dets[:, 4]
                inds = np.where((x2 > x1) & (y2 > y1))[0]
                dets = dets[inds, :]
                if dets == []:
                    continue

                keep = nms(dets, thresh)
                if len(keep) == 0:
                    continue
                nms_boxes.append(temp_boxes[keep].tolist())
        return nms_boxes


    def post_process(self, dets, meta, scale=1):
        dets = generic_post_process(
            self.opt, dets, [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes,
            [meta['calib']], meta['height'], meta['width'])

        if scale != 1:
            for i in range(len(dets[0])):
                for k in ['bbox', 'hps']:
                    if k in dets[0][i]:
                        dets[0][i][k] = (np.array(
                            dets[0][i][k], np.float32) / scale).tolist()
        if self.opt.nms:
            dets = self.apply_nms(dets, self.opt.nms_thresh)
        if len(dets) == 0:
            return dets
        return dets[0]

    def merge_outputs(self, detections):
        assert len(self.opt.test_scales) == 1, 'multi_scale not supported!'
        results = []
        for i in range(len(detections[0])):
            if detections[0][i]['score'] > self.opt.out_thresh:
                results.append(detections[0][i])
        return results
