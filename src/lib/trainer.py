from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import torch
import numpy as np
import torch.nn.functional as F
from progress.bar import Bar
from model.data_parallel import DataParallel
from utils.utils import AverageMeter
from model.losses import FastFocalLoss, RegWeightedL1Loss
from model.losses import BinRotLoss, WeightedBCELoss
from model.decode import generic_decode
from model.utils import _sigmoid
from torch.autograd import Variable
from min_norm_solvers import MinNormSolver, gradient_normalizers


class GenericLoss(torch.nn.Module):
    def __init__(self, opt):
        super(GenericLoss, self).__init__()
        self.crit = FastFocalLoss(opt=opt)
        self.crit_reg = RegWeightedL1Loss()
        if 'rot' in opt.heads:
            self.crit_rot = BinRotLoss()
        if 'nuscenes_att' in opt.heads:
            self.crit_nuscenes_att = WeightedBCELoss()
        self.opt = opt

    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = _sigmoid(output['hm'])
        if 'hm_hp' in output:
            output['hm_hp'] = _sigmoid(output['hm_hp'])
        if 'dep' in output:
            output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
        return output

    def forward(self, outputs, batch, head=None):
        opt = self.opt
        if opt.mtl_loss:
            if head == "hm":
                outputs = _sigmoid(outputs)
                loss_h = self.crit(outputs, batch['hm'], batch['ind'], batch['mask'], batch['cat'])
            else:
                loss_h = self.crit_reg(outputs, batch[head + '_mask'], batch['ind'], batch[head])
            return loss_h
        else:
            losses = {head: 0 for head in opt.heads}

            for s in range(opt.num_stacks):
                output = outputs[s]
                output = self._sigmoid_output(output)

                if 'hm' in output:
                    losses['hm'] += self.crit(
                        output['hm'], batch['hm'], batch['ind'],
                        batch['mask'], batch['cat']) / opt.num_stacks

                regression_heads = [
                    'reg', 'wh', 'tracking', 'ltrb', 'ltrb_amodal', 'hps',
                    'dep', 'dim', 'amodel_offset', 'velocity']

                for head in regression_heads:
                    if head in output:
                        losses[head] += self.crit_reg(
                            output[head], batch[head + '_mask'],
                            batch['ind'], batch[head]) / opt.num_stacks

                if 'hm_hp' in output:
                    losses['hm_hp'] += self.crit(
                        output['hm_hp'], batch['hm_hp'], batch['hp_ind'],
                        batch['hm_hp_mask'], batch['joint']) / opt.num_stacks
                    if 'hp_offset' in output:
                        losses['hp_offset'] += self.crit_reg(
                            output['hp_offset'], batch['hp_offset_mask'],
                            batch['hp_ind'], batch['hp_offset']) / opt.num_stacks

                if 'rot' in output:
                    losses['rot'] += self.crit_rot(
                        output['rot'], batch['rot_mask'], batch['ind'], batch['rotbin'],
                        batch['rotres']) / opt.num_stacks

                if 'nuscenes_att' in output:
                    losses['nuscenes_att'] += self.crit_nuscenes_att(
                        output['nuscenes_att'], batch['nuscenes_att_mask'],
                        batch['ind'], batch['nuscenes_att']) / opt.num_stacks

            losses['tot'] = 0
            for head in opt.heads:
                losses['tot'] += opt.weights[head] * losses[head]

            return losses['tot'], losses

class ModleWithLoss(torch.nn.Module):
    def __init__(self, opt, model, loss):
        super(ModleWithLoss, self).__init__()
        self.opt = opt
        self.model = model
        self.loss = loss
        self._gat_train_count = 0

    def _build_prev_curr_boxes(self, outputs, batch):
        needed = ['ind', 'reg', 'wh', 'mask', 'tracking']
        if not all(k in batch for k in needed):
            return None, None, None
        if 'hm' not in outputs[-1]:
            return None, None, None

        hm = outputs[-1]['hm']   
        B, C, H, W = hm.shape
        ind   = batch['ind']      
        reg   = batch['reg']      
        wh    = batch['wh']       
        mask  = batch['mask']    
        track = batch['tracking'] 
        cat   = batch['cat']      
        cat = cat.long()
        if cat.min() >= 1:
            cat = cat - 1

        x = (ind % W).float()
        y = torch.div(ind, W, rounding_mode='trunc').float()
        ct_feat = torch.stack([x, y], dim=-1) + reg  # [B,M,2]
        pre_ct_feat = ct_feat + track                # [B,M,2]

        down = getattr(self.opt, 'down_ratio', 4)
        ct_pix     = ct_feat * down
        pre_ct_pix = pre_ct_feat * down
        wh_pix     = wh * down

        boxes_cur  = torch.stack([ct_pix[...,0],     ct_pix[...,1],     wh_pix[...,0], wh_pix[...,1]], dim=-1)
        boxes_prev = torch.stack([pre_ct_pix[...,0], pre_ct_pix[...,1], wh_pix[...,0], wh_pix[...,1]], dim=-1)
        m = (mask > 0).view(-1)          # [B*M]
        B, M = ind.shape
        batch_ids = torch.arange(B, device=ind.device).unsqueeze(1).expand(B, M).contiguous().view(-1)  # [B*M]
        cat_flat = cat.view(-1)
        return boxes_prev.view(-1,4)[m], boxes_cur.view(-1,4)[m], batch_ids[m], cat_flat[m]

    def forward(self, batch, head="", rep=None):
        if not self.opt.mtl_loss:
            pre_img = batch['pre_img'] if 'pre_img' in batch else None
            pre_hm  = batch['pre_hm'] if 'pre_hm' in batch else None

            outputs = self.model(batch['image'], pre_img, pre_hm)

            upm_loss_val   = None
            upm_d_hat      = None
            upm_d_gt       = None
            peak_ratio_vec = None
            p_hat          = None
            boxes_prev     = None
            boxes_cur      = None
            batch_ids      = None
            box_classes    = None
            disp_conf_vec  = None
            hm_center_loss_val = None  
            outputs_enh = None
            if getattr(self.opt, 'upm', False) and hasattr(self.model, 'upm'):
                feat_prev = getattr(self.model, 'last_feat_prev', None)
                feat_curr = getattr(self.model, 'last_feat_curr', None)

                if isinstance(feat_prev, torch.Tensor) and isinstance(feat_curr, torch.Tensor):
                    boxes_prev, boxes_cur, batch_ids, box_classes = self._build_prev_curr_boxes(outputs, batch)


                    if boxes_prev is not None and boxes_prev.numel() > 0:
                        upm_out = self.model.upm.train_forward(
                            feat_prev, feat_curr,
                            boxes_prev, boxes_cur,
                            batch_ids=batch_ids,
                        )
                        upm_loss_val   = upm_out['loss']
                        p_hat          = upm_out.get('p_hat', None)
                        peak_ratio_vec = upm_out.get('peak_ratio_vec', None)
                        upm_d_hat      = upm_out.get('d_hat', None)
                        upm_d_gt       = upm_out.get('d_gt', None)
                        disp_conf_vec  = upm_out.get('disp_conf_vec', None)

                else:
                    pass
            cand = []
            if boxes_prev is not None:   cand.append(boxes_prev.shape[0])
            if boxes_cur is not None:    cand.append(boxes_cur.shape[0])
            if p_hat is not None:        cand.append(p_hat.shape[0])
            if batch_ids is not None:    cand.append(batch_ids.shape[0])
            if peak_ratio_vec is not None: cand.append(peak_ratio_vec.shape[0])
            if box_classes is not None:  cand.append(box_classes.shape[0])
            if disp_conf_vec is not None:  cand.append(disp_conf_vec.shape[0])

            if len(cand) > 0:
                N = min(cand)
                if boxes_prev is not None and boxes_prev.shape[0] != N:
                    boxes_prev = boxes_prev[:N]
                if boxes_cur is not None and boxes_cur.shape[0] != N:
                    boxes_cur = boxes_cur[:N]
                if p_hat is not None and p_hat.shape[0] != N:
                    p_hat = p_hat[:N]
                if batch_ids is not None and batch_ids.shape[0] != N:
                    batch_ids = batch_ids[:N]
                if peak_ratio_vec is not None and peak_ratio_vec.shape[0] != N:
                    peak_ratio_vec = peak_ratio_vec[:N]
                if disp_conf_vec is not None and disp_conf_vec.shape[0] != N:
                    disp_conf_vec = disp_conf_vec[:N]
                if box_classes is not None and box_classes.shape[0] != N:
                    box_classes = box_classes[:N]

            if getattr(self.opt, 'gat', False) and hasattr(self.model, 'gat') \
                and (p_hat is not None) \
                and isinstance(getattr(self.model, 'last_feat_prev', None), torch.Tensor) \
                and isinstance(getattr(self.model, 'last_feat_curr', None), torch.Tensor) \
                and (batch_ids is not None):

                F_prev_all = self.model.last_feat_prev
                F_curr_all = self.model.last_feat_curr
                B = F_curr_all.shape[0]
                F_enh_all = F_curr_all.clone()

                epoch_val = getattr(self.opt, 'cur_epoch', None)

                for b in range(B):
                    sel = (batch_ids == b)
                    if sel.sum() == 0:
                        continue

                    boxes_prev_b = boxes_prev[sel]
                    p_hat_b      = p_hat[sel]
                    F_prev_b     = F_prev_all[b:b+1]
                    F_curr_b     = F_curr_all[b:b+1]
                    peak_ratio_b = peak_ratio_vec[sel] if peak_ratio_vec is not None else None
                    disp_conf_b  = disp_conf_vec[sel] if disp_conf_vec is not None else None

                    F_enh_b = self.model.gat.fuse(
                        feat_prev=F_prev_b,
                        feat_curr=F_curr_b,
                        boxes_prev_xywh=boxes_prev_b,
                        p_hat_xy=p_hat_b,
                        disp_conf=disp_conf_b,
                        epoch=epoch_val
                    )

                    F_enh_all[b:b+1] = F_enh_b

                outputs_enh = self.model.heads_from_single(F_enh_all)

                hm_center_w = float(getattr(self.opt, 'gat_hm_center_loss_w', 0.1))
                if hm_center_w > 0.0 and self.training and (outputs_enh is not None):
                    try:
                        hm_bg_ratio  = float(getattr(self.opt, 'gat_hm_bg_ratio', 0.3))
                        hm_bg_margin = float(getattr(self.opt, 'gat_hm_bg_margin', 0.02))

                        with torch.no_grad():
                            outputs_ref_all = self.model.heads_from_single(F_curr_all)
                        if isinstance(outputs_ref_all, list):
                            hm_ref_all = outputs_ref_all[-1]['hm'].detach()
                        else:
                            hm_ref_all = outputs_ref_all['hm'].detach()

                        if isinstance(outputs_enh, list):
                            hm_enh_all = outputs_enh[-1]['hm']
                        else:
                            hm_enh_all = outputs_enh['hm']

                        B_hm, C_hm, H_hm, W_hm = hm_ref_all.shape
                        needed_keys = ['ind', 'reg', 'mask', 'cat']

                        center_terms = []
                        bg_terms = []

                        if all(k in batch for k in needed_keys):
                            for b in range(B_hm):
                                ind_b = batch['ind'][b]
                                reg_b = batch['reg'][b]
                                mask_b = batch['mask'][b] > 0
                                cat_b = batch['cat'][b]

                                if not mask_b.any():
                                    continue

                                ind_b = ind_b[mask_b].long()
                                reg_b = reg_b[mask_b]
                                cat_b = cat_b[mask_b]

                                x = (ind_b % W_hm).float()
                                y = torch.div(ind_b, W_hm, rounding_mode='trunc').float()
                                ct_feat = torch.stack([x, y], dim=-1) + reg_b

                                hm_ref_b = hm_ref_all[b]
                                hm_enh_b = hm_enh_all[b]

                                center_mask = torch.zeros(
                                    (H_hm, W_hm),
                                    dtype=torch.bool,
                                    device=hm_ref_b.device,
                                )

                                for j in range(ct_feat.shape[0]):
                                    cls_id = int(cat_b[j].item())
                                    if cls_id < 0 or cls_id >= C_hm:
                                        continue
                                    cx = ct_feat[j, 0].item()
                                    cy = ct_feat[j, 1].item()
                                    ix = max(0, min(W_hm - 1, int(round(cx))))
                                    iy = max(0, min(H_hm - 1, int(round(cy))))

                                    center_mask[iy, ix] = True

                                    p_ref = torch.sigmoid(hm_ref_b[cls_id, iy, ix])
                                    p_enh = torch.sigmoid(hm_enh_b[cls_id, iy, ix])
                                    center_terms.append(F.relu(p_ref - p_enh))

                                if center_mask.any():
                                    bg_mask = ~center_mask
                                    p_ref_all = torch.sigmoid(hm_ref_b)
                                    p_enh_all = torch.sigmoid(hm_enh_b)

                                    bg_mask_flat = bg_mask.view(-1)
                                    diff_bg = (
                                        p_enh_all.view(C_hm, -1)[:, bg_mask_flat]
                                        - p_ref_all.view(C_hm, -1)[:, bg_mask_flat]
                                    )

                                    if diff_bg.numel() > 0:
                                        bg_pos = F.relu(diff_bg - hm_bg_margin)
                                        if bg_pos.numel() > 0:
                                            bg_terms.append(bg_pos.mean())

                        hm_center_loss_val = None
                        if (len(center_terms) > 0) or (len(bg_terms) > 0):
                            total_hm_loss = 0.0
                            if len(center_terms) > 0:
                                center_loss = torch.stack(center_terms).mean()
                                total_hm_loss = total_hm_loss + center_loss
                            if (len(bg_terms) > 0) and (hm_bg_ratio > 0.0):
                                bg_loss = torch.stack(bg_terms).mean()
                                total_hm_loss = total_hm_loss + hm_bg_ratio * bg_loss

                            hm_center_loss_val = hm_center_w * total_hm_loss
                        else:
                            hm_center_loss_val = None

                    except Exception as e_center:
                        print('[gat_hm_center_loss] failed:', e_center)
                        hm_center_loss_val = None

                outputs = outputs_enh

            loss, loss_stats = self.loss(outputs, batch)

            if hm_center_loss_val is not None:
                loss = loss + hm_center_loss_val
                loss_stats['hm_center'] = hm_center_loss_val.detach()
            else:
                # 保证 loss_stats 里始终有这个 key，方便统计
                zero_scalar = loss.detach().new_zeros(())
                loss_stats['hm_center'] = zero_scalar


            if upm_loss_val is not None:
                upm_w = float(getattr(self.opt, 'upm_loss_w', 1.0))
                loss = loss + upm_w * upm_loss_val
                loss_stats['upm'] = upm_loss_val.detach()
            else:
                zero = loss.detach().new_zeros(())
                loss_stats['upm'] = zero
           
            return outputs[-1], loss, loss_stats
    
        else:
            if head == "encode_1":
                with torch.no_grad():
                    pre_img_volatile = Variable(batch['pre_img']) if 'pre_img' in batch else None
                    pre_hm_volatile = Variable(batch['pre_hm']) if 'pre_hm' in batch else None
                    img_volatile = Variable(batch['image'])
                rep = self.model.forward_encode(img_volatile, pre_img_volatile, pre_hm_volatile)
                return rep
            elif head == "encode_2":
                pre_img = batch['pre_img'] if 'pre_img' in batch else None
                pre_hm = batch['pre_hm'] if 'pre_hm' in batch else None
                rep = self.model.forward_encode(batch['image'], pre_img, pre_hm)
                return rep
            else:
                out_head = self.model.forward_decode(head, rep)
                loss_h = self.loss(out_head, batch, head)
                return out_head, loss_h


class Trainer(object):
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModleWithLoss(opt, model, self.loss)
        self._max_grad_norm = getattr(opt, 'max_grad_norm', 5.0)


    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)


    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        self.model_with_loss.opt.cur_epoch = epoch
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()
    
        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {}
        
        for l in self.loss_stats:
            if l == 'tot':
                avg_loss_stats[l] = AverageMeter()
            elif l.startswith('upm'):  
                avg_loss_stats[l] = AverageMeter()
            elif l == 'hm_center':  
                avg_loss_stats[l] = AverageMeter()
            elif (l in opt.weights) and (opt.weights[l] > 0):
                avg_loss_stats[l] = AverageMeter()

        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)
            if not self.opt.mtl_loss:
                model_with_loss.cur_epoch = getattr(opt, "cur_epoch", 0)
                model_with_loss.cur_iter_in_epoch = iter_id
                model_with_loss.cur_global_step = int(getattr(opt, "cur_epoch", 0) * num_iters + iter_id)
                model_with_loss.cur_lr = float(self.optimizer.param_groups[0]["lr"])

                output, loss, loss_stats = model_with_loss(batch)
                loss = loss.mean()
                if phase == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    if getattr(self, '_max_grad_norm', None) is not None:
                        torch.nn.utils.clip_grad_norm_(self.model_with_loss.parameters(), self._max_grad_norm)
                    self.optimizer.step()
            else:
                if phase == 'train':
                    loss_stats = {}
                    output = {}
                    # loss_h_dict = {}
                    loss_data = {}
                    grads = {}
                    scale = {}
                    self.optimizer.zero_grad()
                    rep = model_with_loss(batch, head='encode_1')
                    rep_variable = Variable(rep.data.clone(), requires_grad=True)
                    for head in self.opt.heads:
                        self.optimizer.zero_grad()
                        out_head, loss_h = model_with_loss(batch, head, rep=rep_variable)
                        # output[head] = out_head
                        loss_h = loss_h.mean()
                        # loss_h_dict[head] = loss_h
                        loss_data[head] = loss_h.data
                        loss_h.backward()
                        grads[head] = []
                        grads[head].append(Variable(rep_variable.grad.data.clone(), requires_grad=False))
                        rep_variable.grad.data.zero_()
                    gn = gradient_normalizers(grads, loss_data, "loss+")
                    for head in self.opt.heads:
                        for gr_i in range(len(grads[head])):
                            grads[head][gr_i] = grads[head][gr_i] / gn[head]
                    sol, min_norm = MinNormSolver.find_min_norm_element([grads[h] for h in self.opt.heads])
                    for i, t in enumerate(self.opt.heads):
                        scale[t] = float(sol[i])

                    self.optimizer.zero_grad()
                    if self.opt.freeze_encoder:
                        rep = rep_variable
                    else:
                        rep = model_with_loss(batch, head="encode_2")
                    # for i, head in enumerate(self.opt.heads):
                    #     if i > 0:
                    #         loss = loss + scale[head] * loss_h_dict[head]
                    #     else:
                    #         loss = scale[head] * loss_h_dict[head]
                    #     loss_stats[head] = scale[head] * loss_h_dict[head]
                    # loss_stats['tot'] = loss
                else:
                    rep = model_with_loss(batch, head="encode_2")
                for i, head in enumerate(self.opt.heads):
                    out_head, loss_h = model_with_loss(batch, head, rep)
                    output[head] = out_head
                    loss_h = loss_h.mean()
                    loss_data[head] = loss_h.data
                    if phase == 'train':
                        if i > 0:
                            loss = loss + scale[head]*loss_h
                        else:
                            loss = scale[head]*loss_h
                        loss_stats[head] = scale[head]*loss_h
                    else:
                        if i > 0:
                            loss = loss + loss_h
                        else:
                            loss = loss_h
                        loss_stats[head] = loss_h
                loss_stats['tot'] = loss
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['image'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                      '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if self.opt.mtl_loss:
                for h in scale:
                    Bar.suffix = Bar.suffix + '|{} {:.4f} '.format('scale-'+h,scale[h])
            if opt.print_iter > 0:  # If not using progress bar
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            del output, loss, loss_stats
            if self.opt.mtl_loss:
                del loss_data, grads
        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        if self.opt.mtl_loss:
            for i, t in enumerate(self.opt.heads):
                ret['scale'+t] = scale[t]
        return ret, results

    def _get_losses(self, opt):
        loss_order = ['hm', 'wh', 'reg', 'ltrb', 'hps', 'hm_hp', \
                      'hp_offset', 'dep', 'dim', 'rot', 'amodel_offset', \
                      'ltrb_amodal', 'tracking', 'nuscenes_att', 'velocity']
        loss_states = ['tot'] + [k for k in loss_order if k in opt.heads]

        if getattr(opt, 'upm', False):
            loss_states += ['upm']
        if float(getattr(opt, 'gat_hm_center_loss_w', 0.1)) > 0.0:
            loss_states += ['hm_center']

        loss = GenericLoss(opt)
        return loss_states, loss

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
