# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils.timer

from layer_utils.snippets import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from utils.visualization import draw_bounding_boxes
from nets.capsule import CapsNet

from torchvision.ops import RoIAlign, RoIPool

from model.config import cfg

import tensorboardX as tb

from scipy.misc import imresize


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self._predictions = {}
        self._losses = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._layers = {}
        self._gt_image = None
        self._act_summaries = {}
        self._score_summaries = {}
        self._event_summaries = {}
        self._image_gt_summaries = {}
        self._variables_to_fix = {}
        self._device = 'cuda'
        self._caps_output = {}

    def _add_gt_image(self):
        # add back mean
        image = self._image_gt_summaries['image'] + cfg.PIXEL_MEANS
        image = imresize(image[0], self._im_info[:2] / self._im_info[2])
        # BGR to RGB (opencv uses BGR)
        self._gt_image = image[np.newaxis, :, :, ::-1].copy(order='C')

    def _add_gt_image_summary(self):
        # use a customized visualization function to visualize the boxes
        self._add_gt_image()
        image = draw_bounding_boxes(\
                          self._gt_image, self._image_gt_summaries['gt_boxes'], self._image_gt_summaries['im_info'])

        return tb.summary.image('GROUND_TRUTH',
                                image[0].astype('float32') / 255.0, dataformats='HWC')

    def _add_reconstruction_summary(self, key = 'rec'):

        image = self._caps_output[key].detach().cpu().numpy() #+ cfg.PIXEL_MEANS
        print(image.shape)
        image = np.reshape(image[0],[3,28,28])#.copy(order='C')
        # image2 = self._caps_output["data"].detach().cpu().numpy()
        # image2 = np.reshape(image2[0],[3,28,28])
        # print('image size:', image.shape)
        # print('image in summary: ', image)
        # image = imresize(image[0], self._im_info[:2] / self._im_info[2])

        return tb.summary.image(key,
                                image.astype('float32'), dataformats='CHW')

    def _add_act_summary(self, key, tensor):
        # return tb.summary.histogram(
        #     'ACT/' + key + '/activations',
        #     tensor.data.cpu().numpy(),
        #     bins='auto'),
        # tb.summary.scalar('ACT/' + key + '/zero_fraction',
        #                   (tensor.data == 0).float().sum() / tensor.numel())
        return tb.summary.histogram(
            'ACT/' + key + '/activations',
            tensor.data.cpu().numpy(),
            bins=100),
        tb.summary.scalar('ACT/' + key + '/zero_fraction',
                          (tensor.data == 0).float().sum() / tensor.numel())

    def _add_score_summary(self, key, tensor):
        # return tb.summary.histogram(
        #     'SCORE/' + key + '/scores', tensor.data.cpu().numpy(), bins='auto')
        return tb.summary.histogram(
            'SCORE/' + key + '/scores', tensor.data.cpu().numpy(), bins=100)

    def _add_train_summary(self, key, var):
        # return tb.summary.histogram(
            # 'TRAIN/' + key, var.data.cpu().numpy(), bins='auto')
        return tb.summary.histogram(
            'TRAIN/' + key, var.data.cpu().numpy(), bins=100)

    def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred):
        rois, rpn_scores = proposal_top_layer(\
                                        rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                         self._feat_stride, self._anchors, self._num_anchors)
        return rois, rpn_scores

    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred):
        rois, rpn_scores = proposal_layer(\
                                        rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                         self._feat_stride, self._anchors, self._num_anchors)

        return rois, rpn_scores

    def _roi_pool_layer(self, bottom, rois):
        return RoIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE),
                       1.0 / 16.0)(bottom, rois)

    def _roi_align_layer(self, bottom, rois):
        # print('bottom size: ', bottom.size())
        return RoIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0,
                        0)(bottom, rois)

    def _roi_align_layer_class(self, bottom, rois, scale = 1.0 / 16.0):
        # print('bottom size: ', bottom.size())

        return RoIAlign((cfg.POOLING_SIZE_CLASS, cfg.POOLING_SIZE_CLASS), scale,
                        0)(bottom, rois)

    def _anchor_target_layer(self, rpn_cls_score):
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
          anchor_target_layer(
          rpn_cls_score.data, self._gt_boxes.data.cpu().numpy(), self._im_info, self._feat_stride, self._anchors.data.cpu().numpy(), self._num_anchors)

        rpn_labels = torch.from_numpy(rpn_labels).float().to(
            self._device)  #.set_shape([1, 1, None, None])
        rpn_bbox_targets = torch.from_numpy(rpn_bbox_targets).float().to(
            self._device)  #.set_shape([1, None, None, self._num_anchors * 4])
        rpn_bbox_inside_weights = torch.from_numpy(
            rpn_bbox_inside_weights).float().to(
                self.
                _device)  #.set_shape([1, None, None, self._num_anchors * 4])
        rpn_bbox_outside_weights = torch.from_numpy(
            rpn_bbox_outside_weights).float().to(
                self.
                _device)  #.set_shape([1, None, None, self._num_anchors * 4])

        rpn_labels = rpn_labels.long()
        self._anchor_targets['rpn_labels'] = rpn_labels
        self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
        self._anchor_targets[
            'rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
        self._anchor_targets[
            'rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

        for k in self._anchor_targets.keys():
            self._score_summaries[k] = self._anchor_targets[k]

        return rpn_labels

    def _proposal_target_layer(self, rois, roi_scores):
        rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
          proposal_target_layer(
          rois, roi_scores, self._gt_boxes, self._num_classes)

        self._proposal_targets['rois'] = rois
        self._proposal_targets['labels'] = labels.long()
        self._proposal_targets['bbox_targets'] = bbox_targets
        self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
        self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

        for k in self._proposal_targets.keys():
            self._score_summaries[k] = self._proposal_targets[k]

        return rois, roi_scores

    def _anchor_component(self, height, width):
        # just to get the shape right
        #height = int(math.ceil(self._im_info.data[0, 0] / self._feat_stride[0]))
        #width = int(math.ceil(self._im_info.data[0, 1] / self._feat_stride[0]))
        anchors, anchor_length = generate_anchors_pre(\
                                              height, width,
                                               self._feat_stride, self._anchor_scales, self._anchor_ratios)
        self._anchors = torch.from_numpy(anchors).to(self._device)
        self._anchor_length = anchor_length

    def _smooth_l1_loss(self,
                        bbox_pred,
                        bbox_targets,
                        bbox_inside_weights,
                        bbox_outside_weights,
                        sigma=1.0,
                        dim=[1]):
        sigma_2 = sigma**2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = torch.abs(in_box_diff)
        smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
        in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = out_loss_box
        for i in sorted(dim, reverse=True):
            loss_box = loss_box.sum(i)
        loss_box = loss_box.mean()
        return loss_box

    def _add_losses(self, sigma_rpn=3.0):
        # RPN, class loss
        rpn_cls_score = self._predictions['rpn_cls_score_reshape'].view(-1, 2)
        rpn_label = self._anchor_targets['rpn_labels'].view(-1)
        rpn_select = (rpn_label.data != -1).nonzero().view(-1)
        rpn_cls_score = rpn_cls_score.index_select(
            0, rpn_select).contiguous().view(-1, 2)
        rpn_label = rpn_label.index_select(0, rpn_select).contiguous().view(-1)
        rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)

        # RPN, bbox loss
        rpn_bbox_pred = self._predictions['rpn_bbox_pred']
        rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
        rpn_bbox_inside_weights = self._anchor_targets[
            'rpn_bbox_inside_weights']
        rpn_bbox_outside_weights = self._anchor_targets[
            'rpn_bbox_outside_weights']
        rpn_loss_box = self._smooth_l1_loss(
            rpn_bbox_pred,
            rpn_bbox_targets,
            rpn_bbox_inside_weights,
            rpn_bbox_outside_weights,
            sigma=sigma_rpn,
            dim=[1, 2, 3])

        # RCNN, class loss
        cls_score = self._predictions["cls_score"]
        label = self._proposal_targets["labels"].view(-1, 1)
        # ind = self._proposal_targets["labels"]#.view(-1, 1)
        # print(ind.size())
        label = torch.zeros(label.size(0), 3).to(self._device).scatter_(1, label, 1) ##### change 3 to more generic
        # label = torch.sparse.torch.eye(3).to(self._device).index_select(dim=0, index=ind)
        # label = torch.
        output = self._caps_output["output"]
        rec = self._caps_output["rec"]
        data = self._caps_output["data"]

        # loss = capsule_net.loss(data, output, target, reconstructions)
        cross_entropy = self.cls_score_net.loss(data, output, label, rec) ### not cross entropy but doesn't matter for now

        # TODO: reverifier min & max image avec bon avg
        # print('max data: ', torch.max(data))
        # print('min data: ', torch.min(data))

        # cross_entropy = F.cross_entropy(
        #     cls_score.view(-1, self._num_classes), label)

        # RCNN, bbox loss
        bbox_pred = self._predictions['bbox_pred']
        bbox_targets = self._proposal_targets['bbox_targets']
        bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
        bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
        loss_box = self._smooth_l1_loss(
            bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

        self._losses['cross_entropy'] = cross_entropy
        self._losses['loss_box'] = loss_box
        self._losses['rpn_cross_entropy'] = rpn_cross_entropy
        self._losses['rpn_loss_box'] = rpn_loss_box

        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
        self._losses['total_loss'] = loss

        for k in self._losses.keys():
            self._event_summaries[k] = self._losses[k]

        return loss

    def _region_proposal(self, net_conv):
        rpn = F.relu(self.rpn_net(net_conv))
        self._act_summaries['rpn'] = rpn

        rpn_cls_score = self.rpn_cls_score_net(
            rpn)  # batch * (num_anchors * 2) * h * w

        # change it so that the score has 2 as its channel size
        rpn_cls_score_reshape = rpn_cls_score.view(
            1, 2, -1,
            rpn_cls_score.size()[-1])  # batch * 2 * (num_anchors*h) * w
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, dim=1)

        # Move channel to the last dimenstion, to fit the input of python functions
        rpn_cls_prob = rpn_cls_prob_reshape.view_as(rpn_cls_score).permute(
            0, 2, 3, 1)  # batch * h * w * (num_anchors * 2)
        rpn_cls_score = rpn_cls_score.permute(
            0, 2, 3, 1)  # batch * h * w * (num_anchors * 2)
        rpn_cls_score_reshape = rpn_cls_score_reshape.permute(
            0, 2, 3, 1).contiguous()  # batch * (num_anchors*h) * w * 2
        rpn_cls_pred = torch.max(rpn_cls_score_reshape.view(-1, 2), 1)[1]

        rpn_bbox_pred = self.rpn_bbox_pred_net(rpn)
        rpn_bbox_pred = rpn_bbox_pred.permute(
            0, 2, 3, 1).contiguous()  # batch * h * w * (num_anchors*4)

        if self._mode == 'TRAIN':
            rois, roi_scores = self._proposal_layer(
                rpn_cls_prob, rpn_bbox_pred)  # rois, roi_scores are varible
            rpn_labels = self._anchor_target_layer(rpn_cls_score)
            rois, _ = self._proposal_target_layer(rois, roi_scores)
        else:
            if cfg.TEST.MODE == 'nms':
                rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred)
            elif cfg.TEST.MODE == 'top':
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred)
            else:
                raise NotImplementedError

        self._predictions["rpn_cls_score"] = rpn_cls_score
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_cls_prob"] = rpn_cls_prob
        self._predictions["rpn_cls_pred"] = rpn_cls_pred
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self._predictions["rois"] = rois

        return rois

    def _region_classification(self, fc7, pool5, pool5_cl):

        means = torch.from_numpy(np.array([102.9801, 115.9465, 122.7717])).view(-1,1,1).float().to(self._device)
        self._caps_output["data"] = (pool5_cl + means) / 255.0 # do something about that and

        output, rec, masked = self.cls_score_net(self._caps_output["data"])#pool5_cl)
        # print("rec size: ", rec.size())
        # print("pool5_cl size: ", pool5_cl.size())
        cls_score = torch.sqrt((output ** 2).sum(2)).view(-1, output.size(1))#torch.sqrt((output ** 2).sum(2))
        # print(cls_score)
        # print(cls_score.size())
        cls_pred = torch.max(masked, 1)[1] # index of max
        # print(cls_score)
        # print(cls_score.size())
        cls_prob = F.softmax(cls_score, dim=1) ## Do not use softmax

        # print('avg proba: ', cls_prob.mean(0))
        # print('avg output: ', cls_score.mean(0))

        # print(cls_prob)
        # print('cls_prob__ size: ', cls_prob__.size())
        # cls_prob = masked#F.softmax(cls_score, dim=2)
        # print('cls_prob: ', cls_prob)
        # print('masked: ', masked)
        # print('cls_pred: ', cls_pred)
        bbox_pred = self.bbox_pred_net(fc7)

        self._caps_output["output"] = output
        self._caps_output["rec"] = rec
        # print(rec)
        # torch.from_numpy(cfg.PIXEL_MEANS)
        # self._caps_output["data"] = (pool5_cl + 110.0) / 255.0
        ### Adding back pixel mean and divide by 255 to get between 0 & 1
        # self._caps_output["output"] = output

        self._predictions["cls_score"] = cls_score
        self._predictions["cls_pred"] = cls_pred
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred

        return cls_prob, bbox_pred

    def _image_to_head(self):
        raise NotImplementedError

    def _head_to_tail(self, pool5):
        raise NotImplementedError

    def create_architecture(self,
                            num_classes,
                            tag=None,
                            anchor_scales=(8, 16, 32),
                            anchor_ratios=(0.5, 1, 2)):
        self._tag = tag

        self._num_classes = num_classes
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)

        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)

        self._num_anchors = self._num_scales * self._num_ratios

        assert tag != None

        # Initialize layers
        self._init_modules()

    def _init_modules(self):
        self._init_head_tail()

        # rpn
        self.rpn_net = nn.Conv2d(
            self._net_conv_channels, cfg.RPN_CHANNELS, [3, 3], padding=1)

        self.rpn_cls_score_net = nn.Conv2d(cfg.RPN_CHANNELS,
                                           self._num_anchors * 2, [1, 1])

        self.rpn_bbox_pred_net = nn.Conv2d(cfg.RPN_CHANNELS,
                                           self._num_anchors * 4, [1, 1])

        # self.cls_score_net = CapsNet(self._net_conv_channels, self._num_classes)
        self.cls_score_net = CapsNet(3, self._num_classes)
        # self.cls_score_net = nn.Linear(self._fc7_channels, self._num_classes) ##### <------- CAPS
        self.bbox_pred_net = nn.Linear(self._fc7_channels,
                                       self._num_classes * 4)

        self.init_weights()

    def _run_summary_op(self, val=False):
        """
    Run the summary operator: feed the placeholders with corresponding newtork outputs(activations)
    """
        summaries = []
        # Add image gt
        summaries.append(self._add_gt_image_summary())
        summaries.append(self._add_reconstruction_summary(key = 'rec'))
        summaries.append(self._add_reconstruction_summary(key = 'data'))
        # Add event_summaries
        for key, var in self._event_summaries.items():
            summaries.append(tb.summary.scalar(key, var.item()))
        self._event_summaries = {}
        if not val:
            # Add score summaries
            for key, var in self._score_summaries.items():
                summaries.append(self._add_score_summary(key, var))
            self._score_summaries = {}
            # Add act summaries
            for key, var in self._act_summaries.items():
                summaries += self._add_act_summary(key, var)
            self._act_summaries = {}
            # Add train summaries
            for k, var in dict(self.named_parameters()).items():

                if var.requires_grad:
                    # print('train_summary values: ', k,'\n', var)
                    summaries.append(self._add_train_summary(k, var))

            self._image_gt_summaries = {}

        return summaries

    def _predict(self):
        # This is just _build_network in tf-faster-rcnn
        torch.backends.cudnn.benchmark = False
        net_conv = self._image_to_head()
        # print("net_conv size: ", net_conv.size())
        # print('net conv size: ', net_conv.size())
        # build the anchors for the image
        self._anchor_component(net_conv.size(2), net_conv.size(3))

        rois = self._region_proposal(net_conv) # quand on appelle region_proposal on enregistre les resultats en même temps
        # print('rois size: ', rois.size())
        if cfg.POOLING_MODE == 'align':
            pool5 = self._roi_align_layer(net_conv, rois)

            # print('mean image ', self._image.mean())

            pool5_cl = self._roi_align_layer_class(self._image, rois, scale=1.0)#net_conv, rois)
            # print('pool5 size:', pool5.size())
        else:
            pool5 = self._roi_pool_layer(net_conv, rois)

        if self._mode == 'TRAIN':
            torch.backends.cudnn.benchmark = True  # benchmark because now the input size are fixed

        # print('pool5 size: ', pool5.size())
        fc7 = self._head_to_tail(pool5)
        # print('fc7 size: ', fc7.size())

        cls_prob, bbox_pred = self._region_classification(fc7, pool5, pool5_cl)
        # print('cls_prob size: ', cls_prob.size())

        for k in self._predictions.keys():
            self._score_summaries[k] = self._predictions[k]

        return rois, cls_prob, bbox_pred

    def forward(self, image, im_info, gt_boxes=None, mode='TRAIN'):
        self._image_gt_summaries['image'] = image
        self._image_gt_summaries['gt_boxes'] = gt_boxes
        self._image_gt_summaries['im_info'] = im_info
        # print('\n\n\n')
        # print('image', image)
        # print('gt_boxes', gt_boxes)
        # print('im_info:', im_info)
        # print('\n\n\n')

        self._image = torch.from_numpy(image.transpose([0, 3, 1,
                                                        2])).to(self._device)
        self._im_info = im_info  # No need to change; actually it can be an list
        self._gt_boxes = torch.from_numpy(gt_boxes).to(
            self._device) if gt_boxes is not None else None

        self._mode = mode

        rois, cls_prob, bbox_pred = self._predict()

        # print('\n\n\n')
        # print('cls_prob', cls_prob)
        # print('rois', rois)
        # print('bbox_pred:', bbox_pred)
        # print('\n\n\n')

        if mode == 'TEST':
            stds = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(
                self._num_classes).unsqueeze(0).expand_as(bbox_pred)
            means = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(
                self._num_classes).unsqueeze(0).expand_as(bbox_pred)
            self._predictions["bbox_pred"] = bbox_pred.mul(stds).add(means)
        else:
            self._add_losses()  # compute losses

    def init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
      weight initalizer: truncated normal and random normal.
      """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

        normal_init(self.rpn_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.rpn_cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.rpn_bbox_pred_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # <----------------------------------------------------------------------------- weight init
        # normal_init(self.cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.bbox_pred_net, 0, 0.001, cfg.TRAIN.TRUNCATED)

    # Extract the head feature maps, for example for vgg16 it is conv5_3
    # only useful during testing mode
    def extract_head(self, image):
        feat = self._layers["head"](torch.from_numpy(
            image.transpose([0, 3, 1, 2])).to(self._device))
        return feat

    # only useful during testing mode
    def test_image(self, image, im_info):
        self.eval()
        with torch.no_grad():
            self.forward(image, im_info, None, mode='TEST')
        cls_score, cls_prob, bbox_pred, rois = self._predictions["cls_score"].data.cpu().numpy(), \
                                                         self._predictions['cls_prob'].data.cpu().numpy(), \
                                                         self._predictions['bbox_pred'].data.cpu().numpy(), \
                                                         self._predictions['rois'].data.cpu().numpy()
        return cls_score, cls_prob, bbox_pred, rois

    def delete_intermediate_states(self):
        # Delete intermediate result to save memory
        for d in [
                self._losses, self._predictions, self._anchor_targets,
                self._proposal_targets
        ]:
            for k in list(d):
                del d[k]

    def get_summary(self, blobs):
        self.eval()
        self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'])
        self.train()
        summary = self._run_summary_op(True)

        return summary

    def train_step(self, blobs, train_op):
        self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'])
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss = self._losses["rpn_cross_entropy"].item(), \
                                                                            self._losses['rpn_loss_box'].item(), \
                                                                            self._losses['cross_entropy'].item(), \
                                                                            self._losses['loss_box'].item(), \
                                                                            self._losses['total_loss'].item()
        #utils.timer.timer.tic('backward')
        train_op.zero_grad()
        self._losses['total_loss'].backward()
        #utils.timer.timer.toc('backward')
        train_op.step()
        # print('lib/nets/networks.py: train_step')

        self.delete_intermediate_states()

        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

    def train_step_with_summary(self, blobs, train_op):
        self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'])
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss = self._losses["rpn_cross_entropy"].item(), \
                                                                            self._losses['rpn_loss_box'].item(), \
                                                                            self._losses['cross_entropy'].item(), \
                                                                            self._losses['loss_box'].item(), \
                                                                            self._losses['total_loss'].item()
        train_op.zero_grad()
        self._losses['total_loss'].backward()
        train_op.step()
        # print('rpn_loss_cls:', rpn_loss_cls)
        # print('rpn_loss_box: ', rpn_loss_box)
        # print('loss_cls: ', loss_cls)
        # print('loss_box: ', loss_box)
        # print('loss: ', loss)
        # print('lib/nets/networks.py: train_step with summary')
        summary = self._run_summary_op()

        self.delete_intermediate_states()

        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

    def train_step_no_return(self, blobs, train_op):
        self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'])
        train_op.zero_grad()
        self._losses['total_loss'].backward()
        train_op.step()
        self.delete_intermediate_states()

    def load_state_dict(self, state_dict):
        """
    Because we remove the definition of fc layer in resnet now, it will fail when loading
    the model trained before.
    To provide back compatibility, we overwrite the load_state_dict
    """
        # for i in state_dict.items():
        #     print(i)
        # for i in self.state_dict():
        #     print(i)
        nn.Module.load_state_dict(
            self, {k: v
                   for k, v in state_dict.items() if k in self.state_dict()}
            ) ########################################################################### ADDED STRICT = FALSE
