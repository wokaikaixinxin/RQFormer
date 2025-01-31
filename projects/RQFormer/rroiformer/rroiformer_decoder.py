from typing import List, Tuple
import torch
from mmengine.structures import InstanceData
from torch import Tensor
from mmdet.models.task_modules.samplers import PseudoSampler
from mmrotate.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi, get_box_tensor
from mmdet.utils import ConfigType, InstanceList, OptConfigType
from mmdet.models.utils import empty_instances, unpack_gt_instances
from mmdet.models.roi_heads import CascadeRoIHead
from mmcv.ops import batched_nms


@MODELS.register_module()
class RRoIFormerDecoder(CascadeRoIHead):
    r"""
    Args:
        num_stages (int): Number of stage whole iterative process.
            Defaults to 6.
        stage_loss_weights (Tuple[float]): The loss
            weight of each stage. By default all stages have
            the same weight 1.
        bbox_roi_extractor (:obj:`ConfigDict` or dict): Config of box
            roi extractor.
        mask_roi_extractor (:obj:`ConfigDict` or dict): Config of mask
            roi extractor.
        bbox_head (:obj:`ConfigDict` or dict): Config of box head.
        mask_head (:obj:`ConfigDict` or dict): Config of mask head.
        train_cfg (:obj:`ConfigDict` or dict, Optional): Configuration
            information in train stage. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, Optional): Configuration
            information in test stage. Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 distinct_cfg: dict = dict(type='nms_rotated', iou_threshold=0.9),
                 num_stages: int = 2,
                 stage_loss_weights: Tuple[float] = (1, 1, 1, 1, 1, 1),
                 selective_query=[0],
                 distinct_query=0,
                 bbox_roi_extractor: ConfigType = dict(
                     type='RotatedSingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlignRotated',
                         out_size=7,
                         sample_num=2,
                         clockwise=True),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 mask_roi_extractor: OptConfigType = None,
                 bbox_head: ConfigType = dict(
                     type='RRoIFormerDecoderLayer',
                     num_classes=20,
                     angle_version='le90',
                     reg_predictor_cfg=dict(type='mmdet.Linear'),
                     cls_predictor_cfg=dict(type='mmdet.Linear'),
                     num_cls_fcs=1,
                     num_reg_fcs=3,
                     dropout=0.1,
                     self_attn_cfg=dict(
                         embed_dims=256,
                         num_heads=8,
                         dropout=0.1),
                     rroi_attn_cfg=dict(
                         type='RRoIAttnv4',
                         embed_dims=256,
                         num_heads=8,
                         roi_pooler_resolution=7,
                     ),
                     ffn_cfg=dict(
                         embed_dims=256,
                         feedforward_channels=2048,
                         num_fcs=2,
                         ffn_drop=0.1,
                         act_cfg=dict(type='ReLU', inplace=True)),
                     loss_bbox=dict(type='mmdet.L1Loss', loss_weight=2.0),
                     loss_iou=dict(type='RotatedIoULoss', mode='linear', loss_weight=5.0),
                     loss_cls=dict(
                         type='mmdet.FocalLoss',
                         use_sigmoid=True,
                         gamma=2.0,
                         alpha=0.25,
                         loss_weight=2.0),
                     bbox_coder=dict(
                         type='DeltaXYWHTRBBoxCoder',
                         angle_version='le90',
                         norm_factor=None,
                         edge_swap=True,
                         proj_xy=True,
                         target_means=(.0, .0, .0, .0, .0),
                         target_stds=(1., 1., 1., 1., 0.08),
                         use_box_type=False)),
                 mask_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None) -> None:
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert len(stage_loss_weights) == num_stages
        self.distinct_cfg = distinct_cfg
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.selective_query = selective_query
        self.distinct_query = distinct_query
        super().__init__(
            num_stages=num_stages,
            stage_loss_weights=stage_loss_weights,
            bbox_roi_extractor=bbox_roi_extractor,
            mask_roi_extractor=mask_roi_extractor,
            bbox_head=bbox_head,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(self.bbox_sampler[stage], PseudoSampler), \
                    'RRoIFormer, DDQRCNN, Sparse R-CNN and QueryInst only support `PseudoSampler`'

    def bbox_loss(self, stage: int, x: Tuple[Tensor],
                  results_list: InstanceList, query: Tensor,
                  query_pos: Tensor, batch_start_index: Tensor,
                  batch_img_metas: List[dict],
                  batch_gt_instances: InstanceList) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            stage (int): The current stage in iterative process.
            x (tuple[Tensor]): List of multi-level img features,
                each level is (bs, c, h, w).
            results_list (List[:obj:`InstanceData`]) : List of region
                proposals.
            query (Tensor): Same as dino, content query,
                has shape (bs, num_query, embed_dims).
            query_pos (Tensor): The positional encoding for query,
                has shape (bs, num_query, embed_dims).
            batch_start_index (Tensor): index query numbers of per img,
                Note: length = batch + 1. [0, 0+N1, 0+N1+N2, ...].
            batch_img_metas (list[dict]): Meta information of each image.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

            - `cls_score` (Tensor): Classification scores,
            - `bbox_pred` (Tensor): Box energies / deltas,
            - `bbox_feats` (Tensor): Extract bbox RoI features.
            - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        proposal_list = [res.bboxes for res in results_list]    # list{[N1, 5], [N2, 5], ...}
        rois = bbox2roi(proposal_list)  # N1+N2, 6
        bbox_results = self._bbox_forward(stage, x, rois, query, query_pos,
                                          batch_start_index, batch_img_metas)
        imgs_whwht = torch.cat(
            [res.imgs_whwht for res in results_list])   # N1+N2, 5
        cls_pred_list = bbox_results['detached_cls_scores'] # {[N1, 80], [N2, 80]}
        proposal_list = bbox_results['detached_proposals']  # {[N1, 4], [N2, 4]}

        sampling_results = []
        bbox_head = self.bbox_head[stage]
        for i in range(len(batch_img_metas)):
            pred_instances = InstanceData()
            # TODO: Enhance the logic
            pred_instances.bboxes = proposal_list[i]  # for assinger
            pred_instances.scores = cls_pred_list[i]
            pred_instances.priors = proposal_list[i]  # for sampler

            assign_result = self.bbox_assigner[stage].assign(
                pred_instances=pred_instances,
                gt_instances=batch_gt_instances[i],
                gt_instances_ignore=None,
                img_meta=batch_img_metas[i])

            sampling_result = self.bbox_sampler[stage].sample(
                assign_result, pred_instances, batch_gt_instances[i])
            sampling_results.append(sampling_result)

        bbox_results.update(sampling_results=sampling_results)

        cls_score = bbox_results['cls_score']
        decoded_bboxes = bbox_results['decoded_bboxes']
        bbox_loss_and_target = bbox_head.loss_and_target(
            cls_score,
            decoded_bboxes,
            sampling_results,
            self.train_cfg[stage],
            imgs_whwht=imgs_whwht,
            concat=True)
        bbox_results.update(bbox_loss_and_target)

        # propose for the new proposal_list
        proposal_list = []
        for idx in range(len(batch_img_metas)):
            results = InstanceData()
            results.imgs_whwht = results_list[idx].imgs_whwht
            results.bboxes = bbox_results['detached_proposals'][idx]
            proposal_list.append(results)
        bbox_results.update(results_list=proposal_list)
        return bbox_results

    def _bbox_forward(self, stage: int, x: Tuple[Tensor], rois: Tensor,
                      query: Tensor, query_pos: Tensor, batch_start_index: Tensor,
                      batch_img_metas: List[dict]) -> dict:
        """Box head forward function used in both training and testing. Returns
        all regression, classification results and a intermediate feature.

        Args:
            stage (int): The current stage in iterative process.
            x (tuple[Tensor]): List of multi-level img features,
                each level is (bs, c, h, w).
            rois (Tensor): RoIs with the shape (bs*num_query, 6) where
                the first column indicates batch id of each RoI.
                Each dimension means (img_index, c_x, c_y, w, h, radian).
            query (Tensor): Same as dino, content query,
                has shape (bs, num_query, embed_dims).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_query, embed_dims).
            batch_start_index (Tensor): index query numbers of per img,
                Note: length = batch + 1. [0, 0+N1, 0+N1+N2, ...].
            batch_img_metas (list[dict]): Meta information of each image.

        Returns:
            dict[str, Tensor]: a dictionary of bbox head outputs,
            Containing the following results:

            - cls_score (Tensor): The score of each class, has
              shape (batch_size, num_proposals, num_classes)
              when use focal loss or
              (batch_size, num_proposals, num_classes+1)
              otherwise.
            - decoded_bboxes (Tensor): The regression results
              with shape (batch_size*num_proposal, 5).
              The last dimension 5 represents
              [cx, cy, w, h, radian].
            - query (Tensor): The object feature extracted
              from current stage, (bs, num_query, 256)
            - detached_cls_scores (list[Tensor]): The detached
              classification results, length is batch_size, and
              each tensor has shape (num_proposal, num_classes).
            - detached_proposals (list[tensor]): The detached
              regression results, length is batch_size, and each
              tensor has shape (num_proposal, 5). The last
              dimension 5 represents [cx, cy, w, h, radian].
        """
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]

        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)   # N1+N2, 256, 7, 7
        cls_score, bbox_pred, query, attn_feats = bbox_head(
            bbox_feats, query, query_pos, batch_start_index)   # [N1+N2, 15], [N1+N2, 5], [N1+N2, 256], [N1+N2, 256]

        fake_bbox_results = dict(
            rois=rois,
            bbox_targets=(rois.new_zeros(len(rois), dtype=torch.long), None),
            bbox_pred=bbox_pred,
            cls_score=cls_score)

        fake_sampling_results = []
        for i in range(len(batch_start_index)-1):
            fake_sampling_results.append(InstanceData(
                pos_is_gt=rois.new_zeros(batch_start_index[i+1]-batch_start_index[i])))

        # refine_bboxes function: delta to bboxes
        results_list = bbox_head.refine_bboxes(
            sampling_results=fake_sampling_results,
            bbox_results=fake_bbox_results,
            batch_img_metas=batch_img_metas)
        proposal_list = [res.bboxes for res in results_list]    # {[N1, 5], [N2, 5]}
        detached_cls_scores = []
        for i in range(len(batch_start_index)-1):
            detached_cls_scores.append(
                cls_score[batch_start_index[i]: batch_start_index[i+1]].detach())

        bbox_results = dict(
            cls_score=cls_score,                        # N1+N2, 80
            decoded_bboxes=torch.cat(proposal_list),    # N1+N2, 4
            query=query,                                # N1+N2, 256
            query_pos=query_pos,                        # N1+N2, 256
            attn_feats=attn_feats,                      # N1+N2, 256
            # detach then use it in label assign
            detached_cls_scores=detached_cls_scores,    # {[N1, 80], [N2, 80]}
            detached_proposals=[item.detach() for item in proposal_list])

        return bbox_results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (List[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: a dictionary of loss components of all stage.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs
        for item in batch_gt_instances:
            item.bboxes = get_box_tensor(item.bboxes)

        initial_num_query = len(rpn_results_list[0])
        batch_start_index = torch.tensor([len(res) for res in rpn_results_list], device=x[0].device)
        batch_start_index = torch.cat([batch_start_index.new_zeros((1,)),
                                       torch.cumsum(batch_start_index, dim=0)])    # [0, 0+N1, 0+N1+N2, ...]

        query = torch.cat([res.pop('query') for res in rpn_results_list], dim=0)            # N1+N2, 256
        query_pos = torch.cat([res.pop('query_pos') for res in rpn_results_list], dim=0)    # N1+N2, 256
        results_list = rpn_results_list
        losses = {}
        query_for_last_layer, cls_score_for_last_layer, \
        proposal_for_last_layer, query_pos_for_last_layer = [], [], [], []
        for stage in range(self.num_stages):
            stage_loss_weight = self.stage_loss_weights[stage]

            # bbox head forward and loss
            bbox_results = self.bbox_loss(
                stage=stage,
                x=x,                                    # tuple 4
                query=query,                            # N1+N2, 256
                query_pos=query_pos,                    # N1+N2, 256
                batch_start_index=batch_start_index,    # [0, 0+N1, 0+N1+N2, ...]
                results_list=results_list,              # list{bs}
                batch_img_metas=batch_img_metas,        # list{bs}
                batch_gt_instances=batch_gt_instances)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{stage}.{name}'] = (
                    value * stage_loss_weight if 'loss' in name else value)

            if self.with_mask:
                raise NotImplementedError('Segmentation not implement! Welcome to finish it!')

            # prepare query and proposal as selective queries. stage: 0, 1
            assert max(self.selective_query) < self.num_stages - 1
            if stage in self.selective_query:
                query_for_last_layer.append(bbox_results['query'])  # {1*[N1+N2,256]}
                cls_score_for_last_layer.append(bbox_results['cls_score'])  # {1*[N1+N2,15]}
                proposal_for_last_layer.append(bbox_results['decoded_bboxes'])  # {1*[N1+N2, 5]}
                query_pos_for_last_layer.append(bbox_results['query_pos'])  # {1*[N1+N2, 256]}

            # filter similar query and proposal by NMS and top-k to obtain distinct query.
            assert self.distinct_query < self.num_stages - 1 and self.distinct_query >= max(self.selective_query)
            if stage == self.distinct_query:
                topk_query, topk_proposal, topk_pos,new_batch_start_index = [], [], [], []
                distinct_cfg = self.distinct_cfg
                for img_id in range(len(batch_img_metas)):
                    per_img_query = torch.cat([q[batch_start_index[img_id]:batch_start_index[img_id+1]]
                                               for q in query_for_last_layer], dim=0)       # 1*N1, 256
                    per_img_proposal = torch.cat([p[batch_start_index[img_id]:batch_start_index[img_id+1]]
                                                  for p in proposal_for_last_layer], dim=0) # 1*N1, 5
                    per_img_scores = torch.cat([s[batch_start_index[img_id]:batch_start_index[img_id+1]]
                                                for s in cls_score_for_last_layer], dim=0)  # 1*N1, 15
                    per_img_pos = torch.cat([p[batch_start_index[img_id]:batch_start_index[img_id+1]]
                                             for p in query_pos_for_last_layer], dim=0)  # 1*N, 256
                    _, keep_idxs = batched_nms(per_img_proposal,
                                               per_img_scores.max(-1).values,
                                               torch.ones(len(per_img_scores)), distinct_cfg)

                    if len(keep_idxs) >= initial_num_query:
                        topk_query.append(per_img_query[keep_idxs][:initial_num_query])       # N1, 256
                        topk_proposal.append(per_img_proposal[keep_idxs][:initial_num_query]) # N1, 4
                        topk_pos.append(per_img_pos[keep_idxs][:initial_num_query])           # N1, 256
                        new_batch_start_index.append(initial_num_query) # due to [:initial_num_query]
                    else:
                        topk_query.append(per_img_query[keep_idxs])       # N1', 256
                        topk_proposal.append(per_img_proposal[keep_idxs]) # N1', 4
                        topk_pos.append(per_img_pos[keep_idxs])           # N1', 256
                        new_batch_start_index.append(len(keep_idxs))

                # update query, query_pos, batch_start_index, results_list
                query = torch.cat(topk_query, dim=0)    # N1'+N2', 256
                query_pos = torch.cat(topk_pos, dim=0)  # N1'+N2', 256
                new_batch_start_index = torch.tensor(new_batch_start_index,
                                                     device=query.device)   # N1', N2'
                batch_start_index = torch.cat([batch_start_index.new_zeros((1,)),
                                               torch.cumsum(new_batch_start_index, dim=0)])  # [0, 0+N1', 0+N1'+N2', ...]
                results_list = []
                for idx in range(len(batch_data_samples)):
                    results = InstanceData()
                    results.imgs_whwht = bbox_results['results_list'][idx].imgs_whwht[:new_batch_start_index[idx]]
                    results.bboxes = topk_proposal[idx]
                    results_list.append(results)
            else:
                # update query and results_list
                query = bbox_results['query']   # N1+N2, 256
                results_list = bbox_results['results_list']

        return losses

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x(tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 5),
              the last dimension 5 arrange as (cx, xy, w, h, radian).
        """
        proposal_list = [res.bboxes for res in rpn_results_list]    # list{[N1, 5], [N2, 5]...}
        initial_num_query = len(rpn_results_list[0])
        batch_start_index = torch.tensor([len(res) for res in rpn_results_list], device=x[0].device)
        batch_start_index = torch.cat([batch_start_index.new_zeros((1,)),
                                       torch.cumsum(batch_start_index, dim=0)])    # [0, 0+N1, 0+N1+N2, ...]

        query = torch.cat([res.pop('query') for res in rpn_results_list], dim=0)            # N1+N2, 256
        query_pos = torch.cat([res.pop('query_pos') for res in rpn_results_list], dim=0)    # N1+N2, 256
        if all([proposal.shape[0] == 0 for proposal in proposal_list]):
            # There is no proposal in the whole batch
            return empty_instances(
                batch_img_metas, x[0].device, task_type='bbox')

        query_for_last_layer, cls_score_for_last_layer, \
        proposal_for_last_layer, query_pos_for_last_layer = [], [], [], []
        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)  # N1+N2, 6
            bbox_results = self._bbox_forward(stage, x, rois, query,
                                              query_pos, batch_start_index, batch_img_metas)

            # prepare query and proposal as selective queries. stage: 0, 1
            assert max(self.selective_query) < self.num_stages - 1
            if stage in self.selective_query:
                query_for_last_layer.append(bbox_results['query'])  # {1*[N1+N2,256]}
                cls_score_for_last_layer.append(bbox_results['cls_score'])  # {1*[N1+N2,80]}
                proposal_for_last_layer.append(bbox_results['decoded_bboxes'])  # {1*[N1+N2, 5]}
                query_pos_for_last_layer.append(bbox_results['query_pos'])  # {1*[N1+N2, 256]}

            # filter similar query and proposal by NMS and top-k to obtain distinct query.
            assert self.distinct_query < self.num_stages - 1 and self.distinct_query >= max(self.selective_query)
            if stage == self.distinct_query:
                topk_query, topk_proposal, topk_pos, topk_score, \
                new_batch_start_index = [], [], [], [], []
                distinct_cfg = self.distinct_cfg
                for img_id in range(len(batch_img_metas)):
                    per_img_query = torch.cat([q[batch_start_index[img_id]:batch_start_index[img_id + 1]]
                                               for q in query_for_last_layer], dim=0)  # 3*N1, 256
                    per_img_proposal = torch.cat([p[batch_start_index[img_id]:batch_start_index[img_id + 1]]
                                                  for p in proposal_for_last_layer], dim=0)  # 3*N1, 5
                    per_img_scores = torch.cat([s[batch_start_index[img_id]:batch_start_index[img_id + 1]]
                                                for s in cls_score_for_last_layer], dim=0)  # 3*N1, 80
                    per_img_pos = torch.cat([p[batch_start_index[img_id]:batch_start_index[img_id + 1]]
                                             for p in query_pos_for_last_layer], dim=0)  # 3*N, 256
                    _, keep_idxs = batched_nms(per_img_proposal,
                                               per_img_scores.max(-1).values,
                                               torch.ones(len(per_img_scores)), distinct_cfg)
                    if len(keep_idxs) >= initial_num_query:
                        topk_query.append(per_img_query[keep_idxs][:initial_num_query])  # N1, 256
                        topk_proposal.append(per_img_proposal[keep_idxs][:initial_num_query])  # N1, 5
                        topk_pos.append(per_img_pos[keep_idxs][:initial_num_query])  # N1, 256
                        topk_score.append(per_img_scores[keep_idxs][:initial_num_query]) # N1, 80
                        new_batch_start_index.append(initial_num_query)  # due to [:initial_num_query]
                    else:
                        topk_query.append(per_img_query[keep_idxs])  # N1', 256
                        topk_proposal.append(per_img_proposal[keep_idxs])  # N1', 5
                        topk_pos.append(per_img_pos[keep_idxs])  # N1', 256
                        topk_score.append(per_img_scores[keep_idxs]) # N1', 80
                        new_batch_start_index.append(len(keep_idxs))

                # update query, query_pos, batch_start_index, proposal_list
                query = torch.cat(topk_query, dim=0)    # N1'+N2', 256
                query_pos = torch.cat(topk_pos, dim=0)  # N1'+N2', 256
                cls_score = torch.cat(topk_score, dim=0) # N1'+N2', 80
                new_batch_start_index = torch.tensor(new_batch_start_index,
                                                     device=query.device)   # N1', N2'
                batch_start_index = torch.cat([batch_start_index.new_zeros((1,)),
                                               torch.cumsum(new_batch_start_index, dim=0)])  # [0, 0+N1', 0+N1'+N2', ...]
                proposal_list = topk_proposal   # {[N1, 5], [N2, 5]}

            else:
                # update query, proposal_list
                query = bbox_results['query']   # N1+N2, 256
                cls_score = bbox_results['cls_score']   # N1+N2, 80
                proposal_list = bbox_results['detached_proposals']  # list{[N1, 4], [N2, 4]}

        num_classes = self.bbox_head[-1].num_classes

        cls_score_ = []
        if self.bbox_head[-1].loss_cls.use_sigmoid:
            for img_id in range(len(batch_img_metas)):
                cls_score_.append(
                    cls_score[batch_start_index[img_id]: batch_start_index[img_id+1]].sigmoid())
        else:
            for img_id in range(len(batch_img_metas)):
                cls_score_.append(cls_score[batch_start_index[img_id]:].softmax(-1)[..., :-1])

        topk_inds_list = []
        results_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score_per_img = cls_score_[img_id]
            scores_per_img, topk_inds = cls_score_per_img.flatten(0, 1).topk(
                batch_start_index[img_id+1]-batch_start_index[img_id], sorted=False)
            labels_per_img = topk_inds % num_classes
            bboxes_per_img = proposal_list[img_id][topk_inds // num_classes]
            topk_inds_list.append(topk_inds)
            if rescale and bboxes_per_img.size(0) > 0:
                assert batch_img_metas[img_id].get('scale_factor') is not None
                scale_factor = bboxes_per_img.new_tensor(
                    batch_img_metas[img_id]['scale_factor']).repeat((1, 2))
                # Notice: Due to keep ratio when resize in data preparation,
                # the angle(radian) will not rescale !
                radian_factor = scale_factor.new_ones((scale_factor.size(0), 1))
                scale_factor = torch.cat([scale_factor, radian_factor], dim=-1)
                bboxes_per_img = (
                    bboxes_per_img.view(bboxes_per_img.size(0), -1, 5) /
                    scale_factor).view(bboxes_per_img.size()[0], -1)

            results = InstanceData()
            results.bboxes = bboxes_per_img
            results.scores = scores_per_img
            results.labels = labels_per_img
            results_list.append(results)
        if self.with_mask:
            raise NotImplementedError('Segmentation not implement! Welcome to finish it!')
        return results_list

    def forward(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
                batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            x (List[Tensor]): Multi-level features that may have different
                resolutions.
            rpn_results_list (List[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns
            tuple: A tuple of features from ``bbox_head`` and ``mask_head``
            forward.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs
        for item in batch_gt_instances:
            item.bboxes = get_box_tensor(item.bboxes)

        initial_num_query = len(rpn_results_list[0])
        batch_start_index = torch.tensor([len(res) for res in rpn_results_list], device=x[0].device)
        batch_start_index = torch.cat([batch_start_index.new_zeros((1,)),
                                       torch.cumsum(batch_start_index, dim=0)])    # [0, 0+N1, 0+N1+N2, ...]

        query = torch.cat([res.pop('query') for res in rpn_results_list], dim=0)            # N1+N2, 256
        query_pos = torch.cat([res.pop('query_pos') for res in rpn_results_list], dim=0)    # N1+N2, 256

        all_stage_bbox_results = []
        query_for_last_layer, cls_score_for_last_layer, \
        proposal_for_last_layer, query_pos_for_last_layer = [], [], [], []
        results_list = rpn_results_list
        if self.with_bbox:
            for stage in range(self.num_stages):
                # bbox head forward and loss
                bbox_results = self.bbox_loss(
                    stage=stage,
                    x=x,  # tuple 4
                    query=query,  # N1+N2, 256
                    query_pos=query_pos,  # N1+N2, 256
                    batch_start_index=batch_start_index,  # [0, 0+N1, 0+N1+N2, ...]
                    results_list=results_list,  # list{bs}
                    batch_img_metas=batch_img_metas,  # list{bs}
                    batch_gt_instances=batch_gt_instances)

                # prepare query and proposal as selective queries. stage: 0, 1
                assert max(self.selective_query) < self.num_stages - 1
                if stage in self.selective_query:
                    query_for_last_layer.append(bbox_results['query'])  # {1*[N1+N2,256]}
                    cls_score_for_last_layer.append(bbox_results['cls_score'])  # {1*[N1+N2,15]}
                    proposal_for_last_layer.append(bbox_results['decoded_bboxes'])  # {1*[N1+N2, 5]}
                    query_pos_for_last_layer.append(bbox_results['query_pos'])  # {1*[N1+N2, 256]}

                # prepare query and proposal as selective queries. stage: 0, 1
                assert self.distinct_query < self.num_stages - 1 and self.distinct_query >= max(self.selective_query)
                if stage == self.distinct_query:
                    topk_query, topk_proposal, topk_pos, new_batch_start_index = [], [], [], []
                    distinct_cfg = self.distinct_cfg
                    for img_id in range(len(batch_img_metas)):
                        per_img_query = torch.cat([q[batch_start_index[img_id]:batch_start_index[img_id + 1]]
                                                   for q in query_for_last_layer], dim=0)  # 1*N1, 256
                        per_img_proposal = torch.cat([p[batch_start_index[img_id]:batch_start_index[img_id + 1]]
                                                      for p in proposal_for_last_layer], dim=0)  # 1*N1, 5
                        per_img_scores = torch.cat([s[batch_start_index[img_id]:batch_start_index[img_id + 1]]
                                                    for s in cls_score_for_last_layer], dim=0)  # 1*N1, 15
                        per_img_pos = torch.cat([p[batch_start_index[img_id]:batch_start_index[img_id + 1]]
                                                 for p in query_pos_for_last_layer], dim=0)  # 1*N, 256
                        _, keep_idxs = batched_nms(per_img_proposal,
                                                   per_img_scores.max(-1).values,
                                                   torch.ones(len(per_img_scores)), distinct_cfg)

                        if len(keep_idxs) >= initial_num_query:
                            topk_query.append(per_img_query[keep_idxs][:initial_num_query])  # N1, 256
                            topk_proposal.append(per_img_proposal[keep_idxs][:initial_num_query])  # N1, 4
                            topk_pos.append(per_img_pos[keep_idxs][:initial_num_query])  # N1, 256
                            new_batch_start_index.append(initial_num_query)  # due to [:initial_num_query]
                        else:
                            topk_query.append(per_img_query[keep_idxs])  # N1', 256
                            topk_proposal.append(per_img_proposal[keep_idxs])  # N1', 4
                            topk_pos.append(per_img_pos[keep_idxs])  # N1', 256
                            new_batch_start_index.append(len(keep_idxs))

                    # update query, query_pos, batch_start_index, results_list
                    query = torch.cat(topk_query, dim=0)  # N1'+N2', 256
                    query_pos = torch.cat(topk_pos, dim=0)  # N1'+N2', 256
                    new_batch_start_index = torch.tensor(new_batch_start_index,
                                                         device=query.device)  # N1', N2'
                    batch_start_index = torch.cat([batch_start_index.new_zeros((1,)),
                                                   torch.cumsum(new_batch_start_index,
                                                                dim=0)])  # [0, 0+N1', 0+N1'+N2', ...]
                    results_list = []
                    for idx in range(len(batch_img_metas)):
                        results = InstanceData()
                        results.imgs_whwht = bbox_results['results_list'][idx].imgs_whwht[:new_batch_start_index[idx]]
                        results.bboxes = topk_proposal[idx]
                        results_list.append(results)
                else:
                    # update query and results_list
                    query = bbox_results['query']  # N1+N2, 256
                    results_list = bbox_results['results_list']

                if self.with_mask:
                    raise NotImplementedError('Segmentation not implement! Welcome to finish it!')

                bbox_results.pop('loss_bbox')
                # torch.jit does not support obj:SamplingResult
                bbox_results.pop('results_list')
                bbox_res = bbox_results.copy()
                bbox_res.pop('sampling_results')
                all_stage_bbox_results.append((bbox_res,))

        return tuple(all_stage_bbox_results)
