from mmrotate.registry import MODELS
from mmdet.models.detectors.two_stage import TwoStageDetector
from mmdet.models.utils.misc import unpack_gt_instances
from mmengine.structures import InstanceData

@MODELS.register_module()
class OrientedDDQRCNN(TwoStageDetector):
    def loss(self,
             batch_inputs,
             batch_data_samples):
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs

        gt_bboxes, gt_labels = [], []
        for i in range(len(batch_gt_instances)):
            gt_bboxes.append(batch_gt_instances[i].bboxes)
            gt_labels.append(batch_gt_instances[i].labels)

        losses = dict()
        x = self.extract_feat(batch_inputs)
        # remove p2 for rpn
        rpn_x = x[1:]
        roi_x = x

        rpn_losses, imgs_whwht, distinc_query_dict, position_embed = \
            self.rpn_head.loss_and_predict(
                rpn_x,
                batch_img_metas,
                gt_bboxes,
                gt_labels)
        proposals = distinc_query_dict['proposals']
        object_feats = distinc_query_dict['object_feats']

        for k, v in rpn_losses.items():
            losses[f'rpn_{k}'] = v

        rpn_results_list = []
        for idx in range(len(batch_img_metas)):
            rpn_results = InstanceData()
            rpn_results.bboxes = proposals[idx]
            rpn_results.imgs_whwht = imgs_whwht[idx].repeat(
                len(proposals[idx]), 1)
            rpn_results.query = object_feats[idx]
            rpn_results.query_pos = position_embed.weight.clone()
            rpn_results_list.append(rpn_results)

        roi_losses = self.roi_head.loss(
            roi_x, rpn_results_list, batch_data_samples)
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs,
                batch_data_samples,
                rescale = True):

        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs

        x = self.extract_feat(batch_inputs)
        # remove p2 for rpn
        rpn_x = x[1:]
        roi_x = x

        rpn_losses, imgs_whwht, distinc_query_dict, position_embed = \
            self.rpn_head.predict(
                rpn_x, batch_img_metas)

        proposals = distinc_query_dict['proposals']
        object_feats = distinc_query_dict['object_feats']

        rpn_results_list = []
        for idx in range(len(batch_img_metas)):
            rpn_results = InstanceData()
            rpn_results.bboxes = proposals[idx]
            rpn_results.imgs_whwht = imgs_whwht[idx].repeat(
                len(proposals[idx]), 1)
            rpn_results.query = object_feats[idx]
            rpn_results.query_pos = position_embed.weight.clone()
            rpn_results_list.append(rpn_results)

        results_list = self.roi_head.predict(roi_x,
                                             rpn_results_list,
                                             batch_data_samples,
                                             rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples


    def _forward(self, batch_inputs, batch_data_samples) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        assert batch_data_samples != None, 'Copy the code get_flops.py from mmdetection-3.x to mmrotate-1.x'
        results = ()
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs

        x = self.extract_feat(batch_inputs)
        # remove p2 for rpn
        rpn_x = x[1:]
        roi_x = x

        rpn_losses, imgs_whwht, distinc_query_dict, position_embed = \
            self.rpn_head.predict(
                rpn_x, batch_img_metas)

        proposals = distinc_query_dict['proposals']
        object_feats = distinc_query_dict['object_feats']

        rpn_results_list = []
        for idx in range(len(batch_img_metas)):
            rpn_results = InstanceData()
            rpn_results.bboxes = proposals[idx]
            rpn_results.imgs_whwht = imgs_whwht[idx].repeat(
                len(proposals[idx]), 1)
            rpn_results.query = object_feats[idx]
            rpn_results.query_pos = position_embed.weight.clone()
            rpn_results_list.append(rpn_results)

        roi_outs = self.roi_head.forward(roi_x, rpn_results_list,
                                         batch_data_samples)
        results = results + (roi_outs, )

        return results