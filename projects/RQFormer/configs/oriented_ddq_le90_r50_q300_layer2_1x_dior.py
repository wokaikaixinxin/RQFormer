_base_ = [
    'mmrotate::_base_/datasets/dior.py',
    'mmrotate::_base_/schedules/schedule_1x.py',
    'mmrotate::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.RRoIFormer.rroiformer'], allow_failed_imports=False)

angle_version = 'le90'
num_stages = 2
num_proposals = 300
num_classes = 20
model = dict(
    type='OrientedDDQRCNN',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedDDQFCNRPN',
        angle_version=angle_version,
        ddq_num_classes=num_classes,
        num_proposals = num_proposals,
        in_channels=256,
        feat_channels=256,
        strides=(4, 8, 16, 32, 64),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        dqs_cfg=dict(type='nms_rotated', iou_threshold=0.7, nms_pre=1000),
        offset=0.5,
        aux_loss=dict(
            loss_cls=dict(
                type='mmdet.QualityFocalLoss',
                use_sigmoid=True,
                activated=True,  # use probability instead of logit as input
                beta=2.0,
                loss_weight=1.0),
            loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=5.0),
            train_cfg=dict(
                assigner=dict(
                    type='TopkHungarianAssigner',
                    topk=8,
                    iou_calculator = dict(type='RBboxOverlaps2D'),
                    cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                    reg_cost=dict(type='RBBoxL1Cost', weight=2.0,
                             box_format='xywht', angle_version=angle_version),
                    iou_cost=dict(type='RotatedIoUCost', iou_mode='iou', weight=5.0)
                ),
                alpha=1,
                beta=6)),
        main_loss=dict(
            loss_cls=dict(
                type='mmdet.QualityFocalLoss',
                use_sigmoid=True,
                activated=True,  # use probability instead of logit as input
                beta=2.0,
                loss_weight=1.0),
            loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=5.0),
            train_cfg=dict(
                assigner=dict(
                    type='TopkHungarianAssigner',
                    topk=8,
                    iou_calculator=dict(type='RBboxOverlaps2D'),
                    cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                    reg_cost=dict(type='RBBoxL1Cost', weight=2.0,
                             box_format='xywht', angle_version=angle_version),
                    iou_cost=dict(type='RotatedIoUCost', iou_mode='iou', weight=5.0)
                ),
                alpha=1,
                beta=6))
    ),
    roi_head=dict(
        type='OrientedSparseRoIHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64]),
        bbox_head=[
            dict(
                type='OrientedDIIHead',
                angle_version=angle_version,
                num_classes=num_classes,
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='mmdet.DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
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
                    angle_version=angle_version,
                    norm_factor=None,
                    edge_swap=True,
                    proj_xy=True,
                    target_means=(.0, .0, .0, .0, .0),
                    target_stds=(1., 1., 1., 1., 1.),
                    use_box_type=False)
            ) for _ in range(num_stages)
        ]),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='mmdet.HungarianAssigner',
                    match_costs=[
                        dict(type='mmdet.FocalLossCost', weight=2.0),
                        dict(type='RBBoxL1Cost', weight=2.0,
                             box_format='xywht', angle_version=angle_version),
                        dict(type='RotatedIoUCost', iou_mode='iou', weight=5.0)
                    ]),
                sampler=dict(type='mmdet.PseudoSampler'),
                pos_weight=1) for _ in range(num_stages)
        ]),
    test_cfg=dict(
        rpn=dict(),
        rcnn=dict(max_per_img=num_proposals)))

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, type='AdamW', lr=4e-05, weight_decay=1e-07),
    clip_grad=dict(max_norm=1, norm_type=2)
)

train_cfg=dict(val_interval=1)
default_hooks = dict(checkpoint=dict(interval=1))