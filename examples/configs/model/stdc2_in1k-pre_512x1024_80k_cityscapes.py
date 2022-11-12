norm_cfg = dict(type='BN', requires_grad=True)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/stdc/stdc2_20220308-7dbd9127.pth'  # noqa: #501
model = dict(type='EncoderDecoder',
             pretrained=None,
             backbone=dict(type='STDCContextPathNet',
                           backbone_cfg=dict(type='STDCNet',
                                             stdc_type='STDCNet2',
                                             in_channels=3,
                                             channels=(32, 64, 256, 512, 1024),
                                             bottleneck_type='cat',
                                             num_convs=4,
                                             norm_cfg=dict(type='BN',
                                                           requires_grad=True),
                                             act_cfg=dict(type='ReLU'),
                                             with_final_conv=False,
                                             init_cfg=dict(
                                                 type='Pretrained',
                                                 checkpoint=checkpoint)),
                           last_in_channels=(1024, 512),
                           out_channels=128,
                           ffm_cfg=dict(in_channels=384,
                                        out_channels=256,
                                        scale_factor=4)),
             decode_head=dict(type='FCNHead',
                              in_channels=256,
                              channels=256,
                              num_convs=1,
                              num_classes=19,
                              in_index=3,
                              concat_input=False,
                              dropout_ratio=0.1,
                              norm_cfg=dict(type='BN', requires_grad=True),
                              align_corners=True,
                              sampler=dict(type='OHEMPixelSampler',
                                           thresh=0.7,
                                           min_kept=10000),
                              loss_decode=dict(type='CrossEntropyLoss',
                                               use_sigmoid=False,
                                               loss_weight=1.0)),
             auxiliary_head=[
                 dict(type='FCNHead',
                      in_channels=128,
                      channels=64,
                      num_convs=1,
                      num_classes=19,
                      in_index=2,
                      norm_cfg=dict(type='BN', requires_grad=True),
                      concat_input=False,
                      align_corners=False,
                      sampler=dict(type='OHEMPixelSampler',
                                   thresh=0.7,
                                   min_kept=10000),
                      loss_decode=dict(type='CrossEntropyLoss',
                                       use_sigmoid=False,
                                       loss_weight=1.0)),
                 dict(type='FCNHead',
                      in_channels=128,
                      channels=64,
                      num_convs=1,
                      num_classes=19,
                      in_index=1,
                      norm_cfg=dict(type='BN', requires_grad=True),
                      concat_input=False,
                      align_corners=False,
                      sampler=dict(type='OHEMPixelSampler',
                                   thresh=0.7,
                                   min_kept=10000),
                      loss_decode=dict(type='CrossEntropyLoss',
                                       use_sigmoid=False,
                                       loss_weight=1.0)),
                 dict(type='STDCHead',
                      in_channels=256,
                      channels=64,
                      num_convs=1,
                      num_classes=2,
                      boundary_threshold=0.1,
                      in_index=0,
                      norm_cfg=dict(type='BN', requires_grad=True),
                      concat_input=False,
                      align_corners=True,
                      loss_decode=[
                          dict(type='CrossEntropyLoss',
                               loss_name='loss_ce',
                               use_sigmoid=True,
                               loss_weight=1.0),
                          dict(type='DiceLoss',
                               loss_name='loss_dice',
                               loss_weight=1.0)
                      ])
             ],
             train_cfg=dict(),
             test_cfg=dict(mode='whole'))
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(2048, 1024),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize',
                  mean=[123.675, 116.28, 103.53],
                  std=[58.395, 57.12, 57.375],
                  to_rgb=True),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img'])
         ])
]
data = dict(samples_per_gpu=12,
            workers_per_gpu=4,
            train=dict(type='CityscapesDataset',
                       data_root='data/cityscapes/',
                       img_dir='leftImg8bit/train',
                       ann_dir='gtFine/train',
                       pipeline=[
                           dict(type='LoadImageFromFile'),
                           dict(type='LoadAnnotations'),
                           dict(type='Resize',
                                img_scale=(2048, 1024),
                                ratio_range=(0.5, 2.0)),
                           dict(type='RandomCrop',
                                crop_size=(512, 1024),
                                cat_max_ratio=0.75),
                           dict(type='RandomFlip', prob=0.5),
                           dict(type='PhotoMetricDistortion'),
                           dict(type='Normalize',
                                mean=[123.675, 116.28, 103.53],
                                std=[58.395, 57.12, 57.375],
                                to_rgb=True),
                           dict(type='Pad',
                                size=(512, 1024),
                                pad_val=0,
                                seg_pad_val=255),
                           dict(type='DefaultFormatBundle'),
                           dict(type='Collect',
                                keys=['img', 'gt_semantic_seg'])
                       ]),
            val=dict(type='CityscapesDataset',
                     data_root='data/cityscapes/',
                     img_dir='leftImg8bit/val',
                     ann_dir='gtFine/val',
                     pipeline=[
                         dict(type='LoadImageFromFile'),
                         dict(type='MultiScaleFlipAug',
                              img_scale=(2048, 1024),
                              flip=False,
                              transforms=[
                                  dict(type='Resize', keep_ratio=True),
                                  dict(type='RandomFlip'),
                                  dict(type='Normalize',
                                       mean=[123.675, 116.28, 103.53],
                                       std=[58.395, 57.12, 57.375],
                                       to_rgb=True),
                                  dict(type='ImageToTensor', keys=['img']),
                                  dict(type='Collect', keys=['img'])
                              ])
                     ]),
            test=dict(type='CityscapesDataset',
                      data_root='data/cityscapes/',
                      img_dir='leftImg8bit/val',
                      ann_dir='gtFine/val',
                      pipeline=[
                          dict(type='LoadImageFromFile'),
                          dict(type='MultiScaleFlipAug',
                               img_scale=(2048, 1024),
                               flip=False,
                               transforms=[
                                   dict(type='Resize', keep_ratio=True),
                                   dict(type='RandomFlip'),
                                   dict(type='Normalize',
                                        mean=[123.675, 116.28, 103.53],
                                        std=[58.395, 57.12, 57.375],
                                        to_rgb=True),
                                   dict(type='ImageToTensor', keys=['img']),
                                   dict(type='Collect', keys=['img'])
                               ])
                      ]))
log_config = dict(interval=50,
                  hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly',
                 power=0.9,
                 min_lr=0.0001,
                 by_epoch=False,
                 warmup='linear',
                 warmup_iters=1000)
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
