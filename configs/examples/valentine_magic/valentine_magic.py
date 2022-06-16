# Copyright (c) OpenMMLab. All rights reserved.
executor_cfg = dict(
    # Basic configurations of the runner
    name='Valentine Magic',
    camera_id=0,
    nodes=[
        dict(
            type='DetectorNode',
            name='detector',
            model_config='model_configs/mmdet/'
            'ssdlite_mobilenetv2_scratch_600e_coco.py',
            model_checkpoint='https://download.openmmlab.com'
            '/mmdetection/v2.0/ssd/'
            'ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_'
            'scratch_600e_coco_20210629_110627-974d9307.pth',
            bbox_thr=0.7,
            input_buffer='_input_',  # `_input_` is an executor-reserved buffer
            output_buffer='det_result'),
        dict(type='TopDownPoseEstimatorNode',
             name='human pose estimator',
             model_config='model_configs/mmpose/'
             'vipnas_mbv3_coco_wholebody_256x192_dark.py',
             model_checkpoint='https://download.openmmlab.com/mmpose/top_down/'
             'vipnas/vipnas_mbv3_coco_wholebody_256x192_dark'
             '-e2158108_20211205.pth',
             labels=['person'],
             smooth=True,
             input_buffer='det_result',
             output_buffer='pose_result'),
        dict(
            type='ObjectAssignerNode',
            name='object assigner',
            frame_buffer='_frame_',  # `_frame_` is a runner-reserved buffer
            object_buffer='pose_result',
            output_buffer='frame'),
        dict(type='ObjectVisualizerNode',
             name='object visualizer',
             enable_key='v',
             enable=False,
             input_buffer='frame',
             output_buffer='vis'),
        dict(
            type='ValentineMagicNode',
            name='valentine magic',
            enable_key='l',
            input_buffer='vis',
            output_buffer='vis_heart',
        ),
        dict(
            type='NoticeBoardNode',
            name='Helper',
            enable_key='h',
            enable=False,
            input_buffer='vis_heart',
            output_buffer='vis_notice',
            content_lines=[
                'This is a demo for pose visualization and simple image '
                'effects. Have fun!', '', 'Hot-keys:',
                '"h": Show help information', '"l": LoveHeart Effect',
                '"v": PoseVisualizer', '"m": Show diagnostic information',
                '"q": Exit'
            ],
        ),
        dict(type='MonitorNode',
             name='monitor',
             enable_key='m',
             enable=False,
             input_buffer='vis_notice',
             output_buffer='display'),  # `_frame_` is a runner-reserved buffer
        dict(type='RecorderNode',
             name='recorder',
             out_video_file='valentine.mp4',
             input_buffer='display',
             output_buffer='_display_')
    ])
