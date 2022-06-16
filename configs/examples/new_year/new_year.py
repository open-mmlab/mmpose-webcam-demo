# Copyright (c) OpenMMLab. All rights reserved.
executor_cfg = dict(
    # Basic configurations of the runner
    name='New Year',
    camera_id=0,
    camera_max_fps=20,
    synchronous=False,
    # Define nodes.
    # The configuration of a node usually includes:
    #   1. 'type': Node class name
    #   2. 'name': Node name
    #   3. I/O buffers (e.g. 'input_buffer', 'output_buffer'): specify the
    #       input and output buffer names. This may depend on the node class.
    #   4. 'enable_key': assign a hot-key to toggle enable/disable this node.
    #       This may depend on the node class.
    #   5. Other class-specific arguments
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
             output_buffer='human_pose'),
        dict(type='TopDownPoseEstimatorNode',
             name='animal pose estimator',
             model_config='model_configs/'
             'mmpose/hrnet_w32_animalpose_256x256.py',
             model_checkpoint='https://download.openmmlab.com/mmpose/animal/'
             'hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth',
             labels=['cat', 'dog', 'horse', 'sheep', 'cow'],
             input_buffer='human_pose',
             output_buffer='animal_pose'),
        dict(
            type='ObjectAssignerNode',
            name='object assigner',
            frame_buffer='_frame_',  # `_frame_` is an executor-reserved buffer
            object_buffer='animal_pose',
            output_buffer='frame'),
        # 'HatNode':
        # This node draw the hat effect in the frame image.
        # Pose results is needed.
        dict(type='HatEffectNode',
             name='hat effect',
             enable_key='t',
             input_buffer='frame',
             output_buffer='vis_hat'),
        # 'FirecrackerNode':
        # This node draw the firecracker effect in the frame image.
        # Pose results is needed.
        dict(type='FirecrackerEffectNode',
             name='firecracker effect',
             enable_key='f',
             input_buffer='vis_hat',
             output_buffer='vis_firecracker'),
        # 'NoticeBoardNode':
        # This node show a notice board with given content, e.g. help
        # information.
        dict(
            type='NoticeBoardNode',
            name='instruction',
            enable_key='h',
            enable=False,
            input_buffer='vis_firecracker',
            output_buffer='vis_notice',
            content_lines=[
                'This is a demo for pose visualization and simple image '
                'effects. Have fun!', '', 'Hot-keys:', '"t": Hat effect',
                '"f": Firecracker effect', '"h": Show help information',
                '"m": Show diagnostic information', '"q": Exit'
            ],
        ),
        # 'MonitorNode':
        # This node show diagnostic information in the frame image. It can
        # be used for debugging or monitoring system resource status.
        dict(type='MonitorNode',
             name='monitor',
             enable_key='m',
             enable=False,
             input_buffer='vis_notice',
             output_buffer='display'),
        # 'RecorderNode':
        # This node save the output video into a file.
        dict(type='RecorderNode',
             name='recorder',
             out_video_file='new_year.mp4',
             input_buffer='display',
             output_buffer='_display_'
             # `_display_` is a runner-reserved buffer
             )
    ])
