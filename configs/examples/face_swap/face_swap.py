# Copyright (c) OpenMMLab. All rights reserved.
executor_cfg = dict(
    name='FaceSwap',
    camera_id=0,
    camera_max_fps=20,
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
        dict(
            type='ObjectAssignerNode',
            name='ResultBinder',
            frame_buffer='_frame_',  # `_frame_` is a runner-reserved buffer
            object_buffer='human_pose',
            output_buffer='frame'),
        dict(type='FaceSwapNode',
             name='FaceSwapper',
             mode_key='s',
             input_buffer='frame',
             output_buffer='face_swap'),
        dict(type='ObjectVisualizerNode',
             name='Visualizer',
             enable_key='v',
             input_buffer='face_swap',
             output_buffer='vis_pose'),
        dict(type='NoticeBoardNode',
             name='Help Information',
             enable_key='h',
             content_lines=[
                 'Swap your faces! ',
                 'Hot-keys:',
                 '"v": Toggle the pose visualization on/off.',
                 '"s": Switch between modes: Shuffle, Clone and None',
                 '"h": Show help information',
                 '"m": Show diagnostic information',
                 '"q": Exit',
             ],
             input_buffer='vis_pose',
             output_buffer='vis_notice'),
        dict(type='MonitorNode',
             name='Monitor',
             enable_key='m',
             enable=False,
             input_buffer='vis_notice',
             output_buffer='display'),
        dict(type='RecorderNode',
             name='Recorder',
             out_video_file='faceswap_output.mp4',
             input_buffer='display',
             output_buffer='_display_')
    ])
