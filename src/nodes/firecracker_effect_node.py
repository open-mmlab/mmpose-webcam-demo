from itertools import groupby
from typing import List, Optional, Union

import cv2
import numpy as np
from mmpose.apis.webcam.nodes import NODES, BaseVisualizerNode
from mmpose.apis.webcam.utils import get_wrist_keypoint_ids


@NODES.register_module()
class FirecrackerEffectNode(BaseVisualizerNode):
    def __init__(self,
                 name: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 src_img_path: Optional[str] = None):

        super().__init__(name, input_buffer, output_buffer, enable_key)

        if src_img_path is None:
            self.src_img_path = 'https://user-images.githubusercontent' \
                                '.com/28900607/149766281-6376055c-ed8b' \
                                '-472b-991f-60e6ae6ee1da.gif'
        src_img = cv2.VideoCapture(self.src_img_path)

        self.frame_list = []
        ret, frame = src_img.read()
        while ret:
            self.frame_list.append(frame)
            ret, frame = src_img.read()
        self.num_frames = len(self.frame_list)
        self.frame_idx = 0
        self.frame_period = 4  # each frame in gif lasts for 4 frames in video

    @staticmethod
    def apply_firecracker_effect(img,
                                 objects,
                                 firecracker_img,
                                 left_wrist_idx,
                                 right_wrist_idx,
                                 kpt_thr=0.5):
        """Apply firecracker effect.
        Args:
            img (np.ndarray): Image data.
            objects (list[dict]): The objects with the following information:
                - "keypoints" (np.ndarray[K,3]): keypoint detection result in
                    [x, y, score]
            firecracker_img (np.ndarray): Firecracker image with white
                background.
            left_wrist_idx (int): Keypoint index of left wrist
            right_wrist_idx (int): Keypoint index of right wrist
            kpt_thr (float): The score threshold of required keypoints.
        """

        hm, wm = firecracker_img.shape[:2]
        # anchor points in the firecracker mask
        pts_src = np.array([[0. * wm, 0. * hm], [0. * wm, 1. * hm],
                            [1. * wm, 0. * hm], [1. * wm, 1. * hm]],
                           dtype=np.float32)

        h, w = img.shape[:2]
        h_tar = h / 3
        w_tar = h_tar / hm * wm

        for obj in objects:
            kpts = obj['keypoints']

            if kpts[left_wrist_idx, 2] > kpt_thr:
                kpt_lwrist = kpts[left_wrist_idx, :2]
                # anchor points in the image by eye positions
                pts_tar = np.vstack([
                    kpt_lwrist - [w_tar / 2, 0],
                    kpt_lwrist - [w_tar / 2, -h_tar],
                    kpt_lwrist + [w_tar / 2, 0],
                    kpt_lwrist + [w_tar / 2, h_tar]
                ])

                h_mat, _ = cv2.findHomography(pts_src, pts_tar)
                patch = cv2.warpPerspective(firecracker_img,
                                            h_mat,
                                            dsize=(img.shape[1], img.shape[0]),
                                            borderValue=(255, 255, 255))
                #  mask the white background area in the patch with
                # a threshold 200
                mask = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                mask = (mask < 240).astype(np.uint8)
                img = cv2.copyTo(patch, mask, img)

            if kpts[right_wrist_idx, 2] > kpt_thr:
                kpt_rwrist = kpts[right_wrist_idx, :2]

                # anchor points in the image by eye positions
                pts_tar = np.vstack([
                    kpt_rwrist - [w_tar / 2, 0],
                    kpt_rwrist - [w_tar / 2, -h_tar],
                    kpt_rwrist + [w_tar / 2, 0],
                    kpt_rwrist + [w_tar / 2, h_tar]
                ])

                h_mat, _ = cv2.findHomography(pts_src, pts_tar)
                patch = cv2.warpPerspective(firecracker_img,
                                            h_mat,
                                            dsize=(img.shape[1], img.shape[0]),
                                            borderValue=(255, 255, 255))
                #  mask the white background area in the patch with
                # a threshold 200
                mask = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                mask = (mask < 240).astype(np.uint8)
                img = cv2.copyTo(patch, mask, img)

        return img

    def draw(self, input_msg):
        canvas = input_msg.get_image()

        objects = input_msg.get_objects(lambda x: 'keypoints' in x)
        frame = self.frame_list[self.frame_idx // self.frame_period]

        for model_cfg, object_group in groupby(objects,
                                               lambda x: x['pose_model_cfg']):

            left_wrist_idx, right_wrist_idx = get_wrist_keypoint_ids(model_cfg)

            canvas = self.apply_firecracker_effect(canvas, object_group, frame,
                                                   left_wrist_idx,
                                                   right_wrist_idx)

        self.frame_idx = (self.frame_idx + 1) % (self.num_frames *
                                                 self.frame_period)

        return canvas
