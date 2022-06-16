# Copyright (c) OpenMMLab. All rights reserved.
from itertools import groupby
from typing import List, Optional, Union

import cv2
import numpy as np
from mmpose.apis.webcam.nodes import NODES, BaseVisualizerNode
from mmpose.apis.webcam.utils import (get_eye_keypoint_ids,
                                      load_image_from_disk_or_url)


@NODES.register_module()
class HatEffectNode(BaseVisualizerNode):
    def __init__(self,
                 name: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 src_img_path: Optional[str] = None):

        super().__init__(name, input_buffer, output_buffer, enable_key)

        if src_img_path is None:
            # The image attributes to:
            # http://616pic.com/sucai/1m9i70p52.html
            src_img_path = 'https://user-images.githubusercontent.' \
                           'com/28900607/149766271-2f591c19-9b67-4' \
                           'd92-8f94-c272396ca141.png'
        self.src_img = load_image_from_disk_or_url(src_img_path,
                                                   cv2.IMREAD_UNCHANGED)

    @staticmethod
    def apply_hat_effect(img,
                         objects,
                         hat_img,
                         left_eye_index,
                         right_eye_index,
                         kpt_thr=0.5):
        """Apply hat effect.
        Args:
            img (np.ndarray): Image data.
            objects (list[dict]): The list of object information containing:
                - "keypoints" ([K,3]): keypoint detection result in
                    [x, y, score]
            hat_img (np.ndarray): Hat image with white alpha channel.
            left_eye_index (int): Keypoint index of left eye
            right_eye_index (int): Keypoint index of right eye
            kpt_thr (float): The score threshold of required keypoints.
        """
        img_orig = img.copy()

        img = img_orig.copy()
        hm, wm = hat_img.shape[:2]
        # anchor points in the sunglasses mask
        a = 0.3
        b = 0.7
        pts_src = np.array([[a * wm, a * hm], [a * wm, b * hm],
                            [b * wm, a * hm], [b * wm, b * hm]],
                           dtype=np.float32)

        for obj in objects:
            kpts = obj['keypoints']

            if kpts[left_eye_index, 2] < kpt_thr or \
                    kpts[right_eye_index, 2] < kpt_thr:
                continue

            kpt_leye = kpts[left_eye_index, :2]
            kpt_reye = kpts[right_eye_index, :2]
            # orthogonal vector to the left-to-right eyes
            vo = 0.5 * (kpt_reye - kpt_leye)[::-1] * [-1, 1]
            veye = 0.5 * (kpt_reye - kpt_leye)

            # anchor points in the image by eye positions
            pts_tar = np.vstack([
                kpt_reye + 1 * veye + 5 * vo, kpt_reye + 1 * veye + 1 * vo,
                kpt_leye - 1 * veye + 5 * vo, kpt_leye - 1 * veye + 1 * vo
            ])

            h_mat, _ = cv2.findHomography(pts_src, pts_tar)
            patch = cv2.warpPerspective(hat_img,
                                        h_mat,
                                        dsize=(img.shape[1], img.shape[0]),
                                        borderValue=(255, 255, 255))
            #  mask the white background area in the patch with a threshold 200
            mask = (patch[:, :, -1] > 128)
            patch = patch[:, :, :-1]
            mask = mask * (cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) > 30)
            mask = mask.astype(np.uint8)

            img = cv2.copyTo(patch, mask, img)
        return img

    def draw(self, input_msg):
        canvas = input_msg.get_image()

        objects = input_msg.get_objects(lambda x: 'keypoints' in x)

        for model_cfg, object_group in groupby(objects,
                                               lambda x: x['pose_model_cfg']):
            left_eye_idx, right_eye_idx = get_eye_keypoint_ids(model_cfg)

            canvas = self.apply_hat_effect(canvas, object_group, self.src_img,
                                           left_eye_idx, right_eye_idx)
        return canvas
