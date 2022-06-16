# Copyright (c) OpenMMLab. All rights reserved.
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from mmpose.apis.webcam.nodes import NODES, BaseVisualizerNode
from mmpose.apis.webcam.utils import (FrameMessage, get_eye_keypoint_ids,
                                      get_hand_keypoint_ids,
                                      get_mouth_keypoint_ids,
                                      load_image_from_disk_or_url)


@dataclass
class HeartInfo():
    """Dataclass for heart information."""
    heart_type: int
    start_time: float
    start_pos: Tuple[int, int]
    end_pos: Tuple[int, int]


@NODES.register_module()
class ValentineMagicNode(BaseVisualizerNode):
    def __init__(self,
                 name: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 kpt_vis_thr: float = 0.3,
                 hand_heart_angle_thr: float = 90.0,
                 longest_duration: float = 2.0,
                 largest_ratio: float = 0.25,
                 hand_heart_img_path: Optional[str] = None,
                 flying_heart_img_path: Optional[str] = None,
                 hand_heart_dis_ratio_thr: float = 1.0,
                 flying_heart_dis_ratio_thr: float = 3.5,
                 num_persons: int = 2):

        super().__init__(name,
                         input_buffer,
                         output_buffer,
                         enable_key=enable_key)

        if hand_heart_img_path is None:
            hand_heart_img_path = 'https://user-images.githubusercontent.com/'\
                           '87690686/149731850-ea946766-a4e8-4efa-82f5'\
                           '-e2f0515db8ae.png'
        if flying_heart_img_path is None:
            flying_heart_img_path = 'https://user-images.githubusercontent.'\
                                    'com/87690686/153554948-937ce496-33dd-4'\
                                    '9ab-9829-0433fd7c13c4.png'

        self.hand_heart = load_image_from_disk_or_url(hand_heart_img_path)
        self.flying_heart = load_image_from_disk_or_url(flying_heart_img_path)

        self.kpt_vis_thr = kpt_vis_thr
        self.hand_heart_angle_thr = hand_heart_angle_thr
        self.hand_heart_dis_ratio_thr = hand_heart_dis_ratio_thr
        self.flying_heart_dis_ratio_thr = flying_heart_dis_ratio_thr
        self.longest_duration = longest_duration
        self.largest_ratio = largest_ratio
        self.num_persons = num_persons

        # record the heart infos for each person
        self.heart_infos = {}

    def _cal_distance(self, p1: np.ndarray, p2: np.ndarray) -> np.float64:
        """calculate the distance of points p1 and p2."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _cal_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                   p4: np.ndarray) -> np.float64:
        """calculate the angle of vectors v1(constructed by points p2 and p1)
        and v2(constructed by points p4 and p3)"""
        v1 = p2 - p1
        v2 = p4 - p3

        vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
        length_prod = np.sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * np.sqrt(
            pow(v2[0], 2) + pow(v2[1], 2))
        cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)

        return (np.arccos(cos) / np.pi) * 180

    def _check_heart(self, obj: Dict[str, np.ndarray], hand_indices: List[int],
                     mouth_index: int, eye_indices: List[int]) -> int:
        """Check the type of Valentine Magic based on the pose results and
        keypoint indices of hand, mouth. and eye.

        Args:
            obj (dict): The object with the following information:
                - "keypoints" (np.ndarray[K,3]): keypoint detection result
                    in [x, y, score]
            hand_indices(list[int]): keypoint indices of hand
            mouth_index(int): keypoint index of mouth
            eye_indices(list[int]): keypoint indices of eyes

        Returns:
            int: a number representing the type of heart pose,
                 0: None, 1: hand heart, 2: left hand blow kiss,
                 3: right hand blow kiss
        """
        kpts = obj['keypoints']

        left_eye_idx, right_eye_idx = eye_indices
        left_eye_pos = kpts[left_eye_idx][:2]
        right_eye_pos = kpts[right_eye_idx][:2]
        eye_dis = self._cal_distance(left_eye_pos, right_eye_pos)

        # these indices are corresoponding to the following keypoints:
        # left_hand_root, left_pinky_finger1,
        # left_pinky_finger3, left_pinky_finger4,
        # right_hand_root, right_pinky_finger1
        # right_pinky_finger3, right_pinky_finger4

        both_hands_vis = True
        for i in [0, 17, 19, 20, 21, 38, 40, 41]:
            if kpts[hand_indices[i]][2] < self.kpt_vis_thr:
                both_hands_vis = False

        if both_hands_vis:
            p1 = kpts[hand_indices[20]][:2]
            p2 = kpts[hand_indices[19]][:2]
            p3 = kpts[hand_indices[17]][:2]
            p4 = kpts[hand_indices[0]][:2]
            left_angle = self._cal_angle(p1, p2, p3, p4)

            p1 = kpts[hand_indices[41]][:2]
            p2 = kpts[hand_indices[40]][:2]
            p3 = kpts[hand_indices[38]][:2]
            p4 = kpts[hand_indices[21]][:2]
            right_angle = self._cal_angle(p1, p2, p3, p4)

            hand_dis = self._cal_distance(kpts[hand_indices[20]][:2],
                                          kpts[hand_indices[41]][:2])

            if (left_angle < self.hand_heart_angle_thr
                    and right_angle < self.hand_heart_angle_thr
                    and hand_dis / eye_dis < self.hand_heart_dis_ratio_thr):
                return 1

        # these indices are corresoponding to the following keypoints:
        # left_middle_finger1, left_middle_finger4,
        left_hand_vis = True
        for i in [9, 12]:
            if kpts[hand_indices[i]][2] < self.kpt_vis_thr:
                left_hand_vis = False
                break
        # right_middle_finger1, right_middle_finger4

        right_hand_vis = True
        for i in [30, 33]:
            if kpts[hand_indices[i]][2] < self.kpt_vis_thr:
                right_hand_vis = False
                break

        mouth_vis = True
        if kpts[mouth_index][2] < self.kpt_vis_thr:
            mouth_vis = False

        if (not left_hand_vis and not right_hand_vis) or not mouth_vis:
            return 0

        mouth_pos = kpts[mouth_index]

        left_mid_hand_pos = (kpts[hand_indices[9]][:2] +
                             kpts[hand_indices[12]][:2]) / 2
        lefthand_mouth_dis = self._cal_distance(left_mid_hand_pos, mouth_pos)

        if lefthand_mouth_dis / eye_dis < self.flying_heart_dis_ratio_thr:
            return 2

        right_mid_hand_pos = (kpts[hand_indices[30]][:2] +
                              kpts[hand_indices[33]][:2]) / 2
        righthand_mouth_dis = self._cal_distance(right_mid_hand_pos, mouth_pos)

        if righthand_mouth_dis / eye_dis < self.flying_heart_dis_ratio_thr:
            return 3

        return 0

    def _get_heart_route(self, heart_type: int, cur_pred: Dict[str,
                                                               np.ndarray],
                         tar_pred: Dict[str,
                                        np.ndarray], hand_indices: List[int],
                         mouth_index: int) -> Tuple[int, int]:
        """get the start and end position of the heart, based on two keypoint
        results and keypoint indices of hand and mouth.

        Args:
            cur_pred(dict): The pose estimation results of current person,
                containing: the following keys:
                - "keypoints" (np.ndarray[K,3]): keypoint detection result
                                                 in [x, y, score]
            tar_pred(dict): The pose estimation results of target person,
                containing: the following keys:
                - "keypoints" (np.ndarray[K,3]): keypoint detection result
                                                 in [x, y, score]
            hand_indices(list[int]): keypoint indices of hand
            mouth_index(int): keypoint index of mouth

        Returns:
            tuple(int): the start position of heart
            tuple(int): the end position of heart
        """
        cur_kpts = cur_pred['keypoints']

        assert heart_type in [1, 2,
                              3], 'Can not determine the type of heart effect'

        if heart_type == 1:
            p1 = cur_kpts[hand_indices[20]][:2]
            p2 = cur_kpts[hand_indices[41]][:2]
        elif heart_type == 2:
            p1 = cur_kpts[hand_indices[9]][:2]
            p2 = cur_kpts[hand_indices[12]][:2]
        elif heart_type == 3:
            p1 = cur_kpts[hand_indices[30]][:2]
            p2 = cur_kpts[hand_indices[33]][:2]

        cur_x, cur_y = (p1 + p2) / 2
        # the mid point of two fingers
        start_pos = (int(cur_x), int(cur_y))

        tar_kpts = tar_pred['keypoints']
        end_pos = tar_kpts[mouth_index][:2]

        return start_pos, end_pos

    def _draw_heart(self, canvas: np.ndarray, heart_info: HeartInfo,
                    t_pass: float) -> np.ndarray:
        """draw the heart according to heart info and time."""
        start_x, start_y = heart_info.start_pos
        end_x, end_y = heart_info.end_pos

        scale = t_pass / self.longest_duration

        max_h, max_w = canvas.shape[:2]
        hm, wm = self.largest_ratio * max_h, self.largest_ratio * max_h
        new_h, new_w = int(hm * scale), int(wm * scale)

        x = int(start_x + scale * (end_x - start_x))
        y = int(start_y + scale * (end_y - start_y))

        y1 = max(0, y - int(new_h / 2))
        y2 = min(max_h - 1, y + int(new_h / 2))

        x1 = max(0, x - int(new_w / 2))
        x2 = min(max_w - 1, x + int(new_w / 2))

        target = canvas[y1:y2 + 1, x1:x2 + 1].copy()
        new_h, new_w = target.shape[:2]

        if new_h == 0 or new_w == 0:
            return canvas

        assert heart_info.heart_type in [
            1, 2, 3
        ], 'Can not determine the type of heart effect'
        if heart_info.heart_type == 1:  # hand heart
            patch = self.hand_heart.copy()
        elif heart_info.heart_type >= 2:  # hand blow kiss
            patch = self.flying_heart.copy()
            if heart_info.start_pos[0] > heart_info.end_pos[0]:
                patch = patch[:, ::-1]

        patch = cv2.resize(patch, (new_w, new_h))
        mask = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        mask = (mask < 100)[..., None].astype(np.float32) * 0.8

        canvas[y1:y2 + 1, x1:x2 + 1] = patch * mask + target * (1 - mask)

        return canvas

    def draw(self, input_msg: FrameMessage) -> np.ndarray:
        canvas = input_msg.get_image()

        persons = input_msg.get_objects(lambda x: 'keypoints' in x)

        if not persons or len(persons) < self.num_persons:
            return canvas

        # Only user the first 2 detected persons
        persons = persons[:self.num_persons]
        ids = [person['track_id'] for person in persons]
        model_cfg = persons[0]['pose_model_cfg']

        for id_ in self.heart_infos.copy():
            if id_ not in ids:
                # if the id of a person not in previous heart_infos,
                # delete the corresponding field
                del self.heart_infos[id_]

        for i, person in enumerate(persons):
            id_ = person['track_id']

            # if the predicted person in previous heart_infos,
            # draw the heart
            if id_ in self.heart_infos.copy():
                t_pass = time.time() - self.heart_infos[id_].start_time

                # the time passed since last heart pose less than
                # longest_duration, continue to draw the heart
                if t_pass < self.longest_duration:
                    canvas = self._draw_heart(canvas, self.heart_infos[id_],
                                              t_pass)
                # reset corresponding heart info
                else:
                    del self.heart_infos[id_]
            else:
                hand_indices = get_hand_keypoint_ids(model_cfg)
                mouth_index = get_mouth_keypoint_ids(model_cfg)
                eye_indices = get_eye_keypoint_ids(model_cfg)

                # check the type of Valentine Magic based on pose results
                # and keypoint indices of hand and mouth
                heart_type = self._check_heart(person, hand_indices,
                                               mouth_index, eye_indices)
                # trigger a Valentine Magic effect
                if heart_type:
                    # get the route of heart
                    start_pos, end_pos = self._get_heart_route(
                        heart_type, person, persons[self.num_persons - 1 - i],
                        hand_indices, mouth_index)
                    start_time = time.time()
                    self.heart_infos[id_] = HeartInfo(heart_type, start_time,
                                                      start_pos, end_pos)

        return canvas
