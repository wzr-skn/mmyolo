# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import unittest

import mmcv
import numpy as np
import torch
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.structures.mask import BitmapMasks

from mmyolo.datasets.transforms import (LetterResize, LoadAnnotations,
                                        YOLOv5HSVRandomAug,
                                        YOLOv5KeepRatioResize,
                                        YOLOv5RandomAffine)


class TestLetterResize(unittest.TestCase):

    def setUp(self):
        """Set up the data info which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.data_info1 = dict(
            img=np.random.random((300, 400, 3)),
            gt_bboxes=np.array([[0, 0, 150, 150]], dtype=np.float32),
            batch_shape=np.array([460, 672], dtype=np.int64),
            gt_masks=BitmapMasks(rng.rand(1, 300, 400), height=300, width=400))
        self.data_info2 = dict(
            img=np.random.random((300, 400, 3)),
            gt_bboxes=np.array([[0, 0, 150, 150]], dtype=np.float32))
        self.data_info3 = dict(
            img=np.random.random((300, 400, 3)),
            batch_shape=np.array([460, 672], dtype=np.int64))
        self.data_info4 = dict(img=np.random.random((300, 400, 3)))

    def test_letter_resize(self):
        # Test allow_scale_up
        transform = LetterResize(scale=(640, 640), allow_scale_up=False)
        results = transform(copy.deepcopy(self.data_info1))
        self.assertEqual(results['img_shape'], (460, 672, 3))
        self.assertTrue(
            (results['gt_bboxes'] == np.array([[136., 80., 286.,
                                                230.]])).all())
        self.assertTrue((results['batch_shape'] == np.array([460, 672])).all())
        self.assertTrue(
            (results['pad_param'] == np.array([80., 80., 136., 136.])).all())
        self.assertTrue((results['scale_factor'] <= 1.).all())

        # Test pad_val
        transform = LetterResize(scale=(640, 640), pad_val=dict(img=144))
        results = transform(copy.deepcopy(self.data_info1))
        self.assertEqual(results['img_shape'], (460, 672, 3))
        self.assertTrue(
            (results['gt_bboxes'] == np.array([[29., 0., 259., 230.]])).all())
        self.assertTrue((results['batch_shape'] == np.array([460, 672])).all())
        self.assertTrue((results['pad_param'] == np.array([0., 0., 29.,
                                                           30.])).all())
        self.assertTrue((results['scale_factor'] > 1.).all())

        # Test use_mini_pad
        transform = LetterResize(scale=(640, 640), use_mini_pad=True)
        results = transform(copy.deepcopy(self.data_info1))
        self.assertEqual(results['img_shape'], (460, 640, 3))
        self.assertTrue(
            (results['gt_bboxes'] == np.array([[13., 0., 243., 230.]])).all())
        self.assertTrue((results['batch_shape'] == np.array([460, 672])).all())
        self.assertTrue((results['pad_param'] == np.array([0., 0., 13.,
                                                           14.])).all())
        self.assertTrue((results['scale_factor'] > 1.).all())

        # Test stretch_only
        transform = LetterResize(scale=(640, 640), stretch_only=True)
        results = transform(copy.deepcopy(self.data_info1))
        self.assertEqual(results['img_shape'], (460, 672, 3))
        self.assertTrue((results['gt_bboxes'] == np.array(
            [[0., 0., 230., 251.99998474121094]])).all())
        self.assertTrue((results['batch_shape'] == np.array([460, 672])).all())
        self.assertTrue((results['pad_param'] == np.array([0, 0, 0, 0])).all())

        # Test
        transform = LetterResize(scale=(640, 640), pad_val=dict(img=144))
        rng = np.random.RandomState(0)
        for _ in range(5):
            input_h, input_w = np.random.randint(100, 700), np.random.randint(
                100, 700)
            output_h, output_w = np.random.randint(100,
                                                   700), np.random.randint(
                                                       100, 700)
            data_info = dict(
                img=np.random.random((input_h, input_w, 3)),
                gt_bboxes=np.array([[0, 0, 10, 10]], dtype=np.float32),
                batch_shape=np.array([output_h, output_w], dtype=np.int64),
                gt_masks=BitmapMasks(
                    rng.rand(1, input_h, input_w),
                    height=input_h,
                    width=input_w))
            results = transform(data_info)
            self.assertEqual(results['img_shape'], (output_h, output_w, 3))
            self.assertTrue(
                (results['batch_shape'] == np.array([output_h,
                                                     output_w])).all())

        # Test without batchshape
        transform = LetterResize(scale=(640, 640), pad_val=dict(img=144))
        rng = np.random.RandomState(0)
        for _ in range(5):
            input_h, input_w = np.random.randint(100, 700), np.random.randint(
                100, 700)
            data_info = dict(
                img=np.random.random((input_h, input_w, 3)),
                gt_bboxes=np.array([[0, 0, 10, 10]], dtype=np.float32),
                gt_masks=BitmapMasks(
                    rng.rand(1, input_h, input_w),
                    height=input_h,
                    width=input_w))
            results = transform(data_info)
            self.assertEqual(results['img_shape'], (640, 640, 3))


class TestYOLOv5KeepRatioResize(unittest.TestCase):

    def setUp(self):
        """Set up the data info which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.data_info1 = dict(
            img=np.random.random((300, 400, 3)),
            gt_bboxes=np.array([[0, 0, 150, 150]], dtype=np.float32),
            gt_masks=BitmapMasks(rng.rand(1, 300, 400), height=300, width=400))
        self.data_info2 = dict(img=np.random.random((300, 400, 3)))

    def test_yolov5_keep_ratio_resize(self):

        # test assertion for invalid keep_ratio
        with self.assertRaises(AssertionError):
            transform = YOLOv5KeepRatioResize(scale=(640, 640))
            transform.keep_ratio = False
            results = transform(copy.deepcopy(self.data_info1))

        # Test with gt_bboxes
        transform = YOLOv5KeepRatioResize(scale=(640, 640))
        results = transform(copy.deepcopy(self.data_info1))
        self.assertTrue(transform.keep_ratio, True)
        self.assertEqual(results['img_shape'], (480, 640))
        self.assertTrue(
            (results['gt_bboxes'] == np.array([[0., 0., 240., 240.]])).all())
        self.assertTrue((results['scale_factor'] == 1.6).all())

        # Test only img
        transform = YOLOv5KeepRatioResize(scale=(640, 640))
        results = transform(copy.deepcopy(self.data_info2))
        self.assertEqual(results['img_shape'], (480, 640))
        self.assertTrue((results['scale_factor'] == 1.6).all())


class TestYOLOv5HSVRandomAug(unittest.TestCase):

    def setUp(self):
        """Set up the data info which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.data_info = dict(
            img=mmcv.imread(
                osp.join(osp.dirname(__file__), '../../data/color.jpg'),
                'color'))

    def test_yolov5_hsv_random_aug(self):
        # Test with gt_bboxes
        transform = YOLOv5HSVRandomAug(
            hue_delta=0.015, saturation_delta=0.7, value_delta=0.4)
        results = transform(copy.deepcopy(self.data_info))
        self.assertTrue(
            results['img'].shape[:2] == self.data_info['img'].shape[:2])


class TestLoadAnnotations(unittest.TestCase):

    def setUp(self):
        """Set up the data info which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        data_prefix = osp.join(osp.dirname(__file__), '../../data')
        seg_map = osp.join(data_prefix, 'gray.jpg')
        self.results = {
            'ori_shape': (300, 400),
            'seg_map_path':
            seg_map,
            'instances': [{
                'bbox': [0, 0, 10, 20],
                'bbox_label': 1,
                'mask': [[0, 0, 0, 20, 10, 20, 10, 0]],
                'ignore_flag': 0
            }, {
                'bbox': [10, 10, 110, 120],
                'bbox_label': 2,
                'mask': [[10, 10, 110, 10, 110, 120, 110, 10]],
                'ignore_flag': 0
            }, {
                'bbox': [50, 50, 60, 80],
                'bbox_label': 2,
                'mask': [[50, 50, 60, 50, 60, 80, 50, 80]],
                'ignore_flag': 1
            }]
        }

    def test_load_bboxes(self):
        transform = LoadAnnotations(
            with_bbox=True,
            with_label=False,
            with_seg=False,
            with_mask=False,
            box_type=None)
        results = transform(copy.deepcopy(self.results))
        self.assertIn('gt_bboxes', results)
        self.assertTrue((results['gt_bboxes'] == np.array([[0, 0, 10, 20],
                                                           [10, 10, 110,
                                                            120]])).all())
        self.assertEqual(results['gt_bboxes'].dtype, np.float32)
        self.assertTrue(
            (results['gt_ignore_flags'] == np.array([False, False])).all())
        self.assertEqual(results['gt_ignore_flags'].dtype, bool)

    def test_load_labels(self):
        transform = LoadAnnotations(
            with_bbox=False,
            with_label=True,
            with_seg=False,
            with_mask=False,
        )
        results = transform(copy.deepcopy(self.results))
        self.assertIn('gt_bboxes_labels', results)
        self.assertTrue((results['gt_bboxes_labels'] == np.array([1,
                                                                  2])).all())
        self.assertEqual(results['gt_bboxes_labels'].dtype, np.int64)


class TestYOLOv5RandomAffine(unittest.TestCase):

    def setUp(self):
        """Setup the data info which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.results = {
            'img':
            np.random.random((224, 224, 3)),
            'img_shape': (224, 224),
            'gt_bboxes_labels':
            np.array([1, 2, 3], dtype=np.int64),
            'gt_bboxes':
            np.array([[10, 10, 20, 20], [20, 20, 40, 40], [40, 40, 80, 80]],
                     dtype=np.float32),
            'gt_ignore_flags':
            np.array([0, 0, 1], dtype=bool),
        }

    def test_transform(self):
        # test assertion for invalid translate_ratio
        with self.assertRaises(AssertionError):
            transform = YOLOv5RandomAffine(max_translate_ratio=1.5)

        # test assertion for invalid scaling_ratio_range
        with self.assertRaises(AssertionError):
            transform = YOLOv5RandomAffine(scaling_ratio_range=(1.5, 0.5))

        with self.assertRaises(AssertionError):
            transform = YOLOv5RandomAffine(scaling_ratio_range=(0, 0.5))

        transform = YOLOv5RandomAffine()
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['img'].shape[:2] == (224, 224))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)

    def test_transform_with_boxlist(self):
        results = copy.deepcopy(self.results)
        results['gt_bboxes'] = HorizontalBoxes(results['gt_bboxes'])

        transform = YOLOv5RandomAffine()
        results = transform(copy.deepcopy(results))
        self.assertTrue(results['img'].shape[:2] == (224, 224))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == torch.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)
