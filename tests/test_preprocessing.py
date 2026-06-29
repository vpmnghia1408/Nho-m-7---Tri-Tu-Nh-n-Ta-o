"""Unit test cho module tiền xử lý của Người 2.

Chạy:  python -m pytest tests/ -v
Hoặc:  python -m unittest tests.test_preprocessing
"""

import unittest

import numpy as np

from src.config import IMG_SIZE, INPUT_SHAPE, NUM_CHANNELS
from src.preprocessing import (
    resize_image,
    to_grayscale,
    normalize_pixels,
    preprocess_image,
    preprocess_batch,
    augment_image,
)


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.raw = rng.integers(0, 256, size=(50, 60, 3), dtype=np.uint8)

    def test_resize(self):
        out = resize_image(self.raw)
        # OpenCV trả (height, width, channel)
        self.assertEqual(out.shape[:2], (IMG_SIZE[1], IMG_SIZE[0]))

    def test_grayscale_keeps_channel_dim(self):
        out = to_grayscale(self.raw)
        self.assertEqual(out.shape[-1], 1)

    def test_normalize_range(self):
        out = normalize_pixels(self.raw)
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)
        self.assertEqual(out.dtype, np.float32)

    def test_preprocess_image_shape(self):
        out = preprocess_image(self.raw)
        self.assertEqual(out.shape, INPUT_SHAPE)

    def test_preprocess_batch_shape(self):
        images = [self.raw for _ in range(4)]
        out = preprocess_batch(images)
        self.assertEqual(out.shape, (4, *INPUT_SHAPE))
        self.assertEqual(out.shape[-1], NUM_CHANNELS)

    def test_augment_keeps_size(self):
        out = augment_image(self.raw, seed=1)
        self.assertEqual(out.shape, self.raw.shape)


if __name__ == "__main__":
    unittest.main()
