import unittest
import numpy as np
import cv2
import os
import sys

# Ensure project root is in path to resolve `src.image_pipeline` imports correctly
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.image_pipeline import ImagePipeline

class TestImagePipeline(unittest.TestCase):
    
    def setUp(self):
        self.camera_id = 0
        self.pipeline = ImagePipeline(self.camera_id, base_dir=base_dir)
        
        # Create a mock grayscale raw image (e.g. 800x600)
        self.mock_frame = np.ones((600, 800, 3), dtype=np.uint8) * 128
        
        # Create a mock mark for alignment testing
        cv2.circle(self.mock_frame, (400, 300), 20, (0, 0, 0), -1)
        
    def test_pipeline_initialization(self):
        self.assertEqual(self.pipeline.camera_id, 0)
        self.assertIsNotNone(self.pipeline.clahe_obj)
        
    def test_process_frame_raw_fallback(self):
        """Test fallback to raw image when alignment is enabled but config is missing"""
        prep_config = {"enable_alignment": True, "enable_box_cropping": False}
        
        # We don't call load_configs, so preproc_config is None
        output_images = self.pipeline.process_frame(self.mock_frame, prep_config)
        
        self.assertEqual(len(output_images), 1)
        self.assertEqual(output_images[0][0], "raw_image")
        self.assertEqual(output_images[0][1].shape, self.mock_frame.shape)

    def test_process_frame_no_processing(self):
        """Test that turning off all flags simply returns the raw frame"""
        prep_config = {
            "enable_alignment": False, 
            "enable_box_cropping": False,
            "enable_shadow_removal": False,
            "enable_grayscale": False,
            "enable_pre_crop": False
        }
        
        output_images = self.pipeline.process_frame(self.mock_frame, prep_config)
        
        self.assertEqual(len(output_images), 1)
        self.assertEqual(output_images[0][0], "raw_image")
        self.assertTrue(np.array_equal(output_images[0][1], self.mock_frame))

if __name__ == '__main__':
    unittest.main()
