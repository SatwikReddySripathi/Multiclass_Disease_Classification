import unittest
import os
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import io
from anomaly_detection import (
    load_dataframe,
    load_schema,
    prepare_data_splits,
    generate_statistics,
    detect_anomalies,
    get_image_dimensions,
    is_grayscale,
    check_image_data,
    check_missing_or_invalid_labels,
    check_class_distribution,
    check_image_dimensions,
    anomalies_detect
)

class TestAnomalyDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up a sample dataframe and schema for testing."""
        # Function to generate a simple dummy image
        def generate_dummy_image(color='red'):
            img = Image.new('RGB', (100, 100), color=color)  # Create a 100x100 red image
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            return img_byte_arr.getvalue()

        # Sample data for testing with valid image byte data
        cls.test_data = {
            'image_data': [generate_dummy_image(), generate_dummy_image(color='green')],
            'image_label': ['[0]', '[1]']  # Labels in the expected format
        }
        cls.df = pd.DataFrame(cls.test_data)


    @classmethod

    def test_prepare_data_splits(self):
        train_df, eval_df, serving_df = prepare_data_splits(self.df)
        self.assertEqual(train_df.shape[0], 1)
        self.assertEqual(eval_df.shape[0], 0)
        self.assertEqual(serving_df.shape[0], 1)

    def test_get_image_dimensions(self):
        # Use the dummy image generated in setUpClass
        img_data = self.test_data['image_data'][0]  # Get the first image data
        dimensions = get_image_dimensions(img_data)
        self.assertEqual(dimensions, (100, 100))  # Check the expected dimensions

    def test_is_grayscale(self):
        # Create a grayscale image for testing
        img_gray = Image.new('L', (100, 100))  # Grayscale image
        img_gray_byte_arr = io.BytesIO()
        img_gray.save(img_gray_byte_arr, format='JPEG')
        img_gray_data = img_gray_byte_arr.getvalue()

        self.assertTrue(is_grayscale(img_gray_data))

        # Create a non-grayscale image
        img_color = Image.new('RGB', (100, 100))
        img_color_byte_arr = io.BytesIO()
        img_color.save(img_color_byte_arr, format='JPEG')
        img_color_data = img_color_byte_arr.getvalue()

        self.assertFalse(is_grayscale(img_color_data))

    def test_check_image_data(self):
        # This should pass as the images are valid
        try:
            check_image_data(self.df)
        except ValueError:
            self.fail("check_image_data raised ValueError unexpectedly!")

    def test_check_missing_or_invalid_labels(self):
        try:
            check_missing_or_invalid_labels(self.df)
        except ValueError:
            self.fail("check_missing_or_invalid_labels raised ValueError unexpectedly!")

    def test_check_class_distribution(self):
        try:
            check_class_distribution(self.df)
        except Exception as e:
            self.fail(f"check_class_distribution raised an exception: {e}")

    def test_check_image_dimensions(self):
        try:
            check_image_dimensions(self.df)
        except Exception as e:
            self.fail(f"check_image_dimensions raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
