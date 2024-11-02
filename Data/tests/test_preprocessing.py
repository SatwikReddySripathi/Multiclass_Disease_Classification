import os
import cv2
import unittest
from Data.scripts.preprocessing import preprocess_image, load_label_indices, augment_generator, apply_augmentation

class TestPreprocessing(unittest.TestCase):

  def test_preprocess_image(self):
    """
    Checks the output image:
    1) Is not None
    2) Has the right shape
    3) Is normalized  
    """
    sample= os.listdir('/content/drive/My Drive/MLOPs Project/sampled_data')[0]
    image_path = os.path.join('/content/drive/My Drive/MLOPs Project/sampled_data', sample)

    processed_image = preprocess_image(image_path)

    self.assertIsNotNone(processed_image)
    self.assertEqual(processed_image.shape, (224, 224))
    self.assertTrue((processed_image >= 0).all() and (processed_image <= 1).all(), "Image normalization failed")

    
  def test_load_label_indices(self):
    
    """Checking that the data read is a dictionary and has expected keys."""

    json_path = '/content/drive/My Drive/MLOPs Project/dummy_label_indices.json'
    label_indices = load_label_indices(json_path)
    self.assertIsInstance(label_indices, dict)
    self.assertIn('14', label_indices)
    self.assertEqual(label_indices['14'], ["00000003_000.png", "00000003_001.png", "00000003_002.png", "00000003_004.png", "00000003_005.png"])

  def test_augment_generator(self):
    generator = augment_generator()
    self.assertIsNotNone(generator)
    self.assertTrue(hasattr(generator, 'flow'))

  def test_apply_augmentation(self):
    """
    Checks if:
    1) if augmented_images is not empty
    2) if augmented_images has the same shape as the original input
    3) if augmented_images are still normalized to [0, 1]
    """
    sample= os.listdir('/content/drive/My Drive/MLOPs Project/sampled_data')[0]
    image_path = os.path.join('/content/drive/My Drive/MLOPs Project/sampled_data', sample)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))

    augmentation_gen = augment_generator()
    augmented_images = apply_augmentation(image, augmentation_gen)

    self.assertTrue(len(augmented_images) > 0, "Augmentation did not produce any images")

    for aug_image in augmented_images:
        self.assertEqual(aug_image.shape, image.shape, "Augmented image shape mismatch")
    
    for aug_image in augmented_images:
        self.assertTrue((aug_image >= 0).all() and (aug_image <= 1).all(), "Augmented image normalization failed")


if __name__ == '__main__':

  """
  In colab, using unittest.main() might not work as it tries to get the tests from root directory and it won't find that in colab.
  So for that, a test suite and test runner has to be created, then the test runner will run that suite. 
  """

  suite = unittest.TestLoader().loadTestsFromTestCase(TestPreprocessing)
  runner = unittest.TextTestRunner()
  runner.run(suite)

  #unittest.main() --> use this if running in local environment