# anomaly_detector.py
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
import tensorflow as tf
from google.cloud import storage
import json

class AnomalyDetector:
    def __init__(self, logger, bucket_name: str):
        self.logger = logger
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        # Define thresholds
        self.thresholds = {
            'missing_labels_ratio': 0.05,
            'corrupt_images_ratio': 0.02,
            'min_image_size': 100 * 100,  # minimum 100x100 pixels
            'max_image_size': 5000 * 5000,  # maximum 1000x1000 pixels
            'class_imbalance_ratio': 0.1  # minimum 10% for any class
        }
        
    def check_data_completeness(self) -> Dict:
        """Check for missing data and labels"""
        try:
            self.logger.log_task_start("check_data_completeness")
            
            # Get all image files
            image_files = set(
                blob.name.split('/')[-1] 
                for blob in self.bucket.list_blobs(prefix='raw/xray/')
                if blob.name.endswith(('.png', '.jpg', '.jpeg'))
            )
            
            # Get all labeled files
            labels_blob = self.bucket.blob('raw/xray/labels.csv')
            labels_df = pd.read_csv(labels_blob.download_as_string())
            labeled_files = set(labels_df['image_id'].values)
            
            # Calculate metrics
            metrics = {
                'total_images': len(image_files),
                'total_labeled': len(labeled_files),
                'missing_labels': len(image_files - labeled_files),
                'extra_labels': len(labeled_files - image_files),
                'timestamp': datetime.now().isoformat()
            }
            
            # Check for anomalies
            missing_ratio = metrics['missing_labels'] / metrics['total_images']
            if missing_ratio > self.thresholds['missing_labels_ratio']:
                self.logger.log_error(
                    "data_completeness",
                    f"High ratio of missing labels: {missing_ratio:.2%}",
                    alert=True
                )
            
            return metrics
            
        except Exception as e:
            self.logger.log_error("check_data_completeness", e)
            raise
            
    def check_image_quality(self, sample_size: int = 100) -> Dict:
        """Check image quality and format"""
        try:
            self.logger.log_task_start("check_image_quality")
            
            # Sample images
            blobs = list(self.bucket.list_blobs(prefix='raw/xray/'))[:sample_size]
            
            metrics = {
                'corrupt_images': 0,
                'invalid_dimensions': 0,
                'invalid_format': 0,
                'samples_checked': len(blobs)
            }
            
            for blob in blobs:
                try:
                    # Try to decode image
                    image_data = blob.download_as_bytes()
                    image = tf.image.decode_image(image_data)
                    
                    # Check dimensions
                    image_size = image.shape[0] * image.shape[1]
                    if (image_size < self.thresholds['min_image_size'] or 
                        image_size > self.thresholds['max_image_size']):
                        metrics['invalid_dimensions'] += 1
                        
                except Exception:
                    metrics['corrupt_images'] += 1
                    
            # Calculate ratios
            metrics['corrupt_ratio'] = metrics['corrupt_images'] / metrics['samples_checked']
            metrics['invalid_dim_ratio'] = metrics['invalid_dimensions'] / metrics['samples_checked']
            
            # Check for anomalies
            if metrics['corrupt_ratio'] > self.thresholds['corrupt_images_ratio']:
                self.logger.log_error(
                    "image_quality",
                    f"High ratio of corrupt images: {metrics['corrupt_ratio']:.2%}",
                    alert=True
                )
                
            return metrics
            
        except Exception as e:
            self.logger.log_error("check_image_quality", e)
            raise
            
    def check_class_distribution(self) -> Dict:
        """Check for class imbalance"""
        try:
            self.logger.log_task_start("check_class_distribution")
            
            # Load labels
            labels_blob = self.bucket.blob('raw/xray/labels.csv')
            labels_df = pd.read_csv(labels_blob.download_as_string())
            
            # Calculate class distribution
            class_dist = labels_df['label'].value_counts()
            total_samples = len(labels_df)
            
            metrics = {
                'class_distribution': class_dist.to_dict(),
                'class_ratios': (class_dist / total_samples).to_dict()
            }
            
            # Check for class imbalance
            for class_name, ratio in metrics['class_ratios'].items():
                if ratio < self.thresholds['class_imbalance_ratio']:
                    self.logger.log_error(
                        "class_distribution",
                        f"Class {class_name} is underrepresented: {ratio:.2%}",
                        alert=True
                    )
                    
            return metrics
            
        except Exception as e:
            self.logger.log_error("check_class_distribution", e)
            raise
