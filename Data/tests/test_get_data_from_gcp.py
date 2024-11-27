import unittest
from unittest.mock import patch, MagicMock, mock_open
from get_data_from_gcp import (
    extract_md5_from_dvc,
    find_md5_hashes,
    get_file_contents_as_dict,
    create_final_json,
    download_and_compress_images
)

class TestGetDataFromGCP(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data="outs:\n  - md5: abc123")
    def test_extract_md5_from_dvc(self, mock_file):
        """Test extracting MD5 from a .dvc file."""
        md5_hash = extract_md5_from_dvc("dummy.dvc")
        self.assertEqual(md5_hash, "abc123", "Failed to extract correct MD5 hash from DVC file.")

    @patch("glob.glob", return_value=["file1.dvc", "file2.dvc"])
    @patch("get_data_from_gcp.extract_md5_from_dvc", side_effect=["md5hash1", "md5hash2"])
    def test_find_md5_hashes(self, mock_extract_md5, mock_glob):
        """Test finding MD5 hashes from multiple .dvc files."""
        md5_hashes = find_md5_hashes("dummy_project_dir")
        self.assertEqual(md5_hashes, ["md5hash1", "md5hash2"], "MD5 hashes list does not match expected values.")

    @patch("get_data_from_gcp.storage.Client")
    def test_get_file_contents_as_dict(self, mock_storage_client):
        """Test reading JSON and CSV content from a GCP bucket."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.download_as_text.side_effect = [
            '{"items": [{"md5": "abc123", "relpath": "file1"}]}',  # JSON content for .dir file
            "Image Index,Labels\nfile1,LabelA\n"  # CSV content
        ]
        mock_bucket.list_blobs.return_value = [mock_blob, mock_blob]
        mock_storage_client.return_value.get_bucket.return_value = mock_bucket

        json_content_dict, csv_data = get_file_contents_as_dict(mock_bucket, ["abc123"])
        
        # Check JSON and CSV data parsed correctly
        self.assertIn("abc123", json_content_dict)
        self.assertEqual(csv_data["file1"], "LabelA", "CSV data mismatch in labels.")

    def test_create_final_json(self):
        """Test creating a final JSON structure from JSON and CSV dictionaries."""
        json_content_dict = {"abc123": [{"md5": "abc123", "relpath": "file1"}]}
        csv_content_dict = {"file1": "LabelA"}
        
        final_data = create_final_json(json_content_dict, csv_content_dict)
        
        self.assertEqual(final_data[0]["md5"], "abc123", "MD5 hash mismatch in final JSON.")
        self.assertEqual(final_data[0]["image_label"], "LabelA", "Image label mismatch in final JSON.")

    @patch("get_data_from_gcp.storage.Client")
    @patch("get_data_from_gcp.Image.open")
    def test_download_and_compress_images(self, mock_open_image, mock_storage_client):
        """Test downloading and compressing images from GCP."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_blob.download_as_bytes.return_value = b"fake_image_data"

        # Simulate an image being opened and saved in JPEG format
        mock_image = MagicMock()
        mock_open_image.return_value = mock_image
        mock_image.convert.return_value = mock_image
        
        md5_image_data = [{"md5": "abc123", "image_index": "file1", "image_label": "LabelA"}]
        output_pickle_file = "compressed_images.pkl"

        with patch("builtins.open", mock_open()) as mocked_file:
            download_and_compress_images(mock_bucket, md5_image_data, output_pickle_file)
            mock_open_image.assert_called_once()
            mock_image.save.assert_called_once_with(mocked_file(), format="JPEG", quality=50)

        mocked_file.assert_called_once_with(output_pickle_file, 'wb')

if __name__ == '__main__':
    unittest.main()
