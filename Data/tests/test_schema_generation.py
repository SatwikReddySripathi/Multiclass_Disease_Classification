import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
from schema_generation import (
    load_data_from_pickle,
    prepare_train_data,
    generate_train_stats,
    generate_serving_stats,
    infer_schema,
    save_schema,
    save_to_pickle,
    save_to_csv,
)

class TestSchemaGeneration(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data=b"pickle_data")
    @patch("pickle.load", return_value={"0": {"feature1": 1, "feature2": 2}})
    def test_load_data_from_pickle(self, mock_pickle_load, mock_file):
        """Test loading data from a pickle file."""
        df = load_data_from_pickle("dummy_path.pkl")
        self.assertIsInstance(df, pd.DataFrame, "Loaded object should be a DataFrame")
        self.assertEqual(len(df), 1, "DataFrame length does not match expected value")

    def test_prepare_train_data(self):
        """Test preparing training, evaluation, and serving datasets."""
        data = {
            "feature1": range(100),
            "feature2": range(100),
            "image_label": ["label"] * 100
        }
        df = pd.DataFrame(data)
        
        train_df, eval_df, serving_df = prepare_train_data(df)
        
        self.assertEqual(len(train_df), 60, "Training set size mismatch")
        self.assertEqual(len(eval_df), 20, "Evaluation set size mismatch")
        self.assertEqual(len(serving_df), 20, "Serving set size mismatch")
        self.assertNotIn("image_label", serving_df.columns, "Serving data should not contain 'image_label' column")

    @patch("schema_generation.tfdv.generate_statistics_from_dataframe")
    def test_generate_train_stats(self, mock_generate_statistics):
        """Test generating statistics from the training dataset."""
        mock_statistics = MagicMock()
        mock_generate_statistics.return_value = mock_statistics
        
        train_df = pd.DataFrame({"feature1": range(10)})
        train_stats = generate_train_stats(train_df)
        
        mock_generate_statistics.assert_called_once_with(train_df)
        self.assertEqual(train_stats, mock_statistics, "Generated train stats do not match expected mock object")

    @patch("schema_generation.tfdv.generate_statistics_from_dataframe")
    def test_generate_serving_stats(self, mock_generate_statistics):
        """Test generating statistics from the serving dataset."""
        mock_statistics = MagicMock()
        mock_generate_statistics.return_value = mock_statistics
        
        serving_df = pd.DataFrame({"feature1": range(10)})
        serving_stats = generate_serving_stats(serving_df)
        
        mock_generate_statistics.assert_called_once_with(serving_df)
        self.assertEqual(serving_stats, mock_statistics, "Generated serving stats do not match expected mock object")

    @patch("schema_generation.tfdv.infer_schema")
    def test_infer_schema(self, mock_infer_schema):
        """Test inferring schema from training data statistics."""
        mock_schema = MagicMock()
        mock_infer_schema.return_value = mock_schema
        
        train_stats = MagicMock()
        schema = infer_schema(train_stats)
        
        mock_infer_schema.assert_called_once_with(statistics=train_stats)
        self.assertEqual(schema, mock_schema, "Inferred schema does not match expected mock schema")

    @patch("schema_generation.tfdv.write_schema_text")
    @patch("os.makedirs")
    @patch("os.path.exists", return_value=False)
    def test_save_schema(self, mock_exists, mock_makedirs, mock_write_schema_text):
        """Test saving the schema to a file."""
        mock_schema = MagicMock()
        schema_file = save_schema(mock_schema, "dummy_dir", suffix="_test")
        
        mock_makedirs.assert_called_once_with("dummy_dir")
        mock_write_schema_text.assert_called_once_with(mock_schema, "dummy_dir/schema_test.pbtxt")
        self.assertEqual(schema_file, "dummy_dir/schema_test.pbtxt", "Schema file path does not match expected")

    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.dump")
    def test_save_to_pickle(self, mock_pickle_dump, mock_file):
        """Test saving an object to a pickle file."""
        obj = {"key": "value"}
        save_to_pickle(obj, "dummy_pickle.pkl")
        
        mock_file.assert_called_once_with("dummy_pickle.pkl", 'wb')
        mock_pickle_dump.assert_called_once_with(obj, mock_file())

    @patch("pandas.DataFrame.to_csv")
    def test_save_to_csv(self, mock_to_csv):
        """Test saving a DataFrame to a CSV file."""
        df = pd.DataFrame({"feature1": range(5)})
        save_to_csv(df, "dummy_file.csv")
        
        mock_to_csv.assert_called_once_with("dummy_file.csv", index=False)

if __name__ == '__main__':
    unittest.main()
