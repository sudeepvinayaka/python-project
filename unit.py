import unittest
import pandas as pd
import numpy as np
from main import DataProcessorBase  # Replace 'main' with your actual module file name

class MockDataProcessor(DataProcessorBase):
    def __init__(self):
        pass  # Override to skip DB loading

class TestDataProcessorBase(unittest.TestCase):

    def setUp(self):
        # Sample data for preprocessing and error calculations
        self.sample_train_data = pd.DataFrame({
            'A': [2, 4, 6],
            'B': [1, 3, 5],
            'C': [0, -1, -2]
        })

        self.sample_ideal_data = pd.DataFrame({
            'X': [2, 5, 8],
            'Y': [1, 4,7],
            'Z': [0, 3, 6]
        })

    def test_preprocess_data(self):
        processor = MockDataProcessor()
        processed_data = processor.preprocess_data(self.sample_train_data)
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertFalse(processed_data.isnull().values.any())  # Ensure no NaNs

    def test_calculate_squared_errors(self):
        processor = MockDataProcessor()
        errors = processor.calculate_squared_errors([1, 2, 3], [4, 5, 6])
        self.assertEqual(errors, 27)

    def test_find_best_fit_column(self):
        processor = MockDataProcessor()
        normalized_train = processor.preprocess_data(self.sample_train_data)
        normalized_ideal = processor.preprocess_data(self.sample_ideal_data)

        best_fit_indices = processor.find_best_fit_column(normalized_train, normalized_ideal)
        self.assertIsInstance(best_fit_indices, np.ndarray)
        self.assertEqual(len(best_fit_indices), 2)  # 2 train cols compared to 2+ ideal cols

if __name__ == '__main__':
    unittest.main()