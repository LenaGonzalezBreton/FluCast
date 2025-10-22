"""
Unit tests for utility functions in utils.py
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import utils
import config


class TestDataValidation(unittest.TestCase):
    """Test data validation functions"""
    
    def setUp(self):
        """Set up test data"""
        self.df_valid = pd.DataFrame({
            'code_departement': ['01', '02', '03'],
            'total_cas_semaine': [100, 200, 150],
            'couv_vacc_grippe_an_passe': [0.6, 0.7, 0.65]
        })
        
        self.df_empty = pd.DataFrame()
        
        self.df_missing_cols = pd.DataFrame({
            'code_departement': ['01', '02'],
            'total_cas_semaine': [100, 200]
        })
    
    def test_validate_dataframe_valid(self):
        """Test validation with valid DataFrame"""
        required = ['code_departement', 'total_cas_semaine']
        is_valid, errors = utils.validate_dataframe(
            self.df_valid, 
            required
        )
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_dataframe_empty(self):
        """Test validation with empty DataFrame"""
        required = ['code_departement']
        is_valid, errors = utils.validate_dataframe(
            self.df_empty, 
            required
        )
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_validate_dataframe_missing_columns(self):
        """Test validation with missing columns"""
        required = ['code_departement', 'total_cas_semaine', 'missing_col']
        is_valid, errors = utils.validate_dataframe(
            self.df_missing_cols, 
            required
        )
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_validate_numeric_range_valid(self):
        """Test numeric range validation with valid data"""
        series = pd.Series([0.5, 0.6, 0.7])
        is_valid, error = utils.validate_numeric_range(
            series, 0.0, 1.0, 'test_col'
        )
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
    
    def test_validate_numeric_range_invalid(self):
        """Test numeric range validation with invalid data"""
        series = pd.Series([0.5, 1.5, 0.7])  # 1.5 is out of range
        is_valid, error = utils.validate_numeric_range(
            series, 0.0, 1.0, 'test_col'
        )
        self.assertFalse(is_valid)
        self.assertNotEqual(error, "")
    
    def test_check_missing_data(self):
        """Test missing data detection"""
        df = pd.DataFrame({
            'col1': [1, 2, None, 4],
            'col2': [1, 2, 3, 4]
        })
        missing = utils.check_missing_data(df)
        
        self.assertIn('col1', missing)
        self.assertEqual(missing['col1'], 25.0)  # 1 out of 4 = 25%
        self.assertNotIn('col2', missing)


class TestDepartmentCode(unittest.TestCase):
    """Test department code functions"""
    
    def test_validate_department_code_valid(self):
        """Test validation with valid codes"""
        self.assertTrue(utils.validate_department_code('01'))
        self.assertTrue(utils.validate_department_code('75'))
        self.assertTrue(utils.validate_department_code('2A'))
        self.assertTrue(utils.validate_department_code('2B'))
    
    def test_validate_department_code_invalid(self):
        """Test validation with invalid codes"""
        self.assertFalse(utils.validate_department_code('99'))
        self.assertFalse(utils.validate_department_code('00'))
        self.assertFalse(utils.validate_department_code('ABC'))
    
    def test_standardize_department_code(self):
        """Test code standardization"""
        self.assertEqual(utils.standardize_department_code('1'), '01')
        self.assertEqual(utils.standardize_department_code('75'), '75')
        self.assertEqual(utils.standardize_department_code('2a'), '2A')
        self.assertEqual(utils.standardize_department_code('2A'), '2A')


class TestDataProcessing(unittest.TestCase):
    """Test data processing helper functions"""
    
    def test_safe_numeric_conversion(self):
        """Test safe numeric conversion"""
        series = pd.Series(['1', '2', 'invalid', '4'])
        result = utils.safe_numeric_conversion(series, fill_value=0.0)
        
        self.assertEqual(result.iloc[0], 1.0)
        self.assertEqual(result.iloc[2], 0.0)  # Invalid converted to 0
        self.assertEqual(result.iloc[3], 4.0)
    
    def test_calculate_percentage_change(self):
        """Test percentage change calculation"""
        current = pd.Series([110, 90, 100])
        previous = pd.Series([100, 100, 0])
        
        result = utils.calculate_percentage_change(current, previous)
        
        self.assertAlmostEqual(result.iloc[0], 0.1)  # 10% increase
        self.assertAlmostEqual(result.iloc[1], -0.1)  # 10% decrease
        self.assertEqual(result.iloc[2], 0)  # Division by zero handled
    
    def test_normalize_to_range(self):
        """Test normalization"""
        series = pd.Series([0, 50, 100])
        result = utils.normalize_to_range(series, 0.0, 1.0)
        
        self.assertEqual(result.iloc[0], 0.0)
        self.assertEqual(result.iloc[1], 0.5)
        self.assertEqual(result.iloc[2], 1.0)
    
    def test_normalize_to_range_constant(self):
        """Test normalization with constant series"""
        series = pd.Series([5, 5, 5])
        result = utils.normalize_to_range(series, 0.0, 1.0)
        
        # All values should be min_val
        self.assertTrue((result == 0.0).all())


class TestFormatting(unittest.TestCase):
    """Test formatting helper functions"""
    
    def test_format_number(self):
        """Test number formatting"""
        self.assertEqual(utils.format_number(1000), "1 000")
        self.assertEqual(utils.format_number(1234567), "1 234 567")
        self.assertEqual(utils.format_number(1000.5, decimals=1), "1 000.5")
    
    def test_format_percentage(self):
        """Test percentage formatting"""
        self.assertEqual(utils.format_percentage(0.5), "50.0%")
        self.assertEqual(utils.format_percentage(0.123, decimals=2), "12.30%")
        self.assertEqual(utils.format_percentage(1.0), "100.0%")


if __name__ == '__main__':
    unittest.main()
