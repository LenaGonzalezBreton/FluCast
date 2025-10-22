"""
Unit tests for configuration in config.py
"""

import unittest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


class TestConfiguration(unittest.TestCase):
    """Test configuration values"""
    
    def test_paths_exist(self):
        """Test that essential path constants are defined"""
        self.assertIsNotNone(config.ROOT_DIR)
        self.assertIsNotNone(config.DATA_DIR)
        self.assertIsNotNone(config.CLEAN_DATA_DIR)
    
    def test_paths_are_path_objects(self):
        """Test that paths are Path objects"""
        self.assertIsInstance(config.ROOT_DIR, Path)
        self.assertIsInstance(config.DATA_DIR, Path)
        self.assertIsInstance(config.CLEAN_DATA_DIR, Path)
    
    def test_geographic_parameters(self):
        """Test geographic configuration"""
        # Check metro codes
        self.assertIsInstance(config.CODES_METRO, list)
        self.assertGreater(len(config.CODES_METRO), 90)
        self.assertIn('01', config.CODES_METRO)
        self.assertIn('75', config.CODES_METRO)
        self.assertIn('2A', config.CODES_METRO)
        self.assertIn('2B', config.CODES_METRO)
        self.assertNotIn('20', config.CODES_METRO)  # Old Corsica code
        
        # Check Grand Est codes
        self.assertIsInstance(config.CODES_GRAND_EST, list)
        self.assertEqual(len(config.CODES_GRAND_EST), 10)
        
        # Check department names
        self.assertIsInstance(config.DEPARTEMENTS_GRAND_EST, dict)
        self.assertEqual(len(config.DEPARTEMENTS_GRAND_EST), 10)
    
    def test_prophet_config(self):
        """Test Prophet model configuration"""
        self.assertIsInstance(config.PROPHET_CONFIG, dict)
        self.assertIn('seasonality_mode', config.PROPHET_CONFIG)
        self.assertIn('yearly_seasonality', config.PROPHET_CONFIG)
        
        # Check types
        self.assertIsInstance(config.PROPHET_CONFIG['yearly_seasonality'], bool)
    
    def test_scoring_parameters(self):
        """Test scoring configuration"""
        self.assertIsInstance(config.SCORE_WEIGHTS, dict)
        self.assertIn('cas_predits', config.SCORE_WEIGHTS)
        self.assertIn('vulnerabilite_vacc', config.SCORE_WEIGHTS)
        
        # Check weights sum to 1.0
        total_weight = sum(config.SCORE_WEIGHTS.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        
        # Check score bounds
        self.assertEqual(config.SCORE_MIN, 0.0)
        self.assertEqual(config.SCORE_MAX, 1.0)
    
    def test_map_config(self):
        """Test map visualization configuration"""
        self.assertIsInstance(config.MAP_CONFIG, dict)
        self.assertIn('center', config.MAP_CONFIG)
        self.assertIn('zoom', config.MAP_CONFIG)
        
        # Check center coordinates
        center = config.MAP_CONFIG['center']
        self.assertIn('lat', center)
        self.assertIn('lon', center)
    
    def test_expected_columns(self):
        """Test expected data columns"""
        self.assertIsInstance(config.EXPECTED_COLUMNS, list)
        self.assertGreater(len(config.EXPECTED_COLUMNS), 5)
        
        # Check essential columns
        essential = [
            'code_departement',
            'total_cas_semaine',
            'couv_vacc_grippe_an_passe'
        ]
        for col in essential:
            self.assertIn(col, config.EXPECTED_COLUMNS)
    
    def test_validation_parameters(self):
        """Test validation configuration"""
        self.assertGreater(config.MIN_DATA_POINTS, 0)
        self.assertGreater(config.MAX_MISSING_PCT, 0)
        self.assertLess(config.MAX_MISSING_PCT, 1)
        
        # Check vaccination coverage range
        self.assertIsInstance(config.VACC_COVERAGE_RANGE, tuple)
        self.assertEqual(len(config.VACC_COVERAGE_RANGE), 2)
        self.assertEqual(config.VACC_COVERAGE_RANGE[0], 0.0)
        self.assertEqual(config.VACC_COVERAGE_RANGE[1], 1.0)


if __name__ == '__main__':
    unittest.main()
