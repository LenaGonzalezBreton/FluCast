# FluCast - Developer Guide

This guide provides information for developers working on the FluCast flu forecasting application.

## Project Structure

```
FluCast/
├── app.py                  # Main application entry point
├── config.py              # Configuration parameters
├── utils.py               # Utility functions (logging, validation, helpers)
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore patterns
│
├── models/               # Machine learning models and scoring
│   ├── __init__.py
│   ├── app_national.py   # National-level forecasting model
│   └── app.py           # Additional model utilities
│
├── views/                # Streamlit UI views
│   ├── __init__.py
│   ├── main-national-view.py   # National dashboard view
│   ├── main-regional-view.py   # Regional dashboard view
│   └── main-dashboard-view.py  # Main dashboard
│
├── data/                 # Data processing and storage
│   ├── clean-data/      # Processed, clean datasets
│   ├── raw-data/        # Raw input data (not in git)
│   ├── processed/       # Intermediate processed data
│   ├── logs/            # Application logs
│   ├── creer_donnees_nationales.py   # National data preparation
│   ├── creer_donnees_finales.py      # Final data preparation
│   └── inspecter_csv.py             # CSV inspection utility
│
└── tests/               # Unit tests
    ├── test_config.py   # Configuration tests
    └── test_utils.py    # Utility function tests
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/LenaGonzalezBreton/FluCast.git
cd FluCast
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation by running tests:
```bash
python -m unittest discover tests -v
```

### Running the Application

To run the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

## Key Components

### Configuration (config.py)

The `config.py` file centralizes all configuration parameters:

- **Paths**: Data directories, file locations
- **Geographic Parameters**: Department codes, region mappings
- **Prophet Model Parameters**: Seasonality, priors, intervals
- **Scoring Parameters**: Weights for tension score calculation
- **Visualization Parameters**: Map settings, colors, zoom levels
- **Validation Parameters**: Data quality thresholds

To modify application behavior, update the relevant parameters in `config.py`.

### Utilities (utils.py)

The `utils.py` module provides reusable functions:

- **Logging**: Consistent logging setup across modules
- **Data Validation**: DataFrame validation, range checking, missing data detection
- **Data Processing**: Safe numeric conversion, percentage calculations, normalization
- **File Handling**: Safe CSV reading/writing with error handling
- **Formatting**: Number and percentage formatting helpers

### Models (models/app_national.py)

The national forecasting model includes:

- **charger_donnees()**: Load and validate data from CSV
- **entrainer_et_predire()**: Train Prophet models for each department
- **calculer_score()**: Calculate tension scores combining predictions and vaccination data
- **get_geojson()**: Download geographic boundaries for map visualization

#### Tension Score Calculation

The tension score combines two components:

1. **Predicted Cases Score** (60% weight): Normalized predicted case numbers
2. **Vaccination Vulnerability Score** (40% weight): Inverse of vaccination coverage

Formula: `Score = 0.6 × (normalized_cases) + 0.4 × (1 - vacc_coverage)`

Weights can be adjusted in `config.SCORE_WEIGHTS`.

## Data Processing

### Data Preparation Scripts

1. **creer_donnees_nationales.py**: Prepares national-level data
   - Merges vaccination and emergency data
   - Adds demographic information (currently simulated - see TODO)
   - Outputs to `clean-data/donnees_analytiques_france.csv`

2. **creer_donnees_finales.py**: Prepares regional data (Grand Est)
   - Similar processing as national script
   - Region-specific filtering
   - Outputs to `clean-data/donnees_analytiques_grand_est.csv`

### Expected Data Format

The application expects CSV files with the following columns:

- `code_departement`: Department code (e.g., '01', '75', '2A')
- `nom_departement`: Department name
- `annee_semaine`: Week in format 'YYYY-SXX' (e.g., '2024-S42')
- `population_totale`: Total population
- `population_plus_65_ans`: Population over 65 years
- `pct_plus_65_ans`: Percentage over 65
- `densite_population`: Population density
- `couv_vacc_grippe_an_passe`: Previous year vaccination coverage (0-1)
- `cas_urgences_semaine`: Emergency cases for the week
- `cas_sos_medecins_semaine`: SOS Médecins cases for the week
- `total_cas_semaine`: Total cases (sum of above)
- `tendance_evolution_cas`: Week-over-week trend

## Testing

### Running Tests

Run all tests:
```bash
python -m unittest discover tests -v
```

Run specific test file:
```bash
python -m unittest tests.test_utils -v
python -m unittest tests.test_config -v
```

### Writing Tests

Tests are located in the `tests/` directory. Follow these conventions:

- Test files should be named `test_*.py`
- Test classes should inherit from `unittest.TestCase`
- Test methods should start with `test_`
- Use descriptive test names that explain what is being tested

Example:
```python
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import utils

class TestMyFeature(unittest.TestCase):
    def test_something_works(self):
        result = utils.some_function()
        self.assertEqual(result, expected_value)
```

## Prophet Model

The application uses Facebook Prophet for time series forecasting:

### Model Configuration

Prophet parameters are configured in `config.PROPHET_CONFIG`:

- `seasonality_mode`: 'additive' - how seasonality combines with trend
- `yearly_seasonality`: True - model annual flu season patterns
- `weekly_seasonality`: False - disabled (not relevant for weekly data)
- `changepoint_prior_scale`: 0.05 - controls trend flexibility
- `seasonality_prior_scale`: 10.0 - controls seasonality flexibility

### Training Process

For each department:

1. Historical data is converted to Prophet format (ds, y)
2. Model is trained on all available historical data
3. Future periods are predicted (1 week ahead by default)
4. Predictions are validated (non-negative values)

### Fallback Mechanism

If Prophet training fails (insufficient data, optimization errors):
- The last observed value is used as prediction
- A warning is logged
- Processing continues for other departments

## Logging

### Log Levels

The application uses Python's logging module with the following levels:

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages (default)
- **WARNING**: Warning messages for recoverable issues
- **ERROR**: Error messages for serious problems

### Configuring Logging

Change log level in `config.py`:
```python
LOG_LEVEL = 'DEBUG'  # or 'INFO', 'WARNING', 'ERROR'
```

### Viewing Logs

Logs are written to:
- Console (always)
- Log files in `data/logs/` (if configured)

## Common Development Tasks

### Adding a New Configuration Parameter

1. Add the parameter to `config.py` with documentation
2. Update `tests/test_config.py` to test the new parameter
3. Use the parameter in your code: `config.NEW_PARAMETER`

### Adding a New Utility Function

1. Add the function to `utils.py` with docstring
2. Add unit tests to `tests/test_utils.py`
3. Run tests to verify: `python -m unittest tests.test_utils`

### Modifying the Scoring Algorithm

1. Update weights in `config.SCORE_WEIGHTS`
2. If changing the formula, update `calculer_score()` in `models/app_national.py`
3. Update documentation in this README
4. Test with different data scenarios

### Adding a New Data Source

1. Create a processing function in the appropriate data script
2. Add validation using `utils.validate_dataframe()`
3. Update `config.EXPECTED_COLUMNS` if adding new columns
4. Document the new data source and format

## Code Style Guidelines

### Python Style

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Use type hints where appropriate

Example function with good style:
```python
def calculate_score(
    predicted_cases: pd.Series,
    vaccination_coverage: pd.Series
) -> pd.Series:
    """
    Calculate tension score from predictions and vaccination data.
    
    Args:
        predicted_cases: Predicted case numbers
        vaccination_coverage: Vaccination coverage rates (0-1)
    
    Returns:
        Tension scores normalized to [0, 1]
    """
    # Implementation...
```

### Error Handling

- Use try/except blocks for operations that might fail
- Log errors with appropriate severity
- Provide user-friendly error messages in the UI
- Don't expose internal errors to users

Example:
```python
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    logger.error(f"File not found: {file_path}")
    st.error("Data file not found. Please check the data directory.")
    return pd.DataFrame()
```

## Troubleshooting

### Prophet Installation Issues

If Prophet installation fails:
```bash
# Install dependencies first
pip install numpy cython
pip install pystan
pip install prophet
```

### Data Loading Errors

Check:
1. File exists in expected location
2. CSV separator matches (`;` for clean data)
3. All required columns are present
4. Department codes are properly formatted

### Model Training Failures

If models fail to train:
1. Check for sufficient historical data (min 3 points)
2. Verify data has no NaN or infinite values
3. Check log files for detailed error messages
4. Consider adjusting Prophet parameters

## Contributing

1. Create a feature branch from `dev`
2. Make your changes with appropriate tests
3. Run all tests to ensure nothing breaks
4. Update documentation if needed
5. Submit a pull request with clear description

## Future Improvements

Areas for enhancement:

- [ ] Replace simulated INSEE data with real data (see TODO in data scripts)
- [ ] Add more sophisticated feature engineering
- [ ] Implement ensemble models combining multiple approaches
- [ ] Add interactive parameter tuning in the UI
- [ ] Expand to other regions beyond Grand Est
- [ ] Add more validation and data quality checks
- [ ] Implement automated data refresh pipeline
- [ ] Add export functionality for predictions

## License

[Add license information here]

## Contact

For questions or support, contact the development team.
