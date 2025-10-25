"""
Utility functions for the FluCast application.
Includes logging, data validation, and helper functions.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import sys
import os
from contextlib import contextmanager

import config


# === CONTEXT MANAGERS ===

@contextmanager
def suppress_stdout_stderr():
    """
    Context manager to suppress stdout and stderr.
    Useful for suppressing verbose library output.
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# === LOGGING SETUP ===

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Name of the logger (usually __name__)
        log_file: Optional path to log file. If None, logs to console only.
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.LOGS_DIR / log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# === DATA VALIDATION ===

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    name: str = "DataFrame"
) -> Tuple[bool, List[str]]:
    """
    Validate that a DataFrame contains required columns and is not empty.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Name of the DataFrame for error messages
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    if df.empty:
        errors.append(f"{name} is empty")
        return False, errors
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        errors.append(f"{name} is missing columns: {', '.join(missing_columns)}")
    
    return len(errors) == 0, errors


def validate_numeric_range(
    series: pd.Series,
    min_val: float,
    max_val: float,
    column_name: str
) -> Tuple[bool, str]:
    """
    Validate that numeric values are within expected range.
    
    Args:
        series: Pandas Series to validate
        min_val: Minimum acceptable value
        max_val: Maximum acceptable value
        column_name: Name of the column for error messages
    
    Returns:
        Tuple of (is_valid, error message)
    """
    numeric_series = pd.to_numeric(series, errors='coerce')
    
    if numeric_series.isna().all():
        return False, f"{column_name} contains no valid numeric values"
    
    out_of_range = (numeric_series < min_val) | (numeric_series > max_val)
    out_of_range = out_of_range & ~numeric_series.isna()
    
    if out_of_range.any():
        count = out_of_range.sum()
        return False, f"{column_name} has {count} values out of range [{min_val}, {max_val}]"
    
    return True, ""


def check_missing_data(
    df: pd.DataFrame,
    max_missing_pct: float = config.MAX_MISSING_PCT
) -> Dict[str, float]:
    """
    Check for missing data in DataFrame and return percentage per column.
    
    Args:
        df: DataFrame to check
        max_missing_pct: Maximum acceptable percentage of missing values
    
    Returns:
        Dictionary mapping column names to missing data percentage
    """
    missing_pct = {}
    for col in df.columns:
        pct = df[col].isna().sum() / len(df)
        if pct > 0:
            missing_pct[col] = round(pct * 100, 2)
    
    return missing_pct


def validate_department_code(code: str) -> bool:
    """
    Validate that a department code is valid for metropolitan France.
    
    Args:
        code: Department code to validate
    
    Returns:
        True if valid, False otherwise
    """
    return code in config.CODES_METRO


def standardize_department_code(code: str) -> str:
    """
    Standardize department code format (zero-padded, handle Corsica).
    
    Args:
        code: Raw department code
    
    Returns:
        Standardized department code
    """
    code = str(code).strip().upper()
    
    # Handle Corsica special cases
    if code in ['2A', '2B']:
        return code
    
    # Zero-pad numeric codes
    if code.isdigit():
        code = code.zfill(2)
    
    return code


# === DATA PROCESSING HELPERS ===

def safe_numeric_conversion(
    series: pd.Series,
    fill_value: float = 0.0
) -> pd.Series:
    """
    Safely convert a series to numeric, filling invalid values.
    
    Args:
        series: Series to convert
        fill_value: Value to use for invalid entries
    
    Returns:
        Numeric series with invalid values filled
    """
    return pd.to_numeric(series, errors='coerce').fillna(fill_value)


def calculate_percentage_change(
    current: pd.Series,
    previous: pd.Series
) -> pd.Series:
    """
    Calculate percentage change, handling division by zero.
    
    Args:
        current: Current values
        previous: Previous values
    
    Returns:
        Percentage change, with inf values replaced by 0
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        change = (current - previous) / previous
    
    # Replace inf and -inf with 0
    change = change.replace([np.inf, -np.inf], 0)
    change = change.fillna(0)
    
    return change


def normalize_to_range(
    series: pd.Series,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> pd.Series:
    """
    Normalize a series to a specified range using min-max normalization.
    
    Args:
        series: Series to normalize
        min_val: Minimum value of output range
        max_val: Maximum value of output range
    
    Returns:
        Normalized series
    """
    series_min = series.min()
    series_max = series.max()
    
    if series_max == series_min:
        return pd.Series([min_val] * len(series), index=series.index)
    
    normalized = (series - series_min) / (series_max - series_min)
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized


# === FILE HANDLING ===

def ensure_directory_exists(path: Path) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to directory
    """
    path.mkdir(parents=True, exist_ok=True)


def safe_read_csv(
    file_path: Path,
    separator: str = ',',
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> Optional[pd.DataFrame]:
    """
    Safely read a CSV file with error handling.
    
    Args:
        file_path: Path to CSV file
        separator: CSV separator character
        logger: Optional logger for error messages
        **kwargs: Additional arguments to pass to pd.read_csv
    
    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        df = pd.read_csv(file_path, sep=separator, **kwargs)
        if logger:
            logger.info(f"Successfully loaded {file_path}")
        return df
    except FileNotFoundError:
        if logger:
            logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        if logger:
            logger.error(f"Error reading {file_path}: {e}")
        return None


def safe_write_csv(
    df: pd.DataFrame,
    file_path: Path,
    separator: str = ';',
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> bool:
    """
    Safely write a DataFrame to CSV with error handling.
    
    Args:
        df: DataFrame to write
        file_path: Path to output CSV file
        separator: CSV separator character
        logger: Optional logger for messages
        **kwargs: Additional arguments to pass to df.to_csv
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        ensure_directory_exists(file_path.parent)
        
        df.to_csv(file_path, sep=separator, index=False, **kwargs)
        if logger:
            logger.info(f"Successfully saved to {file_path}")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Error writing to {file_path}: {e}")
        return False


# === FORMATTING HELPERS ===

def format_number(num: float, decimals: int = 0) -> str:
    """
    Format a number with thousands separator.
    
    Args:
        num: Number to format
        decimals: Number of decimal places
    
    Returns:
        Formatted string
    """
    return f"{num:,.{decimals}f}".replace(',', ' ')


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a value as percentage.
    
    Args:
        value: Value to format (0.0 to 1.0)
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"
