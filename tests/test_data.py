import pandas as pd
import pytest
import os

def test_column_names():
    # Check if the master file exists
    # (Note: You'll need to save the master_df to a CSV first to test it like this)
    df = pd.read_csv('data/master_climate_data.csv')

    # Professional check: Do we have the critical columns?
    required_columns = ['Country', 'T2M', 'PRECTOTCORR', 'Date']
    for col in required_columns:
        assert col in df.columns, f"Missing critical column: {col}"


def test_no_extreme_outliers():
    df = pd.read_csv("data/master_climate_data.csv")
    # Physics check: Temperature on Earth shouldn't be 100C 0r -100C in these regions
    assert df['T2M'].max() < 60, "Temperature to high! Check for data errors"
    assert df['T2M'].min() > -10, "Temperature too low! Check for data errors"