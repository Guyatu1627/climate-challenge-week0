import pandas as pd
import pytest
import os
import numpy as np
from datetime import datetime

class TestClimateDataValidation:
    """Test suite for climate data validation and processing"""
    
    @pytest.fixture
    def master_df(self):
        """Load master climate dataset for testing"""
        df = pd.read_csv('data/master_climate_data.csv')
        if 'Year' not in df.columns or 'Month' not in df.columns:
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
            elif 'YEAR' in df.columns and 'DOY' in df.columns:
                df['Date'] = pd.to_datetime(
                    df['YEAR'].astype(str) + '-' + df['DOY'].astype(str),
                    format='%Y-%j',
                    errors='coerce'
                )
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
        return df
    
    @pytest.fixture
    def ethiopia_df(self):
        """Load Ethiopia climate dataset for testing"""
        if os.path.exists('data/ethiopia_clean.csv'):
            return pd.read_csv('data/ethiopia_clean.csv')
        return pd.read_csv('data/ethiopia.csv')

    def test_master_data_exists(self, master_df):
        """Test that master dataset exists and has data"""
        assert len(master_df) > 0, "Master dataset is empty"
        assert isinstance(master_df, pd.DataFrame), "Master data is not a pandas DataFrame"

    def test_required_columns_present(self, master_df):
        """Test that all required columns are present"""
        required_columns = [
            'Country', 'T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 
            'RH2M', 'WS2M', 'Date', 'Year', 'Month'
        ]
        
        for col in required_columns:
            assert col in master_df.columns, f"Missing critical column: {col}"

    def test_country_coverage(self, master_df):
        """Test that expected countries are present"""
        expected_countries = ['Ethiopia', 'Kenya', 'Nigeria', 'Sudan', 'Tanzania']
        actual_countries = master_df['Country'].unique()
        
        for country in expected_countries:
            assert country in actual_countries, f"Missing country data: {country}"

    def test_temperature_ranges(self, master_df):
        """Test temperature values are physically reasonable for East Africa"""
        # East Africa temperatures should be reasonable
        assert master_df['T2M'].max() < 50, f"Max temperature too high: {master_df['T2M'].max()}°C"
        assert master_df['T2M'].min() > -5, f"Min temperature too low: {master_df['T2M'].min()}°C"
        
        # Max temperature should be >= mean temperature
        assert (master_df['T2M_MAX'] >= master_df['T2M']).all(), "T2M_MAX should be >= T2M"
        
        # Min temperature should be <= mean temperature  
        assert (master_df['T2M_MIN'] <= master_df['T2M']).all(), "T2M_MIN should be <= T2M"

    def test_rainfall_ranges(self, master_df):
        """Test rainfall values are physically reasonable"""
        # Rainfall should be non-negative
        assert (master_df['PRECTOTCORR'] >= 0).all(), "Rainfall should be non-negative"
        
        # Extreme rainfall events should be reasonable (not > 500mm/day)
        assert master_df['PRECTOTCORR'].max() < 500, f"Extreme rainfall: {master_df['PRECTOTCORR'].max()}mm/day"

    def test_humidity_ranges(self, master_df):
        """Test relative humidity values are valid"""
        assert (master_df['RH2M'] >= 0).all(), "Relative humidity should be non-negative"
        assert (master_df['RH2M'] <= 100).all(), "Relative humidity should not exceed 100%"

    def test_wind_speed_ranges(self, master_df):
        """Test wind speed values are reasonable"""
        assert (master_df['WS2M'] >= 0).all(), "Wind speed should be non-negative"
        assert master_df['WS2M'].max() < 50, f"Extreme wind speed: {master_df['WS2M'].max()}m/s"

    def test_date_consistency(self, master_df):
        """Test date fields are consistent and valid"""
        # Test Year is reasonable range
        assert master_df['Year'].min() >= 2015, "Data should start from 2015 or later"
        assert master_df['Year'].max() <= 2030, "Data should not extend beyond 2030"
        
        # Test Month is valid
        assert master_df['Month'].min() >= 1, "Month should start from 1"
        assert master_df['Month'].max() <= 12, "Month should not exceed 12"
        
        # Test Date format consistency
        date_sample = master_df['Date'].head(10)
        for date_val in date_sample:
            if isinstance(date_val, pd.Timestamp):
                date_str = date_val.strftime('%Y-%m-%d')
            else:
                date_str = str(date_val)
            try:
                # Try to parse date format
                datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                pytest.fail(f"Invalid date format: {date_str}")

    def test_data_completeness(self, master_df):
        """Test for missing values in critical columns"""
        critical_columns = ['T2M', 'PRECTOTCORR', 'RH2M', 'WS2M']
        
        for col in critical_columns:
            null_count = master_df[col].isnull().sum()
            total_count = len(master_df)
            null_percentage = (null_count / total_count) * 100
            
            assert null_percentage < 5, f"Too many null values in {col}: {null_percentage:.2f}%"

    def test_country_data_balance(self, master_df):
        """Test that countries have reasonable data amounts"""
        country_counts = master_df['Country'].value_counts()
        
        # Each country should have substantial data
        for country, count in country_counts.items():
            assert count > 1000, f"Insufficient data for {country}: {count} records"
        
        # Data distribution should be relatively balanced (within 50%)
        max_count = country_counts.max()
        min_count = country_counts.min()
        balance_ratio = min_count / max_count
        
        assert balance_ratio > 0.5, f"Unbalanced data distribution: ratio {balance_ratio:.2f}"

    def test_ethiopia_specific_data(self, ethiopia_df):
        """Test Ethiopia-specific data characteristics"""
        if len(ethiopia_df) > 0:
            # Ethiopia should have expected altitude-related temperature patterns
            assert ethiopia_df['T2M'].mean() < 30, "Ethiopia average temperature seems too high"
            assert ethiopia_df['T2M'].mean() > 10, "Ethiopia average temperature seems too low"

    def test_pandas_workflow_efficiency(self, master_df):
        """Test that we're using efficient pandas operations"""
        # Test vectorized operations work properly
        temp_mean = master_df['T2M'].mean()  # Should use vectorized operation
        rainfall_sum = master_df['PRECTOTCORR'].sum()  # Should use vectorized operation
        
        assert isinstance(temp_mean, (float, np.float64)), "Mean calculation should return numeric"
        assert isinstance(rainfall_sum, (float, np.float64)), "Sum calculation should return numeric"
        
        # Test groupby operations work
        country_means = master_df.groupby('Country')['T2M'].mean()
        assert len(country_means) == 5, "Groupby should return data for all 5 countries"

    def test_data_types(self, master_df):
        """Test that columns have appropriate data types"""
        expected_types = {
            'Country': 'object',
            'T2M': 'float64',
            'PRECTOTCORR': 'float64',
            'RH2M': 'float64',
            'WS2M': 'float64',
            'Year': 'int',
            'Month': 'int'
        }
        
        for col, expected_type in expected_types.items():
            if col in master_df.columns:
                actual_dtype = master_df[col].dtype
                if expected_type == 'object':
                    assert actual_dtype == object or str(actual_dtype) == 'object' or pd.api.types.is_string_dtype(actual_dtype), f"Column {col} has wrong type: {actual_dtype}"
                elif expected_type == 'float64':
                    assert np.issubdtype(actual_dtype, np.floating), f"Column {col} has wrong type: {actual_dtype}"
                elif expected_type == 'int':
                    assert np.issubdtype(actual_dtype, np.integer), f"Column {col} has wrong type: {actual_dtype}"
                else:
                    assert expected_type in str(actual_dtype), f"Column {col} has wrong type: {actual_dtype}"
