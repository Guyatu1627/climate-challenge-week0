"""
Climate Data Processor - Core Data Processing Module

This module handles loading, cleaning, and preprocessing of climate data
using efficient pandas workflows for the EthioClimate Analytics Engine.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClimateDataProcessor:
    """
    Professional climate data processing class using pandas-based workflows.
    
    This class provides efficient, vectorized operations for climate data
    analysis, avoiding inefficient loops and following pandas best practices.
    """
    
    def __init__(self, data_path: str = "data/"):
        """
        Initialize the Climate Data Processor.
        
        Args:
            data_path (str): Path to the data directory
        """
        self.data_path = Path(data_path)
        self.master_df = None
        self.processed_data = {}
        
    def load_climate_data(self, countries: List[str] = None) -> pd.DataFrame:
        """
        Load and combine climate data from multiple country files.
        
        Args:
            countries (List[str]): List of countries to load. Defaults to all available.
            
        Returns:
            pd.DataFrame: Combined climate dataset
        """
        if countries is None:
            countries = ['ethiopia', 'kenya', 'nigeria', 'sudan', 'tanzania']
        
        dataframes = []
        
        for country in countries:
            file_path = self.data_path / f"{country}.csv"
            if file_path.exists():
                logger.info(f"Loading data for {country.title()}")
                df = pd.read_csv(file_path)
                df['Country'] = country.title()
                dataframes.append(df)
            else:
                logger.warning(f"Data file not found: {file_path}")
        
        if not dataframes:
            raise ValueError("No climate data files found")
        
        # Combine all dataframes efficiently
        self.master_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Combined dataset: {len(self.master_df)} records")
        
        return self.master_df
    
    def clean_climate_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Clean climate data using vectorized pandas operations.
        
        Args:
            df (pd.DataFrame): DataFrame to clean. Uses master_df if None.
            
        Returns:
            pd.DataFrame: Cleaned climate dataset
        """
        if df is None:
            df = self.master_df.copy()
        
        logger.info("Starting data cleaning process")
        
        # Replace invalid values (-999) with NaN using vectorized operation
        numeric_columns = ['T2M', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 
                          'PRECTOTCORR', 'RH2M', 'WS2M', 'WS2M_MAX', 'PS', 'QV2M']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].replace(-999, np.nan)
        
        # Convert date columns efficiently
        if 'YEAR' in df.columns and 'DOY' in df.columns:
            df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + 
                                      df['DOY'].astype(str), format='%Y-%j', errors='coerce')
        
        # Extract year and month using vectorized operations
        if 'Date' in df.columns:
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
        
        # Remove rows with critical missing values
        critical_columns = ['T2M', 'PRECTOTCORR']
        available_columns = [col for col in critical_columns if col in df.columns]
        
        if available_columns:
            initial_count = len(df)
            df = df.dropna(subset=available_columns)
            final_count = len(df)
            logger.info(f"Removed {initial_count - final_count} rows with missing critical data")
        
        logger.info(f"Cleaned dataset: {len(df)} records")
        return df
    
    def calculate_derived_variables(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate derived climate variables using vectorized operations.
        
        Args:
            df (pd.DataFrame): DataFrame to process. Uses master_df if None.
            
        Returns:
            pd.DataFrame: DataFrame with additional derived variables
        """
        if df is None:
            df = self.master_df.copy()
        
        logger.info("Calculating derived variables")
        
        # Temperature range validation using vectorized operations
        if 'T2M_MAX' in df.columns and 'T2M_MIN' in df.columns:
            calculated_range = df['T2M_MAX'] - df['T2M_MIN']
            if 'T2M_RANGE' in df.columns:
                # Validate existing range calculation
                df['T2M_RANGE_Valid'] = np.isclose(calculated_range, df['T2M_RANGE'], rtol=0.01)
            else:
                df['T2M_RANGE'] = calculated_range
        
        # Heat stress indicators
        if 'T2M' in df.columns:
            df['Heat_Stress_Days'] = (df['T2M'] > 35).astype(int)
            df['Cold_Stress_Days'] = (df['T2M'] < 10).astype(int)
        
        # Rainfall categories
        if 'PRECTOTCORR' in df.columns:
            conditions = [
                df['PRECTOTCORR'] == 0,
                (df['PRECTOTCORR'] > 0) & (df['PRECTOTCORR'] <= 2.5),
                (df['PRECTOTCORR'] > 2.5) & (df['PRECTOTCORR'] <= 10),
                (df['PRECTOTCORR'] > 10) & (df['PRECTOTCORR'] <= 50),
                df['PRECTOTCORR'] > 50
            ]
            categories = ['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain', 'Extreme Rain']
            df['Rainfall_Category'] = np.select(conditions, categories, default='Unknown')
        
        logger.info("Derived variables calculated successfully")
        return df
    
    def aggregate_by_period(self, df: pd.DataFrame = None, 
                           period: str = 'monthly') -> pd.DataFrame:
        """
        Aggregate climate data by time period using efficient pandas operations.
        
        Args:
            df (pd.DataFrame): DataFrame to aggregate. Uses master_df if None.
            period (str): Aggregation period ('daily', 'monthly', 'yearly')
            
        Returns:
            pd.DataFrame: Aggregated climate data
        """
        if df is None:
            df = self.master_df.copy()
        
        logger.info(f"Aggregating data by {period}")
        
        # Define aggregation functions for different variables
        agg_functions = {
            'T2M': 'mean',
            'T2M_MAX': 'max',
            'T2M_MIN': 'min',
            'PRECTOTCORR': 'sum',
            'RH2M': 'mean',
            'WS2M': 'mean'
        }
        
        # Filter to available columns
        available_columns = {col: func for col, func in agg_functions.items() 
                          if col in df.columns}
        
        if period == 'monthly':
            grouped = df.groupby(['Country', 'Year', 'Month'])
        elif period == 'yearly':
            grouped = df.groupby(['Country', 'Year'])
        else:  # daily
            grouped = df.groupby(['Country', 'Date'])
        
        aggregated = grouped[available_columns.keys()].agg(available_columns)
        
        # Reset index for easier analysis
        aggregated = aggregated.reset_index()
        
        logger.info(f"Aggregated to {len(aggregated)} records")
        return aggregated
    
    def detect_outliers(self, df: pd.DataFrame = None, 
                       method: str = 'iqr') -> pd.DataFrame:
        """
        Detect outliers in climate data using statistical methods.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze. Uses master_df if None.
            method (str): Outlier detection method ('iqr', 'zscore')
            
        Returns:
            pd.DataFrame: DataFrame with outlier flags
        """
        if df is None:
            df = self.master_df.copy()
        
        logger.info(f"Detecting outliers using {method} method")
        
        numeric_columns = ['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'RH2M', 'WS2M']
        available_columns = [col for col in numeric_columns if col in df.columns]
        
        for col in available_columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[f'{col}_Outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df[f'{col}_Outlier'] = z_scores > 3
        
        outlier_summary = {}
        for col in available_columns:
            outlier_count = df[f'{col}_Outlier'].sum()
            outlier_percentage = (outlier_count / len(df)) * 100
            outlier_summary[col] = {
                'count': outlier_count,
                'percentage': outlier_percentage
            }
        
        logger.info(f"Outlier detection complete: {outlier_summary}")
        return df
    
    def get_data_summary(self, df: pd.DataFrame = None) -> Dict:
        """
        Generate comprehensive data summary using pandas describe and custom metrics.
        
        Args:
            df (pd.DataFrame): DataFrame to summarize. Uses master_df if None.
            
        Returns:
            Dict: Comprehensive data summary
        """
        if df is None:
            df = self.master_df.copy()
        
        logger.info("Generating data summary")
        
        summary = {
            'basic_info': {
                'total_records': len(df),
                'date_range': {
                    'start': df['Date'].min() if 'Date' in df.columns else None,
                    'end': df['Date'].max() if 'Date' in df.columns else None
                },
                'countries': df['Country'].unique().tolist() if 'Country' in df.columns else [],
                'columns': df.columns.tolist()
            },
            'climate_variables': {},
            'data_quality': {
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_records': df.duplicated().sum()
            }
        }
        
        # Climate variable statistics
        climate_vars = ['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'RH2M', 'WS2M']
        for var in climate_vars:
            if var in df.columns:
                summary['climate_variables'][var] = {
                    'mean': df[var].mean(),
                    'std': df[var].std(),
                    'min': df[var].min(),
                    'max': df[var].max(),
                    'median': df[var].median()
                }
        
        return summary


# Utility function for quick data loading
def load_master_climate_data(data_path: str = "data/") -> pd.DataFrame:
    """
    Quick utility function to load and clean master climate dataset.
    
    Args:
        data_path (str): Path to data directory
        
    Returns:
        pd.DataFrame: Cleaned master climate dataset
    """
    processor = ClimateDataProcessor(data_path)
    df = processor.load_climate_data()
    df_clean = processor.clean_climate_data(df)
    return df_clean
