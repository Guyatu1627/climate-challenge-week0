"""
Climate Analyzer - Advanced Climate Analysis Module

This module provides sophisticated climate analysis capabilities using
vectorized pandas operations for the EthioClimate Analytics Engine.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy import stats
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


class ClimateAnalyzer:
    """
    Advanced climate analysis class with pandas-based workflows.
    
    Provides comprehensive climate trend analysis, correlation analysis,
    and statistical insights for policy decision support.
    """
    
    def __init__(self, climate_data: pd.DataFrame):
        """
        Initialize the Climate Analyzer.
        
        Args:
            climate_data (pd.DataFrame): Cleaned climate dataset
        """
        self.df = climate_data.copy()
        self.analysis_results = {}
        
    def calculate_trend_analysis(self, variable: str, 
                               by_country: bool = True) -> Dict:
        """
        Calculate climate trends using linear regression.
        
        Args:
            variable (str): Climate variable to analyze
            by_country (bool): Whether to analyze by country
            
        Returns:
            Dict: Trend analysis results
        """
        logger.info(f"Calculating trend analysis for {variable}")
        
        results = {}
        
        if by_country:
            for country in self.df['Country'].unique():
                country_data = self.df[self.df['Country'] == country].copy()
                
                # Prepare data for regression
                if 'Date' in country_data.columns:
                    country_data = country_data.sort_values('Date')
                    # Convert dates to numeric for regression
                    country_data['Date_Numeric'] = pd.to_datetime(country_data['Date']).astype(int) / 10**9
                    
                    # Remove NaN values
                    clean_data = country_data[['Date_Numeric', variable]].dropna()
                    
                    if len(clean_data) > 1:
                        # Linear regression
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            clean_data['Date_Numeric'], clean_data[variable]
                        )
                        
                        results[country] = {
                            'slope': slope,
                            'intercept': intercept,
                            'r_squared': r_value**2,
                            'p_value': p_value,
                            'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                            'significance': 'significant' if p_value < 0.05 else 'not_significant'
                        }
        
        self.analysis_results[f'{variable}_trends'] = results
        return results
    
    def calculate_seasonal_patterns(self, variable: str) -> Dict:
        """
        Analyze seasonal climate patterns using vectorized operations.
        
        Args:
            variable (str): Climate variable to analyze
            
        Returns:
            Dict: Seasonal pattern analysis
        """
        logger.info(f"Calculating seasonal patterns for {variable}")
        
        # Monthly averages by country
        monthly_patterns = self.df.groupby(['Country', 'Month'])[variable].mean().unstack()
        
        # Calculate seasonal statistics
        seasonal_stats = {}
        for country in monthly_patterns.index:
            country_data = monthly_patterns.loc[country]
            
            seasonal_stats[country] = {
                'monthly_means': country_data.to_dict(),
                'peak_month': country_data.idxmax(),
                'lowest_month': country_data.idxmin(),
                'annual_range': country_data.max() - country_data.min(),
                'coefficient_of_variation': country_data.std() / country_data.mean()
            }
        
        # Identify regional patterns
        regional_peak = monthly_patterns.mean().idxmax()
        regional_low = monthly_patterns.mean().idxmin()
        
        results = {
            'country_patterns': seasonal_stats,
            'regional_peak_month': regional_peak,
            'regional_low_month': regional_low,
            'monthly_regional_averages': monthly_patterns.mean().to_dict()
        }
        
        self.analysis_results[f'{variable}_seasonal'] = results
        return results
    
    def calculate_correlations(self, variables: List[str] = None) -> Dict:
        """
        Calculate correlation matrix for climate variables.
        
        Args:
            variables (List[str]): Variables to correlate. Uses default if None.
            
        Returns:
            Dict: Correlation analysis results
        """
        logger.info("Calculating climate variable correlations")
        
        if variables is None:
            variables = ['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'RH2M', 'WS2M']
        
        # Filter to available variables
        available_vars = [var for var in variables if var in self.df.columns]
        
        # Calculate overall correlation matrix
        correlation_matrix = self.df[available_vars].corr()
        
        # Calculate correlations by country
        country_correlations = {}
        for country in self.df['Country'].unique():
            country_data = self.df[self.df['Country'] == country]
            country_corr = country_data[available_vars].corr()
            country_correlations[country] = country_corr
        
        # Find strongest correlations
        strongest_correlations = []
        for i, var1 in enumerate(available_vars):
            for j, var2 in enumerate(available_vars):
                if i < j:  # Avoid duplicate pairs
                    corr_value = correlation_matrix.loc[var1, var2]
                    strongest_correlations.append({
                        'variables': (var1, var2),
                        'correlation': corr_value,
                        'strength': 'strong' if abs(corr_value) > 0.7 else 
                                   'moderate' if abs(corr_value) > 0.3 else 'weak'
                    })
        
        # Sort by absolute correlation value
        strongest_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        results = {
            'overall_correlation_matrix': correlation_matrix,
            'country_correlations': country_correlations,
            'strongest_correlations': strongest_correlations[:10]  # Top 10
        }
        
        self.analysis_results['correlations'] = results
        return results
    
    def detect_extreme_events(self, variable: str, 
                            threshold_method: str = 'percentile') -> Dict:
        """
        Detect extreme climate events using various threshold methods.
        
        Args:
            variable (str): Climate variable to analyze
            threshold_method (str): Method for threshold determination
            
        Returns:
            Dict: Extreme event analysis
        """
        logger.info(f"Detecting extreme events for {variable} using {threshold_method}")
        
        results = {}
        
        for country in self.df['Country'].unique():
            country_data = self.df[self.df['Country'] == country].copy()
            
            if threshold_method == 'percentile':
                # Use 95th and 5th percentiles
                upper_threshold = country_data[variable].quantile(0.95)
                lower_threshold = country_data[variable].quantile(0.05)
                
            elif threshold_method == 'std':
                # Use 2 standard deviations
                mean_val = country_data[variable].mean()
                std_val = country_data[variable].std()
                upper_threshold = mean_val + 2 * std_val
                lower_threshold = mean_val - 2 * std_val
            
            # Detect extreme events
            extreme_high = country_data[country_data[variable] > upper_threshold]
            extreme_low = country_data[country_data[variable] < lower_threshold]
            
            results[country] = {
                'upper_threshold': upper_threshold,
                'lower_threshold': lower_threshold,
                'extreme_high_events': len(extreme_high),
                'extreme_low_events': len(extreme_low),
                'extreme_high_percentage': (len(extreme_high) / len(country_data)) * 100,
                'extreme_low_percentage': (len(extreme_low) / len(country_data)) * 100,
                'extreme_high_dates': extreme_high['Date'].tolist() if 'Date' in extreme_high.columns else [],
                'extreme_low_dates': extreme_low['Date'].tolist() if 'Date' in extreme_low.columns else []
            }
        
        self.analysis_results[f'{variable}_extreme_events'] = results
        return results
    
    def calculate_climate_indices(self) -> Dict:
        """
        Calculate various climate indices for policy analysis.
        
        Returns:
            Dict: Climate indices calculation
        """
        logger.info("Calculating climate indices")
        
        results = {}
        
        for country in self.df['Country'].unique():
            country_data = self.df[self.df['Country'] == country].copy()
            
            country_indices = {}
            
            # Temperature indices
            if 'T2M' in country_data.columns:
                country_indices['mean_annual_temp'] = country_data.groupby('Year')['T2M'].mean().mean()
                country_indices['temp_annual_range'] = country_data.groupby('Year')['T2M'].max().mean() - \
                                                       country_data.groupby('Year')['T2M'].min().mean()
                
                # Growing degree days (base 10°C)
                country_indices['growing_degree_days'] = (country_data['T2M'] - 10).clip(lower=0).sum()
            
            # Rainfall indices
            if 'PRECTOTCORR' in country_data.columns:
                country_indices['mean_annual_rainfall'] = country_data.groupby('Year')['PRECTOTCORR'].sum().mean()
                country_indices['rainy_days'] = (country_data['PRECTOTCORR'] > 0.1).sum()
                
                # Dry spell analysis (consecutive days with <1mm rain)
                country_data['dry_day'] = country_data['PRECTOTCORR'] < 1.0
                country_data['dry_spell'] = country_data['dry_day'].groupby(
                    (country_data['dry_day'] != country_data['dry_day'].shift()).cumsum()
                ).cumcount() + 1
                
                max_dry_spell = country_data[country_data['dry_day']]['dry_spell'].max()
                country_indices['max_dry_spell'] = max_dry_spell if not pd.isna(max_dry_spell) else 0
            
            # Humidity indices
            if 'RH2M' in country_data.columns:
                country_indices['mean_relative_humidity'] = country_data['RH2M'].mean()
                country_indices['high_humidity_days'] = (country_data['RH2M'] > 80).sum()
            
            # Wind indices
            if 'WS2M' in country_data.columns:
                country_indices['mean_wind_speed'] = country_data['WS2M'].mean()
                country_indices['calm_days'] = (country_data['WS2M'] < 1.0).sum()
            
            results[country] = country_indices
        
        self.analysis_results['climate_indices'] = results
        return results
    
    def generate_policy_insights(self, variable: str = 'T2M') -> Dict:
        """
        Generate policy-relevant insights from climate analysis.
        
        Args:
            variable (str): Primary variable for policy insights
            
        Returns:
            Dict: Policy insights and recommendations
        """
        logger.info("Generating policy insights")
        
        insights = {
            'temperature_trends': {},
            'water_security': {},
            'agricultural_impacts': {},
            'extreme_events': {}
        }
        
        # Temperature trend insights
        if f'{variable}_trends' in self.analysis_results:
            trends = self.analysis_results[f'{variable}_trends']
            for country, trend_data in trends.items():
                if trend_data['significance'] == 'significant':
                    if trend_data['trend_direction'] == 'increasing':
                        insights['temperature_trends'][country] = {
                            'status': 'warming',
                            'confidence': 'high',
                            'policy_implication': 'Heat stress adaptation needed'
                        }
                    else:
                        insights['temperature_trends'][country] = {
                            'status': 'cooling',
                            'confidence': 'high',
                            'policy_implication': 'Cold stress monitoring required'
                        }
        
        # Water security insights
        if 'climate_indices' in self.analysis_results:
            indices = self.analysis_results['climate_indices']
            for country, country_indices in indices.items():
                if 'mean_annual_rainfall' in country_indices:
                    rainfall = country_indices['mean_annual_rainfall']
                    if rainfall < 500:  # Low rainfall threshold
                        insights['water_security'][country] = {
                            'status': 'water_stressed',
                            'recommendation': 'Invest in water harvesting and conservation'
                        }
                    elif 'max_dry_spell' in country_indices and country_indices['max_dry_spell'] > 30:
                        insights['water_security'][country] = {
                            'status': 'seasonal_drought_risk',
                            'recommendation': 'Develop drought early warning systems'
                        }
        
        # Agricultural impact insights
        if 'climate_indices' in self.analysis_results:
            indices = self.analysis_results['climate_indices']
            for country, country_indices in indices.items():
                agri_insights = {}
                
                if 'growing_degree_days' in country_indices:
                    gdd = country_indices['growing_degree_days']
                    if gdd > 3000:  # High GDD threshold
                        agri_insights['growing_season'] = 'extended'
                    elif gdd < 2000:  # Low GDD threshold
                        agri_insights['growing_season'] = 'limited'
                
                if 'temp_annual_range' in country_indices:
                    temp_range = country_indices['temp_annual_range']
                    if temp_range > 15:  # High temperature variability
                        agri_insights['climate_risk'] = 'high_temperature_variability'
                
                if agri_insights:
                    insights['agricultural_impacts'][country] = agri_insights
        
        # Extreme events insights
        if f'{variable}_extreme_events' in self.analysis_results:
            extreme_events = self.analysis_results[f'{variable}_extreme_events']
            for country, event_data in extreme_events.items():
                total_extreme = event_data['extreme_high_percentage'] + event_data['extreme_low_percentage']
                if total_extreme > 15:  # More than 15% extreme events
                    insights['extreme_events'][country] = {
                        'risk_level': 'high',
                        'recommendation': 'Develop comprehensive disaster risk reduction strategies'
                    }
        
        return insights
    
    def export_analysis_summary(self) -> Dict:
        """
        Export comprehensive analysis summary for reporting.
        
        Returns:
            Dict: Complete analysis summary
        """
        logger.info("Exporting analysis summary")
        
        summary = {
            'analysis_metadata': {
                'total_records': len(self.df),
                'countries_analyzed': self.df['Country'].unique().tolist(),
                'date_range': {
                    'start': self.df['Date'].min() if 'Date' in self.df.columns else None,
                    'end': self.df['Date'].max() if 'Date' in self.df.columns else None
                },
                'variables_analyzed': self.df.select_dtypes(include=[np.number]).columns.tolist()
            },
            'analysis_results': self.analysis_results,
            'policy_insights': self.generate_policy_insights()
        }
        
        return summary


# Utility function for quick analysis
def quick_climate_analysis(climate_data: pd.DataFrame, 
                         primary_variable: str = 'T2M') -> Dict:
    """
    Quick utility function for comprehensive climate analysis.
    
    Args:
        climate_data (pd.DataFrame): Cleaned climate dataset
        primary_variable (str): Primary variable for analysis
        
    Returns:
        Dict: Complete analysis results
    """
    analyzer = ClimateAnalyzer(climate_data)
    
    # Run all analyses
    analyzer.calculate_trend_analysis(primary_variable)
    analyzer.calculate_seasonal_patterns(primary_variable)
    analyzer.calculate_correlations()
    analyzer.detect_extreme_events(primary_variable)
    analyzer.calculate_climate_indices()
    
    return analyzer.export_analysis_summary()
