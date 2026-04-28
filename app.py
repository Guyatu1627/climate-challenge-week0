import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
from src.processor import ClimateDataProcessor

# 1. Set the Page Title
st.set_page_config(page_title="EthioClimate COP32 Portal", layout="wide", initial_sidebar_state="expanded")

# 2. Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
    }
    .insight-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 3. Header Section
st.markdown("""
<div class="main-header">
    <h1>🌍 EthioClimate Analytics: COP32 Evidence Portal</h1>
    <p style="font-size: 1.1rem; margin-top: 1rem;">
        **Negotiation-grade climate evidence** for COP32 summit preparations
    </p>
    <p style="opacity: 0.9;">Real-time climate analytics across Ethiopia, Kenya, Nigeria, Sudan, and Tanzania</p>
</div>
""", unsafe_allow_html=True)

# 4. Executive Summary
st.markdown("### 📊 Executive Summary for Climate Negotiators")
col1, col2, col3, col4 = st.columns(4)

# 5. Load the Master Data
processor = ClimateDataProcessor("data/master_climate_data.csv")
df = processor.load_and_clean()

# Country-level KPIs from the climate engine
selected_country = st.sidebar.selectbox(
    "🌍 Select Country for Engine KPIs:",
    options=df['Country'].unique(),
    index=0,
    help="Select a country to view climate KPIs powered by the ClimateDataProcessor engine"
)
metrics = processor.get_country_metrics(selected_country)
trend_label, trend_diff = processor.predict_next_season_trend(selected_country)

st.markdown("### 🔧 Climate Engine Country KPIs")
col1, col2, col3 = st.columns(3)
col1.metric("Average Temperature", f"{metrics['avg_temp']:.2f} °C")
col2.metric("Total Rainfall", f"{metrics['total_rainfall']:.1f} mm")
col3.metric("Rainy Days", metrics['rain_days'])

st.markdown(f"**Trend Outlook for {selected_country}:** {trend_label} ({trend_diff:+.2f}°C difference)")

# Calculate key metrics for executive summary
latest_year = df['Year'].max()
avg_temp = df['T2M'].mean()
avg_rainfall = df['PRECTOTCORR'].mean()
total_countries = df['Country'].nunique()

with col1:
    st.metric("📈 Data Coverage", f"{latest_year}", "Latest Year")
with col2:
    st.metric("🌡️ Avg Temperature", f"{avg_temp:.1f}°C", "Regional Mean")
with col3:
    st.metric("💧 Avg Rainfall", f"{avg_rainfall:.1f}mm", "Daily Mean")
with col4:
    st.metric("🌍 Countries", f"{total_countries}", "Monitored")

# 6. Strategic Insights
st.markdown("### 🎯 Key Climate Insights for Policy Makers")
with st.expander("📖 Read Strategic Briefing", expanded=True):
    st.markdown("""
    <div class="insight-box">
        <strong>🔥 Critical Finding:</strong> Regional temperatures show consistent warming trends with
        seasonal variations that impact agricultural planning across all monitored countries.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Policy Implications:**
    - **Agriculture:** Temperature increases require climate-resilient crop varieties
    - **Water Resources:** Rainfall patterns show significant seasonal variation
    - **Regional Cooperation:** Shared climate challenges need coordinated responses
    """)

# 7. Interactive Control Panel
st.sidebar.markdown("### 🎛️ Analysis Controls")

# Country Selection
country_choice = st.sidebar.multiselect(
    "🌍 Select Countries for Comparison:",
    options=df['Country'].unique(),
    default=['Ethiopia', 'Kenya'],
    help="Choose countries to compare climate patterns"
)

# Time Period Selection
min_year = int(df['Year'].min())
max_year = int(df['Year'].max())
year_range = st.sidebar.slider(
    "📅 Select Time Period:",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    help="Filter data by specific time period"
)

# Climate Variable Selection
variable_options = {
    'Temperature': 'T2M',
    'Maximum Temperature': 'T2M_MAX',
    'Minimum Temperature': 'T2M_MIN',
    'Rainfall': 'PRECTOTCORR',
    'Relative Humidity': 'RH2M',
    'Wind Speed': 'WS2M'
}

variable_display = st.sidebar.selectbox(
    "📊 Choose Climate Variable:",
    options=list(variable_options.keys()),
    index=0,
    help="Select which climate variable to visualize"
)

variable_choice = variable_options[variable_display]

# Analysis Type
analysis_type = st.sidebar.radio(
    "🔍 Analysis Type:",
    ["Time Series", "Country Comparison", "Seasonal Patterns", "Correlation Analysis"],
    help="Choose the type of analysis to perform"
)

# Apply filters
filtered_df = df[
    (df['Country'].isin(country_choice)) &
    (df['Year'].between(year_range[0], year_range[1]))
]

# 8. Dynamic Visualization Area
if not country_choice:
    st.warning("⚠️ Please select at least one country to begin analysis")
else:
    # Time Series Analysis
    if analysis_type == "Time Series":
        st.markdown("### 📈 Climate Trends Over Time")
        
        fig = px.line(
            filtered_df, 
            x="Date", 
            y=variable_choice, 
            color="Country",
            title=f"{variable_display} Trends ({year_range[0]}-{year_range[1]})",
            labels={variable_choice: variable_display, "Date": "Date"}
        )
        fig.update_layout(height=500, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # Second Chart: Rainfall Analysis
        st.markdown("### 💧 Rainfall Patterns Analysis")
        
        # Create rainfall bar chart
        rainfall_data = filtered_df.groupby(['Country', 'Month'])['PRECTOTCORR'].mean().reset_index()
        
        fig_rainfall = px.bar(
            rainfall_data,
            x="Month",
            y="PRECTOTCORR",
            color="Country",
            title=f"Monthly Average Rainfall by Country ({year_range[0]}-{year_range[1]})",
            labels={"Month": "Month (1=Jan, 12=Dec)", "PRECTOTCORR": "Rainfall (mm/day)"},
            barmode="group"
        )
        fig_rainfall.update_layout(height=400, hovermode="x unified")
        st.plotly_chart(fig_rainfall, use_container_width=True)
        
        # Rainfall Insights
        st.markdown("#### 🌧️ Rainfall Insights")
        peak_rainfall = rainfall_data.loc[rainfall_data['PRECTOTCORR'].idxmax()]
        total_rainfall_by_country = rainfall_data.groupby('Country')['PRECTOTCORR'].sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                f"🌊 Peak Rainfall", 
                f"{peak_rainfall['PRECTOTCORR']:.2f}mm/day",
                f"{peak_rainfall['Country']} - Month {peak_rainfall['Month']}"
            )
        
        with col2:
            highest_total_country = total_rainfall_by_country.idxmax()
            highest_total = total_rainfall_by_country.max()
            st.metric(
                "📊 Highest Annual Rainfall", 
                f"{highest_total:.1f}mm/year",
                highest_total_country
            )
        
        # Trend Analysis
        st.markdown("#### 📊 Trend Analysis")
        trend_data = filtered_df.groupby(['Country', 'Year'])[variable_choice].mean().reset_index()
        
        for country in country_choice:
            country_data = trend_data[trend_data['Country'] == country]
            if len(country_data) > 1:
                correlation = np.corrcoef(country_data['Year'], country_data[variable_choice])[0, 1]
                trend_direction = "📈 Increasing" if correlation > 0.1 else "📉 Decreasing" if correlation < -0.1 else "➡️ Stable"
                st.write(f"**{country}**: {trend_direction} trend (correlation: {correlation:.3f})")
    
    # Country Comparison
    elif analysis_type == "Country Comparison":
        st.markdown("### 🌍 Country Comparison Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box Plot
            fig_box = px.box(
                filtered_df, 
                x="Country", 
                y=variable_choice,
                title=f"{variable_display} Distribution by Country",
                color="Country"
            )
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # Bar Chart of Averages
            avg_by_country = filtered_df.groupby('Country')[variable_choice].mean().reset_index()
            fig_bar = px.bar(
                avg_by_country,
                x="Country",
                y=variable_choice,
                title=f"Average {variable_display} by Country",
                color="Country"
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Statistical Summary Table
        st.markdown("#### 📋 Statistical Summary")
        stats_table = filtered_df.groupby('Country')[variable_choice].agg([
            'mean', 'std', 'min', 'max'
        ]).round(2)
        stats_table.columns = ['Mean', 'Std Dev', 'Minimum', 'Maximum']
        st.dataframe(stats_table, use_container_width=True)
    
    # Seasonal Patterns
    elif analysis_type == "Seasonal Patterns":
        st.markdown("### 🌺 Seasonal Climate Patterns")
        
        # Monthly Averages
        monthly_data = filtered_df.groupby(['Country', 'Month'])[variable_choice].mean().reset_index()
        
        fig_seasonal = px.line(
            monthly_data,
            x="Month",
            y=variable_choice,
            color="Country",
            title=f"Seasonal {variable_display} Patterns",
            labels={"Month": "Month (1=Jan, 12=Dec)", variable_choice: variable_display}
        )
        fig_seasonal.update_layout(height=500)
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # Seasonal Insights
        st.markdown("#### 🎯 Seasonal Insights")
        peak_month = monthly_data.loc[monthly_data[variable_choice].idxmax()]
        low_month = monthly_data.loc[monthly_data[variable_choice].idxmin()]
        
        st.write(f"**Peak {variable_display}**: {peak_month['Country']} in month {peak_month['Month']}")
        st.write(f"**Lowest {variable_display}**: {low_month['Country']} in month {low_month['Month']}")
    
    # Correlation Analysis
    elif analysis_type == "Correlation Analysis":
        st.markdown("### 🔗 Climate Variable Correlations")
        
        # Select second variable for correlation
        second_var = st.selectbox(
            "Select second variable for correlation:",
            options=['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'RH2M', 'WS2M'],
            index=2,
            key="second_var"
        )
        
        # Scatter Plot
        fig_scatter = px.scatter(
            filtered_df,
            x=variable_choice,
            y=second_var,
            color="Country",
            title=f"{variable_display} vs {second_var.replace('_', ' ').title()}",
            hover_data=['Date'],
            opacity=0.7
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Correlation Matrix
        st.markdown("#### 📊 Correlation Coefficients")
        correlation_data = []
        for country in country_choice:
            country_data = filtered_df[filtered_df['Country'] == country]
            if len(country_data) > 1:
                corr = np.corrcoef(country_data[variable_choice], country_data[second_var])[0, 1]
                correlation_data.append({
                    'Country': country,
                    'Correlation': corr,
                    'Strength': 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
                })
        
        corr_df = pd.DataFrame(correlation_data)
        st.dataframe(corr_df, use_container_width=True)

# 9. Policy Recommendations Section
st.markdown("---")
st.markdown("### 🏛️ Policy Recommendations")

recommendations = {
    'Temperature': [
        "Invest in heat-resistant crop varieties for agricultural resilience",
        "Develop early warning systems for extreme heat events",
        "Create urban cooling strategies and green infrastructure"
    ],
    'Rainfall': [
        "Implement water harvesting and storage systems",
        "Develop drought-resistant agricultural practices",
        "Create flood management infrastructure for high-rainfall areas"
    ],
    'Humidity': [
        "Monitor health impacts of humidity changes",
        "Adjust building designs for humidity management",
        "Develop humidity-sensitive agricultural planning"
    ]
}

var_category = 'Temperature' if 'T2M' in variable_choice else 'Rainfall' if 'PREC' in variable_choice else 'Humidity'

if var_category in recommendations:
    st.markdown(f"#### 📋 Recommendations for {var_category} Management")
    for i, rec in enumerate(recommendations[var_category], 1):
        st.write(f"{i}. {rec}")

# 10. Export Functionality
st.markdown("---")
st.markdown("### 📤 Export Analysis Results")

col1, col2 = st.columns(2)
with col1:
    if st.button("📊 Download Summary Data"):
        summary_data = filtered_df.groupby(['Country', 'Year'])[variable_choice].mean().reset_index()
        csv = summary_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"climate_summary_{variable_choice}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📋 Generate Report"):
        st.info("📄 Report generation feature coming soon! This will create a PDF summary for ministerial briefings.")

# 11. Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🌍 <strong>EthioClimate COP32 Portal</strong> | Evidence-Based Climate Analytics</p>
    <p style='font-size: 0.9em;'>Data Sources: NASA POWER | Updated: {}</p>
</div>
""".format(datetime.now().strftime("%B %Y")), unsafe_allow_html=True)