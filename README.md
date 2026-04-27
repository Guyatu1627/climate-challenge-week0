# EthioClimate Analytics Engine for COP32

## 🌍 Project Overview
This repository contains the **Climate Analytics Engine for COP32**, focusing on East African climate trends (2015-2026). We analyze temperature, rainfall, and climate patterns across Ethiopia, Kenya, Nigeria, Sudan, and Tanzania to provide **negotiation-grade evidence** for climate policy decisions.

## 🚀 Quick Start
```bash
# Clone the repository
git clone https://github.com/Guyatu1627/climate-challenge-week0.git
cd climate-challenge-week0

# Create and activate virtual environment
python -m venv venv
# Windows: .\venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the interactive dashboard
streamlit run app.py
```

## 📊 Repository Structure
```
climate-challenge-week0/
├── .github/workflows/    # GitHub Actions for CI/CD (Task 1)
├── data/                 # Raw and cleaned climate datasets
├── notebooks/            # EDA and data profiling (Task 2)
│   ├── 01_eda_ethiopia.ipynb
│   └── 02_comparative_analysis.ipynb
├── src/                  # Core Python processing logic
├── tests/                # Unit tests for data validation (Task 3)
├── app.py                # Interactive Streamlit dashboard (Task 4)
├── README.md             # This file
├── requirements.txt      # Python dependencies
└── .gitignore           # Version control exclusions
```

## 📋 Task Breakdown

### **Task 1: Git & Environment Setup**
- ✅ Virtual environment configuration
- ✅ Dependencies managed via `requirements.txt`
- ✅ GitHub Actions workflow for automated testing
- ✅ Proper `.gitignore` for clean version control

### **Task 2: Data Profiling, Cleaning & EDA**
- ✅ **Ethiopia Climate EDA**: `notebooks/01_eda_ethiopia.ipynb`
  - Data loading and profiling with `df.info()`
  - Missing value analysis and cleaning
  - Temperature and rainfall visualizations
- ✅ **Comparative Analysis**: `notebooks/02_comparative_analysis.ipynb`
  - Multi-country climate comparisons
  - Statistical analysis and correlations

### **Task 3: Repository & Code Best Practices**
- ✅ Clean pandas-based workflows
- ✅ Vectorized operations (no inefficient loops)
- ✅ Proper error handling and data validation
- ✅ Unit tests for critical data processing functions

### **Task 4: Interactive Dashboard**
- ✅ Professional Streamlit application at `app.py`
- ✅ Minister-friendly interface with executive summary
- ✅ Interactive country selection and time filtering
- ✅ Multiple visualization types (time series, comparisons, correlations)

## 🛠️ Technical Implementation

### Data Processing
```python
# Example: Clean pandas workflow
import pandas as pd
df = pd.read_csv('data/master_climate_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
monthly_avg = df.groupby(['Country', 'Month'])['T2M'].mean()
```

### Key Features
- **Temperature Analysis**: Daily, monthly, and seasonal trends
- **Rainfall Patterns**: Precipitation analysis and drought indicators  
- **Country Comparisons**: Side-by-side climate metrics
- **Policy Insights**: Actionable recommendations for decision makers

## 📈 Key Climate Findings

### Temperature Trends
- Regional warming patterns identified across all monitored countries
- Seasonal variations impact agricultural planning
- Correlation analysis reveals climate relationships

### Rainfall Patterns  
- Significant seasonal variation in precipitation
- Country-specific rainfall patterns identified
- Water resource implications documented

## 🧪 Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src
```

## 📊 Dashboard Features
- **Executive Summary**: Key metrics for negotiators
- **Interactive Controls**: Country/time period selection
- **Multiple Visualizations**: Line charts, bar charts, scatter plots
- **Export Functionality**: Download analysis results
- **Policy Recommendations**: Tailored insights for climate action

## 🤝 Contributing
This project demonstrates best practices in:
- Climate data analysis with pandas
- Interactive visualization with Streamlit
- Professional repository organization
- Evidence-based policy communication

## 📚 Dependencies
See `requirements.txt` for complete list including:
- `pandas` - Data analysis
- `streamlit` - Interactive dashboard
- `plotly` - Advanced visualizations
- `numpy` - Numerical operations
- `pytest` - Testing framework

---

**Note**: This repository is designed for climate policy analysis and COP32 negotiation support. All analysis uses NASA POWER climate data and follows scientific best practices.