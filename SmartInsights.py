import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import time
import numpy as np
import plotly.graph_objects as go
import base64
import json
import io
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config with custom icon
st.set_page_config(page_title=" Smart Insights ", layout="wide", initial_sidebar_state="expanded")

# Enhanced Styling and Animations (keeping your original styling)
st.markdown(
    """
    <style>
        /* Global animations */
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(-20px);}
            to {opacity: 1; transform: translateY(0);}
        }
        @keyframes slideIn {
            from {opacity: 0; transform: translateX(-50px);}
            to {opacity: 1; transform: translateX(0);}
        }
        @keyframes pulse {
            0% {transform: scale(1);}
            50% {transform: scale(1.05);}
            100% {transform: scale(1);}
        }
        @keyframes glow {
            0% {text-shadow: 0 0 5px rgba(106, 90, 205, 0.5);}
            50% {text-shadow: 0 0 20px rgba(106, 90, 205, 0.8);}
            100% {text-shadow: 0 0 5px rgba(106, 90, 205, 0.5);}
        }
        @keyframes shimmer {
            0% {background-position: -1000px 0;}
            100% {background-position: 1000px 0;}
        }
        
        /* Global styling */
        body {
            background: linear-gradient(135deg, #1A1A1D 0%, #2E2E3D 100%);
            color: #E0E0FF;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Header styling */
        .title {
            color: #9D72FF;
            text-align: center;
            font-size: 52px;
            font-weight: bold;
            margin-bottom: 25px;
            padding: 20px;
            border-radius: 10px;
            background: linear-gradient(90deg, rgba(106, 90, 205, 0.1), rgba(106, 90, 205, 0.3), rgba(106, 90, 205, 0.1));
            animation: fadeIn 1.5s ease-in-out, glow 3s infinite;
            text-shadow: 0 0 10px rgba(106, 90, 205, 0.7);
        }
        
        /* Greeting styling */
        .greeting {
            font-size: 72px;
            font-weight: bold;
            text-align: left;
            background: linear-gradient(45deg, #D4AF37, #FFE16B, #D4AF37);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: fadeIn 2s ease-in-out, shimmer 5s infinite linear;
            background-size: 200% 100%;
            margin-top: 10px;
            margin-left: 30px;
            text-shadow: 0 0 15px rgba(212, 175, 55, 0.5);
            border-bottom: 3px solid rgba(212, 175, 55, 0.3);
            padding-bottom: 15px;
        }
        
        /* Section styles */
        .upload-section {
            text-align: center;
            animation: slideIn 1.5s;
            margin-top: 20px;
            padding: 25px;
            border-radius: 15px;
            background: linear-gradient(to right, rgba(44, 47, 51, 0.7), rgba(75, 78, 85, 0.7));
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            transition: all 0.3s ease;
        }
        .upload-section:hover {
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.57);
            transform: translateY(-5px);
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #2C2F33 0%, #23272A 100%);
            border-right: 2px solid rgba(106, 90, 205, 0.3);
        }
        
        /* Widget styling */
        .stSlider > div > div {
            background-color: #6A5ACD !important;
        }
        .stSlider > div > div > div > div {
            background-color: #FFD700 !important;
            animation: pulse 2s infinite;
        }
        .slider-label {
            color: #B19CD9;
            font-weight: bold;
            letter-spacing: 1px;
        }
        
        /* Dataframe styling */
        .dataframe {
            animation: fadeIn 1s;
            border-radius: 10px !important;
            overflow: hidden !important;
        }
        
        /* Input field styling */
        input[type=text] {
            border-radius: 10px !important;
            border: 2px solid rgba(106, 90, 205, 0.5) !important;
            background-color: rgba(40, 42, 54, 0.8) !important;
            color: #E0E0FF !important;
            transition: all 0.3s !important;
        }
        input[type=text]:focus {
            border: 2px solid #9D72FF !important;
            box-shadow: 0 0 15px rgba(106, 90, 205, 0.5) !important;
        }
        
        /* Section headers */
        h2, h3 {
            color: #B19CD9;
            border-bottom: 2px solid rgba(106, 90, 205, 0.3);
            padding-bottom: 8px;
            animation: fadeIn 1.5s;
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(45deg, #6A5ACD, #9370DB) !important;
            color: white !important;
            border-radius: 10px !important;
            border: none !important;
            padding: 10px 25px !important;
            transition: all 0.3s !important;
            animation: fadeIn 1.5s, pulse 3s infinite !important;
            font-weight: bold !important;
            letter-spacing: 1px !important;
        }
        .stButton>button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 7px 20px rgba(106, 90, 205, 0.5) !important;
        }
        
        /* Metrics */
        .metric-container {
            background: linear-gradient(135deg, rgba(106, 90, 205, 0.2), rgba(147, 112, 219, 0.2)) !important;
            border-radius: 10px !important;
            padding: 10px !important;
            margin-bottom: 15px !important;
            border: 1px solid rgba(106, 90, 205, 0.3) !important;
            animation: fadeIn 1.5s !important;
            transition: all 0.3s !important;
        }
        .metric-container:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 7px 20px rgba(106, 90, 205, 0.3) !important;
        }
        
        /* File uploader */
        .uploadedFile {
            background: linear-gradient(135deg, rgba(106, 90, 205, 0.2), rgba(147, 112, 219, 0.2)) !important;
            border-radius: 10px !important;
            border: 2px dashed rgba(106, 90, 205, 0.5) !important;
            padding: 15px !important;
            animation: fadeIn 1.5s !important;
        }
        
        /* Selectbox styling */
        .stSelectbox>div>div {
            background-color: rgba(40, 42, 54, 0.8) !important;
            border: 2px solid rgba(106, 90, 205, 0.5) !important;
            border-radius: 10px !important;
            color: #E0E0FF !important;
            transition: all 0.3s !important;
        }
        .stSelectbox>div>div:hover {
            border: 2px solid #9D72FF !important;
            box-shadow: 0 0 15px rgba(106, 90, 205, 0.5) !important;
        }
        
        /* Power BI Style Filter Pane */
        .filter-pane {
            background: linear-gradient(135deg, rgba(60, 63, 68, 0.9), rgba(40, 42, 54, 0.9));
            border-radius: 10px;
            border-left: 3px solid #9D72FF;
            padding: 15px;
            box-shadow: -5px 0 15px rgba(0, 0, 0, 0.2);
            height: 100%;
        }
        
        /* KPI Cards */
        .kpi-card {
            background: linear-gradient(135deg, rgba(60, 63, 68, 0.9), rgba(40, 42, 54, 0.9));
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid;
            transition: all 0.3s;
        }
        .kpi-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 7px 15px rgba(0, 0, 0, 0.3);
        }
        .kpi-green {
            border-color: #4CAF50;
        }
        .kpi-yellow {
            border-color: #FFC107;
        }
        .kpi-red {
            border-color: #F44336;
        }
        
        /* Tooltips */
        .custom-tooltip {
            background-color: rgba(40, 42, 54, 0.95) !important;
            border: 1px solid #9D72FF !important;
            border-radius: 5px !important;
            padding: 10px !important;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3) !important;
            color: #E0E0FF !important;
        }
        
        /* Bookmark Panel */
        .bookmark-panel {
            background: linear-gradient(135deg, rgba(60, 63, 68, 0.9), rgba(40, 42, 54, 0.9));
            border-radius: 10px;
            border-top: 3px solid #FFD700;
            padding: 15px;
            margin-bottom: 20px;
        }
        
        /* AI Insights Panel */
        .ai-insights {
            background: linear-gradient(135deg, rgba(106, 90, 205, 0.1), rgba(147, 112, 219, 0.2));
            border-radius: 10px;
            border-left: 4px solid #00BCD4;
            padding: 15px;
            margin: 20px 0;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        .insight-item {
            padding: 10px;
            margin: 10px 0;
            border-radius: 8px;
            background: rgba(60, 63, 68, 0.5);
            border-left: 3px solid;
        }
        .high-priority {
            border-color: #F44336;
        }
        .medium-priority {
            border-color: #FFC107;
        }
        .low-priority {
            border-color: #4CAF50;
        }
        
        /* Q&A Section */
        .qa-section {
            background: linear-gradient(135deg, rgba(60, 63, 68, 0.9), rgba(40, 42, 54, 0.9));
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            border-left: 4px solid #9D72FF;
        }
        .qa-input {
            background-color: rgba(40, 42, 54, 0.8) !important;
            border: 2px solid rgba(106, 90, 205, 0.5) !important;
            border-radius: 20px !important;
            padding: 10px 15px !important;
            color: #E0E0FF !important;
        }
        
        /* Grid Layout */
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .grid-item {
            background: rgba(60, 63, 68, 0.5);
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.3s;
        }
        .grid-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom-styled sidebar for navigation
st.sidebar.markdown(
    """
    <div style='animation: fadeIn 1.5s; text-align: center;'>
        <h1 style='color: #9D72FF; text-shadow: 0 0 10px rgba(106, 90, 205, 0.5); font-size: 28px; margin-bottom: 30px;'>
             Dashboard Navigation 
        </h1>
        <p style='color: #B19CD9; font-style: italic; margin-bottom: 20px; padding: 10px; border-radius: 10px; background: rgba(106, 90, 205, 0.1);'>
            Use the options below to explore your data insights.
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Enhanced title with animated sparkles
st.markdown(
    """
    <div class='title'>
        <span style='animation: pulse 3s infinite;'></span> 
        Smart Insights
        <span style='animation: pulse 3s infinite;'></span>
    </div>
    """, 
    unsafe_allow_html=True
)

# Get current time to determine greeting with emoji
time_now = datetime.datetime.now().hour
if time_now < 12:
    greeting_text = "Good Morning 🌅"
elif time_now < 18:
    greeting_text = "Good Afternoon ☀️"
else:
    greeting_text = "Good Evening 🌙"

# User input name with enhanced animation
st.markdown(
    """
    <div style='animation: slideIn 1.5s; padding: 15px; margin-bottom: 20px; border-radius: 10px; background: linear-gradient(to right, rgba(44, 47, 51, 0.7), rgba(75, 78, 85, 0.7));'>
        <p style='color: #B19CD9; font-size: 20px; margin-bottom: 10px;'>Please enter your name below:</p>
    </div>
    """,
    unsafe_allow_html=True
)
name = st.text_input("", "", key="name_input", placeholder="Type your name here...")

# Function to display KPI
def display_kpi(title, value, subtitle):
    st.markdown(
        f"""
        <div style='padding: 10px; border-radius: 5px; background: rgba(60, 63, 68, 0.7); margin-bottom: 10px;'>
            <h4 style='color: #9D72FF; font-size: 16px; margin-bottom: 5px;'>{title}</h4>
            <p style='color: #E0E0FF; font-size: 24px; margin: 0;'>{value}</p>
            <p style='color: #B19CD9; font-size: 14px; margin: 0;'>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Function to generate AI insights from data
def generate_ai_insights(df):
    insights = []
    
    # Check for numeric columns for analysis
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) > 0:
        try:
            # 1. Find outliers/anomalies using Isolation Forest
            if len(df) > 10:  # Need enough data for meaningful anomaly detection
                # Select numeric columns for anomaly detection
                data_for_anomaly = df[numeric_cols].copy()
                # Fill NaN values with column means
                data_for_anomaly = data_for_anomaly.fillna(data_for_anomaly.mean())
                
                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data_for_anomaly)
                
                # Train Isolation Forest
                iso_forest = IsolationForest(contamination=0.05, random_state=42)
                outliers = iso_forest.fit_predict(scaled_data)
                
                # Count outliers
                num_outliers = np.sum(outliers == -1)
                if num_outliers > 0:
                    outlier_indices = np.where(outliers == -1)[0]
                    outlier_rows = df.iloc[outlier_indices]
                    
                    # Identify which columns have the most extreme values in outliers
                    problematic_columns = []
                    for col in numeric_cols:
                        if col in df.columns:
                            col_mean = df[col].mean()
                            col_std = df[col].std()
                            if col_std > 0:  # Avoid division by zero
                                extreme_values = outlier_rows[abs((outlier_rows[col] - col_mean) / col_std) > 2]
                                if len(extreme_values) > 0:
                                    problematic_columns.append((col, len(extreme_values)))
                    
                    if problematic_columns:
                        problematic_columns.sort(key=lambda x: x[1], reverse=True)
                        insights.append({
                            'title': f"🚨 Detected {num_outliers} outliers in your data",
                            'description': f"The most affected columns are: {', '.join([col[0] for col in problematic_columns[:3]])}. These outliers may be skewing your analysis.",
                            'recommendation': "Review these outliers to determine if they're errors or valuable insights. Consider using robust statistical methods.",
                            'priority': 'high' if num_outliers > len(df) * 0.1 else 'medium'
                        })
            
            # 2. Identify columns with high null values
            null_percentage = df.isnull().sum() / len(df) * 100
            high_null_cols = null_percentage[null_percentage > 15].index.tolist()
            
            if high_null_cols:
                insights.append({
                    'title': f"⚠️ High missing values detected in {len(high_null_cols)} columns",
                    'description': f"Columns with >15% missing values: {', '.join(high_null_cols)}",
                    'recommendation': "Consider imputation techniques or evaluate if these columns can be dropped.",
                    'priority': 'high' if any(null_percentage > 40) else 'medium'
                })
            
            # 3. Identify low-performing metrics based on statistical analysis
            if len(numeric_cols) >= 2:
                # Calculate correlation matrix
                corr_matrix = df[numeric_cols].corr().abs()
                
                # Find highly correlated pairs (could indicate redundant features)
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr_pairs = [(upper_tri.index[i], upper_tri.columns[j], upper_tri.iloc[i, j]) 
                                  for i, j in zip(*np.where(upper_tri > 0.9))]
                
                if high_corr_pairs:
                    insights.append({
                        'title': f"Found {len(high_corr_pairs)} highly correlated metrics",
                        'description': f"These pairs show >90% correlation, suggesting potential redundancy: " + 
                                      f"{', '.join([f'{pair[0]} & {pair[1]}' for pair in high_corr_pairs[:3]])}",
                        'recommendation': "Consider consolidating these metrics for cleaner analysis.",
                        'priority': 'medium'
                    })
                
                # Identify metrics with high variance (potential instability)
                cv_values = df[numeric_cols].std() / df[numeric_cols].mean()
                high_variance_cols = cv_values[cv_values > 1].index.tolist()
                
                if high_variance_cols:
                    insights.append({
                        'title': f"📈 Detected {len(high_variance_cols)} metrics with high variability",
                        'description': f"These metrics show unusually high coefficient of variation: {', '.join(high_variance_cols[:3])}",
                        'recommendation': "Investigate the causes of volatility and consider segmentation for more stable metrics.",
                        'priority': 'medium'
                    })
                
                # Identify skewed distributions
                skewed_cols = []
                for col in numeric_cols:
                    if df[col].skew() > 1.5 or df[col].skew() < -1.5:
                        skewed_cols.append((col, df[col].skew()))
                
                if skewed_cols:
                    insights.append({
                        'title': f"⚖️ Found {len(skewed_cols)} metrics with skewed distributions",
                        'description': f"These metrics may benefit from transformation: {', '.join([col[0] for col in skewed_cols[:3]])}",
                        'recommendation': "Consider log or square-root transformations for more accurate statistical analysis.",
                        'priority': 'low'
                    })
                    
                # Check for potential seasonality/trends (if any date columns exist)
                date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols and len(date_cols) > 0:
                    insights.append({
                        'title': "Time-based analysis opportunity detected",
                        'description': f"Your data contains date/time information in columns: {', '.join(date_cols[:3])}",
                        'recommendation': "Analyze temporal patterns to uncover seasonality or trends in your metrics.",
                        'priority': 'medium'
                    })
            
            # 4. Performance insights (for sales data specifically)
            if any('sale' in col.lower() for col in df.columns) or any('revenue' in col.lower() for col in df.columns):
                sales_cols = [col for col in df.columns if 'sale' in col.lower() or 'revenue' in col.lower()]
                if sales_cols:
                    sales_col = sales_cols[0]  # Use the first sales column found
                    if df[sales_col].mean() > 0:
                        quartiles = df[sales_col].quantile([0.25, 0.75]).values
                        bottom_performers = df[df[sales_col] < quartiles[0]]
                        
                        if len(bottom_performers) > 0:
                            # Look for patterns in underperforming sales
                            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                            pattern_insights = []
                            
                            for cat_col in categorical_cols[:3]:  # Check first 3 categorical columns
                                if cat_col in bottom_performers.columns:
                                    value_counts = bottom_performers[cat_col].value_counts(normalize=True)
                                    if not value_counts.empty and value_counts.iloc[0] > 0.3:  # If any category represents >30% of poor performers
                                        pattern_insights.append(f"{cat_col}='{value_counts.index[0]}'")
                            
                            if pattern_insights:
                                insights.append({
                                    'title': f"💡 Performance improvement opportunity identified",
                                    'description': f"25% of your sales/revenue data falls below {quartiles[0]:.2f}. " +
                                                  f"Common patterns in low performers: {', '.join(pattern_insights)}",
                                    'recommendation': "Focus improvement efforts on these specific segments to boost overall performance.",
                                    'priority': 'high'
                                })
        except Exception as e:
            # Fallback generic insights if analysis fails
            insights.append({
                'title': "General data quality assessment",
                'description': f"Your dataset contains {len(df)} records and {len(df.columns)} variables.",
                'recommendation': "Ensure data completeness and accuracy for reliable analysis.",
                'priority': 'medium'
            })
    
    # If we couldn't generate insights or not enough were generated
    if len(insights) == 0:
        insights.append({
            'title': "👋 Welcome to AI Insights",
            'description': "Upload more comprehensive data to receive detailed performance analysis.",
            'recommendation': "For best results, include sales/revenue metrics, dates, and categorical dimensions.",
            'priority': 'low'
        })
    
    return insights

# Function to create date slicer
def create_date_slicer(df):
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    if not date_cols:
        st.markdown(
            """
            <div style='padding: 15px; border-radius: 10px; background: rgba(60, 63, 68, 0.7); margin-bottom: 15px;'>
                <p style='color: #B19CD9; font-size: 16px;'>No date columns detected for time-based filtering.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return None
    
    # Select first date column
    date_col = date_cols[0]
    
    try:
        # Convert to datetime if not already
        if df[date_col].dtype != 'datetime64[ns]':
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Get min and max dates
        min_date = df[date_col].min().date()
        max_date = df[date_col].max().date()
        
        st.markdown(
            """
            <div style='padding: 15px; border-radius: 10px; background: rgba(60, 63, 68, 0.7); margin-bottom: 15px;'>
                <h4 style='color: #9D72FF; font-size: 18px; margin-bottom: 10px;'>Date Range Filter</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Create date range slider
        date_range = st.date_input(
            "",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Return filtered dataframe if date range selected
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = df[(df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)]
            return filtered_df
    except:
        st.warning(f"Could not convert '{date_col}' to date format.", icon="⚠️")
    
    return df

# Function to create KPI cards
def create_kpi_cards(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        return
    
    # Select up to 3 numeric columns for KPIs
    kpi_cols = numeric_cols[:3]
    
    st.markdown(
        """
        <div style='padding: 15px; border-radius: 10px; background: rgba(60, 63, 68, 0.7); margin-bottom: 15px;'>
            <h4 style='color: #9D72FF; font-size: 18px; margin-bottom: 10px;'>Key Performance Indicators</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    for col in kpi_cols:
        current_value = df[col].mean()
        
        # Determine if KPI is good, average, or concerning
        if df[col].skew() > 0:  # Right-skewed (higher values better)
            percentile_75 = df[col].quantile(0.75)
            percentile_25 = df[col].quantile(0.25)
            
            if current_value >= percentile_75:
                kpi_class = "kpi-green"
                kpi_icon = "↗️"
                kpi_status = "Excellent"
            elif current_value >= percentile_25:
                kpi_class = "kpi-yellow"
                kpi_icon = "→"
                kpi_status = "Average"
            else:
                kpi_class = "kpi-red"
                kpi_icon = "↘️"
                kpi_status = "Needs Attention"
        else:  # Left-skewed or symmetric (lower values might be better)
            percentile_75 = df[col].quantile(0.75)
            percentile_25 = df[col].quantile(0.25)
            
            if current_value <= percentile_25:
                kpi_class = "kpi-green"
                kpi_icon = "↘️"
                kpi_status = "Excellent"
            elif current_value <= percentile_75:
                kpi_class = "kpi-yellow" 
                kpi_icon = "→"
                kpi_status = "Average"
            else:
                kpi_class = "kpi-red"
                kpi_icon = "↗️"
                kpi_status = "Needs Attention"
        
        # Format value appropriately
        if abs(current_value) > 1000:
            formatted_value = f"{current_value:,.0f}"
        elif abs(current_value) > 10:
            formatted_value = f"{current_value:.1f}"
        else:
            formatted_value = f"{current_value:.2f}"
            
        st.markdown(
            f"""
            <div class='kpi-card {kpi_class}'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <h4 style='margin: 0; color: #E0E0FF;'>{col.replace('_', ' ').title()}</h4>
                    <span style='font-size: 24px;'>{kpi_icon}</span>
                </div>
                <h2 style='margin: 10px 0; font-size: 32px; color: #E0E0FF;'>{formatted_value}</h2>
                <p style='margin: 0; color: #B19CD9;'>{kpi_status}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Function to create interactive visualizations
def create_visualizations(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numeric_cols:
        return
    
    st.markdown(
        """
        <div style='padding: 15px; border-radius: 10px; background: rgba(60, 63, 68, 0.7); margin-bottom: 15px;'>
            <h4 style='color: #9D72FF; font-size: 18px; margin-bottom: 10px;'>Interactive Data Visualizations</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Create tabs for different visualization types
    viz_tabs = st.tabs(["Distribution Analysis", "Correlation Analysis", "Trend Analysis"])
    
    with viz_tabs[0]:
        # Distribution Analysis
        if len(numeric_cols) > 0:
            st.markdown("<h5 style='color: #B19CD9;'>Distribution of Key Metrics</h5>", unsafe_allow_html=True)
            selected_metric = st.selectbox("Select metric to analyze:", numeric_cols)
            
            # Create histogram with KDE
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=df[selected_metric],
                    name="Frequency",
                    marker=dict(color="rgba(106, 90, 205, 0.6)"),
                    nbinsx=20
                )
            )
            
            # Add KDE (Kernel Density Estimation)
            try:
                from scipy.stats import gaussian_kde
                import numpy as np
                
                # Remove NaN values
                data = df[selected_metric].dropna()
                
                if len(data) > 1:  # Need at least 2 points for KDE
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 1000)
                    y_kde = kde(x_range)
                    
                    # Scale KDE to match histogram height
                    hist, bin_edges = np.histogram(data, bins=20)
                    max_hist_height = max(hist)
                    scaling_factor = max_hist_height / max(y_kde)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_range, 
                            y=y_kde * scaling_factor,
                            mode="lines",
                            line=dict(color="rgba(255, 215, 0, 0.8)", width=3),
                            name="Density"
                        ),
                        secondary_y=False
                    )
            except:
                pass
            
            # Add box plot on secondary y-axis
            fig.add_trace(
                go.Box(
                    x=df[selected_metric],
                    name="Distribution",
                    marker=dict(color="rgba(157, 114, 255, 0.7)"),
                    boxpoints="outliers",
                    orientation="h"
                ),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(40, 42, 54, 0.8)",
                paper_bgcolor="rgba(40, 42, 54, 0)",
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                title={
                    "text": f"Distribution Analysis of {selected_metric}",
                    "y": 0.95,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                    "font": {"color": "#9D72FF", "size": 18}
                }
            )
            
            # Update y-axes
            fig.update_yaxes(title_text="Frequency", secondary_y=False, showgrid=True, gridwidth=1, gridcolor="rgba(107, 114, 142, 0.2)")
            fig.update_yaxes(title_text="", secondary_y=True, showticklabels=False)
            
            # Update x-axis
            fig.update_xaxes(title_text=selected_metric, showgrid=True, gridwidth=1, gridcolor="rgba(107, 114, 142, 0.2)")
            
            # Show plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Add statistics summary
            st.markdown("<h5 style='color: #B19CD9;'>Statistical Summary</h5>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{df[selected_metric].mean():.2f}")
            with col2:
                st.metric("Median", f"{df[selected_metric].median():.2f}")
            with col3:
                st.metric("Std Dev", f"{df[selected_metric].std():.2f}")
            with col4:
                st.metric("Skewness", f"{df[selected_metric].skew():.2f}")
            
    with viz_tabs[1]:
        # Correlation Analysis
        if len(numeric_cols) >= 2:
            st.markdown("<h5 style='color: #B19CD9;'>Correlation Between Metrics</h5>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("Select X-axis metric:", numeric_cols)
            with col2:
                y_var = st.selectbox("Select Y-axis metric:", [col for col in numeric_cols if col != x_var])
            
            # Create scatter plot with trendline
            fig = px.scatter(
                df, 
                x=x_var, 
                y=y_var,
                trendline="ols",
                color_discrete_sequence=["rgba(157, 114, 255, 0.7)"],
                trendline_color_override="rgba(255, 215, 0, 0.8)"
            )
            
            # Calculate correlation coefficient
            corr = df[x_var].corr(df[y_var])
            
            # Update layout
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(40, 42, 54, 0.8)",
                paper_bgcolor="rgba(40, 42, 54, 0)",
                margin=dict(l=20, r=20, t=50, b=20),
                title={
                    "text": f"Correlation: {corr:.2f}",
                    "y": 0.95,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                    "font": {"color": "#9D72FF", "size": 18}
                }
            )
            
            # Update axes
            fig.update_xaxes(title_text=x_var, showgrid=True, gridwidth=1, gridcolor="rgba(107, 114, 142, 0.2)")
            fig.update_yaxes(title_text=y_var, showgrid=True, gridwidth=1, gridcolor="rgba(107, 114, 142, 0.2)")
            
            # Show plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Add correlation interpretation
            correlation_text = ""
            if abs(corr) < 0.3:
                correlation_text = "Weak correlation"
            elif abs(corr) < 0.7:
                correlation_text = "Moderate correlation"
            else:
                correlation_text = "Strong correlation"
                
            if corr > 0:
                correlation_text += " (positive)"
            else:
                correlation_text += " (negative)"
            
            st.markdown(
                f"""
                <div style='background: rgba(60, 63, 68, 0.7); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                    <h5 style='color: #B19CD9; margin-top: 0;'>Correlation Insight</h5>
                    <p style='color: #E0E0FF;'>{correlation_text} detected between {x_var} and {y_var}.</p>
                    <p style='color: #B19CD9;'>
                        R² value: {corr**2:.2f} - This means approximately {(corr**2 * 100):.1f}% of the variation 
                        in {y_var} can be explained by {x_var}.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    with viz_tabs[2]:
        # Trend Analysis
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if date_cols and numeric_cols:
            st.markdown("<h5 style='color: #B19CD9;'>Trend Analysis Over Time</h5>", unsafe_allow_html=True)
            
            # Select date column and metric
            date_col = st.selectbox("Select date column:", date_cols)
            trend_metric = st.selectbox("Select metric to track:", numeric_cols, key="trend_metric")
            
            # Convert to datetime if needed
            if df[date_col].dtype != 'datetime64[ns]':
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Group by date and calculate metrics
            try:
                # Determine appropriate date grouping based on date range
                date_range = (df[date_col].max() - df[date_col].min()).days
                
                if date_range > 365*2:  # More than 2 years
                    grouper = pd.Grouper(key=date_col, freq='Q')
                    group_name = "Quarterly"
                elif date_range > 90:  # More than 3 months
                    grouper = pd.Grouper(key=date_col, freq='M')
                    group_name = "Monthly"
                elif date_range > 21:  # More than 3 weeks
                    grouper = pd.Grouper(key=date_col, freq='W')
                    group_name = "Weekly"
                else:
                    grouper = pd.Grouper(key=date_col, freq='D')
                    group_name = "Daily"
                
                # Group and aggregate
                trend_df = df.groupby(grouper)[trend_metric].agg(['mean', 'min', 'max']).reset_index()
                trend_df.columns = [date_col, 'Average', 'Minimum', 'Maximum']
                
                # Create area plot
                fig = go.Figure()
                
                # Add min-max range
                fig.add_trace(
                    go.Scatter(
                        x=trend_df[date_col],
                        y=trend_df['Maximum'],
                        fill=None,
                        mode='lines',
                        line_color='rgba(157, 114, 255, 0.1)',
                        name='Maximum'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=trend_df[date_col],
                        y=trend_df['Minimum'],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(157, 114, 255, 0.1)',
                        name='Minimum'
                    )
                )
                
                # Add average line
                fig.add_trace(
                    go.Scatter(
                        x=trend_df[date_col],
                        y=trend_df['Average'],
                        mode='lines+markers',
                        line=dict(color='rgba(255, 215, 0, 0.8)', width=3),
                        marker=dict(size=8, color='rgba(255, 215, 0, 0.9)'),
                        name='Average'
                    )
                )
                
                # Update layout
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(40, 42, 54, 0.8)",
                    paper_bgcolor="rgba(40, 42, 54, 0)",
                    margin=dict(l=20, r=20, t=50, b=20),
                    title={
                        "text": f"{group_name} Trend of {trend_metric}",
                        "y": 0.95,
                        "x": 0.5,
                        "xanchor": "center",
                        "yanchor": "top",
                        "font": {"color": "#9D72FF", "size": 18}
                    },
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Update axes
                fig.update_xaxes(title_text="Date", showgrid=True, gridwidth=1, gridcolor="rgba(107, 114, 142, 0.2)")
                fig.update_yaxes(title_text=trend_metric, showgrid=True, gridwidth=1, gridcolor="rgba(107, 114, 142, 0.2)")
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate trend statistics
                first_avg = trend_df['Average'].iloc[0] if not trend_df.empty else 0
                last_avg = trend_df['Average'].iloc[-1] if not trend_df.empty else 0
                pct_change = ((last_avg - first_avg) / first_avg * 100) if first_avg != 0 else 0
                
                # Add trend summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Start Value", f"{first_avg:.2f}")
                with col2:
                    st.metric("End Value", f"{last_avg:.2f}")
                with col3:
                    st.metric("Overall Change", f"{pct_change:.1f}%",
                             delta=f"{pct_change:.1f}%",
                             delta_color="normal")
            except Exception as e:
                st.error(f"Could not create trend analysis: {str(e)}")

# Function to render AI insights panel
def render_ai_insights(insights):
    st.markdown(
        """
        <div style='padding: 15px; border-radius: 10px; background: rgba(60, 63, 68, 0.7); margin-bottom: 15px;'>
            <h4 style='color: #9D72FF; font-size: 18px; margin-bottom: 10px;'>AI-Generated Insights</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Create expandable sections for each insight
    for i, insight in enumerate(insights):
        priority_class = f"{insight['priority']}-priority"
        
        st.markdown(
            f"""
            <div class='insight-item {priority_class}'>
                <h5 style='color: #E0E0FF; margin-top: 0;'>{insight['title']}</h5>
                <p style='color: #B19CD9;'>{insight['description']}</p>
                <p style='color: #9D72FF; font-style: italic;'><strong>Recommendation:</strong> {insight['recommendation']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Function to display Q&A section
def display_qa_section(df):
    st.markdown(
        """
        <div class='qa-section'>
            <h4 style='color: #9D72FF; margin-top: 0;'>Ask Questions About Your Data</h4>
            <p style='color: #B19CD9;'>Type your question below to get insights about your data.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    query = st.text_input("", "", key="qa_input", placeholder="e.g., What is the average sales value?")
    
    if query:
        with st.spinner("Analyzing your question..."):
            time.sleep(1)  # Simulate processing
            
            try:
                # Simple keyword-based response system
                query_lower = query.lower()
                
                if 'average' in query_lower or 'mean' in query_lower:
                    # Extract column name from query
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    matched_col = None
                    
                    for col in numeric_cols:
                        if col.lower() in query_lower:
                            matched_col = col
                            break
                    
                    if matched_col:
                        avg_value = df[matched_col].mean()
                        st.markdown(
                            f"""
                            <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                <p style='color: #E0E0FF;'>The average {matched_col} is <strong>{avg_value:.2f}</strong></p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        # If no specific column mentioned, show averages for all numeric columns
                        st.markdown(
                            f"""
                            <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                <p style='color: #E0E0FF;'>Here are the averages for all numeric columns:</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        avg_df = pd.DataFrame(df.mean()).reset_index()
                        avg_df.columns = ['Column', 'Average']
                        st.dataframe(avg_df)
                
                elif 'maximum' in query_lower or 'max' in query_lower:
                    # Extract column name from query
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    matched_col = None
                    
                    for col in numeric_cols:
                        if col.lower() in query_lower:
                            matched_col = col
                            break
                    
                    if matched_col:
                        max_value = df[matched_col].max()
                        st.markdown(
                            f"""
                            <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                <p style='color: #E0E0FF;'>The maximum {matched_col} is <strong>{max_value:.2f}</strong></p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        # If no specific column mentioned, explain the limitation
                        st.markdown(
                            f"""
                            <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                <p style='color: #E0E0FF;'>Please specify which column you'd like to find the maximum value for.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                elif 'minimum' in query_lower or 'min' in query_lower:
                    # Extract column name from query
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    matched_col = None
                    
                    for col in numeric_cols:
                        if col.lower() in query_lower:
                            matched_col = col
                            break
                    
                    if matched_col:
                        min_value = df[matched_col].min()
                        st.markdown(
                            f"""
                            <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                <p style='color: #E0E0FF;'>The minimum  {matched_col} is <strong>{min_value:.2f}</strong></p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        # If no specific column mentioned, explain the limitation
                        st.markdown(
                            f"""
                            <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                <p style='color: #E0E0FF;'>Please specify which column you'd like to find the minimum value for.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                elif 'correlation' in query_lower or 'correlate' in query_lower:
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    
                    if len(numeric_cols) >= 2:
                        corr_matrix = df[numeric_cols].corr().abs()
                        
                        st.markdown(
                            f"""
                            <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                <p style='color: #E0E0FF;'>Here's the correlation matrix between numeric variables:</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Plot heatmap
                        fig = px.imshow(
                            corr_matrix,
                            color_continuous_scale="Viridis",
                            labels=dict(color="Correlation")
                        )
                        
                        fig.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="rgba(40, 42, 54, 0.8)",
                            paper_bgcolor="rgba(40, 42, 54, 0)",
                            margin=dict(l=20, r=20, t=50, b=20),
                            title={
                                "text": "Correlation Matrix",
                                "y": 0.95,
                                "x": 0.5,
                                "xanchor": "center",
                                "yanchor": "top",
                                "font": {"color": "#9D72FF", "size": 18}
                            }
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Find highest correlation
                        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                        highest_corr = upper_tri.stack().nlargest(1)
                        
                        for idx, value in highest_corr.items():
                            st.markdown(
                                f"""
                                <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                    <p style='color: #E0E0FF;'>The strongest correlation is between <strong>{idx[0]}</strong> and <strong>{idx[1]}</strong> with a value of <strong>{value:.2f}</strong></p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown(
                            f"""
                            <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                <p style='color: #E0E0FF;'>Need at least two numeric columns to calculate correlations.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                elif 'top' in query_lower or 'highest' in query_lower:
                    # Extract column name and number of results
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    matched_col = None
                    
                    for col in numeric_cols:
                        if col.lower() in query_lower:
                            matched_col = col
                            break
                    
                    # Try to extract a number from the query
                    import re
                    num_match = re.search(r'\b(\d+)\b', query_lower)
                    num_results = int(num_match.group(1)) if num_match else 5
                    
                    if matched_col:
                        top_values = df.nlargest(num_results, matched_col)
                        
                        st.markdown(
                            f"""
                            <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                <p style='color: #E0E0FF;'>Top {num_results} highest {matched_col} values:</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        st.dataframe(top_values)
                    else:
                        # If no specific column mentioned, explain the limitation
                        st.markdown(
                            f"""
                            <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                                <p style='color: #E0E0FF;'>Please specify which column you'd like to see top values for.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                else:
                    # Generic response for unrecognized queries
                    st.markdown(
                        f"""
                        <div style='background: rgba(106, 90, 205, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                            <p style='color: #E0E0FF;'>I understand you're asking about: "{query}"</p>
                            <p style='color: #B19CD9;'>Try asking about averages, maximums, correlations, or top values in your data.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")

    st.markdown("<h3 style='color: #9D72FF;'>Excel Data Preview</h3>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# NLP ENGINE — Generates plain-English explanations for any chart or result
# ═══════════════════════════════════════════════════════════════════════════════
def nlp_explain_chart(chart_type, df=None, col=None, col2=None, series=None, extra=None):
    """Returns a plain-English paragraph explaining what the chart shows."""
    try:
        if chart_type == "distribution" and series is not None:
            s = series.dropna()
            if len(s) == 0: return ""
            mean_v, med_v = s.mean(), s.median()
            std_v, skew_v = s.std(), s.skew()
            mn, mx = s.min(), s.max()
            cv = std_v / (abs(mean_v) + 1e-9)
            skew_txt = (
                "roughly symmetric — meaning values are evenly spread around the average" if abs(skew_v) < 0.5
                else ("right-skewed — most values cluster on the lower end, but a few very high values pull the average up" if skew_v > 0
                      else "left-skewed — most values are high, but a few very low values pull the average down")
            )
            spread_txt = "low" if cv < 0.2 else ("very high" if cv > 1.0 else "moderate")
            pct_above = (s > mean_v).mean() * 100
            return (
                f"This histogram shows how <strong>{col}</strong> is distributed across your dataset. "
                f"Values range from <strong>{mn:,.2f}</strong> to <strong>{mx:,.2f}</strong>. "
                f"The average is <strong>{mean_v:,.2f}</strong> and the median is <strong>{med_v:,.2f}</strong>. "
                f"{'Since mean ≈ median, there are no extreme outliers distorting the picture.' if abs(mean_v - med_v) / (abs(mean_v) + 1e-9) < 0.1 else 'The mean and median differ, suggesting some extreme values are pulling the average away from the typical value.'} "
                f"The shape is <strong>{skew_txt}</strong>. "
                f"Variability is <strong>{spread_txt}</strong> (standard deviation = {std_v:,.2f}). "
                f"About <strong>{pct_above:.1f}%</strong> of records are above the average."
            )

        elif chart_type == "correlation" and df is not None and col and col2:
            corr = df[[col, col2]].dropna().corr().iloc[0, 1]
            r2 = corr ** 2
            strength = "strong" if abs(corr) >= 0.7 else ("moderate" if abs(corr) >= 0.4 else "weak")
            direction = "positive" if corr >= 0 else "negative"
            meaning = (
                f"As <strong>{col}</strong> increases, <strong>{col2}</strong> tends to {'increase' if corr > 0 else 'decrease'} as well."
                if abs(corr) >= 0.4
                else f"There is no meaningful linear relationship — knowing <strong>{col}</strong> does not reliably predict <strong>{col2}</strong>."
            )
            return (
                f"This scatter plot shows the relationship between <strong>{col}</strong> and <strong>{col2}</strong>. "
                f"The Pearson correlation is <strong>{corr:.2f}</strong> — a <strong>{strength} {direction} relationship</strong>. "
                f"{meaning} "
                f"The R² value is <strong>{r2:.2f}</strong>, meaning {col} explains about <strong>{r2*100:.1f}%</strong> of the variation in {col2}. "
                f"{'The remaining variation is driven by other factors not captured here.' if r2 < 0.9 else 'This is a very strong predictor.'}"
            )

        elif chart_type == "timeseries" and series is not None:
            s = series.dropna()
            if len(s) < 2: return ""
            first_v, last_v = s.iloc[0], s.iloc[-1]
            mx_v, mn_v = s.max(), s.min()
            pct = ((last_v - first_v) / (abs(first_v) + 1e-9)) * 100
            trend = "growing" if pct > 5 else ("declining" if pct < -5 else "relatively stable")
            vol = s.pct_change().dropna().std() * 100
            vol_txt = "low" if vol < 5 else ("high" if vol > 20 else "moderate")
            return (
                f"This time series chart tracks <strong>{col}</strong> over time. "
                f"The overall trend is <strong>{trend}</strong>, moving from <strong>{first_v:,.2f}</strong> at the start "
                f"to <strong>{last_v:,.2f}</strong> at the end — a change of <strong>{'+'if pct>0 else ''}{pct:.1f}%</strong>. "
                f"The highest recorded value was <strong>{mx_v:,.2f}</strong> and the lowest was <strong>{mn_v:,.2f}</strong>. "
                f"Period-to-period volatility is <strong>{vol_txt}</strong> (avg swing ~{vol:.1f}%). "
                f"{'The dotted moving average line helps smooth out short-term noise and shows the underlying direction.' if len(s) > 5 else ''}"
            )

        elif chart_type == "sector_bar" and series is not None:
            s = series.sort_values(ascending=False).dropna()
            if len(s) == 0: return ""
            total = s.sum()
            top_n, bot_n = s.index[0], s.index[-1]
            top_v, bot_v = s.iloc[0], s.iloc[-1]
            top_share = top_v / total * 100 if total else 0
            top3 = s.head(3)
            top3_share = top3.sum() / total * 100 if total else 0
            gap = top_v / (bot_v + 1e-9)
            return (
                f"This bar chart ranks each category in <strong>{col}</strong> by <strong>{col2}</strong>. "
                f"<strong>{top_n}</strong> leads the pack with <strong>{top_v:,.2f}</strong>, accounting for <strong>{top_share:.1f}%</strong> of the total. "
                f"The bottom performer is <strong>{bot_n}</strong> at <strong>{bot_v:,.2f}</strong> — the top is <strong>{gap:.1f}x larger</strong>. "
                f"The top 3 categories together represent <strong>{top3_share:.1f}%</strong> of all value. "
                f"{'This is a highly concentrated distribution — a small number of categories dominate.' if top3_share > 70 else 'Distribution is spread relatively evenly across categories.'}"
            )

        elif chart_type == "sector_treemap" and series is not None:
            s = series.sort_values(ascending=False).dropna()
            total = s.sum()
            top_n = s.index[0]; top_share = s.iloc[0] / total * 100 if total else 0
            return (
                f"The treemap visualizes proportional size — larger tiles represent larger shares of the total <strong>{col2}</strong>. "
                f"<strong>{top_n}</strong> occupies the most space with <strong>{top_share:.1f}%</strong> of the total. "
                f"Hover over any tile to see exact values. This view makes it easy to spot which categories dominate at a glance, "
                f"without needing to read numbers — the bigger the tile, the bigger the contribution."
            )

        elif chart_type == "sector_radar" and extra is not None:
            metrics = extra.get("metrics", [])
            return (
                f"The radar (spider) chart compares sectors across <strong>{len(metrics)} metrics simultaneously</strong>: "
                f"{', '.join(metrics[:4])}{'...' if len(metrics) > 4 else ''}. "
                f"Each axis represents one metric, normalized to 0–100 so metrics with different units can be compared fairly. "
                f"A larger, more filled polygon means that sector performs well across all dimensions. "
                f"Narrow or lopsided polygons indicate a sector that excels in some areas but lags in others — a candidate for targeted improvement."
            )

        elif chart_type == "forecast" and extra is not None:
            model_name = extra.get("model", "ML model")
            r2 = extra.get("r2", 0)
            mae = extra.get("mae", 0)
            periods = extra.get("periods", 0)
            last_actual = extra.get("last_actual", 0)
            last_forecast = extra.get("last_forecast", 0)
            pct = ((last_forecast - last_actual) / (abs(last_actual) + 1e-9)) * 100
            quality = "excellent" if r2 > 0.85 else ("good" if r2 > 0.65 else ("fair" if r2 > 0.45 else "limited"))
            direction = "increase" if pct > 0 else "decrease"
            return (
                f"The <strong>{model_name}</strong> was trained on historical patterns in your data to forecast future values of <strong>{col}</strong>. "
                f"Model accuracy is <strong>{quality}</strong> — R² = {r2:.3f} (how well it fits past data; 1.0 = perfect), MAE = {mae:,.2f} (average prediction error). "
                f"Over the next <strong>{periods}</strong> period(s), the model projects a <strong>{direction} of {abs(pct):.1f}%</strong>. "
                f"The shaded confidence band shows the range of likely outcomes — wider bands mean more uncertainty. "
                f"{'Higher R² means you can trust this forecast more.' if r2 > 0.7 else 'The moderate R² suggests treat this forecast as directional guidance, not a precise prediction.'} "
                f"Always consider factors your dataset may not capture — market changes, seasonality shifts, or one-time events."
            )

        elif chart_type == "regression_actual_vs_pred" and extra is not None:
            r2 = extra.get("r2", 0)
            mae = extra.get("mae", 0)
            rmse = extra.get("rmse", 0)
            model_name = extra.get("model", "model")
            quality = "excellent" if r2 > 0.85 else ("good" if r2 > 0.65 else ("fair" if r2 > 0.45 else "limited"))
            return (
                f"This scatter plot shows how well the <strong>{model_name}</strong> predicted values versus what actually happened (test set). "
                f"Points close to the diagonal line = accurate predictions. Points far from it = prediction errors. "
                f"The model has <strong>{quality}</strong> accuracy: R² = {r2:.3f}, meaning it explains {r2*100:.1f}% of variation in the target. "
                f"The average prediction error (MAE) is <strong>{mae:,.2f}</strong> and RMSE is <strong>{rmse:,.2f}</strong>. "
                f"{'You can rely on this model for predictions.' if r2 > 0.7 else 'Consider adding more features or trying a different model to improve accuracy.'}"
            )

        elif chart_type == "feature_importance" and extra is not None:
            top_feat = extra.get("top_feature", "")
            top_imp = extra.get("top_importance", 0)
            return (
                f"This chart shows which input features the model relied on most heavily when making predictions. "
                f"<strong>{top_feat}</strong> is the most influential variable with an importance score of <strong>{top_imp:.3f}</strong>. "
                f"Higher importance = the model used that feature more to split decisions and reduce errors. "
                f"Low-importance features contribute little — you could potentially remove them without hurting accuracy. "
                f"This also helps detect if the model is relying too heavily on a single feature, which could make it brittle."
            )

    except Exception:
        return ""
    return ""


def render_nlp_panel(explanation):
    """Renders a clean, styled explanation box."""
    if not explanation:
        return
    st.markdown(
        f"""
        <div style='background: linear-gradient(135deg, rgba(0,188,212,0.07), rgba(106,90,205,0.10));
                    border-left: 3px solid #00BCD4; border-radius: 8px;
                    padding: 14px 18px; margin: 8px 0 20px 0;'>
            <span style='color: #00BCD4; font-size: 12px; font-weight: 700;
                         letter-spacing: 1.5px; text-transform: uppercase;'>
                Plain-English Explanation
            </span>
            <p style='color: #D0D0F0; font-size: 14px; line-height: 1.75;
                      margin: 8px 0 0 0;'>
                {explanation}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# NLP INSIGHTS PAGE — Dedicated panel for user-selected chart explanations
# ═══════════════════════════════════════════════════════════════════════════════
def render_nlp_insights_page(df):
    st.markdown("""
        <div style='padding:20px 0 10px 0;'>
            <h1 style='color:#9D72FF; margin:0; font-size:32px;'>NLP Insights Center</h1>
            <p style='color:#B19CD9; margin:6px 0 0 0; font-size:15px;'>
                Select any chart type below. The system will generate a full plain-English explanation of
                exactly what that chart is showing, what the numbers mean, and what actions to consider.
            </p>
        </div>
    """, unsafe_allow_html=True)

    num_cols  = df.select_dtypes(include='number').columns.tolist()
    cat_cols  = df.select_dtypes(include='object').columns.tolist()
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]

    # Guide header
    st.markdown("""
        <div style='background:rgba(106,90,205,0.12); border-radius:10px;
                    border-left:4px solid #9D72FF; padding:14px 18px; margin:12px 0 20px 0;'>
            <p style='color:#B19CD9; margin:0; font-size:14px;'>
                <strong style='color:#9D72FF;'>How to use this page:</strong>
                Choose a chart type from the dropdown, select the columns you want to analyze,
                then click <em>Generate Explanation</em>. A full chart + plain-English narrative will appear below.
            </p>
        </div>
    """, unsafe_allow_html=True)

    chart_options = ["Distribution / Histogram", "Correlation / Scatter", "Time Series Trend",
                     "Category Bar Ranking", "Category Treemap"]
    selected_chart = st.selectbox("Select chart type to explain", chart_options, key="nlp_chart_select")

    explanation = ""
    fig = None

    # ── Distribution
    if selected_chart == "Distribution / Histogram":
        if not num_cols:
            st.warning("No numeric columns found.")
            return
        col = st.selectbox("Select column", num_cols, key="nlp_dist_col")
        bins = st.slider("Number of bins", 5, 60, 20, key="nlp_bins")
        if st.button("Generate Explanation", key="nlp_dist_btn"):
            s = df[col].dropna()
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=s, nbinsx=bins,
                marker=dict(color="rgba(106,90,205,0.7)", line=dict(color='rgba(157,114,255,0.9)', width=1))))
            # Mean line
            fig.add_vline(x=s.mean(), line_dash="dash", line_color="#FFD700", annotation_text=f"Mean: {s.mean():,.2f}")
            fig.add_vline(x=s.median(), line_dash="dot", line_color="#64FFDA", annotation_text=f"Median: {s.median():,.2f}")
            fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(40,42,54,0.8)",
                paper_bgcolor="rgba(40,42,54,0)", margin=dict(l=20,r=20,t=50,b=20),
                title={"text": f"Distribution of {col}", "x":0.5, "font":{"color":"#9D72FF","size":18}},
                xaxis_title=col, yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
            explanation = nlp_explain_chart("distribution", series=s, col=col)
            render_nlp_panel(explanation)
            # Stats
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Mean", f"{s.mean():,.2f}")
            c2.metric("Median", f"{s.median():,.2f}")
            c3.metric("Std Dev", f"{s.std():,.2f}")
            c4.metric("Skewness", f"{s.skew():,.2f}")

    # ── Correlation
    elif selected_chart == "Correlation / Scatter":
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
            return
        c1, c2 = st.columns(2)
        with c1: col  = st.selectbox("X-axis column", num_cols, key="nlp_corr_x")
        with c2: col2 = st.selectbox("Y-axis column", [c for c in num_cols if c != col], key="nlp_corr_y")
        color_by = "None"
        if cat_cols:
            color_by = st.selectbox("Color points by (optional)", ["None"] + cat_cols, key="nlp_corr_color")
        if st.button("Generate Explanation", key="nlp_corr_btn"):
            fig = px.scatter(df, x=col, y=col2,
                color=color_by if color_by != "None" else None,
                trendline="ols",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                trendline_color_override="#FFD700",
                opacity=0.7)
            corr = df[[col, col2]].dropna().corr().iloc[0,1]
            fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(40,42,54,0.8)",
                paper_bgcolor="rgba(40,42,54,0)", margin=dict(l=20,r=20,t=50,b=20),
                title={"text": f"Correlation: {col} vs {col2} (r = {corr:.2f})",
                       "x":0.5, "font":{"color":"#9D72FF","size":18}})
            st.plotly_chart(fig, use_container_width=True)
            explanation = nlp_explain_chart("correlation", df=df, col=col, col2=col2)
            render_nlp_panel(explanation)
            st.metric("Pearson r", f"{corr:.3f}")

    # ── Time Series
    elif selected_chart == "Time Series Trend":
        if not date_cols:
            st.info("No date/time columns detected in your dataset.")
            return
        if not num_cols:
            st.warning("No numeric columns found.")
            return
        c1, c2, c3 = st.columns(3)
        with c1: date_col = st.selectbox("Date column", date_cols, key="nlp_ts_date")
        with c2: col = st.selectbox("Metric to track", num_cols, key="nlp_ts_col")
        with c3: period = st.selectbox("Group by", ["Day", "Week", "Month", "Quarter", "Year"], index=2, key="nlp_ts_period")
        freq_map = {"Day": "D", "Week": "W", "Month": "ME", "Quarter": "QE", "Year": "YE"}
        if st.button("Generate Explanation", key="nlp_ts_btn"):
            tmp = df[[date_col, col]].copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
            tmp = tmp.dropna().sort_values(date_col)
            agg = tmp.groupby(pd.Grouper(key=date_col, freq=freq_map[period]))[col].sum().reset_index().dropna()
            if len(agg) < 2:
                st.warning("Not enough data points for time series.")
                return
            win = min(3, len(agg)-1)
            agg['MA'] = agg[col].rolling(win, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=agg[date_col], y=agg[col], mode='lines+markers', name=col,
                line=dict(color='#9D72FF', width=2), marker=dict(size=7)))
            fig.add_trace(go.Scatter(x=agg[date_col], y=agg['MA'], mode='lines', name=f'{win}-Period MA',
                line=dict(color='#FFD700', width=2.5, dash='dot')))
            fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(40,42,54,0.8)",
                paper_bgcolor="rgba(40,42,54,0)", margin=dict(l=20,r=20,t=50,b=20),
                title={"text": f"{period} Trend — {col}", "x":0.5, "font":{"color":"#9D72FF","size":18}},
                legend=dict(orientation='h', y=1.05))
            st.plotly_chart(fig, use_container_width=True)
            explanation = nlp_explain_chart("timeseries", series=agg[col], col=col)
            render_nlp_panel(explanation)

    # ── Category Bar
    elif selected_chart == "Category Bar Ranking":
        if not cat_cols or not num_cols:
            st.warning("Need both categorical and numeric columns.")
            return
        c1, c2, c3 = st.columns(3)
        with c1: col  = st.selectbox("Category column", cat_cols, key="nlp_bar_cat")
        with c2: col2 = st.selectbox("Metric", num_cols, key="nlp_bar_num")
        with c3: agg_f = st.selectbox("Aggregation", ["Sum", "Mean", "Count", "Median"], key="nlp_bar_agg")
        top_n = st.slider("Show top N categories", 3, 30, 10, key="nlp_bar_n")
        agg_map = {"Sum":"sum","Mean":"mean","Count":"count","Median":"median"}
        if st.button("Generate Explanation", key="nlp_bar_btn"):
            grouped = df.groupby(col)[col2].agg(agg_map[agg_f]).reset_index()
            grouped.columns = [col, col2]
            grouped = grouped.sort_values(col2, ascending=False).head(top_n)
            fig = go.Figure(go.Bar(
                y=grouped[col], x=grouped[col2], orientation='h',
                marker=dict(color=grouped[col2], colorscale='Viridis', showscale=True),
                text=grouped[col2].apply(lambda v: f"{v:,.1f}"), textposition='outside'
            ))
            fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(40,42,54,0.8)",
                paper_bgcolor="rgba(40,42,54,0)", margin=dict(l=20,r=80,t=50,b=20),
                yaxis=dict(categoryorder='total ascending'),
                title={"text": f"Top {top_n}: {agg_f} of {col2} by {col}",
                       "x":0.5, "font":{"color":"#9D72FF","size":18}},
                height=max(350, top_n * 44))
            st.plotly_chart(fig, use_container_width=True)
            s = grouped.set_index(col)[col2]
            explanation = nlp_explain_chart("sector_bar", series=s, col=col, col2=col2)
            render_nlp_panel(explanation)

    # ── Treemap
    elif selected_chart == "Category Treemap":
        if not cat_cols or not num_cols:
            st.warning("Need both categorical and numeric columns.")
            return
        c1, c2 = st.columns(2)
        with c1: col  = st.selectbox("Category column", cat_cols, key="nlp_tree_cat")
        with c2: col2 = st.selectbox("Metric", num_cols, key="nlp_tree_num")
        if st.button("Generate Explanation", key="nlp_tree_btn"):
            grouped = df.groupby(col)[col2].sum().reset_index()
            grouped.columns = [col, col2]
            fig = px.treemap(grouped, path=[col], values=col2, color=col2,
                color_continuous_scale="Viridis",
                title=f"Treemap — {col2} by {col}")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(40,42,54,0)",
                margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)
            s = grouped.set_index(col)[col2]
            explanation = nlp_explain_chart("sector_treemap", series=s, col=col, col2=col2)
            render_nlp_panel(explanation)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTOR ANALYSIS PAGE
# ═══════════════════════════════════════════════════════════════════════════════

SECTOR_TEMPLATES = {
    "Healthcare / Pharma": {
        "description": "Analyze patient data, drug sales, hospital metrics, or clinical trial results.",
        "typical_segments": ["Department", "Drug", "Condition", "Region", "Age Group", "Insurance Type"],
        "typical_metrics": ["Revenue", "Patients", "Cost", "Recovery Rate", "Readmission Rate", "Units Sold"],
        "guide": "For healthcare data, focus on cost-per-patient ratios and regional performance gaps. High variability in recovery rates often signals quality differences between departments."
    },
    "Retail / E-Commerce": {
        "description": "Analyze product sales, store performance, customer segments, and inventory.",
        "typical_segments": ["Product Category", "Store", "Region", "Customer Segment", "Channel"],
        "typical_metrics": ["Revenue", "Units Sold", "Profit", "Returns", "Margin", "Basket Size"],
        "guide": "Look for category concentration — if one product line drives 60%+ of revenue, you're exposed to risk. Compare online vs in-store channels."
    },
    "Finance / Banking": {
        "description": "Analyze loan portfolios, transaction volumes, account performance, or risk metrics.",
        "typical_segments": ["Loan Type", "Branch", "Customer Tier", "Region", "Product"],
        "typical_metrics": ["Balance", "Interest Income", "Defaults", "Transactions", "Fee Revenue"],
        "guide": "Watch for concentration risk in loan types. High default rates in specific segments warrant deeper investigation."
    },
    "Stock / Investments": {
        "description": "Analyze portfolio performance, sector allocation, returns, and risk metrics.",
        "typical_segments": ["Sector", "Asset Class", "Exchange", "Market Cap Tier", "Geography"],
        "typical_metrics": ["Return", "Volume", "Price", "P/E Ratio", "Volatility", "Dividend Yield"],
        "guide": "Diversification matters — check if returns are too concentrated in one sector. Compare risk-adjusted returns across asset classes."
    },
    "Manufacturing / Supply Chain": {
        "description": "Analyze production output, defect rates, supplier performance, and logistics.",
        "typical_segments": ["Plant", "Product Line", "Supplier", "Shift", "Region"],
        "typical_metrics": ["Output", "Defects", "Cost", "Downtime", "Lead Time", "Yield"],
        "guide": "Focus on defect rates by production line and supplier — small quality differences compound at scale."
    },
    "Education": {
        "description": "Analyze student performance, course outcomes, faculty metrics, or enrollment data.",
        "typical_segments": ["Department", "Course", "Grade Level", "Campus", "Student Cohort"],
        "typical_metrics": ["Score", "Enrollment", "Pass Rate", "Attendance", "Graduation Rate"],
        "guide": "Look for performance gaps between departments or demographic groups — these often indicate resource allocation issues."
    },
    "General / Other": {
        "description": "Works with any dataset. The tool auto-detects your categories and metrics.",
        "typical_segments": ["Any categorical column"],
        "typical_metrics": ["Any numeric column"],
        "guide": "Pick the categorical column that represents your main grouping (e.g. Region, Type, Category) and the numeric column that measures performance."
    }
}

def render_sector_analysis(df):
    st.markdown("""
        <div style='padding:20px 0 10px 0;'>
            <h1 style='color:#9D72FF; margin:0; font-size:32px;'>Sector Analysis</h1>
            <p style='color:#B19CD9; margin:6px 0 0 0; font-size:15px;'>
                Compare performance across categories, segments, and sectors with interactive visualizations.
            </p>
        </div>
    """, unsafe_allow_html=True)

    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    num_cols = df.select_dtypes(include='number').columns.tolist()

    if not cat_cols:
        st.warning("No categorical columns found. Sector analysis needs at least one text/category column (e.g. Region, Product, Department).")
        return
    if not num_cols:
        st.warning("No numeric columns found. Please upload a dataset with measurable metrics.")
        return

    # ── STEP 1: Sector Template Guide
    st.markdown("""
        <div style='background:rgba(106,90,205,0.10); border-radius:10px;
                    border-left:4px solid #9D72FF; padding:14px 18px; margin:0 0 18px 0;'>
            <p style='color:#9D72FF; font-weight:700; margin:0 0 4px 0; font-size:14px;'>
                STEP 1 — Choose your industry template
            </p>
            <p style='color:#B19CD9; margin:0; font-size:13px;'>
                Select the industry that best matches your data. This provides context-aware guidance throughout the analysis.
            </p>
        </div>
    """, unsafe_allow_html=True)

    sector_template = st.selectbox("Industry / Domain", list(SECTOR_TEMPLATES.keys()), key="sa_template")
    tmpl = SECTOR_TEMPLATES[sector_template]

    # Show template info
    st.markdown(f"""
        <div style='background:rgba(40,42,54,0.7); border-radius:8px; padding:14px 18px; margin-bottom:18px;'>
            <p style='color:#E0E0FF; margin:0 0 6px 0; font-size:14px;'>{tmpl['description']}</p>
            <p style='color:#00BCD4; margin:0; font-size:13px;'>
                <strong>Analyst tip:</strong> {tmpl['guide']}
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ── STEP 2: Column Selection
    st.markdown("""
        <div style='background:rgba(106,90,205,0.10); border-radius:10px;
                    border-left:4px solid #9D72FF; padding:14px 18px; margin:0 0 18px 0;'>
            <p style='color:#9D72FF; font-weight:700; margin:0 0 4px 0; font-size:14px;'>
                STEP 2 — Select columns and aggregation
            </p>
        </div>
    """, unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        sector_col = st.selectbox("Category / Sector column", cat_cols, key="sa_cat",
            help="This is the column that defines your groups — e.g. Region, Product, Department")
    with sc2:
        metric_col = st.selectbox("Metric to measure", num_cols, key="sa_metric",
            help="This is the number you want to compare across groups — e.g. Revenue, Units, Score")
    with sc3:
        agg_choice = st.selectbox("Aggregation method", ["Sum", "Mean", "Count", "Median", "Max"], key="sa_agg",
            help="Sum = total per group | Mean = average per group | Count = how many records per group")

    agg_map = {"Sum":"sum","Mean":"mean","Count":"count","Median":"median","Max":"max"}
    grouped = df.groupby(sector_col)[metric_col].agg(agg_map[agg_choice]).reset_index()
    grouped.columns = [sector_col, metric_col]
    grouped = grouped.sort_values(metric_col, ascending=False).reset_index(drop=True)
    total = grouped[metric_col].sum()

    # ── KPI Row
    st.markdown("<br>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""<div class='kpi-card kpi-green'>
            <h5 style='margin:0;color:#B19CD9;font-size:12px;text-transform:uppercase;letter-spacing:1px;'>Top Performer</h5>
            <h3 style='margin:8px 0 4px 0;color:#E0E0FF;font-size:18px;'>{grouped[sector_col].iloc[0]}</h3>
            <p style='margin:0;color:#64FFDA;font-size:13px;'>{grouped[metric_col].iloc[0]:,.2f}</p>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class='kpi-card kpi-red'>
            <h5 style='margin:0;color:#B19CD9;font-size:12px;text-transform:uppercase;letter-spacing:1px;'>Bottom Performer</h5>
            <h3 style='margin:8px 0 4px 0;color:#E0E0FF;font-size:18px;'>{grouped[sector_col].iloc[-1]}</h3>
            <p style='margin:0;color:#FF5370;font-size:13px;'>{grouped[metric_col].iloc[-1]:,.2f}</p>
        </div>""", unsafe_allow_html=True)
    with k3:
        top_share = grouped[metric_col].iloc[0] / (total + 1e-9) * 100
        st.markdown(f"""<div class='kpi-card kpi-yellow'>
            <h5 style='margin:0;color:#B19CD9;font-size:12px;text-transform:uppercase;letter-spacing:1px;'>Top Share</h5>
            <h3 style='margin:8px 0 4px 0;color:#E0E0FF;font-size:18px;'>{top_share:.1f}%</h3>
            <p style='margin:0;color:#FFC107;font-size:13px;'>of total {metric_col}</p>
        </div>""", unsafe_allow_html=True)
    with k4:
        avg_val = grouped[metric_col].mean()
        st.markdown(f"""<div class='kpi-card kpi-yellow'>
            <h5 style='margin:0;color:#B19CD9;font-size:12px;text-transform:uppercase;letter-spacing:1px;'>Group Average</h5>
            <h3 style='margin:8px 0 4px 0;color:#E0E0FF;font-size:18px;'>{avg_val:,.2f}</h3>
            <p style='margin:0;color:#FFC107;font-size:13px;'>mean across all sectors</p>
        </div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(f"""<div class='kpi-card kpi-yellow'>
            <h5 style='margin:0;color:#B19CD9;font-size:12px;text-transform:uppercase;letter-spacing:1px;'>Total Sectors</h5>
            <h3 style='margin:8px 0 4px 0;color:#E0E0FF;font-size:18px;'>{len(grouped)}</h3>
            <p style='margin:0;color:#FFC107;font-size:13px;'>unique categories</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs
    s_tabs = st.tabs(["Bar Rankings", "Treemap + Pie", "Multi-Metric Radar", "Sector vs Sector", "Ranking Table"])

    with s_tabs[0]:
        top_n = st.slider("Show top N sectors", 3, min(30, len(grouped)), min(10, len(grouped)), key="sa_topn")
        disp = grouped.head(top_n)
        bar_fig = go.Figure(go.Bar(
            y=disp[sector_col], x=disp[metric_col], orientation='h',
            marker=dict(color=disp[metric_col], colorscale='Viridis', showscale=True),
            text=disp[metric_col].apply(lambda v: f"{v:,.1f}"), textposition='outside'
        ))
        # Average line
        bar_fig.add_vline(x=avg_val, line_dash="dash", line_color="#FFD700",
            annotation_text=f"Avg: {avg_val:,.1f}", annotation_font_color="#FFD700")
        bar_fig.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(40,42,54,0.8)", paper_bgcolor="rgba(40,42,54,0)",
            margin=dict(l=20,r=100,t=50,b=20), yaxis=dict(categoryorder='total ascending'),
            title={"text": f"{agg_choice} of {metric_col} by {sector_col}",
                   "x":0.5, "font":{"color":"#9D72FF","size":18}},
            height=max(380, top_n * 46))
        st.plotly_chart(bar_fig, use_container_width=True)
        s = grouped.set_index(sector_col)[metric_col]
        render_nlp_panel(nlp_explain_chart("sector_bar", series=s, col=sector_col, col2=metric_col))

    with s_tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            treemap_fig = px.treemap(grouped, path=[sector_col], values=metric_col,
                color=metric_col, color_continuous_scale="Viridis")
            treemap_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(40,42,54,0)",
                margin=dict(l=10,r=10,t=50,b=10),
                title={"text":f"Treemap — {metric_col}", "x":0.5, "font":{"color":"#9D72FF","size":16}})
            st.plotly_chart(treemap_fig, use_container_width=True)
        with c2:
            pie_data = grouped.head(12)
            pie_fig = px.pie(pie_data, names=sector_col, values=metric_col, hole=0.42,
                color_discrete_sequence=px.colors.sequential.Plasma_r)
            pie_fig.update_traces(textposition='inside', textinfo='percent+label')
            pie_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(40,42,54,0)",
                margin=dict(l=10,r=10,t=50,b=10),
                title={"text":f"Share of {metric_col}", "x":0.5, "font":{"color":"#9D72FF","size":16}})
            st.plotly_chart(pie_fig, use_container_width=True)
        s = grouped.set_index(sector_col)[metric_col]
        render_nlp_panel(nlp_explain_chart("sector_treemap", series=s, col=sector_col, col2=metric_col))

    with s_tabs[2]:
        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns for multi-metric radar comparison.")
        else:
            multi_metrics = st.multiselect("Select 2–6 metrics to compare across sectors",
                num_cols, default=num_cols[:min(4, len(num_cols))], key="sa_radar_metrics")
            max_sectors = st.slider("Max sectors to show on radar", 2, min(12, len(grouped)), min(6, len(grouped)), key="sa_radar_n")
            if multi_metrics and len(multi_metrics) >= 2:
                multi_df = df.groupby(sector_col)[multi_metrics].agg(agg_map[agg_choice]).reset_index()
                multi_df = multi_df.head(max_sectors)
                radar_fig = go.Figure()
                for _, row in multi_df.iterrows():
                    normalized = [(row[m] / (multi_df[m].max() + 1e-9)) * 100 for m in multi_metrics]
                    radar_fig.add_trace(go.Scatterpolar(
                        r=normalized + [normalized[0]],
                        theta=multi_metrics + [multi_metrics[0]],
                        fill='toself', name=str(row[sector_col]), opacity=0.65
                    ))
                radar_fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                    template="plotly_dark", paper_bgcolor="rgba(40,42,54,0)",
                    title={"text": "Multi-Metric Comparison (Normalized 0–100)",
                           "x":0.5, "font":{"color":"#9D72FF","size":16}},
                    margin=dict(l=30,r=30,t=60,b=30), legend=dict(orientation='h', y=-0.15)
                )
                st.plotly_chart(radar_fig, use_container_width=True)
                render_nlp_panel(nlp_explain_chart("sector_radar", extra={"metrics": multi_metrics}))

    with s_tabs[3]:
        if len(grouped) < 2:
            st.info("Need at least 2 sectors to compare.")
        else:
            sector_options = grouped[sector_col].tolist()
            c1, c2 = st.columns(2)
            with c1: s1 = st.selectbox("Sector A", sector_options, index=0, key="sa_cmp1")
            with c2: s2 = st.selectbox("Sector B", sector_options, index=min(1, len(sector_options)-1), key="sa_cmp2")
            compare_metrics = st.multiselect("Metrics to compare",
                num_cols, default=num_cols[:min(5, len(num_cols))], key="sa_cmp_metrics")
            if compare_metrics and s1 != s2:
                row1 = df[df[sector_col] == s1][compare_metrics].agg(agg_map[agg_choice])
                row2 = df[df[sector_col] == s2][compare_metrics].agg(agg_map[agg_choice])
                cmp_df = pd.DataFrame({s1: row1, s2: row2}).T.reset_index()
                cmp_df.columns = ['Sector'] + compare_metrics
                # Grouped bar
                melted = cmp_df.melt(id_vars='Sector', var_name='Metric', value_name='Value')
                cmp_fig = px.bar(melted, x='Metric', y='Value', color='Sector',
                    barmode='group', color_discrete_sequence=['#9D72FF','#64FFDA'],
                    title=f"{s1} vs {s2} — Head-to-Head Comparison")
                cmp_fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(40,42,54,0.8)",
                    paper_bgcolor="rgba(40,42,54,0)", margin=dict(l=20,r=20,t=60,b=20))
                st.plotly_chart(cmp_fig, use_container_width=True)
                # Table
                diff_row = {}
                for m in compare_metrics:
                    v1, v2 = row1[m], row2[m]
                    pct = ((v1 - v2) / (abs(v2) + 1e-9)) * 100
                    diff_row[m] = f"{'+' if pct > 0 else ''}{pct:.1f}%"
                diff_df = pd.DataFrame([{sector_col: f"{s1} vs {s2} (% diff)"} | diff_row])
                final_table = pd.concat([cmp_df, diff_df], ignore_index=True)
                st.dataframe(final_table, use_container_width=True)

    with s_tabs[4]:
        display = grouped.copy()
        display.insert(0, "Rank", range(1, len(display)+1))
        display["Share %"] = (display[metric_col] / (total + 1e-9) * 100).round(2)
        display["vs Average"] = ((display[metric_col] - avg_val) / (avg_val + 1e-9) * 100).round(1)
        display["vs Average"] = display["vs Average"].apply(lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%")
        st.dataframe(display, use_container_width=True)
        render_nlp_panel(
            f"This ranking table shows all {len(grouped)} sectors sorted by {agg_choice.lower()} {metric_col}. "
            f"<strong>Share %</strong> shows each sector's contribution to the total. "
            f"<strong>vs Average</strong> shows how far above or below the group average each sector sits — "
            f"positive = outperforming, negative = underperforming. "
            f"Sectors with negative 'vs Average' scores are candidates for deeper investigation or resource reallocation."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ML FORECASTING PAGE
# ═══════════════════════════════════════════════════════════════════════════════
def render_ml_forecasting(df):
    st.markdown("""
        <div style='padding:20px 0 10px 0;'>
            <h1 style='color:#9D72FF; margin:0; font-size:32px;'>ML Predictive Forecasting</h1>
            <p style='color:#B19CD9; margin:6px 0 0 0; font-size:15px;'>
                Train machine learning models on your data and forecast future values or predict custom inputs.
            </p>
        </div>
    """, unsafe_allow_html=True)

    num_cols  = df.select_dtypes(include='number').columns.tolist()
    cat_cols  = df.select_dtypes(include='object').columns.tolist()
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]

    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns to train a forecasting model.")
        return

    ml_tabs = st.tabs(["Model Guide", "Time-Series Forecast", "Regression Predictor", "Custom Prediction"])

    # ── TAB 0: Model Guide
    with ml_tabs[0]:
        st.markdown("<h4 style='color:#B19CD9;'>Which model should I use?</h4>", unsafe_allow_html=True)
        st.markdown("""
            <div style='background:rgba(40,42,54,0.7); border-radius:10px; padding:20px; margin-bottom:16px;'>
                <p style='color:#E0E0FF; font-size:14px; line-height:1.8; margin:0;'>
                    This guide helps you pick the right approach for your data.
                    Read the descriptions below and match them to your situation.
                </p>
            </div>
        """, unsafe_allow_html=True)

        models_info = [
            {
                "name": "Time-Series Forecast",
                "when": "When your data has a date/time column and you want to predict future values over time.",
                "example": "Forecasting next month's revenue, predicting weekly sales for the next 4 weeks.",
                "requirements": "A date column + at least one numeric metric. Works best with 10+ time periods.",
                "models": "Linear Regression (best for clear trends), Gradient Boosting (best for complex patterns), Random Forest (best for noisy data)",
                "color": "#9D72FF"
            },
            {
                "name": "Regression Predictor",
                "when": "When you want to understand which factors influence an outcome, and predict that outcome from new inputs.",
                "example": "Predicting customer lifetime value from purchase history, estimating price from features.",
                "requirements": "Multiple numeric columns. One column is your 'target' (what you want to predict); others are 'features' (inputs).",
                "models": "Random Forest (most accurate, handles non-linear patterns), Gradient Boosting (very accurate, slower), Linear Regression (fastest, best for linear relationships, most interpretable)",
                "color": "#64FFDA"
            }
        ]
        for m in models_info:
            st.markdown(f"""
                <div style='background:rgba(40,42,54,0.8); border-radius:10px;
                            border-left:4px solid {m['color']}; padding:18px; margin-bottom:14px;'>
                    <h4 style='color:{m['color']}; margin:0 0 10px 0;'>{m['name']}</h4>
                    <p style='color:#E0E0FF; margin:0 0 6px 0; font-size:14px;'><strong>Use when:</strong> {m['when']}</p>
                    <p style='color:#B19CD9; margin:0 0 6px 0; font-size:13px;'><strong>Example:</strong> {m['example']}</p>
                    <p style='color:#B19CD9; margin:0 0 6px 0; font-size:13px;'><strong>Data requirements:</strong> {m['requirements']}</p>
                    <p style='color:#9D72FF; margin:0; font-size:13px;'><strong>Available models:</strong> {m['models']}</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("""
            <div style='background:rgba(0,188,212,0.08); border-radius:10px;
                        border-left:4px solid #00BCD4; padding:16px; margin-top:8px;'>
                <p style='color:#00BCD4; font-weight:700; margin:0 0 6px 0; font-size:13px;'>
                    UNDERSTANDING MODEL QUALITY METRICS
                </p>
                <p style='color:#B19CD9; font-size:13px; line-height:1.8; margin:0;'>
                    <strong style='color:#E0E0FF;'>R² Score (0 to 1):</strong>
                    How much of the variation in your target the model explains.
                    0.90+ = excellent, 0.70–0.90 = good, 0.50–0.70 = fair, below 0.50 = limited predictive power.<br>
                    <strong style='color:#E0E0FF;'>MAE (Mean Absolute Error):</strong>
                    The average gap between predicted and actual values, in the same units as your target.
                    Lower is better. If MAE = 200 and your target is Revenue in dollars, predictions are off by ~$200 on average.<br>
                    <strong style='color:#E0E0FF;'>RMSE:</strong>
                    Similar to MAE but penalizes large errors more heavily. Useful when big mistakes are especially costly.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # ── TAB 1: Time-Series
    with ml_tabs[1]:
        st.markdown("<h4 style='color:#B19CD9;'>Time-Series Forecast</h4>", unsafe_allow_html=True)
        if not date_cols:
            st.markdown("""
                <div style='background:rgba(255,193,7,0.1); border-left:4px solid #FFC107;
                            border-radius:8px; padding:14px; margin-bottom:12px;'>
                    <p style='color:#FFC107; margin:0; font-size:13px;'>
                        No date/time columns detected in your dataset.
                        This tab requires a column with dates or timestamps (e.g. "Date", "Month", "Timestamp").
                        Switch to the <strong>Regression Predictor</strong> tab to use feature-based forecasting.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Inline guide
            st.markdown("""
                <div style='background:rgba(106,90,205,0.10); border-radius:8px;
                            border-left:3px solid #9D72FF; padding:12px 16px; margin-bottom:14px;'>
                    <p style='color:#B19CD9; font-size:13px; margin:0; line-height:1.7;'>
                        <strong style='color:#9D72FF;'>How this works:</strong>
                        Select your date column and the metric you want to forecast.
                        The model learns patterns in your historical data (trends, momentum) and extrapolates them forward.
                        A confidence band shows the range of likely outcomes.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            tc1, tc2, tc3 = st.columns(3)
            with tc1: date_col = st.selectbox("Date column", date_cols, key="ts_date")
            with tc2: target_col = st.selectbox("Metric to forecast", num_cols, key="ts_target")
            with tc3:
                model_choice = st.selectbox("Model", ["Gradient Boosting", "Random Forest", "Linear Regression"], key="ts_model",
                    help="Gradient Boosting: best for complex patterns | Random Forest: best for noisy data | Linear Regression: best for clear upward/downward trends")

            tc4, tc5 = st.columns(2)
            with tc4:
                period = st.selectbox("Group data by", ["Day", "Week", "Month", "Quarter"], index=2, key="ts_period",
                    help="Choose how to aggregate your data before forecasting. Monthly is usually best for business data.")
            with tc5:
                forecast_periods = st.slider("Forecast periods ahead", 1, 30, 6, key="ts_fwd",
                    help="How many future periods to predict. Don't go too far — forecasts become less reliable farther out.")

            freq_map = {"Day":"D","Week":"W","Month":"ME","Quarter":"QE"}

            if st.button("Run Forecast", key="ts_run"):
                with st.spinner("Training model and generating forecast..."):
                    try:
                        tmp = df[[date_col, target_col]].copy()
                        tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
                        tmp = tmp.dropna().sort_values(date_col)
                        agg = tmp.groupby(pd.Grouper(key=date_col, freq=freq_map[period]))[target_col].sum().reset_index().dropna()

                        if len(agg) < 6:
                            st.warning(f"Only {len(agg)} {period.lower()} periods found. Need at least 6 for a reliable forecast. Try switching to a shorter period (e.g. Day or Week).")
                        else:
                            agg['t']  = np.arange(len(agg))
                            agg['t2'] = agg['t'] ** 2
                            agg['lag1'] = agg[target_col].shift(1)
                            agg['lag2'] = agg[target_col].shift(2)
                            agg['rm3'] = agg[target_col].rolling(3, min_periods=1).mean()
                            agg = agg.dropna()
                            fcols = ['t','t2','lag1','lag2','rm3']
                            X = agg[fcols].values; y = agg[target_col].values
                            split = max(2, int(len(X)*0.8))
                            X_tr, X_te = X[:split], X[split:]; y_tr, y_te = y[:split], y[split:]

                            mdl_map = {
                                "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, random_state=42),
                                "Random Forest":     RandomForestRegressor(n_estimators=150, random_state=42),
                                "Linear Regression": Ridge(alpha=1.0)
                            }
                            mdl = mdl_map[model_choice]
                            mdl.fit(X_tr, y_tr)
                            y_hat = mdl.predict(X_te)
                            mae = mean_absolute_error(y_te, y_hat) if len(y_te) > 0 else 0
                            r2  = r2_score(y_te, y_hat) if len(y_te) > 1 else 0

                            last_vals = list(agg[target_col].values[-3:])
                            last_t = agg['t'].max()
                            preds = []
                            for s in range(1, forecast_periods+1):
                                nt = last_t+s
                                l1 = last_vals[-1]; l2 = last_vals[-2] if len(last_vals)>=2 else l1
                                rm = np.mean(last_vals[-3:]) if len(last_vals)>=3 else l1
                                p = mdl.predict(np.array([[nt,nt**2,l1,l2,rm]]))[0]
                                preds.append(p); last_vals.append(p)

                            last_date = agg[date_col].max()
                            off_map = {"Day":pd.Timedelta(days=1),"Week":pd.Timedelta(weeks=1),
                                       "Month":pd.DateOffset(months=1),"Quarter":pd.DateOffset(months=3)}
                            fdates = [last_date + off_map[period]*(i+1) for i in range(forecast_periods)]
                            std_e = np.std(y_te - y_hat) if len(y_te)>1 else agg[target_col].std()*0.15

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=agg[date_col], y=agg[target_col],
                                mode='lines+markers', name='Historical',
                                line=dict(color='#9D72FF',width=2), marker=dict(size=6)))
                            if len(y_te)>0:
                                fig.add_trace(go.Scatter(x=agg[date_col].iloc[split:], y=y_hat,
                                    mode='lines+markers', name='Model Fit on Test Data',
                                    line=dict(color='#FFD700',width=2,dash='dot'), marker=dict(size=5)))
                            fig.add_trace(go.Scatter(x=fdates, y=preds,
                                mode='lines+markers', name='Forecast',
                                line=dict(color='#64FFDA',width=3), marker=dict(size=8,symbol='diamond')))
                            fig.add_trace(go.Scatter(
                                x=fdates+fdates[::-1],
                                y=[v+std_e*1.5 for v in preds]+[v-std_e*1.5 for v in preds][::-1],
                                fill='toself', fillcolor='rgba(100,255,218,0.08)',
                                line=dict(color='rgba(0,0,0,0)'), name='Confidence Band'))
                            fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(40,42,54,0.8)",
                                paper_bgcolor="rgba(40,42,54,0)",
                                title={"text":f"{model_choice} — {period} Forecast of {target_col}",
                                       "x":0.5,"font":{"color":"#9D72FF","size":18}},
                                margin=dict(l=20,r=20,t=60,b=20),
                                legend=dict(orientation='h',y=1.05))
                            st.plotly_chart(fig, use_container_width=True)

                            mc1,mc2,mc3 = st.columns(3)
                            mc1.metric("R² Score", f"{r2:.3f}")
                            mc2.metric("MAE", f"{mae:,.2f}")
                            mc3.metric("Forecast Periods", f"{forecast_periods}")

                            render_nlp_panel(nlp_explain_chart("forecast", col=target_col, extra={
                                "model": model_choice, "r2": r2, "mae": mae, "periods": forecast_periods,
                                "last_actual": float(agg[target_col].iloc[-1]),
                                "last_forecast": float(preds[-1])
                            }))

                            with st.expander("View Forecast Table"):
                                fc_df = pd.DataFrame({"Period": fdates,
                                    f"Forecast ({target_col})": [round(v,2) for v in preds]})
                                st.dataframe(fc_df, use_container_width=True)

                    except Exception as e:
                        st.error(f"Forecasting error: {str(e)}")

    # ── TAB 2: Regression Predictor
    with ml_tabs[2]:
        st.markdown("<h4 style='color:#B19CD9;'>Regression Predictor</h4>", unsafe_allow_html=True)
        st.markdown("""
            <div style='background:rgba(106,90,205,0.10); border-radius:8px;
                        border-left:3px solid #9D72FF; padding:12px 16px; margin-bottom:14px;'>
                <p style='color:#B19CD9; font-size:13px; margin:0; line-height:1.7;'>
                    <strong style='color:#9D72FF;'>How this works:</strong>
                    Choose a <strong>target variable</strong> (the value you want to predict) and
                    <strong>feature columns</strong> (the inputs the model will use to make predictions).
                    The model trains on 80% of your data and reports accuracy on the remaining 20%.
                    Once trained, you can enter custom inputs on the <strong>Custom Prediction</strong> tab.
                </p>
            </div>
        """, unsafe_allow_html=True)

        rc1, rc2 = st.columns(2)
        with rc1:
            target_reg = st.selectbox("Target variable (what to predict)", num_cols, key="reg_target",
                help="This is the outcome you want to predict — e.g. Revenue, Score, Price")
        with rc2:
            feat_options = [c for c in num_cols if c != target_reg]
            sel_feats = st.multiselect("Feature columns (model inputs)", feat_options,
                default=feat_options[:min(5, len(feat_options))], key="reg_feats",
                help="These are the variables the model uses to predict the target. More relevant features = better predictions.")

        reg_model = st.selectbox("Model", ["Random Forest", "Gradient Boosting", "Linear Regression"], key="reg_model",
            help="Random Forest: handles complex patterns well | Gradient Boosting: usually most accurate | Linear Regression: simple, fast, interpretable")
        test_pct = st.slider("Test set size (%)", 10, 40, 20, key="reg_test",
            help="The portion of data held back for testing accuracy. 20% is the standard.")

        if st.button("Train Model", key="reg_run"):
            if not sel_feats:
                st.warning("Please select at least one feature column.")
            else:
                with st.spinner("Training model..."):
                    try:
                        clean = df[sel_feats + [target_reg]].dropna()
                        X = clean[sel_feats].values
                        y = clean[target_reg].values
                        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_pct/100, random_state=42)
                        sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

                        mdl_map = {
                            "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42),
                            "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
                            "Linear Regression": Ridge(alpha=1.0)
                        }
                        mdl = mdl_map[reg_model]
                        mdl.fit(X_tr_s, y_tr)
                        y_hat  = mdl.predict(X_te_s)
                        mae    = mean_absolute_error(y_te, y_hat)
                        rmse   = np.sqrt(mean_squared_error(y_te, y_hat))
                        r2     = r2_score(y_te, y_hat)

                        st.session_state['ml_model']    = mdl
                        st.session_state['ml_scaler']   = sc
                        st.session_state['ml_features'] = sel_feats
                        st.session_state['ml_target']   = target_reg
                        st.session_state['ml_df']       = df

                        ap = go.Figure()
                        ap.add_trace(go.Scatter(x=y_te, y=y_hat, mode='markers',
                            marker=dict(color='#9D72FF',size=8,opacity=0.7), name='Predicted vs Actual'))
                        ln = [min(y_te.min(),y_hat.min()), max(y_te.max(),y_hat.max())]
                        ap.add_trace(go.Scatter(x=ln,y=ln,mode='lines',
                            line=dict(color='#64FFDA',dash='dash',width=2),name='Perfect Prediction Line'))
                        ap.update_layout(template="plotly_dark", plot_bgcolor="rgba(40,42,54,0.8)",
                            paper_bgcolor="rgba(40,42,54,0)",
                            title={"text":"Actual vs Predicted Values (Test Set)","x":0.5,"font":{"color":"#9D72FF","size":16}},
                            xaxis_title="Actual", yaxis_title="Predicted", margin=dict(l=20,r=20,t=60,b=20))
                        st.plotly_chart(ap, use_container_width=True)

                        rc1,rc2,rc3 = st.columns(3)
                        rc1.metric("R² Score", f"{r2:.3f}")
                        rc2.metric("MAE", f"{mae:,.2f}")
                        rc3.metric("RMSE", f"{rmse:,.2f}")

                        render_nlp_panel(nlp_explain_chart("regression_actual_vs_pred",
                            extra={"model":reg_model,"r2":r2,"mae":mae,"rmse":rmse}))

                        if hasattr(mdl, 'feature_importances_'):
                            fi_df = pd.DataFrame({'Feature':sel_feats,'Importance':mdl.feature_importances_}).sort_values('Importance')
                            fi = go.Figure(go.Bar(y=fi_df['Feature'],x=fi_df['Importance'],orientation='h',
                                marker=dict(color=fi_df['Importance'],colorscale='Viridis',showscale=False)))
                            fi.update_layout(template="plotly_dark",plot_bgcolor="rgba(40,42,54,0.8)",
                                paper_bgcolor="rgba(40,42,54,0)",
                                title={"text":"Feature Importance","x":0.5,"font":{"color":"#9D72FF","size":16}},
                                margin=dict(l=20,r=20,t=50,b=20))
                            st.plotly_chart(fi, use_container_width=True)
                            top_f = fi_df.iloc[-1]
                            render_nlp_panel(nlp_explain_chart("feature_importance",
                                extra={"top_feature":top_f['Feature'],"top_importance":top_f['Importance']}))

                        st.success("Model trained. Go to Custom Prediction to predict new values.")

                    except Exception as e:
                        st.error(f"Training error: {str(e)}")

    # ── TAB 3: Custom Prediction
    with ml_tabs[3]:
        st.markdown("<h4 style='color:#B19CD9;'>Custom Prediction</h4>", unsafe_allow_html=True)
        if 'ml_model' not in st.session_state:
            st.markdown("""
                <div style='background:rgba(106,90,205,0.10); border-radius:8px;
                            border-left:3px solid #9D72FF; padding:14px 18px;'>
                    <p style='color:#B19CD9; font-size:14px; margin:0;'>
                        No model trained yet. Go to the <strong>Regression Predictor</strong> tab,
                        select your target variable and features, then click <strong>Train Model</strong>.
                        Once trained, come back here to enter custom values and get predictions.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            mdl      = st.session_state['ml_model']
            sc       = st.session_state['ml_scaler']
            features = st.session_state['ml_features']
            target   = st.session_state['ml_target']
            src_df   = st.session_state.get('ml_df', df)

            st.markdown(f"""
                <div style='background:rgba(40,42,54,0.7); border-radius:8px; padding:14px 18px; margin-bottom:16px;'>
                    <p style='color:#B19CD9; font-size:13px; margin:0;'>
                        Model: <strong style='color:#9D72FF;'>{mdl.__class__.__name__}</strong> —
                        Predicting: <strong style='color:#64FFDA;'>{target}</strong><br>
                        Enter the values below and click <strong>Predict</strong>.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            input_vals = {}
            n = min(len(features), 3)
            cols = st.columns(n)
            for i, feat in enumerate(features):
                with cols[i % n]:
                    if feat in src_df.columns:
                        mn = float(src_df[feat].min())
                        mx = float(src_df[feat].max())
                        av = float(src_df[feat].mean())
                    else:
                        mn, mx, av = 0.0, 100.0, 50.0
                    input_vals[feat] = st.number_input(feat, value=round(av,2),
                        min_value=mn*0.5, max_value=mx*2.0, key=f"cp_{feat}",
                        help=f"Dataset range: {mn:,.2f} – {mx:,.2f} | Average: {av:,.2f}")

            if st.button("Predict", key="cp_run"):
                try:
                    row = np.array([[input_vals[f] for f in features]])
                    pred = mdl.predict(sc.transform(row))[0]
                    st.markdown(f"""
                        <div style='background:linear-gradient(135deg,rgba(100,255,218,0.12),rgba(106,90,205,0.12));
                                    border-left:4px solid #64FFDA; border-radius:12px;
                                    padding:24px; margin:20px 0; text-align:center;'>
                            <p style='color:#64FFDA; font-size:13px; letter-spacing:1px;
                                      text-transform:uppercase; margin:0;'>Predicted {target}</p>
                            <h1 style='color:#E0E0FF; font-size:56px; margin:12px 0;'>{pred:,.2f}</h1>
                            <p style='color:#B19CD9; font-size:13px; margin:0;'>
                                Using {mdl.__class__.__name__} trained on {len(src_df)} records
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    render_nlp_panel(
                        f"Based on the values you entered, the <strong>{mdl.__class__.__name__}</strong> model "
                        f"predicts <strong>{target}</strong> = <strong>{pred:,.2f}</strong>. "
                        f"This estimate comes from patterns learned across {len(src_df)} training records. "
                        f"To explore different scenarios, adjust the input values above and click Predict again. "
                        f"The prediction is most reliable when your input values fall within the dataset's historical range."
                    )
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# INTELLIGENCE REPORT ENGINE
# Full automated analysis → professional downloadable HTML report
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_to_base64(fig):
    """Convert a plotly figure to a base64 PNG string for embedding in HTML."""
    try:
        img_bytes = fig.to_image(format="png", width=900, height=420, scale=2)
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception:
        return None


def _run_full_analysis(df, report_title, analyst_name):
    """
    Runs the complete automated analysis pipeline.
    Returns a structured dict of findings used to build the HTML report.
    """
    findings = {}
    num_cols  = df.select_dtypes(include='number').columns.tolist()
    cat_cols  = df.select_dtypes(include='object').columns.tolist()
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]

    findings['meta'] = {
        'title':     report_title,
        'analyst':   analyst_name,
        'rows':      len(df),
        'cols':      len(df.columns),
        'num_cols':  len(num_cols),
        'cat_cols':  len(cat_cols),
        'date_cols': len(date_cols),
        'generated': datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        'filename':  st.session_state.get('file_name', 'Dataset')
    }

    # ── 1. Data Quality Audit
    missing_pct  = (df.isnull().sum() / len(df) * 100).round(2)
    dup_count    = df.duplicated().sum()
    quality_score = 100
    quality_issues = []
    if missing_pct.max() > 0:
        high_miss = missing_pct[missing_pct > 10]
        if not high_miss.empty:
            quality_score -= min(30, len(high_miss) * 8)
            quality_issues.append(f"{len(high_miss)} column(s) have >10% missing values: {', '.join(high_miss.index.tolist()[:4])}")
    if dup_count > 0:
        pct_dup = dup_count / len(df) * 100
        quality_score -= min(20, int(pct_dup))
        quality_issues.append(f"{dup_count} duplicate rows detected ({pct_dup:.1f}% of data)")
    if not num_cols:
        quality_score -= 15
        quality_issues.append("No numeric columns found — limited analytical capability")

    findings['quality'] = {
        'score':      max(0, quality_score),
        'missing':    missing_pct.to_dict(),
        'duplicates': int(dup_count),
        'issues':     quality_issues,
        'grade':      ('A' if quality_score >= 90 else 'B' if quality_score >= 75
                       else 'C' if quality_score >= 60 else 'D')
    }

    # ── 2. Descriptive Statistics
    if num_cols:
        desc = df[num_cols].describe().T
        desc['skewness'] = df[num_cols].skew()
        desc['kurtosis'] = df[num_cols].kurtosis()
        desc['cv']       = (desc['std'] / (desc['mean'].abs() + 1e-9) * 100).round(1)
        findings['descriptive'] = desc.round(4).to_dict()
        findings['num_cols'] = num_cols
    else:
        findings['descriptive'] = {}
        findings['num_cols'] = []

    # ── 3. Anomaly / Outlier Detection (Isolation Forest)
    anomaly_findings = {}
    if num_cols and len(df) > 20:
        try:
            clean = df[num_cols].fillna(df[num_cols].median())
            sc    = StandardScaler()
            scaled = sc.fit_transform(clean)
            iso   = IsolationForest(contamination=0.05, random_state=42)
            labels = iso.fit_predict(scaled)
            n_anom = int((labels == -1).sum())
            anom_pct = n_anom / len(df) * 100

            # Per-column outlier count using IQR
            col_outliers = {}
            for c in num_cols:
                q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
                iqr = q3 - q1
                n_out = int(((df[c] < q1 - 1.5*iqr) | (df[c] > q3 + 1.5*iqr)).sum())
                if n_out > 0:
                    col_outliers[c] = {'count': n_out, 'pct': round(n_out/len(df)*100, 1)}

            anomaly_findings = {
                'total':      n_anom,
                'pct':        round(anom_pct, 1),
                'severity':   'High' if anom_pct > 10 else 'Medium' if anom_pct > 5 else 'Low',
                'col_detail': col_outliers,
                'labels':     labels.tolist()
            }
        except Exception:
            anomaly_findings = {'total': 0, 'pct': 0, 'severity': 'Unknown', 'col_detail': {}, 'labels': []}
    findings['anomalies'] = anomaly_findings

    # ── 4. Correlation Matrix + Key Pairs
    corr_findings = {}
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        strong_pos = []
        strong_neg = []
        for col in upper.columns:
            for idx in upper.index:
                val = upper.loc[idx, col]
                if pd.notna(val):
                    if val >= 0.7:
                        strong_pos.append({'pair': f"{idx} × {col}", 'r': round(val, 3)})
                    elif val <= -0.7:
                        strong_neg.append({'pair': f"{idx} × {col}", 'r': round(val, 3)})
        corr_findings = {
            'matrix':     corr_matrix.round(3).to_dict(),
            'strong_pos': sorted(strong_pos, key=lambda x: -x['r'])[:6],
            'strong_neg': sorted(strong_neg, key=lambda x: x['r'])[:6]
        }
    findings['correlations'] = corr_findings

    # ── 5. Trend Detection (if date column exists)
    trend_findings = {}
    if date_cols and num_cols:
        try:
            dc = date_cols[0]
            tmp = df[[dc] + num_cols[:4]].copy()
            tmp[dc] = pd.to_datetime(tmp[dc], errors='coerce')
            tmp = tmp.dropna(subset=[dc]).sort_values(dc)
            date_range_days = (tmp[dc].max() - tmp[dc].min()).days
            freq = 'ME' if date_range_days > 60 else 'W'

            trends_by_col = {}
            for nc in num_cols[:4]:
                try:
                    agg = tmp.groupby(pd.Grouper(key=dc, freq=freq))[nc].sum().dropna()
                    if len(agg) >= 4:
                        t = np.arange(len(agg))
                        slope, intercept = np.polyfit(t, agg.values, 1)
                        pct_change = ((agg.iloc[-1] - agg.iloc[0]) / (abs(agg.iloc[0]) + 1e-9)) * 100
                        vol = agg.pct_change().dropna().std() * 100
                        direction = 'Growing' if slope > 0 else 'Declining'
                        trends_by_col[nc] = {
                            'slope':      round(float(slope), 4),
                            'pct_change': round(float(pct_change), 1),
                            'direction':  direction,
                            'volatility': round(float(vol), 1),
                            'peak_date':  str(agg.idxmax().date()),
                            'peak_val':   round(float(agg.max()), 2),
                            'periods':    len(agg)
                        }
                except Exception:
                    pass
            trend_findings = {'date_col': dc, 'freq': freq, 'trends': trends_by_col}
        except Exception:
            pass
    findings['trends'] = trend_findings

    # ── 6. Segment / Category Breakdown
    segment_findings = {}
    if cat_cols and num_cols:
        best_cat = None
        best_score = 0
        for cc in cat_cols:
            n_unique = df[cc].nunique()
            if 2 <= n_unique <= 20:
                score = 20 - abs(n_unique - 8)
                if score > best_score:
                    best_score = score
                    best_cat = cc
        if best_cat:
            target_num = num_cols[0]
            seg = df.groupby(best_cat)[target_num].agg(['sum','mean','count']).reset_index()
            seg.columns = [best_cat, 'Total', 'Average', 'Count']
            seg = seg.sort_values('Total', ascending=False)
            total_sum = seg['Total'].sum()
            seg['Share%'] = (seg['Total'] / (total_sum + 1e-9) * 100).round(1)
            top3_share = seg['Share%'].head(3).sum()
            segment_findings = {
                'category':   best_cat,
                'metric':     target_num,
                'table':      seg.head(10).round(2).to_dict('records'),
                'top3_share': round(top3_share, 1),
                'n_segments': len(seg),
                'top_name':   seg[best_cat].iloc[0],
                'top_val':    round(float(seg['Total'].iloc[0]), 2),
                'bot_name':   seg[best_cat].iloc[-1],
                'bot_val':    round(float(seg['Total'].iloc[-1]), 2)
            }
    findings['segments'] = segment_findings

    # ── 7. Executive Summary + Risk Flags + Recommendations (generated from findings)
    summary_bullets = []
    risk_flags = []
    recommendations = []

    # Summary
    summary_bullets.append(f"Dataset contains {findings['meta']['rows']:,} records across {findings['meta']['cols']} columns ({len(num_cols)} numeric, {len(cat_cols)} categorical).")
    if findings['quality']['score'] >= 85:
        summary_bullets.append(f"Data quality is strong (Grade {findings['quality']['grade']}, score {findings['quality']['score']}/100) — analysis results are reliable.")
    else:
        summary_bullets.append(f"Data quality needs attention (Grade {findings['quality']['grade']}, score {findings['quality']['score']}/100) — results should be interpreted cautiously.")

    if anomaly_findings.get('total', 0) > 0:
        summary_bullets.append(f"Anomaly detection flagged {anomaly_findings['total']} records ({anomaly_findings['pct']}%) as statistical outliers using Isolation Forest.")

    if trend_findings.get('trends'):
        growing = [k for k,v in trend_findings['trends'].items() if v['direction'] == 'Growing']
        declining = [k for k,v in trend_findings['trends'].items() if v['direction'] == 'Declining']
        if growing:
            summary_bullets.append(f"Positive growth trend detected in: {', '.join(growing[:3])}.")
        if declining:
            summary_bullets.append(f"Declining trend detected in: {', '.join(declining[:3])} — warrants attention.")

    if corr_findings.get('strong_pos'):
        pair = corr_findings['strong_pos'][0]
        summary_bullets.append(f"Strongest positive correlation: {pair['pair']} (r={pair['r']}) — these variables move together.")

    if segment_findings:
        summary_bullets.append(
            f"{segment_findings['top_name']} is the top-performing segment in {segment_findings['category']}, "
            f"contributing {segment_findings['top3_share']}% of total {segment_findings['metric']} across the top 3 groups."
        )

    # Risk Flags
    if findings['quality']['issues']:
        for issue in findings['quality']['issues']:
            risk_flags.append({'level': 'HIGH', 'flag': issue})
    if anomaly_findings.get('severity') in ['High', 'Medium']:
        risk_flags.append({'level': anomaly_findings['severity'].upper(),
                           'flag': f"{anomaly_findings['total']} anomalous records ({anomaly_findings['pct']}%) detected — review before making decisions."})
    if corr_findings.get('strong_pos') and len(corr_findings['strong_pos']) > 2:
        risk_flags.append({'level': 'MEDIUM', 'flag': f"{len(corr_findings['strong_pos'])} highly correlated variable pairs detected — multicollinearity may affect ML model accuracy."})
    for nc in (findings['num_cols'] or []):
        if findings['descriptive'] and nc in findings['descriptive'].get('skewness', {}):
            skew = findings['descriptive']['skewness'][nc]
            if abs(skew) > 2:
                risk_flags.append({'level': 'LOW', 'flag': f"{nc} is heavily skewed (skewness={skew:.2f}) — consider log transformation before modelling."})
    if not risk_flags:
        risk_flags.append({'level': 'LOW', 'flag': 'No critical data risks detected. Dataset appears analytically sound.'})

    # Recommendations
    if findings['quality']['duplicates'] > 0:
        recommendations.append("Remove duplicate rows before running downstream analysis or model training.")
    if any(v > 20 for v in (missing_pct.values if hasattr(missing_pct, 'values') else [])):
        recommendations.append("Impute or drop columns with >20% missing values. Consider median imputation for numeric columns and mode for categorical.")
    if trend_findings.get('trends'):
        for col, t in trend_findings['trends'].items():
            if t['direction'] == 'Declining' and abs(t['pct_change']) > 15:
                recommendations.append(f"Investigate the declining trend in {col} ({t['pct_change']:+.1f}% change). Examine segment breakdowns to identify root cause.")
    if segment_findings and segment_findings.get('top3_share', 0) > 70:
        recommendations.append(f"High concentration risk: top 3 segments in {segment_findings['category']} represent {segment_findings['top3_share']}% of total. Diversification may reduce vulnerability.")
    if anomaly_findings.get('col_detail'):
        top_anom_col = max(anomaly_findings['col_detail'], key=lambda k: anomaly_findings['col_detail'][k]['count'])
        recommendations.append(f"Prioritize review of {top_anom_col} — highest outlier density ({anomaly_findings['col_detail'][top_anom_col]['pct']}% of records). Validate data collection process.")
    if corr_findings.get('strong_pos'):
        pair = corr_findings['strong_pos'][0]
        recommendations.append(f"Leverage the strong correlation between {pair['pair']} (r={pair['r']}) for feature engineering or as a leading indicator in forecasting models.")
    if len(recommendations) < 3:
        recommendations.append("Run the ML Forecasting module to build predictive models on this dataset for forward-looking intelligence.")
    if len(recommendations) < 4:
        recommendations.append("Use the Sector Analysis module to benchmark individual segment performance against group averages.")

    findings['summary']         = summary_bullets
    findings['risk_flags']      = risk_flags
    findings['recommendations'] = recommendations
    return findings


def _build_html_report(findings, df):
    """Builds a complete, self-contained HTML report from findings."""
    num_cols = findings.get('num_cols', [])
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    meta = findings['meta']

    # ── Generate charts as base64 PNGs
    charts_html = {}

    # Chart A: Distribution grid (up to 4 numeric cols)
    if num_cols:
        n = min(4, len(num_cols))
        cols_to_plot = num_cols[:n]
        rows = (n + 1) // 2
        dist_fig = make_subplots(rows=rows, cols=2,
            subplot_titles=[f"Distribution: {c}" for c in cols_to_plot],
            vertical_spacing=0.15, horizontal_spacing=0.1)
        for i, c in enumerate(cols_to_plot):
            r, col_ = divmod(i, 2)
            dist_fig.add_trace(go.Histogram(x=df[c].dropna(), name=c,
                marker_color='rgba(106,90,205,0.75)',
                marker_line=dict(color='rgba(157,114,255,0.9)', width=1)),
                row=r+1, col=col_+1)
        dist_fig.update_layout(showlegend=False, template="plotly_white",
            paper_bgcolor="white", plot_bgcolor="#F8F9FA",
            font=dict(family="Segoe UI", size=11), height=320*rows,
            margin=dict(l=30, r=30, t=50, b=30))
        b64 = _fig_to_base64(dist_fig)
        if b64:
            charts_html['dist'] = f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:8px;">'

    # Chart B: Correlation heatmap
    if len(num_cols) >= 2:
        corr = df[num_cols[:10]].corr()
        heat_fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            colorbar=dict(title="r", thickness=15)))
        heat_fig.update_layout(template="plotly_white", paper_bgcolor="white",
            font=dict(family="Segoe UI", size=10),
            height=max(350, len(num_cols[:10]) * 40),
            margin=dict(l=20, r=20, t=40, b=20))
        b64 = _fig_to_base64(heat_fig)
        if b64:
            charts_html['corr'] = f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:8px;">'

    # Chart C: Time series (if available)
    if findings.get('trends', {}).get('trends'):
        dc   = findings['trends']['date_col']
        freq = findings['trends']['freq']
        target_nc = list(findings['trends']['trends'].keys())[0]
        try:
            tmp = df[[dc, target_nc]].copy()
            tmp[dc] = pd.to_datetime(tmp[dc], errors='coerce')
            tmp = tmp.dropna().sort_values(dc)
            agg = tmp.groupby(pd.Grouper(key=dc, freq=freq))[target_nc].sum().reset_index().dropna()
            win = min(3, len(agg)-1)
            agg['MA'] = agg[target_nc].rolling(win, min_periods=1).mean()
            ts_fig = go.Figure()
            ts_fig.add_trace(go.Scatter(x=agg[dc], y=agg[target_nc], mode='lines+markers',
                name=target_nc, line=dict(color='#6A5ACD', width=2), marker=dict(size=5)))
            ts_fig.add_trace(go.Scatter(x=agg[dc], y=agg['MA'], mode='lines',
                name='Moving Avg', line=dict(color='#FF8C00', width=2, dash='dot')))
            ts_fig.update_layout(template="plotly_white", paper_bgcolor="white",
                plot_bgcolor="#F8F9FA", font=dict(family="Segoe UI", size=11),
                height=320, margin=dict(l=30, r=30, t=40, b=30),
                legend=dict(orientation='h', y=1.1))
            b64 = _fig_to_base64(ts_fig)
            if b64:
                charts_html['ts'] = f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:8px;">'
        except Exception:
            pass

    # Chart D: Segment bar chart
    if findings.get('segments'):
        seg_data = findings['segments']
        top10 = seg_data['table'][:10]
        cats  = [r[seg_data['category']] for r in top10]
        vals  = [r['Total'] for r in top10]
        shrs  = [r['Share%'] for r in top10]
        seg_fig = go.Figure(go.Bar(
            y=cats, x=vals, orientation='h',
            marker=dict(color=vals, colorscale='Viridis', showscale=True,
                        colorbar=dict(title=seg_data['metric'], thickness=14)),
            text=[f"{s:.1f}%" for s in shrs], textposition='outside'))
        seg_fig.update_layout(template="plotly_white", paper_bgcolor="white",
            plot_bgcolor="#F8F9FA", font=dict(family="Segoe UI", size=11),
            yaxis=dict(categoryorder='total ascending'),
            height=max(320, len(top10)*44),
            margin=dict(l=20, r=100, t=40, b=20))
        b64 = _fig_to_base64(seg_fig)
        if b64:
            charts_html['seg'] = f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:8px;">'

    # Chart E: Anomaly scatter (first 2 numeric cols)
    if num_cols and len(num_cols) >= 2 and findings['anomalies'].get('labels'):
        labels = findings['anomalies']['labels']
        plot_df = df[num_cols[:2]].copy().iloc[:len(labels)]
        plot_df['_label'] = ['Anomaly' if l == -1 else 'Normal' for l in labels]
        colors = ['#E53935' if l == 'Anomaly' else 'rgba(106,90,205,0.45)' for l in plot_df['_label']]
        anom_fig = go.Figure()
        anom_fig.add_trace(go.Scatter(
            x=plot_df[plot_df['_label']=='Normal'][num_cols[0]],
            y=plot_df[plot_df['_label']=='Normal'][num_cols[1]],
            mode='markers', name='Normal',
            marker=dict(color='rgba(106,90,205,0.45)', size=5)))
        anom_fig.add_trace(go.Scatter(
            x=plot_df[plot_df['_label']=='Anomaly'][num_cols[0]],
            y=plot_df[plot_df['_label']=='Anomaly'][num_cols[1]],
            mode='markers', name='Anomaly',
            marker=dict(color='#E53935', size=8, symbol='x', line=dict(width=1.5))))
        anom_fig.update_layout(template="plotly_white", paper_bgcolor="white",
            plot_bgcolor="#F8F9FA", font=dict(family="Segoe UI", size=11),
            xaxis_title=num_cols[0], yaxis_title=num_cols[1],
            height=320, margin=dict(l=30,r=20,t=40,b=30),
            legend=dict(orientation='h', y=1.1))
        b64 = _fig_to_base64(anom_fig)
        if b64:
            charts_html['anom'] = f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:8px;">'

    # ── Build quality badge
    grade = findings['quality']['grade']
    score = findings['quality']['score']
    grade_color = {'A':'#2E7D32','B':'#1565C0','C':'#E65100','D':'#B71C1C'}.get(grade, '#555')

    # ── Build risk badge rows
    def risk_row(level, text):
        color = {'HIGH':'#C62828','MEDIUM':'#E65100','LOW':'#1565C0'}.get(level,'#555')
        return f"""<tr>
          <td style="padding:8px 12px;white-space:nowrap;">
            <span style="background:{color};color:#fff;font-size:11px;font-weight:700;
                         padding:3px 10px;border-radius:20px;letter-spacing:1px;">{level}</span>
          </td>
          <td style="padding:8px 12px;color:#37474F;font-size:13px;line-height:1.6;">{text}</td>
        </tr>"""

    risk_rows_html = "".join(risk_row(r['level'], r['flag']) for r in findings['risk_flags'])

    # ── Build stats table rows
    def stat_rows():
        rows_html = ""
        desc = findings.get('descriptive', {})
        if not desc or 'mean' not in desc: return ""
        for nc in (findings['num_cols'] or []):
            mean_v  = desc.get('mean',{}).get(nc,'–')
            std_v   = desc.get('std',{}).get(nc,'–')
            min_v   = desc.get('min',{}).get(nc,'–')
            max_v   = desc.get('max',{}).get(nc,'–')
            skew_v  = desc.get('skewness',{}).get(nc,'–')
            cv_v    = desc.get('cv',{}).get(nc,'–')
            def fmt(v):
                try: return f"{float(v):,.3f}"
                except: return str(v)
            skew_flag = ""
            try:
                if abs(float(skew_v)) > 2:
                    skew_flag = f" <span style='color:#E65100;font-size:11px;font-weight:700;'>⚠ Skewed</span>"
            except: pass
            rows_html += f"""<tr>
              <td style="padding:7px 12px;font-weight:600;color:#37474F;border-bottom:1px solid #EEE;">{nc}</td>
              <td style="padding:7px 12px;border-bottom:1px solid #EEE;">{fmt(mean_v)}</td>
              <td style="padding:7px 12px;border-bottom:1px solid #EEE;">{fmt(std_v)}</td>
              <td style="padding:7px 12px;border-bottom:1px solid #EEE;">{fmt(min_v)}</td>
              <td style="padding:7px 12px;border-bottom:1px solid #EEE;">{fmt(max_v)}</td>
              <td style="padding:7px 12px;border-bottom:1px solid #EEE;">{fmt(skew_v)}{skew_flag}</td>
              <td style="padding:7px 12px;border-bottom:1px solid #EEE;">{fmt(cv_v)}%</td>
            </tr>"""
        return rows_html

    # ── Build trend table rows
    def trend_rows():
        rows_html = ""
        for nc, t in (findings.get('trends', {}).get('trends', {}) or {}).items():
            dir_color = '#2E7D32' if t['direction'] == 'Growing' else '#C62828'
            dir_arrow = '▲' if t['direction'] == 'Growing' else '▼'
            rows_html += f"""<tr>
              <td style="padding:7px 12px;font-weight:600;border-bottom:1px solid #EEE;">{nc}</td>
              <td style="padding:7px 12px;color:{dir_color};font-weight:700;border-bottom:1px solid #EEE;">{dir_arrow} {t['direction']}</td>
              <td style="padding:7px 12px;border-bottom:1px solid #EEE;">{t['pct_change']:+.1f}%</td>
              <td style="padding:7px 12px;border-bottom:1px solid #EEE;">{t['volatility']}%</td>
              <td style="padding:7px 12px;border-bottom:1px solid #EEE;">{t['peak_date']}</td>
              <td style="padding:7px 12px;border-bottom:1px solid #EEE;">{t['peak_val']:,.2f}</td>
            </tr>"""
        return rows_html

    # ── Build segment table rows
    def segment_rows():
        rows_html = ""
        for i, row in enumerate(findings.get('segments', {}).get('table', [])):
            cat   = findings['segments']['category']
            nc    = findings['segments']['metric']
            bg    = "#F3F6FF" if i % 2 == 0 else "#FFFFFF"
            share = row.get('Share%', 0)
            bar_w = int(share * 2)
            bar_w = min(bar_w, 100)
            rows_html += f"""<tr style="background:{bg};">
              <td style="padding:7px 12px;font-weight:600;border-bottom:1px solid #EEE;">{i+1}</td>
              <td style="padding:7px 12px;border-bottom:1px solid #EEE;">{row[cat]}</td>
              <td style="padding:7px 12px;border-bottom:1px solid #EEE;">{row['Total']:,.2f}</td>
              <td style="padding:7px 12px;border-bottom:1px solid #EEE;">{row['Average']:,.2f}</td>
              <td style="padding:7px 12px;border-bottom:1px solid #EEE;">{int(row['Count'])}</td>
              <td style="padding:7px 12px;border-bottom:1px solid #EEE;">
                <div style="display:flex;align-items:center;gap:8px;">
                  <div style="background:#E3E8FF;border-radius:4px;height:8px;width:100px;">
                    <div style="background:#6A5ACD;border-radius:4px;height:8px;width:{bar_w}px;"></div>
                  </div>
                  <span style="font-size:12px;font-weight:600;">{share}%</span>
                </div>
              </td>
            </tr>"""
        return rows_html

    # ── Build recommendation rows
    rec_html = "".join(f"""
        <div style="display:flex;gap:14px;align-items:flex-start;padding:12px 0;border-bottom:1px solid #EEE;">
          <div style="min-width:28px;height:28px;border-radius:50%;background:#6A5ACD;color:#fff;
                      font-weight:700;font-size:13px;display:flex;align-items:center;justify-content:center;">{i+1}</div>
          <p style="margin:0;color:#37474F;font-size:13px;line-height:1.7;">{r}</p>
        </div>""" for i, r in enumerate(findings['recommendations']))

    # ── Build missing values rows
    missing_rows = ""
    missing_dict = findings['quality'].get('missing', {})
    for col_name, pct in sorted(missing_dict.items(), key=lambda x: -x[1]):
        if pct > 0:
            bar_w = int(min(pct, 100))
            color = "#C62828" if pct > 30 else ("#E65100" if pct > 10 else "#1565C0")
            missing_rows += f"""<tr>
              <td style="padding:6px 12px;border-bottom:1px solid #EEE;font-size:13px;">{col_name}</td>
              <td style="padding:6px 12px;border-bottom:1px solid #EEE;">
                <div style="display:flex;align-items:center;gap:8px;">
                  <div style="background:#EEE;border-radius:4px;height:8px;width:120px;">
                    <div style="background:{color};border-radius:4px;height:8px;width:{bar_w}px;"></div>
                  </div>
                  <span style="font-size:12px;font-weight:600;color:{color};">{pct:.1f}%</span>
                </div>
              </td>
            </tr>"""
    if not missing_rows:
        missing_rows = '<tr><td colspan="2" style="padding:12px;color:#2E7D32;font-size:13px;">No missing values detected. Dataset is complete.</td></tr>'

    # ── Corr pairs
    corr_pairs_html = ""
    all_pairs = (findings['correlations'].get('strong_pos', []) +
                 findings['correlations'].get('strong_neg', []))
    for p in all_pairs[:8]:
        r_val = p['r']
        color = '#2E7D32' if r_val > 0 else '#C62828'
        bar_w = int(abs(r_val) * 80)
        corr_pairs_html += f"""<div style="display:flex;justify-content:space-between;align-items:center;
            padding:8px 0;border-bottom:1px solid #EEE;">
          <span style="font-size:13px;color:#37474F;font-weight:500;">{p['pair']}</span>
          <div style="display:flex;align-items:center;gap:8px;">
            <div style="background:#EEE;border-radius:4px;height:8px;width:80px;">
              <div style="background:{color};border-radius:4px;height:8px;width:{bar_w}px;"></div>
            </div>
            <span style="font-weight:700;color:{color};font-size:13px;min-width:40px;text-align:right;">r={r_val:.3f}</span>
          </div>
        </div>"""
    if not corr_pairs_html:
        corr_pairs_html = '<p style="color:#666;font-size:13px;">No strong correlations (|r| ≥ 0.70) detected.</p>'

    # ── Summary bullets
    summary_html = "".join(f'<li style="margin-bottom:8px;line-height:1.7;color:#37474F;">{b}</li>'
                           for b in findings['summary'])

    # ── Anomaly summary
    anom = findings['anomalies']
    anom_detail = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #EEE;">'
        f'<span style="color:#37474F;font-size:13px;font-weight:600;">{k}</span>'
        f'<span style="color:#E53935;font-size:13px;">{v["count"]} records ({v["pct"]}%)</span></div>'
        for k, v in list(anom.get('col_detail', {}).items())[:8]
    ) or '<p style="color:#2E7D32;font-size:13px;">No per-column outliers detected at the 1.5×IQR threshold.</p>'

    # ── ASSEMBLE THE FULL HTML
    # Section heading helper
    def section(title, subtitle, content, icon_color="#6A5ACD"):
        return f"""
        <div style="margin:40px 0 0 0;">
          <div style="border-left:4px solid {icon_color};padding:0 0 0 14px;margin-bottom:18px;">
            <h2 style="margin:0;font-size:20px;font-weight:700;color:#1A1A2E;">{title}</h2>
            <p style="margin:4px 0 0 0;color:#607D8B;font-size:13px;">{subtitle}</p>
          </div>
          {content}
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{meta['title']}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #F5F6FA; color: #212121; }}
  .page {{ max-width: 1100px; margin: 0 auto; background: #fff; }}
  .cover {{
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%);
    padding: 70px 60px 60px 60px; color: #fff;
  }}
  .cover-tag {{ font-size:12px; letter-spacing:3px; text-transform:uppercase;
                color:#9D72FF; font-weight:700; margin-bottom:20px; }}
  .cover-title {{ font-size:38px; font-weight:800; line-height:1.25;
                  color:#fff; margin-bottom:16px; }}
  .cover-sub {{ font-size:16px; color:#B0BEC5; line-height:1.6; max-width:600px; }}
  .cover-meta {{ margin-top:50px; display:flex; gap:40px; }}
  .cover-meta-item {{ border-top:2px solid rgba(157,114,255,0.5); padding-top:12px; }}
  .cover-meta-label {{ font-size:11px; letter-spacing:2px; color:#9D72FF;
                        text-transform:uppercase; font-weight:700; }}
  .cover-meta-value {{ font-size:15px; color:#E0E0FF; margin-top:4px; }}
  .cover-kpis {{ display:flex; gap:0; margin-top:40px; border:1px solid rgba(255,255,255,0.1);
                 border-radius:12px; overflow:hidden; }}
  .cover-kpi {{ flex:1; padding:20px 24px; border-right:1px solid rgba(255,255,255,0.08); }}
  .cover-kpi:last-child {{ border-right:none; }}
  .cover-kpi-val {{ font-size:28px; font-weight:800; color:#9D72FF; }}
  .cover-kpi-label {{ font-size:12px; color:#90A4AE; margin-top:4px; }}
  .body-content {{ padding: 50px 60px; }}
  .exec-box {{ background: linear-gradient(135deg, #EDE7F6, #F3E5F5);
               border-radius:12px; padding:28px 32px; margin-bottom:10px; }}
  .exec-box ul {{ padding-left:18px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ background:#1A1A2E; color:#fff; padding:9px 12px;
        text-align:left; font-size:12px; letter-spacing:0.5px; }}
  .card {{ background:#F8F9FF; border-radius:10px; padding:22px 26px;
           border:1px solid #E8EAF6; }}
  .quality-badge {{
    display:inline-flex; align-items:center; gap:16px;
    background:{grade_color}; color:#fff; border-radius:12px; padding:14px 24px;
    font-size:22px; font-weight:800; margin-bottom:16px;
  }}
  .footer {{ background:#1A1A2E; color:#90A4AE; padding:28px 60px;
             font-size:12px; display:flex; justify-content:space-between; }}
  @media print {{
    body {{ background:#fff; }}
    .cover {{ -webkit-print-color-adjust:exact; print-color-adjust:exact; }}
  }}
</style>
</head>
<body>
<div class="page">

<!-- COVER PAGE -->
<div class="cover">
  <div class="cover-tag">Automated Intelligence Report</div>
  <div class="cover-title">{meta['title']}</div>
  <div class="cover-sub">
    A comprehensive, automated analytical report generated from your dataset.
    Includes statistical profiling, anomaly detection, trend analysis,
    segment performance, correlation mapping, and strategic recommendations.
  </div>
  <div class="cover-kpis">
    <div class="cover-kpi">
      <div class="cover-kpi-val">{meta['rows']:,}</div>
      <div class="cover-kpi-label">Total Records</div>
    </div>
    <div class="cover-kpi">
      <div class="cover-kpi-val">{meta['cols']}</div>
      <div class="cover-kpi-label">Columns</div>
    </div>
    <div class="cover-kpi">
      <div class="cover-kpi-val">{meta['num_cols']}</div>
      <div class="cover-kpi-label">Numeric Features</div>
    </div>
    <div class="cover-kpi">
      <div class="cover-kpi-val">{findings['quality']['grade']}</div>
      <div class="cover-kpi-label">Data Quality Grade</div>
    </div>
    <div class="cover-kpi">
      <div class="cover-kpi-val">{anom.get('total', 0)}</div>
      <div class="cover-kpi-label">Anomalies Detected</div>
    </div>
  </div>
  <div class="cover-meta">
    <div class="cover-meta-item">
      <div class="cover-meta-label">Source File</div>
      <div class="cover-meta-value">{meta['filename']}</div>
    </div>
    <div class="cover-meta-item">
      <div class="cover-meta-label">Analyst</div>
      <div class="cover-meta-value">{meta['analyst'] or 'Not specified'}</div>
    </div>
    <div class="cover-meta-item">
      <div class="cover-meta-label">Generated</div>
      <div class="cover-meta-value">{meta['generated']}</div>
    </div>
  </div>
</div>

<!-- BODY -->
<div class="body-content">

{section("Executive Summary",
  "High-level findings and takeaways from the automated analysis",
  f'<div class="exec-box"><ul>{summary_html}</ul></div>',
  "#6A5ACD")}

{section("Risk Flags",
  "Issues and concerns that require attention before proceeding with analysis or decisions",
  f'''<table><thead><tr><th style="width:100px;">Severity</th><th>Issue</th></tr></thead>
  <tbody>{risk_rows_html}</tbody></table>''',
  "#C62828")}

{section("Data Quality Audit",
  "Completeness, integrity, and structural assessment of the dataset",
  f'''<div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:20px;">
    <div class="card" style="flex:1;min-width:200px;">
      <div class="quality-badge">Grade {grade} &nbsp;·&nbsp; {score}/100</div>
      <p style="color:#607D8B;font-size:13px;margin-top:8px;">
        {"Excellent data quality — analysis results are highly reliable." if score >= 90 else
         "Good quality — minor issues present but results are trustworthy." if score >= 75 else
         "Moderate quality — results should be cross-validated." if score >= 60 else
         "Poor quality — significant data cleaning required before analysis."}
      </p>
    </div>
    <div class="card" style="flex:1;min-width:200px;">
      <p style="font-size:12px;color:#9D72FF;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;">Duplicate Records</p>
      <p style="font-size:28px;font-weight:800;color:#1A1A2E;">{findings['quality']['duplicates']:,}</p>
      <p style="font-size:12px;color:#607D8B;margin-top:4px;">
        {f"{findings['quality']['duplicates']/meta['rows']*100:.1f}% of dataset" if meta['rows'] else ""}
      </p>
    </div>
  </div>
  <p style="font-size:13px;font-weight:600;color:#37474F;margin-bottom:10px;">Missing Values by Column</p>
  <table><thead><tr><th>Column</th><th>Missing Rate</th></tr></thead>
  <tbody>{missing_rows}</tbody></table>''',
  "#1565C0")}

{section("Statistical Profiling",
  "Descriptive statistics for all numeric variables — central tendency, spread, and shape",
  f'''{'<p style="color:#607D8B;font-size:13px;">No numeric columns available.</p>' if not findings['num_cols'] else f"""
  <table><thead><tr>
    <th>Column</th><th>Mean</th><th>Std Dev</th>
    <th>Min</th><th>Max</th><th>Skewness</th><th>CV%</th>
  </tr></thead><tbody>{stat_rows()}</tbody></table>
  <p style="font-size:12px;color:#607D8B;margin-top:10px;">
    CV% = Coefficient of Variation (Std Dev / Mean × 100). Higher = more variable.
    Skewness &gt;2 or &lt;-2 indicates a heavily skewed distribution.
  </p>"""}''',
  "#00796B")}

{(section("Distribution Overview",
  "Frequency distribution of numeric variables",
  charts_html.get('dist', '<p style="color:#607D8B;">Chart not available.</p>'),
  "#00796B")) if 'dist' in charts_html else ""}

{(section("Anomaly Detection",
  "Records identified as statistical anomalies using Isolation Forest (5% contamination threshold)",
  f'''<div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:20px;">
    <div class="card" style="flex:1;min-width:160px;">
      <p style="font-size:12px;color:#9D72FF;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;">Anomalies</p>
      <p style="font-size:32px;font-weight:800;color:#E53935;">{anom.get("total",0):,}</p>
      <p style="font-size:12px;color:#607D8B;">{anom.get("pct",0):.1f}% of records</p>
    </div>
    <div class="card" style="flex:1;min-width:160px;">
      <p style="font-size:12px;color:#9D72FF;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;">Severity</p>
      <p style="font-size:28px;font-weight:800;color:{"#C62828" if anom.get("severity")=="High" else "#E65100" if anom.get("severity")=="Medium" else "#1565C0"};">{anom.get("severity","–")}</p>
    </div>
  </div>
  <p style="font-size:13px;font-weight:600;color:#37474F;margin-bottom:10px;">Outliers by Column (IQR Method)</p>
  {anom_detail}
  <div style="margin-top:20px;">{charts_html.get("anom","")}</div>''',
  "#E53935")) if anom.get('total', 0) >= 0 else ""}

{(section("Trend Analysis",
  f"Time-series patterns detected using {findings['trends']['date_col']} — grouped by period",
  f'''<table><thead><tr>
    <th>Metric</th><th>Direction</th><th>Change %</th>
    <th>Volatility</th><th>Peak Date</th><th>Peak Value</th>
  </tr></thead><tbody>{trend_rows()}</tbody></table>
  <div style="margin-top:20px;">{charts_html.get("ts","")}</div>''',
  "#F57F17")) if findings.get('trends', {}).get('trends') else ""}

{(section("Correlation Analysis",
  "Linear relationships between numeric variables — only strong correlations (|r| ≥ 0.70) shown",
  f'''<div style="display:flex;gap:20px;flex-wrap:wrap;">
    <div style="flex:1;min-width:260px;">
      <p style="font-size:13px;font-weight:600;color:#37474F;margin-bottom:12px;">Strong Relationship Pairs</p>
      {corr_pairs_html}
    </div>
    <div style="flex:2;min-width:300px;">{charts_html.get("corr","")}</div>
  </div>''',
  "#1A237E")) if findings.get('correlations') else ""}

{(section("Segment Performance",
  f"Breakdown of {findings['segments']['metric']} by {findings['segments']['category']} — top 10 segments ranked",
  f'''<div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:20px;">
    <div class="card" style="flex:1;min-width:180px;">
      <p style="font-size:12px;color:#9D72FF;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;">Top Segment</p>
      <p style="font-size:20px;font-weight:800;color:#1A1A2E;">{findings["segments"]["top_name"]}</p>
      <p style="font-size:13px;color:#607D8B;">{findings["segments"]["top_val"]:,.2f}</p>
    </div>
    <div class="card" style="flex:1;min-width:180px;">
      <p style="font-size:12px;color:#9D72FF;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;">Bottom Segment</p>
      <p style="font-size:20px;font-weight:800;color:#C62828;">{findings["segments"]["bot_name"]}</p>
      <p style="font-size:13px;color:#607D8B;">{findings["segments"]["bot_val"]:,.2f}</p>
    </div>
    <div class="card" style="flex:1;min-width:180px;">
      <p style="font-size:12px;color:#9D72FF;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;">Top 3 Concentration</p>
      <p style="font-size:28px;font-weight:800;color:#1A1A2E;">{findings["segments"]["top3_share"]}%</p>
    </div>
  </div>
  <table><thead><tr><th>#</th><th>{findings["segments"]["category"]}</th>
    <th>Total</th><th>Average</th><th>Count</th><th>Share</th>
  </tr></thead><tbody>{segment_rows()}</tbody></table>
  <div style="margin-top:20px;">{charts_html.get("seg","")}</div>''',
  "#00695C")) if findings.get('segments') else ""}

{section("Strategic Recommendations",
  "Prioritized action items derived from the automated analysis findings",
  f'<div class="card">{rec_html}</div>',
  "#4A148C")}

</div>

<!-- FOOTER -->
<div class="footer">
  <span>Smart Insights Platform &nbsp;·&nbsp; Intelligence Report Engine</span>
  <span>Generated: {meta['generated']} &nbsp;·&nbsp; Analyst: {meta['analyst'] or 'Unspecified'}</span>
</div>

</div>
</body>
</html>"""
    return html


def render_intelligence_report(df):
    """Main render function for the Intelligence Report page."""
    st.markdown("""
        <div style='padding:20px 0 10px 0;'>
            <h1 style='color:#9D72FF;margin:0;font-size:32px;'>Intelligence Report Engine</h1>
            <p style='color:#B19CD9;margin:6px 0 20px 0;font-size:15px;'>
                Automatically analyzes your entire dataset and generates a professional,
                downloadable executive report — including statistical profiling, anomaly detection,
                trend analysis, segment performance, correlation mapping, and strategic recommendations.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # What this feature does
    st.markdown("""
        <div style='background:rgba(106,90,205,0.10);border-radius:10px;
                    border-left:4px solid #9D72FF;padding:16px 20px;margin-bottom:24px;'>
            <p style='color:#9D72FF;font-weight:700;font-size:13px;
                      letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>
                What this generates
            </p>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
                <div style='color:#B19CD9;font-size:13px;line-height:1.7;'>
                    Cover page with dataset KPIs<br>
                    Executive summary (auto-written)<br>
                    Risk flags with severity ratings<br>
                    Data quality audit with missing value map
                </div>
                <div style='color:#B19CD9;font-size:13px;line-height:1.7;'>
                    Statistical profiling (all numeric columns)<br>
                    Anomaly detection with per-column breakdown<br>
                    Time-series trend analysis (if dates present)<br>
                    Correlation heatmap + segment performance
                </div>
            </div>
            <p style='color:#00BCD4;font-size:13px;margin:12px 0 0 0;'>
                The final output is a self-contained HTML file you can open in any browser,
                print to PDF, or share with stakeholders — no tools required.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Report configuration
    st.markdown("<p style='color:#9D72FF;font-size:13px;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;'>Report Configuration</p>", unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)
    with rc1:
        report_title = st.text_input("Report Title",
            value=f"Intelligence Report — {st.session_state.get('file_name','Dataset')}",
            key="ir_title")
    with rc2:
        analyst_name = st.text_input("Analyst Name (appears in report)",
            value="", placeholder="e.g. Data Analytics Team", key="ir_analyst")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Generate Intelligence Report", key="ir_generate"):
        # Multi-step progress
        progress_bar  = st.progress(0)
        status_text   = st.empty()

        steps = [
            (5,  "Running data quality audit..."),
            (18, "Computing descriptive statistics..."),
            (32, "Running anomaly detection (Isolation Forest)..."),
            (48, "Computing correlation matrix..."),
            (62, "Detecting time-series trends..."),
            (75, "Analysing segment performance..."),
            (85, "Generating charts and embedding as PNG..."),
            (94, "Writing executive narrative and recommendations..."),
            (100,"Report complete."),
        ]

        with st.spinner(""):
            findings = None
            html_report = None

            try:
                for pct, msg in steps[:-2]:
                    progress_bar.progress(pct)
                    status_text.markdown(f"<p style='color:#B19CD9;font-size:13px;'>{msg}</p>",
                        unsafe_allow_html=True)
                    time.sleep(0.15)

                # Run the actual analysis
                findings    = _run_full_analysis(df, report_title, analyst_name)
                progress_bar.progress(85)
                status_text.markdown("<p style='color:#B19CD9;font-size:13px;'>Generating charts and embedding as PNG...</p>",
                    unsafe_allow_html=True)
                time.sleep(0.1)

                html_report = _build_html_report(findings, df)
                progress_bar.progress(100)
                status_text.markdown("<p style='color:#64FFDA;font-size:13px;font-weight:700;'>Report generated successfully.</p>",
                    unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Report generation error: {str(e)}")
                return

        if findings and html_report:
            # Store for in-app preview
            st.session_state['ir_findings'] = findings
            st.session_state['ir_html']     = html_report

    # If report is ready — show download + in-app summary
    if 'ir_findings' in st.session_state and 'ir_html' in st.session_state:
        findings    = st.session_state['ir_findings']
        html_report = st.session_state['ir_html']

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Download button (prominent)
        b64_report = base64.b64encode(html_report.encode('utf-8')).decode()
        file_name  = f"intelligence_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html"
        st.markdown(f"""
            <div style='background:linear-gradient(135deg,rgba(100,255,218,0.10),rgba(106,90,205,0.15));
                        border-left:4px solid #64FFDA;border-radius:12px;
                        padding:22px 26px;margin-bottom:28px;
                        display:flex;align-items:center;justify-content:space-between;gap:20px;'>
                <div>
                    <p style='color:#64FFDA;font-weight:700;font-size:14px;margin:0 0 4px 0;'>
                        Report Ready
                    </p>
                    <p style='color:#B19CD9;font-size:13px;margin:0;'>
                        Self-contained HTML · Open in browser · Print to PDF
                    </p>
                </div>
                <a href="data:text/html;base64,{b64_report}" download="{file_name}"
                   style='background:linear-gradient(135deg,#6A5ACD,#9370DB);
                          color:#fff;text-decoration:none;border-radius:8px;
                          padding:12px 28px;font-weight:700;font-size:14px;
                          white-space:nowrap;letter-spacing:0.5px;'>
                    Download Report
                </a>
            </div>
        """, unsafe_allow_html=True)

        # ── In-app preview of key findings
        st.markdown("<p style='color:#9D72FF;font-weight:700;font-size:14px;letter-spacing:1px;text-transform:uppercase;margin-bottom:14px;'>In-App Preview</p>", unsafe_allow_html=True)

        prev_tabs = st.tabs(["Summary", "Data Quality", "Statistics", "Anomalies", "Trends", "Correlations", "Segments"])

        with prev_tabs[0]:
            st.markdown("<p style='color:#B19CD9;font-size:13px;margin-bottom:12px;'>Auto-generated executive summary:</p>", unsafe_allow_html=True)
            for bullet in findings['summary']:
                st.markdown(f"""
                    <div style='display:flex;gap:12px;align-items:flex-start;margin-bottom:10px;'>
                        <div style='min-width:8px;height:8px;margin-top:6px;border-radius:50%;background:#9D72FF;'></div>
                        <p style='color:#D0D0F0;font-size:14px;line-height:1.7;margin:0;'>{bullet}</p>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("<p style='color:#B19CD9;font-size:13px;font-weight:700;margin:20px 0 10px 0;'>Risk Flags</p>", unsafe_allow_html=True)
            for rf in findings['risk_flags']:
                lv_color = {'HIGH':'#C62828','MEDIUM':'#E65100','LOW':'#1565C0'}.get(rf['level'],'#555')
                st.markdown(f"""
                    <div style='display:flex;gap:12px;align-items:flex-start;margin-bottom:8px;
                                padding:10px 14px;border-radius:8px;background:rgba(40,42,54,0.6);'>
                        <span style='background:{lv_color};color:#fff;font-size:10px;font-weight:700;
                                     padding:2px 8px;border-radius:20px;letter-spacing:1px;
                                     white-space:nowrap;margin-top:2px;'>{rf['level']}</span>
                        <p style='color:#D0D0F0;font-size:13px;line-height:1.7;margin:0;'>{rf['flag']}</p>
                    </div>
                """, unsafe_allow_html=True)

        with prev_tabs[1]:
            q = findings['quality']
            grade_color = {'A':'#2E7D32','B':'#1565C0','C':'#E65100','D':'#B71C1C'}.get(q['grade'],'#555')
            st.markdown(f"""
                <div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:20px;'>
                    <div style='background:rgba(40,42,54,0.7);border-radius:10px;padding:18px 24px;flex:1;min-width:160px;'>
                        <p style='color:#B19CD9;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin:0 0 6px 0;'>Quality Grade</p>
                        <p style='color:{grade_color};font-size:36px;font-weight:800;margin:0;'>{q['grade']}</p>
                        <p style='color:#B19CD9;font-size:13px;margin:4px 0 0 0;'>{q['score']}/100</p>
                    </div>
                    <div style='background:rgba(40,42,54,0.7);border-radius:10px;padding:18px 24px;flex:1;min-width:160px;'>
                        <p style='color:#B19CD9;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin:0 0 6px 0;'>Duplicates</p>
                        <p style='color:#E0E0FF;font-size:36px;font-weight:800;margin:0;'>{q['duplicates']:,}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if q['issues']:
                for iss in q['issues']:
                    st.warning(iss)
            miss_df = pd.DataFrame(list(findings['quality']['missing'].items()), columns=['Column','Missing %'])
            miss_df = miss_df[miss_df['Missing %'] > 0].sort_values('Missing %', ascending=False)
            if not miss_df.empty:
                st.dataframe(miss_df, use_container_width=True)
            else:
                st.success("No missing values found.")

        with prev_tabs[2]:
            if findings['num_cols']:
                desc_rows = []
                desc = findings['descriptive']
                for nc in findings['num_cols']:
                    desc_rows.append({
                        'Column':   nc,
                        'Mean':     round(desc.get('mean',{}).get(nc,0), 3),
                        'Std Dev':  round(desc.get('std',{}).get(nc,0), 3),
                        'Min':      round(desc.get('min',{}).get(nc,0), 3),
                        'Max':      round(desc.get('max',{}).get(nc,0), 3),
                        'Skewness': round(desc.get('skewness',{}).get(nc,0), 3),
                        'CV%':      round(desc.get('cv',{}).get(nc,0), 1),
                    })
                st.dataframe(pd.DataFrame(desc_rows), use_container_width=True)
            else:
                st.info("No numeric columns to profile.")

        with prev_tabs[3]:
            anom = findings['anomalies']
            a1, a2 = st.columns(2)
            a1.metric("Total Anomalies", f"{anom.get('total',0):,}")
            a2.metric("Anomaly Rate", f"{anom.get('pct',0):.1f}%")
            if anom.get('col_detail'):
                anom_df = pd.DataFrame([
                    {'Column': k, 'Outlier Count': v['count'], 'Outlier %': v['pct']}
                    for k, v in anom['col_detail'].items()
                ]).sort_values('Outlier Count', ascending=False)
                st.dataframe(anom_df, use_container_width=True)

        with prev_tabs[4]:
            trends = findings.get('trends', {}).get('trends', {})
            if trends:
                trend_rows_data = []
                for nc, t in trends.items():
                    trend_rows_data.append({
                        'Metric':    nc,
                        'Direction': t['direction'],
                        'Change %':  f"{t['pct_change']:+.1f}%",
                        'Volatility':f"{t['volatility']}%",
                        'Peak Date': t['peak_date'],
                        'Peak Value':f"{t['peak_val']:,.2f}"
                    })
                st.dataframe(pd.DataFrame(trend_rows_data), use_container_width=True)
            else:
                st.info("No date columns found — trend analysis not available for this dataset.")

        with prev_tabs[5]:
            corr = findings.get('correlations', {})
            all_pairs = corr.get('strong_pos', []) + corr.get('strong_neg', [])
            if all_pairs:
                st.markdown("<p style='color:#B19CD9;font-size:13px;'>Variable pairs with |r| ≥ 0.70:</p>", unsafe_allow_html=True)
                for p in all_pairs:
                    r_val = p['r']
                    color = '#64FFDA' if r_val > 0 else '#FF5370'
                    st.markdown(f"""
                        <div style='display:flex;justify-content:space-between;align-items:center;
                                    padding:8px 12px;border-radius:8px;background:rgba(40,42,54,0.6);margin-bottom:6px;'>
                            <span style='color:#E0E0FF;font-size:13px;'>{p['pair']}</span>
                            <span style='color:{color};font-weight:700;font-size:14px;'>r = {r_val:.3f}</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No strong correlations (|r| ≥ 0.70) detected.")

        with prev_tabs[6]:
            seg = findings.get('segments', {})
            if seg:
                seg_df = pd.DataFrame(seg['table'])
                st.dataframe(seg_df, use_container_width=True)
            else:
                st.info("No categorical columns suitable for segment analysis.")

        # Recommendations
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='color:#9D72FF;font-weight:700;font-size:14px;letter-spacing:1px;text-transform:uppercase;margin-bottom:12px;'>Strategic Recommendations</p>", unsafe_allow_html=True)
        for i, rec in enumerate(findings['recommendations']):
            st.markdown(f"""
                <div style='display:flex;gap:14px;align-items:flex-start;
                            padding:12px 16px;border-radius:10px;
                            background:rgba(40,42,54,0.6);margin-bottom:8px;'>
                    <div style='min-width:26px;height:26px;border-radius:50%;
                                background:#6A5ACD;color:#fff;font-weight:700;
                                font-size:12px;display:flex;align-items:center;justify-content:center;'>{i+1}</div>
                    <p style='color:#D0D0F0;font-size:13px;line-height:1.75;margin:0;'>{rec}</p>
                </div>
            """, unsafe_allow_html=True)


# Sidebar navigation
navigation = st.sidebar.radio(
    "",
    ["Upload Data", "Dashboard", "Advanced Analysis", "Sector Analysis", "ML Forecasting", "NLP Insights", "Intelligence Report"],
    key="navigation"
)

# Initialize session state for data storage
if 'data' not in st.session_state:
    st.session_state.data = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'insights' not in st.session_state:
    st.session_state.insights = []

# Upload Data Page
if navigation == "Upload Data":
    if name:
        st.markdown(f"<div class='greeting'>{greeting_text}, {name}!</div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class='upload-section'>
            <h2 style='color: #9D72FF; margin-bottom: 15px;'>Upload Your Sales Data</h2>
            <p style='color: #B19CD9; margin-bottom: 20px;'>
                Upload CSV, Excel, or JSON files to begin your data exploration journey.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # File uploader
    uploaded_file = st.file_uploader("", type=["csv", "xlsx", "xls", "json"])
    
    if uploaded_file is not None:
        try:
            # Display progress
            progress_text = st.markdown("**Processing your data...**")
            progress_bar = st.progress(0)
            
            # Update progress
            for i in range(10):
                progress_bar.progress((i+1) * 10)
                time.sleep(0.05)    
            
            # Read file based on type
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            if file_ext == 'csv':
                df = pd.read_csv(uploaded_file, encoding='latin1')
            elif file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_ext == 'json':
                df = pd.read_json(uploaded_file, encoding='latin1')
            
            # Save to session state
            st.session_state.data = df
            st.session_state.file_name = uploaded_file.name
            
            # Generate AI insights
            st.session_state.insights = generate_ai_insights(df)
            
            # Update progress to completion
            progress_bar.progress(100)
            progress_text.text("Data processed successfully!")
            
            # Show success message with animation
            st.markdown(
                f"""
                <div style='animation: fadeIn 1s; background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(76, 175, 80, 0.1)); 
                     padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #4CAF50;'>
                    <h3 style='color: #4CAF50; margin-top: 0;'>Success!</h3>
                    <p style='color: #E0E0FF;'>
                        Successfully loaded <strong>{len(df)}</strong> records with <strong>{len(df.columns)}</strong> columns from <strong>{uploaded_file.name}</strong>
                    </p>
                    <p style='color: #B19CD9;'>
                        Generated <strong>{len(st.session_state.insights)}</strong> AI-powered insights about your data.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Preview data
            st.markdown("<h3 style='color: #9D72FF;'>Data Preview</h3>", unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
            
            # Show column info
            st.markdown("<h3 style='color: #9D72FF;'>Column Information</h3>", unsafe_allow_html=True)
            
            # Create column info table
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Missing Values': df.isnull().sum(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            
            st.dataframe(col_info, use_container_width=True)
            
            # Navigation hint
            st.markdown(
                """
                <div style='animation: fadeIn 1.5s; text-align: center; margin-top: 30px;'>
                    <p style='color: #B19CD9; font-size: 18px;'>
                        You're all set! Navigate to the <strong>Dashboard</strong> tab to explore your data.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        # Show sample data option
        st.markdown(
            """
            <div style='margin-top: 30px; text-align: center; animation: fadeIn 1.5s;'>
                <p style='color: #B19CD9; font-size: 18px;'>
                    Don't have data? Try our <span style='cursor: pointer; text-decoration: underline; color: #9D72FF;' id='load-sample'>sample dataset</span> to see the dashboard in action.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # JavaScript to handle sample data loading
        st.markdown(
            """
            <script>
                document.getElementById('load-sample').addEventListener('click', function() {
                    // This is handled by Streamlit components
                });
            </script>
            """,
            unsafe_allow_html=True
        )
        
        if st.button("Load Sample Dataset"):
            # Create sample sales data
            sample_data = {
                'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
                'Product': np.random.choice(['Widget A', 'Widget B', 'Widget C', 'Widget D'], 100),
                'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
                'Sales': np.random.normal(loc=1000, scale=200, size=100).round(2),
                'Units': np.random.randint(10, 100, 100),
                'Customer_Satisfaction': np.random.uniform(3.5, 5, 100).round(1),
                'Marketing_Spend': np.random.uniform(50, 200, 100).round(2),
                'Profit_Margin': np.random.uniform(0.1, 0.3, 100).round(3)
            }
            
            df = pd.DataFrame(sample_data)
            
            # Calculate some derived metrics
            df['Revenue'] = df['Sales'] * df['Units']
            df['Marketing_ROI'] = df['Revenue'] / df['Marketing_Spend']
            df['Profit'] = df['Revenue'] * df['Profit_Margin']
            
            # Save to session state
            st.session_state.data = df
            st.session_state.file_name = "sample_sales_data.csv"
            
            # Generate AI insights
            st.session_state.insights = generate_ai_insights(df)
            
            # Show success message
            st.markdown(
                f"""
                <div style='animation: fadeIn 1s; background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(76, 175, 80, 0.1)); 
                     padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #4CAF50;'>
                    <h3 style='color: #4CAF50; margin-top: 0;'>Sample Data Loaded!</h3>
                    <p style='color: #E0E0FF;'>
                        Successfully loaded <strong>{len(df)}</strong> records with <strong>{len(df.columns)}</strong> columns from the sample dataset.
                    </p>
                    <p style='color: #B19CD9;'>
                        Generated <strong>{len(st.session_state.insights)}</strong> AI-powered insights about your data.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Preview data
            st.markdown("<h3 style='color: #9D72FF;'>Data Preview</h3>", unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
            
            # Navigation hint
            st.markdown(
                """
                <div style='animation: fadeIn 1.5s; text-align: center; margin-top: 30px;'>
                    <p style='color: #B19CD9; font-size: 18px;'>
                        All set! Navigate to the <strong>Dashboard</strong> tab to explore your data.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

# Dashboard Page
elif navigation == "Dashboard":
    if st.session_state.data is not None:
        # Dashboard header
        st.markdown(
            f"""
            <div class='dashboard-header'>
                <h1 style='color: #9D72FF; margin-bottom: 5px;'>Sales Analytics Dashboard</h1>
                <p style='color: #B19CD9; margin-top: 0;'>
                    Analyzing data from <strong>{st.session_state.file_name}</strong> • 
                    <span style='color: #E0E0FF;'>{len(st.session_state.data)} records</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # KPI Metrics Row
        st.markdown(
            """
            <div style='padding: 15px; border-radius: 10px; background: rgba(60, 63, 68, 0.7); margin-bottom: 15px;'>
                <h4 style='color: #9D72FF; font-size: 18px; margin-bottom: 10px;'>Key Performance Indicators</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Find revenue/sales column
        df = st.session_state.data
        revenue_col = None
        
        potential_revenue_cols = ['Revenue', 'Sales', 'Income', 'Amount']
        for col in potential_revenue_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                revenue_col = col
                break
        
        # Find profit column
        profit_col = None
        potential_profit_cols = ['Profit', 'Margin', 'Earnings', 'Net_Income', 'Profit_Margin']
        for col in potential_profit_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                profit_col = col
                break
        
        # Create KPI metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Total Revenue
            if revenue_col:
                total_revenue = df[revenue_col].sum()
                display_kpi("Total Revenue", f"${total_revenue:,.2f}", "Up 12% from last period")
            else:
                # Find any numeric column for demo
                num_cols = df.select_dtypes(include=['number']).columns
                if len(num_cols) > 0:
                    total_val = df[num_cols[0]].sum()
                    display_kpi("Total " + num_cols[0], f"{total_val:,.2f}", "Key metric overview")
                else:
                    display_kpi("Total Records", f"{len(df):,}", "Data volume indicator")
        
        with col2:
            # Average Order Value
            if revenue_col:
                avg_value = df[revenue_col].mean()
                display_kpi("Average Order", f"${avg_value:,.2f}", "Up 3.5% from last period")
            else:
                # Use another numeric column
                num_cols = df.select_dtypes(include=['number']).columns
                if len(num_cols) > 0:
                    avg_val = df[num_cols[0]].mean()
                    display_kpi("Average " + num_cols[0], f"{avg_val:,.2f}", "Trend is stable")
                else:
                    display_kpi("Unique Values", f"{df.nunique().sum():,}", "Data diversity")
        
        with col3:
            # Profit Margin
            if profit_col and revenue_col:
                if 'Margin' in profit_col and df[profit_col].max() <= 1:
                    # It's already a margin percentage
                    margin = df[profit_col].mean() * 100
                else:
                    # Calculate margin from profit and revenue
                    margin = (df[profit_col].sum() / df[revenue_col].sum()) * 100
                display_kpi("Profit Margin", f"{margin:.1f}%", "Down 1.2% from last period")
            else:
                # Use another metric
                num_cols = df.select_dtypes(include=['number']).columns
                if len(num_cols) > 1:
                    ratio = (df[num_cols[0]].sum() / df[num_cols[1]].sum()) * 100
                    display_kpi("Ratio Metric", f"{ratio:.1f}%", "Key performance ratio")
                else:
                    display_kpi("Missing Values", f"{df.isna().sum().sum():,}", "Data quality metric")
        
        with col4:
            # Customer Satisfaction
            satisfaction_col = None
            potential_satisfaction_cols = ['Satisfaction', 'Rating', 'Score', 'NPS', 'Customer_Satisfaction']
            for col in potential_satisfaction_cols:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    satisfaction_col = col
                    break
            
            if satisfaction_col:
                avg_satisfaction = df[satisfaction_col].mean()
                max_possible = df[satisfaction_col].max()
                
                if max_possible <= 5:
                    # Assume 5-star scale
                    display_kpi("Satisfaction", f"{avg_satisfaction:.1f}/5 ★", "Up 0.3 points")
                elif max_possible <= 10:
                    # Assume 10-point scale
                    display_kpi("Satisfaction", f"{avg_satisfaction:.1f}/10", "Trending positive")
                else:
                    display_kpi("Satisfaction", f"{avg_satisfaction:.1f} pts", "Customer happiness")
            else:
                # Use another unique metric
                date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols:
                    try:
                        time_range = (pd.to_datetime(df[date_cols[0]]).max() - pd.to_datetime(df[date_cols[0]]).min()).days
                        display_kpi("Date Range", f"{time_range} days", "Time period analyzed")
                    except:
                        display_kpi("Unique Categories", f"{df.select_dtypes(include=['object']).nunique().sum():,}", "Categorical breakdown")
                else:
                    display_kpi("Unique Categories", f"{df.select_dtypes(include=['object']).nunique().sum():,}", "Categorical breakdown")
        
        # Create horizontal spacer
        st.markdown("<hr style='margin: 30px 0; opacity: 0.3;'>", unsafe_allow_html=True)
        
        # Main dashboard charts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Time Series Chart
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols and revenue_col:
                st.markdown("<h5 style='color: #B19CD9;'>Sales Performance Over Time</h5>", unsafe_allow_html=True)
                
                # Convert to datetime
                date_col = date_cols[0]
                if df[date_col].dtype != 'datetime64[ns]':
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                # Group by date
                try:
                    # Determine appropriate date grouping
                    date_range = (df[date_col].max() - df[date_col].min()).days
                    
                    if date_range > 365:
                        grouper = pd.Grouper(key=date_col, freq='M')
                        period = "Monthly"
                    elif date_range > 60:
                        grouper = pd.Grouper(key=date_col, freq='W')
                        period = "Weekly"
                    else:
                        grouper = pd.Grouper(key=date_col, freq='D')
                        period = "Daily"
                    
                    # Group and plot
                    time_series = df.groupby(grouper)[revenue_col].sum().reset_index()
                    
                    fig = px.line(
                        time_series,
                        x=date_col,
                        y=revenue_col,
                        title=f"{period} {revenue_col} Performance",
                        color_discrete_sequence=["#9D72FF"]
                    )
                    
                    # Add markers
                    fig.update_traces(mode="lines+markers", marker=dict(size=8, opacity=0.7))
                    
                    # Calculate rolling average
                    window = 3 if len(time_series) > 5 else 2
                    if len(time_series) > window:
                        time_series['Rolling_Avg'] = time_series[revenue_col].rolling(window=window).mean()
                        
                        fig.add_trace(
                            go.Scatter(
                                x=time_series[date_col],
                                y=time_series['Rolling_Avg'],
                                mode="lines",
                                line=dict(color="rgba(255, 215, 0, 0.8)", width=3, dash="dot"),
                                name=f"{window}-Period Moving Average"
                            )
                        )
                    
                    # Update layout
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(40, 42, 54, 0.8)",
                        paper_bgcolor="rgba(40, 42, 54, 0)",
                        margin=dict(l=20, r=20, t=50, b=20),
                        title={
                            "y": 0.95,
                            "x": 0.5,
                            "xanchor": "center",
                            "yanchor": "top",
                            "font": {"color": "#9D72FF", "size": 18}
                        },
                        xaxis=dict(
                            title="",
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="rgba(107, 114, 142, 0.2)"
                        ),
                        yaxis=dict(
                            title=revenue_col,
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="rgba(107, 114, 142, 0.2)"
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Could not create time series: {str(e)}")
            else:
                # Fallback to distribution chart
                st.markdown("<h5 style='color: #B19CD9;'>Value Distribution</h5>", unsafe_allow_html=True)
                
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    selected_col = st.selectbox("Select metric:", numeric_cols)
                    
                    fig = go.Figure()
                    
                    # Add histogram
                    fig.add_trace(
                        go.Histogram(
                            x=df[selected_col],
                            marker=dict(color="rgba(157, 114, 255, 0.7)"),
                            nbinsx=20
                        )
                    )
                    
                    # Update layout
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(40, 42, 54, 0.8)",
                        paper_bgcolor="rgba(40, 42, 54, 0)",
                        margin=dict(l=20, r=20, t=50, b=20),
                        title={
                            "text": f"Distribution of {selected_col}",
                            "y": 0.95,
                            "x": 0.5,
                            "xanchor": "center",
                            "yanchor": "top",
                            "font": {"color": "#9D72FF", "size": 18}
                        }
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No numeric columns found for visualization")
        
        with col2:
            # Category breakdown chart
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if categorical_cols and revenue_col:
                st.markdown("<h5 style='color: #B19CD9;'>Revenue by Category</h5>", unsafe_allow_html=True)
                
                # Select categorical column with the most appropriate number of categories
                best_cat_col = None
                best_cat_count = 0
                
                for col in categorical_cols:
                    cat_count = df[col].nunique()
                    if 3 <= cat_count <= 10:
                        best_cat_col = col
                        break
                    elif cat_count > best_cat_count:
                        best_cat_col = col
                        best_cat_count = cat_count
                
                if best_cat_col:
                    # If too many categories, take the top ones
                    if df[best_cat_col].nunique() > 7:
                        top_cats = df.groupby(best_cat_col)[revenue_col].sum().nlargest(7).index.tolist()
                        filtered_df = df[df[best_cat_col].isin(top_cats)].copy()
                        filtered_df.loc[~filtered_df[best_cat_col].isin(top_cats), best_cat_col] = 'Other'
                    else:
                        filtered_df = df
                    
                    # Aggregate data
                    cat_data = filtered_df.groupby(best_cat_col)[revenue_col].sum().reset_index()
                    
                    # Create pie chart
                    fig = px.pie(
                        cat_data,
                        values=revenue_col,
                        names=best_cat_col,
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.Agsunset
                    )
                    
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(40, 42, 54, 0.8)",
                        paper_bgcolor="rgba(40, 42, 54, 0)",
                        margin=dict(l=10, r=10, t=30, b=10),
                        title={
                            "text": f"{revenue_col} by {best_cat_col}",
                            "y": 0.95,
                            "x": 0.5,
                            "xanchor": "center",
                            "yanchor": "top",
                            "font": {"color": "#9D72FF", "size": 16}
                        },
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.2,
                            xanchor="center",
                            x=0.5
                        )
                    )
                    
                    # Update traces
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add insights below the chart
                    top_category = cat_data.iloc[cat_data[revenue_col].argmax()][best_cat_col]
                    top_percent = cat_data[revenue_col].max() / cat_data[revenue_col].sum() * 100
                    
                    st.markdown(
                        f"""
                        <div style='background: rgba(60, 63, 68, 0.7); padding: 15px; border-radius: 10px; margin-top: 15px;'>
                            <p style='color: #E0E0FF; margin: 0;'>
                                <strong>{top_category}</strong> accounts for <strong>{top_percent:.1f}%</strong> of total {revenue_col}.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.write("No suitable categorical columns found")
            else:
                # Fallback to another categorical visualization
                if categorical_cols:
                    st.markdown("<h5 style='color: #B19CD9;'>Category Distribution</h5>", unsafe_allow_html=True)
                    
                    # Find categorical column with reasonable number of categories
                    suitable_cols = [col for col in categorical_cols if 2 <= df[col].nunique() <= 10]
                    if suitable_cols:
                        selected_cat = suitable_cols[0]
                        
                        # Create count plot
                        cat_counts = df[selected_cat].value_counts().reset_index()
                        cat_counts.columns = [selected_cat, 'Count']
                        
                        fig = px.bar(
                            cat_counts,
                            x=selected_cat,
                            y='Count',
                            color='Count',
                            color_continuous_scale="Purp"
                        )
                        
                        fig.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="rgba(40, 42, 54, 0.8)",
                            paper_bgcolor="rgba(40, 42, 54, 0)",
                            margin=dict(l=20, r=20, t=50, b=20),
                            title={
                                "text": f"Distribution of {selected_cat}",
                                "y": 0.95,
                                "x": 0.5,
                                "xanchor": "center",
                                "yanchor": "top",
                                "font": {"color": "#9D72FF", "size": 16}
                            }
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("Categorical columns have too many unique values for visualization")
                else:
                    # If no categorical columns, show data summary
                    st.markdown("<h5 style='color: #B19CD9;'>Data Summary</h5>", unsafe_allow_html=True)
                    
                    numeric_df = df.select_dtypes(include=['number'])
                    if not numeric_df.empty:
                        summary_stats = numeric_df.describe().T[['mean', 'min', 'max', 'std']].reset_index()
                        summary_stats.columns = ['Metric', 'Mean', 'Min', 'Max', 'Std Dev']
                        
                        st.dataframe(summary_stats, use_container_width=True)
                    else:
                        st.write("No numeric data available for summary statistics")
        
        # AI Insights Panel
        if st.session_state.insights:
            render_ai_insights(st.session_state.insights)
        
        # Interactive Q&A Section
        display_qa_section(df)
    else:
        # Prompt to upload data
        st.markdown(
            """
            <div style='text-align: center; padding: 50px 20px; animation: fadeIn 1s;'>
                <img src="https://cdn-icons-png.flaticon.com/512/6571/6571582.png" style="width: 100px; opacity: 0.5;">
                <h2 style='color: #9D72FF; margin-top: 20px;'>Upload Data to View Dashboard</h2>
                <p style='color: #B19CD9; margin-bottom: 30px;'>
                    Please navigate to the Upload Data tab to load your dataset.
                </p>
                <a href="#" style='
                    display: inline-block;
                    padding: 10px 20px;
                    background: linear-gradient(90deg, #9D72FF 0%, #B19CD9 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                    font-weight: 600;
                    transition: all 0.3s ease;
                    cursor: pointer;
                '>Go to Upload Data</a>
            </div>
            """,
            unsafe_allow_html=True
        )

# Advanced Analysis Page
elif navigation == "Advanced Analysis":
    if st.session_state.data is not None:
        # Advanced Analysis Header
        st.markdown(
            """
            <div class='dashboard-header'>
                <h1 style='color: #9D72FF; margin-bottom: 5px;'>Advanced Analytics Tools</h1>
                <p style='color: #B19CD9; margin-top: 0;'>
                    Deep dive tools for comprehensive data exploration and analysis
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Create tabs for different analysis tools
        analysis_tabs = st.tabs(["Data Explorer", "Correlation Analysis", "Segment Analysis", "Time Series"])
        
        with analysis_tabs[0]:
            # Data Explorer
            st.markdown("<h4 style='color: #B19CD9;'>Interactive Data Explorer</h4>", unsafe_allow_html=True)
            
            df = st.session_state.data
            
            # Filter controls
            st.markdown("<p style='color: #9D72FF;'><strong>Data Filters</strong></p>", unsafe_allow_html=True)
            
            # Dynamically create filters based on data types
            filters_applied = False
            filtered_df = df.copy()
            
            # Categorical filters
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            if cat_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    if cat_cols:
                        cat_filter_col = st.selectbox("Select category to filter", ["None"] + cat_cols)
                        
                        if cat_filter_col != "None":
                            cat_values = ["All"] + sorted(df[cat_filter_col].unique().tolist())
                            selected_cats = st.multiselect("Select values", cat_values, ["All"])
                            
                            if selected_cats and "All" not in selected_cats:
                                filtered_df = filtered_df[filtered_df[cat_filter_col].isin(selected_cats)]
                                filters_applied = True
                
                with col2:
                    # Second categorical filter if available
                    remaining_cat_cols = [col for col in cat_cols if col != cat_filter_col]
                    if remaining_cat_cols:
                        cat_filter_col2 = st.selectbox("Select second category", ["None"] + remaining_cat_cols)
                        
                        if cat_filter_col2 != "None":
                            cat_values2 = ["All"] + sorted(df[cat_filter_col2].unique().tolist())
                            selected_cats2 = st.multiselect("Select values", cat_values2, ["All"], key="second_cat")
                            
                            if selected_cats2 and "All" not in selected_cats2:
                                filtered_df = filtered_df[filtered_df[cat_filter_col2].isin(selected_cats2)]
                                filters_applied = True
            
            # Numeric range filters
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            if num_cols:
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    if num_cols:
                        num_filter_col = st.selectbox("Select numeric range to filter", ["None"] + num_cols)
                        
                        if num_filter_col != "None":
                            min_val = float(df[num_filter_col].min())
                            max_val = float(df[num_filter_col].max())
                            
                            num_range = st.slider(
                                f"Range for {num_filter_col}",
                                min_val,
                                max_val,
                                (min_val, max_val)
                            )
                            
                            if num_range[0] > min_val or num_range[1] < max_val:
                                filtered_df = filtered_df[
                                    (filtered_df[num_filter_col] >= num_range[0]) & 
                                    (filtered_df[num_filter_col] <= num_range[1])
                                ]
                                filters_applied = True
                
                with col2:
                    # Second numeric filter if available
                    remaining_num_cols = [col for col in num_cols if col != num_filter_col]
                    if remaining_num_cols:
                        num_filter_col2 = st.selectbox("Select second numeric range", ["None"] + remaining_num_cols)
                        
                        if num_filter_col2 != "None":
                            min_val2 = float(df[num_filter_col2].min())
                            max_val2 = float(df[num_filter_col2].max())
                            
                            num_range2 = st.slider(
                                f"Range for {num_filter_col2}",
                                min_val2,
                                max_val2,
                                (min_val2, max_val2),
                                key="second_num"
                            )
                            
                            if num_range2[0] > min_val2 or num_range2[1] < max_val2:
                                filtered_df = filtered_df[
                                    (filtered_df[num_filter_col2] >= num_range2[0]) & 
                                    (filtered_df[num_filter_col2] <= num_range2[1])
                                ]
                                filters_applied = True
            
            # Date range filter if date columns exist
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                st.markdown("<br>", unsafe_allow_html=True)
                date_filter_col = st.selectbox("Select date range to filter", ["None"] + date_cols)
                
                if date_filter_col != "None":
                    # Convert to datetime if needed
                    if df[date_filter_col].dtype != 'datetime64[ns]':
                        df[date_filter_col] = pd.to_datetime(df[date_filter_col], errors='coerce')
                        filtered_df[date_filter_col] = pd.to_datetime(filtered_df[date_filter_col], errors='coerce')
                    
                    min_date = df[date_filter_col].min().date()
                    max_date = df[date_filter_col].max().date()
                    
                    date_range = st.date_input(
                        f"Range for {date_filter_col}",
                        [min_date, max_date]
                    )
                    
                    if len(date_range) == 2:
                        if date_range[0] > min_date or date_range[1] < max_date:
                            filtered_df = filtered_df[
                                (filtered_df[date_filter_col].dt.date >= date_range[0]) & 
                                (filtered_df[date_filter_col].dt.date <= date_range[1])
                            ]
                            filters_applied = True
            
            # Display filter summary
            # Display filter summary
            if filters_applied:
                st.markdown(
                    f"""
                    <div style='background: rgba(157, 114, 255, 0.1); padding: 10px 15px; border-radius: 5px; 
                         border-left: 3px solid #9D72FF; margin: 15px 0;'>
                        <p style='color: #B19CD9; margin: 0;'>
                            <strong>Filters Applied:</strong> Showing {len(filtered_df)} of {len(df)} records ({(len(filtered_df)/len(df)*100):.1f}%)
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            
            # Data table display
            st.markdown("<hr style='margin: 30px 0; opacity: 0.2;'>", unsafe_allow_html=True)
            st.markdown("<p style='color: #9D72FF;'><strong>Filtered Data Preview</strong></p>", unsafe_allow_html=True)
            
            # Column selector
            all_cols = filtered_df.columns.tolist()
            selected_cols = st.multiselect("Select columns to display", all_cols, all_cols[:6] if len(all_cols) > 6 else all_cols)
            
            if selected_cols:
                st.dataframe(filtered_df[selected_cols].head(100), use_container_width=True)
                
                # Download option
                csv = filtered_df[selected_cols].to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                
                st.markdown(
                    f"""
                    <div style='text-align: right; margin-top: 10px;'>
                        <a href="data:file/csv;base64,{b64}" download="filtered_data.csv" style='
                            display: inline-block;
                            padding: 8px 15px;
                            background: rgba(157, 114, 255, 0.2);
                            color: #9D72FF;
                            text-decoration: none;
                            border-radius: 5px;
                            font-size: 14px;
                            border: 1px solid rgba(157, 114, 255, 0.5);
                        '>
                            <span style='display: flex; align-items: center; gap: 5px;'>
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                                    <polyline points="7 10 12 15 17 10"></polyline>
                                    <line x1="12" y1="15" x2="12" y2="3"></line>
                                </svg>
                                Download Filtered Data
                            </span>
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            # Data display options
            st.markdown("<hr style='margin: 20px 0; opacity: 0.2;'>", unsafe_allow_html=True)
            st.markdown("<p style='color: #9D72FF;'><strong>Data Visualization</strong></p>", unsafe_allow_html=True)
            
            # Choose columns for visualization
            vis_options = st.columns(2)
            
            with vis_options[0]:
                num_features = df.select_dtypes(include=['number']).columns.tolist()
                x_axis = st.selectbox("X-axis (Feature)", num_features if num_features else ["None"])
            
            with vis_options[1]:
                y_axis = st.selectbox("Y-axis (Target)", [col for col in num_features if col != x_axis] if len(num_features) > 1 else ["None"])
            
            if x_axis != "None" and y_axis != "None":
                # Create visualization
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Create color option if categorical columns exist
                color_by = None
                cat_cols = filtered_df.select_dtypes(include=['object']).columns.tolist()
                if cat_cols:
                    color_options = ["None"] + cat_cols
                    color_by = st.selectbox("Color points by category", color_options)
                
                # Create scatter plot
                fig = px.scatter(
                    filtered_df,
                    x=x_axis,
                    y=y_axis,
                    color=color_by if color_by and color_by != "None" else None,
                    opacity=0.7,
                    size_max=10,
                    color_discrete_sequence=px.colors.qualitative.Pastel if color_by and color_by != "None" else ["#9D72FF"]
                )
                
                # Add trendline if no color grouping
                if not color_by or color_by == "None":
                    fig.update_layout(
                        shapes=[{
                            'type': 'line',
                            'x0': min(filtered_df[x_axis]),
                            'y0': filtered_df[y_axis].mean(),
                            'x1': max(filtered_df[x_axis]),
                            'y1': filtered_df[y_axis].mean(),
                            'line': {
                                'color': 'rgba(255, 255, 255, 0.3)',
                                'width': 2,
                                'dash': 'dash'
                            }
                        }]
                    )
                
                # Update layout
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(40, 42, 54, 0.8)",
                    paper_bgcolor="rgba(40, 42, 54, 0)",
                    margin=dict(l=20, r=20, t=50, b=20),
                    title={
                        "text": f"Relationship between {x_axis} and {y_axis}",
                        "y": 0.95,
                        "x": 0.5,
                        "xanchor": "center",
                        "yanchor": "top",
                        "font": {"color": "#9D72FF", "size": 18}
                    },
                    xaxis=dict(
                        title=x_axis,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor="rgba(107, 114, 142, 0.2)"
                    ),
                    yaxis=dict(
                        title=y_axis,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor="rgba(107, 114, 142, 0.2)"
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add correlation information
                correlation = filtered_df[[x_axis, y_axis]].corr().iloc[0, 1]
                
                # Interpret correlation strength
                corr_strength = "strong"
                if abs(correlation) < 0.3:
                    corr_strength = "weak"
                elif abs(correlation) < 0.7:
                    corr_strength = "moderate"
                
                # Interpret direction
                corr_direction = "positive" if correlation > 0 else "negative"
                
                st.markdown(
                    f"""
                    <div style='background: rgba(40, 42, 54, 0.5); padding: 15px; border-radius: 5px; margin: 15px 0;'>
                        <p style='color: #E0E0FF; margin: 0;'>
                            <strong>Correlation Analysis:</strong> There is a <span style='color: {"#64FFDA" if correlation > 0 else "#FF5370"};'>{corr_strength} {corr_direction}</span> 
                            correlation (r = {correlation:.3f}) between {x_axis} and {y_axis}.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            

        with analysis_tabs[1]:
            # Correlation Analysis
            st.markdown("<h4 style='color: #B19CD9;'>Correlation Matrix & Insights</h4>", unsafe_allow_html=True)
            
            df = st.session_state.data
            
            # Get numeric columns
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(num_cols) > 1:
                # Column selection
                selected_corr_cols = st.multiselect(
                    "Select columns for correlation analysis",
                    num_cols,
                    num_cols[:5] if len(num_cols) > 5 else num_cols
                )
                
                if selected_corr_cols and len(selected_corr_cols) > 1:
                    # Generate correlation matrix
                    corr_matrix = df[selected_corr_cols].corr()
                    
                    # Create heatmap
                    fig = px.imshow(
                        corr_matrix,
                        text_auto='.2f',
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1,
                        aspect="auto"
                    )
                    
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(40, 42, 54, 0.8)",
                        paper_bgcolor="rgba(40, 42, 54, 0)",
                        margin=dict(l=20, r=20, t=50, b=20),
                        title={
                            "text": "Correlation Matrix Heatmap",
                            "y": 0.95,
                            "x": 0.5,
                            "xanchor": "center",
                            "yanchor": "top",
                            "font": {"color": "#9D72FF", "size": 18}
                        }
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Find top correlations
                    corr_pairs = []
                    for i in range(len(selected_corr_cols)):
                        for j in range(i+1, len(selected_corr_cols)):
                            col1 = selected_corr_cols[i]
                            col2 = selected_corr_cols[j]
                            corr_value = corr_matrix.iloc[i, j]
                            corr_pairs.append((col1, col2, corr_value))
                    
                    # Sort by absolute correlation value
                    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                    
                    # Display top correlations
                    st.markdown("<p style='color: #9D72FF;'><strong>Top Correlations</strong></p>", unsafe_allow_html=True)
                    
                    for i, (col1, col2, corr) in enumerate(corr_pairs[:5]):
                        # Determine correlation strength
                        if abs(corr) >= 0.7:
                            strength = "Strong"
                            color = "#64FFDA" if corr > 0 else "#FF5370"
                        elif abs(corr) >= 0.3:
                            strength = "Moderate"
                            color = "#C3E88D" if corr > 0 else "#F78C6C"
                        else:
                            strength = "Weak"
                            color = "#B2CCD6" if corr > 0 else "#EEFFFF"
                        
                        direction = "positive" if corr > 0 else "negative"
                        
                        st.markdown(
                            f"""
                            <div style='background: rgba(40, 42, 54, 0.5); padding: 15px; border-radius: 5px; margin: 10px 0;'>
                                <p style='color: #E0E0FF; margin: 0;'>
                                    <span style='display: inline-block; width: 24px; height: 24px; text-align: center; 
                                           background: {color}; color: rgba(40, 42, 54, 1); border-radius: 12px; 
                                           margin-right: 10px; font-weight: bold;'>{i+1}</span>
                                    <strong>{col1}</strong> and <strong>{col2}</strong>: 
                                    <span style='color: {color};'>{strength} {direction}</span> correlation (r = {corr:.3f})
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # If it's the top correlation, show scatterplot
                        if i == 0:
                            # Create scatter plot for top correlation
                            scatter_fig = px.scatter(
                                df,
                                x=col1,
                                y=col2,
                                trendline="ols",
                                opacity=0.7,
                                color_discrete_sequence=["#9D72FF"]
                            )
                            
                            scatter_fig.update_layout(
                                template="plotly_dark",
                                plot_bgcolor="rgba(40, 42, 54, 0.8)",
                                paper_bgcolor="rgba(40, 42, 54, 0)",
                                margin=dict(l=20, r=20, t=50, b=20),
                                title={
                                    "text": f"Strongest Relationship: {col1} vs {col2}",
                                    "y": 0.95,
                                    "x": 0.5,
                                    "xanchor": "center",
                                    "yanchor": "top",
                                    "font": {"color": "#9D72FF", "size": 16}
                                }
                            )
                            
                            st.plotly_chart(scatter_fig, use_container_width=True)
                else:
                    st.info("Please select at least two columns to generate correlation matrix")
            else:
                st.write("Not enough numeric columns for correlation analysis.")

        with analysis_tabs[2]:
            # Segment Analysis
            st.markdown("<h4 style='color: #B19CD9;'>Customer/Product Segment Analysis</h4>", unsafe_allow_html=True)
            
            df = st.session_state.data
            
            # Find appropriate categorical column for segmentation
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if cat_cols and num_cols:
                # Segment selection
                segment_col = st.selectbox("Select segment column", cat_cols)
                
                # Metric selection 
                metric_col = st.selectbox("Select metric to analyze", num_cols)
                
                if segment_col and metric_col:
                    # Get segment data
                    segment_data = df.groupby(segment_col)[metric_col].agg(['mean', 'sum', 'count']).reset_index()
                    segment_data.columns = [segment_col, 'Average', 'Total', 'Count']
                    
                    # Calculate percentage of total
                    segment_data['Percentage'] = (segment_data['Total'] / segment_data['Total'].sum() * 100).round(1)
                    
                    # Sort by total value
                    segment_data = segment_data.sort_values('Total', ascending=False)
                    
                    # Display segment insights
                    st.markdown(
                        f"""
                        <div style='background: rgba(40, 42, 54, 0.5); padding: 20px; border-radius: 5px; margin: 15px 0;'>
                            <h5 style='color: #9D72FF; margin-top: 0;'>Segment Overview: {segment_col} by {metric_col}</h5>
                            <p style='color: #E0E0FF;'>
                                The analysis shows <strong>{len(segment_data)}</strong> distinct segments. 
                                The top segment accounts for <strong>{segment_data['Percentage'].max():.1f}%</strong> of the total {metric_col}.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Visualization tabs
                    viz_tabs = st.tabs(["Bar Chart", "Pie Chart", "Treemap", "Data Table",])
                    
                    with viz_tabs[0]:
                        # Create bar chart
                        fig = px.bar(
                            segment_data,
                            x=segment_col,
                            y='Total',
                            text='Percentage',
                            color='Average',
                            color_continuous_scale="Purp",
                            labels={'Total': f'Total {metric_col}', 'Average': f'Avg {metric_col}'}
                        )
                        
                        fig.update_traces(texttemplate='%{text}%', textposition='outside')
                        
                        fig.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="rgba(40, 42, 54, 0.8)",
                            paper_bgcolor="rgba(40, 42, 54, 0)",
                            margin=dict(l=20, r=20, t=50, b=20),
                            title={
                                "text": f"{segment_col} Segments by {metric_col}",
                                "y": 0.95,
                                "x": 0.5,
                                "xanchor": "center",
                                "yanchor": "top",
                                "font": {"color": "#9D72FF", "size": 18}
                            }
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_tabs[1]:  # Pie Chart Tab
                        st.markdown("<h5 style='color: #B19CD9;'>Segment Distribution (Pie Chart)</h5>", unsafe_allow_html=True)

                        fig_pie = px.pie(
                            segment_data,
                            names=segment_col,
                            values='Total',
                            color_discrete_sequence=px.colors.sequential.Plasma,
                            title=f"Distribution of {segment_col}"
                        )
                        fig_pie.update_traces(textinfo='percent+label', pull=[0.1 if i == 0 else 0 for i in range(len(segment_data))])

                        fig_pie.update_layout(
                            template="plotly_dark",
                            margin=dict(t=50, b=20, l=20, r=20),
                            title=dict(x=0.5, xanchor="center", font=dict(size=18, color="#9D72FF"))
                        )

                        st.plotly_chart(fig_pie, use_container_width=True)

                    with viz_tabs[2]:
                        # Create treemap for hierarchical view
                        fig = px.treemap(
                            segment_data,
                            path=[segment_col],
                            values='Total',
                            color='Average',
                            color_continuous_scale="Purp",
                            hover_data=['Percentage', 'Count']
                        )
                        
                        fig.update_layout(
                            template="plotly_dark",
                            margin=dict(l=20, r=20, t=50, b=20),
                            title={
                                "text": f"Treemap of {segment_col} Segments",
                                "y": 0.95,
                                "x": 0.5,
                                "xanchor": "center",
                                "yanchor": "top",
                                "font": {"color": "#9D72FF", "size": 18}
                            }
                        )
                        
                        fig.update_traces(textinfo="label+value+percent parent")
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_tabs[3]:
                        # Format the dataframe for display
                        display_df = segment_data.copy()
                        
                        # Format columns
                        display_df['Average'] = display_df['Average'].round(2)
                        display_df['Percentage'] = display_df['Percentage'].astype(str) + '%'
                        
                        # Add styling
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Calculate concentration metrics
                        top_segments = segment_data.head(3)
                        concentration = top_segments['Total'].sum() / segment_data['Total'].sum() * 100
                        
                        st.markdown(
                            f"""
                            <div style='background: rgba(40, 42, 54, 0.5); padding: 15px; border-radius: 5px; margin: 15px 0;'>
                                <p style='color: #E0E0FF; margin: 0;'>
                                    <strong>Segment Concentration:</strong> Top 3 segments account for <strong>{concentration:.1f}%</strong> of total {metric_col}.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    
            else:
                st.write("Need both categorical and numeric columns for segment analysis.")

        with analysis_tabs[3]:
            # Time Series Analysis
            st.markdown("<h4 style='color: #B19CD9;'>Time Series Analysis</h4>", unsafe_allow_html=True)
            
            df = st.session_state.data
            
            # Look for date columns
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            
            if date_cols:
                # Date column selection
                date_col = st.selectbox("Select date column", date_cols)
                
                # Convert to datetime if needed
                if df[date_col].dtype != 'datetime64[ns]':
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                # Drop rows with invalid dates
                valid_df = df.dropna(subset=[date_col])
                
                if not valid_df.empty:
                    # Select metric for time series
                    num_cols = valid_df.select_dtypes(include=['number']).columns.tolist()
                    if num_cols:
                        metric_col = st.selectbox("Select metric for time analysis", num_cols)
                        
                        # Group by options
                        date_range = (valid_df[date_col].max() - valid_df[date_col].min()).days
                        
                        if date_range > 365*2:
                            default_period = 'Year'
                            period_options = ['Year', 'Quarter', 'Month']
                        elif date_range > 180:
                            default_period = 'Month'
                            period_options = ['Quarter', 'Month', 'Week']
                        elif date_range > 60:
                            default_period = 'Week'
                            period_options = ['Month', 'Week', 'Day']
                        else:
                            default_period = 'Day'
                            period_options = ['Week', 'Day']
                        
                        period = st.radio("Group by:", period_options, period_options.index(default_period) if default_period in period_options else 0)
                        
                        # Define frequency mapping
                        freq_map = {
                            'Year': 'Y',
                            'Quarter': 'Q',
                            'Month': 'M',
                            'Week': 'W',
                            'Day': 'D'
                        }
                        
                        # Aggregate data
                        time_df = valid_df.set_index(date_col)
                        time_df = time_df.resample(freq_map[period])[metric_col].agg(['sum', 'mean', 'count'])
                        time_df.reset_index(inplace=True)
                        
                        # Calculate period-over-period change
                        time_df['pct_change'] = time_df['sum'].pct_change() * 100
                        
                        # Create time series visualization
                        fig = go.Figure()
                        
                        # Add bar chart for sum
                        fig.add_trace(
                            go.Bar(
                                x=time_df[date_col],
                                y=time_df['sum'],
                                name=f"Total {metric_col}",
                                marker_color="rgba(157, 114, 255, 0.7)"
                            )
                        )
                        
                        # Add line for the mean
                        fig.add_trace(
                            go.Scatter(
                                x=time_df[date_col],
                                y=time_df['mean'],
                                mode='lines+markers',
                                name=f"Average {metric_col}",
                                line=dict(color='rgba(100, 255, 218, 0.8)', width=3),
                                marker=dict(size=8, line=dict(width=2, color='#64FFDA')),
                                yaxis="y2"
                            )
                        )
                        
                        # Update layout with double y-axis
                        fig.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="rgba(40, 42, 54, 0.8)",
                            paper_bgcolor="rgba(40, 42, 54, 0)",
                            margin=dict(l=20, r=20, t=50, b=20),
                            title={
                                "text": f"{metric_col} Over Time (by {period})",
                                "y": 0.95,
                                "x": 0.5,
                                "xanchor": "center",
                                "yanchor": "top",
                                "font": {"color": "#9D72FF", "size": 18}
                            },
                            xaxis=dict(
                                title=f"{period}",
                                showgrid=True,
                                gridwidth=1,
                                gridcolor="rgba(107, 114, 142, 0.2)"
                            ),
                            yaxis=dict(
                                title=f"Total {metric_col}",
                                showgrid=True,
                                gridwidth=1,
                                gridcolor="rgba(107, 114, 142, 0.2)"
                            ),
                            yaxis2=dict(
                                title=f"Average {metric_col}",
                                overlaying="y",
                                side="right"
                            ),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Growth analysis
                        growth_tabs = st.tabs(["Growth Analysis", "Trend Details"])
                        
                        with growth_tabs[0]:
                            # Create growth visualization
                            growth_fig = go.Figure()
                            
                            # Add bar chart for period-over-period change
                            growth_fig.add_trace(
                                go.Bar(
                                    x=time_df[date_col][1:],  # Skip first entry since pct_change creates NaN
                                    y=time_df['pct_change'][1:],
                                    name="% Change",
                                    marker_color=["rgba(255, 83, 112, 0.7)" if x < 0 else "rgba(100, 255, 218, 0.7)" 
                                                for x in time_df['pct_change'][1:]]
                                )
                            )
                            
                            # Add zero line
                            growth_fig.add_shape(
                                type="line",
                                x0=time_df[date_col].iloc[1],
                                y0=0,
                                x1=time_df[date_col].iloc[-1],
                                y1=0,
                                line=dict(color="rgba(255, 255, 255, 0.5)", width=2, dash="dot")
                            )
                            
                            # Update layout
                            growth_fig.update_layout(
                                template="plotly_dark",
                                plot_bgcolor="rgba(40, 42, 54, 0.8)",
                                paper_bgcolor="rgba(40, 42, 54, 0)",
                                margin=dict(l=20, r=20, t=50, b=20),
                                title={
                                    "text": f"{period}-over-{period} Growth Rate",
                                    "y": 0.95,
                                    "x": 0.5,
                                    "xanchor": "center",
                                    "yanchor": "top",
                                    "font": {"color": "#9D72FF", "size": 18}
                                },
                                yaxis=dict(
                                    title="% Change",
                                    showgrid=True,
                                    zeroline=False,
                                    gridwidth=1,
                                    gridcolor="rgba(107, 114, 142, 0.2)"
                                )
                            )
                            
                            st.plotly_chart(growth_fig, use_container_width=True)
                            
                            # Calculate growth metrics
                            avg_growth = time_df['pct_change'][1:].mean()
                            positive_periods = (time_df['pct_change'] > 0).sum()
                            total_periods = len(time_df) - 1  # Subtract 1 because first period has no growth rate
                            
                            st.markdown(
                                f"""
                                <div style='background: rgba(40, 42, 54, 0.5); padding: 15px; border-radius: 5px; margin: 15px 0;'>
                                    <p style='color: #E0E0FF; margin: 0;'>
                                        <strong>Growth Insights:</strong> Average {period.lower()}-over-{period.lower()} growth rate is 
                                        <span style='color: {"#64FFDA" if avg_growth > 0 else "#FF5370"};'>{avg_growth:.2f}%</span>.
                                        Positive growth in {positive_periods} out of {total_periods} periods 
                                        ({(positive_periods/total_periods*100):.1f}% of the time).
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with growth_tabs[1]:
                            # Time series decomposition if enough periods
                            if len(time_df) >= 6:
                                try:
                                    # Resample to regular frequency if needed
                                    ts_data = time_df.set_index(date_col)['sum']
                                    
                                    # Get trend using moving average
                                    window_size = min(5, len(ts_data) // 2)
                                    if window_size % 2 == 0:  # Make window size odd
                                        window_size += 1
                                    
                                    ts_data_clean = ts_data.dropna()
                                    if len(ts_data_clean) >= window_size:
                                        # Calculate trend
                                        trend = ts_data_clean.rolling(window=window_size, center=True).mean()
                                        
                                        # Create decomposition plot
                                        decomp_fig = go.Figure()
                                        
                                        # Add original data
                                        decomp_fig.add_trace(
                                            go.Scatter(
                                                x=ts_data_clean.index,
                                                y=ts_data_clean.values,
                                                mode='lines+markers',
                                                name="Original Data",
                                                line=dict(color='rgba(157, 114, 255, 0.8)', width=2),
                                                marker=dict(size=6)
                                            )
                                        )
                                        
                                        # Add trend
                                        decomp_fig.add_trace(
                                            go.Scatter(
                                                x=trend.index,
                                                y=trend.values,
                                                mode='lines',
                                                name="Trend Component",
                                                                                                line=dict(color='rgba(100, 255, 218, 0.8)', width=3)
                                            )
                                        )
                                        
                                        # Update layout
                                        decomp_fig.update_layout(
                                            template="plotly_dark",
                                            plot_bgcolor="rgba(40, 42, 54, 0.8)",
                                            paper_bgcolor="rgba(40, 42, 54, 0)",
                                            margin=dict(l=20, r=20, t=50, b=20),
                                            title={
                                                "text": f"Trend Analysis of {metric_col} Over Time",
                                                "y": 0.95,
                                                "x": 0.5,
                                                "xanchor": "center",
                                                "yanchor": "top",
                                                "font": {"color": "#9D72FF", "size": 18}
                                            },
                                            xaxis=dict(
                                                title=f"{period}",
                                                showgrid=True,
                                                gridwidth=1,
                                                gridcolor="rgba(107, 114, 142, 0.2)"
                                            ),
                                            yaxis=dict(
                                                title=f"{metric_col}",
                                                showgrid=True,
                                                gridwidth=1,
                                                gridcolor="rgba(107, 114, 142, 0.2)"
                                            ),
                                            legend=dict(
                                                orientation="h",
                                                yanchor="bottom",
                                                y=1.02,
                                                xanchor="right",
                                                x=1
                                            )
                                        )
                                        
                                        st.plotly_chart(decomp_fig, use_container_width=True)
                                        
                                        # Calculate trend metrics
                                        trend_growth = (trend.iloc[-1] - trend.iloc[0]) / trend.iloc[0] * 100
                                        st.markdown(
                                            f"""
                                            <div style='background: rgba(40, 42, 54, 0.5); padding: 15px; border-radius: 5px; margin: 15px 0;'>
                                                <p style='color: #E0E0FF; margin: 0;'>
                                                    <strong>Trend Insights:</strong> The overall trend shows a 
                                                    <span style='color: {"#64FFDA" if trend_growth > 0 else "#FF5370"};'>{trend_growth:.2f}%</span> 
                                                    change from the beginning to the end of the period.
                                                </p>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )
                                    else:
                                        st.warning("Not enough data points to calculate trend.")
                                except Exception as e:
                                    st.error(f"Error in trend analysis: {e}")
                            else:
                                st.warning("At least 6 periods are required for trend analysis.")
                    else:
                        st.warning("No numeric columns available for time series analysis.")
                else:
                    st.warning("No valid dates found in the selected column.")
            else:
                st.warning("No date or time columns found for time series analysis.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE HANDLERS — Sector Analysis, ML Forecasting, NLP Insights
# ═══════════════════════════════════════════════════════════════════════════════
elif navigation == "Sector Analysis":
    if st.session_state.data is not None:
        render_sector_analysis(st.session_state.data)
    else:
        st.markdown("""
            <div style='text-align:center; padding:80px 20px;'>
                <h2 style='color:#9D72FF;'>No Data Loaded</h2>
                <p style='color:#B19CD9; font-size:16px;'>
                    Please go to <strong>Upload Data</strong> in the sidebar and load a dataset first.
                </p>
            </div>
        """, unsafe_allow_html=True)

elif navigation == "ML Forecasting":
    if st.session_state.data is not None:
        render_ml_forecasting(st.session_state.data)
    else:
        st.markdown("""
            <div style='text-align:center; padding:80px 20px;'>
                <h2 style='color:#9D72FF;'>No Data Loaded</h2>
                <p style='color:#B19CD9; font-size:16px;'>
                    Please go to <strong>Upload Data</strong> in the sidebar and load a dataset first.
                </p>
            </div>
        """, unsafe_allow_html=True)

elif navigation == "NLP Insights":
    if st.session_state.data is not None:
        render_nlp_insights_page(st.session_state.data)
    else:
        st.markdown("""
            <div style='text-align:center; padding:80px 20px;'>
                <h2 style='color:#9D72FF;'>No Data Loaded</h2>
                <p style='color:#B19CD9; font-size:16px;'>
                    Please go to <strong>Upload Data</strong> in the sidebar and load a dataset first.
                </p>
            </div>
        """, unsafe_allow_html=True)

elif navigation == "Intelligence Report":
    if st.session_state.data is not None:
        render_intelligence_report(st.session_state.data)
    else:
        st.markdown("""
            <div style='text-align:center; padding:80px 20px;'>
                <h2 style='color:#9D72FF;'>No Data Loaded</h2>
                <p style='color:#B19CD9; font-size:16px;'>
                    Please go to <strong>Upload Data</strong> in the sidebar and load a dataset first.
                </p>
            </div>
        """, unsafe_allow_html=True)
