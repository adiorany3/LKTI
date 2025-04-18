import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import io
from matplotlib.figure import Figure
import seaborn as sns

# Set page configuration and title
st.set_page_config(
    page_title="LKTI Data Analysis",
    page_icon="📊",
    layout="wide"
)

# Main title
st.title("LKTI Data Analysis Dashboard")
st.write("Interactive analysis of dosage effects on Body Weight Gain, Carcass Percentage, and FCR")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    confidence_level = st.slider("Confidence Level for Statistical Tests", 
                                min_value=0.80, max_value=0.99, value=0.95, step=0.01)
    alpha = 1 - confidence_level
    st.write(f"Alpha level (significance threshold): {alpha:.2f}")

# Function to find the best matching column based on keywords
def find_column(keywords, df_columns):
    for keyword in keywords:
        matches = [col for col in df_columns if keyword.lower() in col.lower()]
        if matches:
            return matches[0]
    return None

# Function to load and prepare data
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Function to analyze data
def analyze_data(df):
    # Find the relevant columns
    dosage_column = find_column(['dosage', 'dose', 'dosis'], df.columns)
    bwg_column = find_column(['body weight gain', 'weight gain', 'bwg'], df.columns)
    carcass_column = find_column(['carcass', 'carcas', 'karkas'], df.columns)
    fcr_column = find_column(['fcr', 'feed conversion'], df.columns)
    
    # Check if all necessary columns were found
    missing = []
    if not dosage_column: missing.append("Dosage")
    if not bwg_column: missing.append("Body Weight Gain")
    if not carcass_column: missing.append("Carcass Percentage")
    if not fcr_column: missing.append("FCR")
    
    if missing:
        st.warning(f"⚠️ Could not find columns for: {', '.join(missing)}")
        st.info(f"Available columns: {', '.join(df.columns)}")
        return None
    
    # Prepare dataframe with only the columns we need
    analysis_df = df[[dosage_column, bwg_column, carcass_column, fcr_column]].copy()
    
    # For readability, rename the columns to standard names
    analysis_df.columns = ['Dosage', 'Body_Weight_Gain', 'Carcass_Percentage', 'FCR']
    
    # Try to convert to numeric, coercing errors to NaN
    for col in analysis_df.columns:
        analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')
    
    # Remove rows with NaN values after conversion
    analysis_df = analysis_df.dropna()
    
    return analysis_df

# Main processing logic - use only the existing file
file_path = "Data LKTI.xlsx"

if os.path.exists(file_path):
    df = load_data(file_path)
    st.success(f"✅ Loaded data file: {file_path}")
else:
    st.error(f"⚠️ Data file '{file_path}' not found. Please ensure the file exists in the current directory.")
    st.stop()

# Display raw data with toggleable view
with st.expander("View Raw Data", expanded=False):
    st.dataframe(df, use_container_width=True)
    st.download_button(
        label="Download Data as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='data_export.csv',
        mime='text/csv'
    )

# Process data for analysis
analysis_df = analyze_data(df)

if analysis_df is None or len(analysis_df) < 2:
    st.error("⚠️ Not enough valid data for analysis. Please check your data file.")
    st.stop()

# Display cleaned analysis data
with st.expander("View Analysis Data", expanded=False):
    st.dataframe(analysis_df, use_container_width=True)
    st.write(f"Number of rows after cleaning: {len(analysis_df)}")
    
    # Show basic statistics
    st.subheader("Statistical Summary")
    st.dataframe(analysis_df.describe(), use_container_width=True)

# Determine if Dosage is categorical or continuous
unique_dosages = analysis_df['Dosage'].unique()
is_categorical = len(unique_dosages) < 10  # Assuming fewer than 10 values means categorical

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["Data Visualization", "Statistical Analysis", "Correlation Analysis"])

with tab1:
    st.header("Data Visualization")
    
    # Select visualization type
    viz_type = st.radio(
        "Select plot type:",
        ["Bar Chart", "Scatter Plot", "Box Plot"]
    )
    
    # Select variables for plotting
    metric = st.selectbox(
        "Select metric to visualize:",
        ["Body_Weight_Gain", "Carcass_Percentage", "FCR"]
    )
    
    # Create the visualization
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    if viz_type == "Bar Chart":
        if is_categorical:
            # Group data by dosage and calculate mean
            grouped = analysis_df.groupby('Dosage')[metric].mean().reset_index()
            ax.bar(grouped['Dosage'].astype(str), grouped[metric])
            ax.set_xlabel('Dosage')
            ax.set_ylabel(metric.replace('_', ' '))
            ax.set_title(f'{metric.replace("_", " ")} by Dosage')
        else:
            st.warning("Bar charts are better for categorical dosages. Consider using a scatter plot instead.")
            # Create a histogram instead
            ax.hist(analysis_df['Dosage'], bins=10, alpha=0.7)
            ax.set_xlabel('Dosage')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Dosage')
    
    elif viz_type == "Scatter Plot":
        ax.scatter(analysis_df['Dosage'], analysis_df[metric])
        ax.set_xlabel('Dosage')
        ax.set_ylabel(metric.replace('_', ' '))
        ax.set_title(f'{metric.replace("_", " ")} vs Dosage')
        
        # Add regression line if not categorical
        if not is_categorical:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                analysis_df['Dosage'], analysis_df[metric]
            )
            x = np.array([min(analysis_df['Dosage']), max(analysis_df['Dosage'])])
            ax.plot(x, intercept + slope * x, 'r', label=f'y={slope:.2f}x+{intercept:.2f}')
            ax.legend()
    
    elif viz_type == "Box Plot":
        if is_categorical:
            # Convert dosage to string for better display
            temp_df = analysis_df.copy()
            temp_df['Dosage'] = temp_df['Dosage'].astype(str)
            sns.boxplot(data=temp_df, x='Dosage', y=metric, ax=ax)
            ax.set_title(f'Distribution of {metric.replace("_", " ")} by Dosage')
        else:
            st.warning("Box plots work better with categorical data. Using histogram instead.")
            # Create a histogram
            ax.hist(analysis_df[metric], bins=10, alpha=0.7)
            ax.set_xlabel(metric.replace('_', ' '))
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {metric.replace("_", " ")}')
    
    # Display the plot
    fig.tight_layout()
    st.pyplot(fig)
    
    # Option to download the figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="Download Figure",
        data=buf.getvalue(),
        file_name=f"{metric.lower()}_plot.png",
        mime="image/png"
    )

with tab2:
    st.header("Statistical Analysis")
    
    if is_categorical:
        st.subheader("Group Analysis by Dosage")
        
        # Group by dosage
        grouped = analysis_df.groupby('Dosage').agg({
            'Body_Weight_Gain': ['mean', 'std', 'count'],
            'Carcass_Percentage': ['mean', 'std', 'count'],
            'FCR': ['mean', 'std', 'count']
        })
        
        st.dataframe(grouped, use_container_width=True)
        
        # ANOVA Analysis
        st.subheader("ANOVA Analysis")
        
        try:
            from scipy import stats
            
            for metric in ['Body_Weight_Gain', 'Carcass_Percentage', 'FCR']:
                # Create groups for ANOVA
                groups = [analysis_df[analysis_df['Dosage'] == dose][metric] for dose in unique_dosages]
                
                # Remove empty groups
                groups = [g for g in groups if len(g) > 0]
                
                if len(groups) >= 2:  # Need at least 2 groups for ANOVA
                    f_val, p_val = stats.f_oneway(*groups)
                    
                    # Create a collapsible section for each metric
                    with st.expander(f"ANOVA for {metric.replace('_', ' ')}"):
                        st.write(f"**F-value:** {f_val:.4f}")
                        st.write(f"**p-value:** {p_val:.4f}")
                        
                        # Interpret the results
                        if p_val < alpha:
                            st.success(f"✅ Significant difference detected (α = {alpha:.2f})")
                            
                            # Optional: Add post-hoc test if significant
                            st.write("**Post-hoc Analysis (Tukey HSD):**")
                            try:
                                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                                
                                # Prepare data for Tukey's test
                                data = []
                                labels = []
                                for i, dosage in enumerate(unique_dosages):
                                    group_data = analysis_df[analysis_df['Dosage'] == dosage][metric]
                                    data.extend(group_data)
                                    labels.extend([str(dosage)] * len(group_data))
                                
                                # Perform Tukey's test
                                tukey = pairwise_tukeyhsd(data, labels, alpha=alpha)
                                
                                # Display results
                                st.text(str(tukey))
                            except ImportError:
                                st.info("Tukey's HSD test requires statsmodels package.")
                        else:
                            st.info(f"No significant difference detected (α = {alpha:.2f})")
        except ImportError:
            st.warning("ANOVA analysis requires SciPy. Install it with: pip install scipy")
    else:
        st.subheader("Regression Analysis")
        
        try:
            from scipy import stats
            
            for metric in ['Body_Weight_Gain', 'Carcass_Percentage', 'FCR']:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    analysis_df['Dosage'], analysis_df[metric]
                )
                
                with st.expander(f"Regression for Dosage vs {metric.replace('_', ' ')}"):
                    st.write(f"**Equation:** {metric.replace('_', ' ')} = {slope:.4f} × Dosage + {intercept:.4f}")
                    st.write(f"**R-squared:** {r_value**2:.4f}")
                    st.write(f"**p-value:** {p_value:.4f}")
                    
                    # Interpret the results
                    if p_value < alpha:
                        st.success(f"✅ Significant relationship detected (α = {alpha:.2f})")
                    else:
                        st.info(f"No significant relationship detected (α = {alpha:.2f})")
                    
                    # Create and display regression plot
                    fig = Figure(figsize=(8, 5))
                    ax = fig.add_subplot(111)
                    
                    # Scatter plot
                    ax.scatter(analysis_df['Dosage'], analysis_df[metric])
                    
                    # Regression line
                    x = np.array([min(analysis_df['Dosage']), max(analysis_df['Dosage'])])
                    ax.plot(x, intercept + slope * x, 'r', 
                            label=f'y = {slope:.4f}x + {intercept:.4f} (R² = {r_value**2:.4f})')
                    
                    ax.set_xlabel('Dosage')
                    ax.set_ylabel(metric.replace('_', ' '))
                    ax.set_title(f'Regression: {metric.replace("_", " ")} vs Dosage')
                    ax.legend()
                    
                    fig.tight_layout()
                    st.pyplot(fig)
        except ImportError:
            st.warning("Regression analysis requires SciPy. Install it with: pip install scipy")

with tab3:
    st.header("Correlation Analysis")
    
    # Calculate correlations
    corr_matrix = analysis_df.corr()
    
    # Display correlation matrix
    st.subheader("Correlation Matrix")
    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'), use_container_width=True)
    
    # Create correlation heatmap
    fig = Figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # Generate heatmap
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(im)
    
    # Add labels
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45)
    ax.set_yticklabels(corr_matrix.columns)
    
    # Add correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                          ha="center", va="center", color="black")
    
    ax.set_title('Correlation Matrix')
    fig.tight_layout()
    
    st.pyplot(fig)
    
    # Download correlation matrix
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="Download Correlation Matrix",
        data=buf.getvalue(),
        file_name="correlation_matrix.png",
        mime="image/png"
    )
    
    # Interpretation of correlations
    st.subheader("Correlation Interpretation")
    
    correlations = corr_matrix['Dosage'].drop('Dosage')
    
    for metric in ['Body_Weight_Gain', 'Carcass_Percentage', 'FCR']:
        corr = correlations[metric]
        if abs(corr) > 0.7:
            strength = "strong"
        elif abs(corr) > 0.3:
            strength = "moderate"
        else:
            strength = "weak"
            
        direction = "positive" if corr > 0 else "negative"
        
        st.write(f"- {metric.replace('_', ' ')} has a **{strength} {direction}** correlation ({corr:.2f}) with Dosage.")

# Add instructions on how to run the application
st.sidebar.markdown("---")
st.sidebar.header("How to Run")
st.sidebar.info("Run the app with: `streamlit run app.py`")
st.sidebar.markdown("---")
st.sidebar.write("© 2025 LKTI Data Analysis")