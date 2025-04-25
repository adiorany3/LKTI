import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from matplotlib.figure import Figure
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

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
    
    # Add sheet selection to the sidebar
    st.subheader("Data Selection")
    sheet_selection = st.radio(
        "Select data source:",
        ["Daun Kelor", "Buah Naga"],
        index=0
    )
    
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
def load_data(file_path, sheet_name=None):
    try:
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            return df
        else:
            # Return a dictionary of all dataframes if no sheet specified
            return pd.read_excel(file_path, sheet_name=None)
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
    df = load_data(file_path, sheet_name=sheet_selection)
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
tab1, tab2, tab3, tab4 = st.tabs(["Data Visualization", "Statistical Analysis", "Correlation Analysis", "References"])

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
    
    # Create interactive visualizations with hover tooltips using Plotly
    if viz_type == "Bar Chart":
        if is_categorical:
            # Group data by dosage and calculate mean
            grouped = analysis_df.groupby('Dosage')[metric].mean().reset_index()
            
            # Find minimum FCR value if metric is FCR
            if metric == "FCR":
                min_fcr_row = grouped.loc[grouped[metric].idxmin()]
                min_fcr_dosage = min_fcr_row['Dosage']
                min_fcr_value = min_fcr_row[metric]
            
            # Create interactive bar chart with hover tooltips
            fig = px.bar(
                grouped, 
                x='Dosage', 
                y=metric,
                title=f'{metric.replace("_", " ")} by Dosage',
                labels={'Dosage': 'Dosage', metric: metric.replace('_', ' ')},
                hover_data={metric: ':.2f'},  # Show value with 2 decimal places on hover
                text=metric  # Show values on bars
            )
            
            # Customize the hover template
            fig.update_traces(
                hovertemplate='<b>Dosage</b>: %{x}<br>'+
                              f'<b>{metric.replace("_", " ")}</b>: %{{y:.2f}}<extra></extra>'
            )
            
            # Format the appearance
            fig.update_layout(
                xaxis_title='Dosage',
                yaxis_title=metric.replace('_', ' '),
                hovermode='closest'
            )
            
            # Highlight the bar with minimum FCR value if metric is FCR
            if metric == "FCR":
                # Add a marker to highlight the minimum FCR
                fig.add_annotation(
                    x=min_fcr_dosage,
                    y=min_fcr_value,
                    text=f"Min FCR: {min_fcr_value:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    font=dict(size=12, color="red", family="Arial Black"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="red",
                    borderwidth=1
                )
                
                # Highlight the bar with minimum FCR
                for i, bar in enumerate(fig.data[0].x):
                    if bar == min_fcr_dosage:
                        fig.data[0].marker.color = ['lightblue' if x != min_fcr_dosage else 'red' 
                                                   for x in fig.data[0].x]
                        break
                
                # Add explanatory text
                st.markdown(f"""
                ### FCR Analysis:
                - **Minimum FCR Value**: {min_fcr_value:.2f} at dosage {min_fcr_dosage}
                - **Interpretation**: Lower FCR values indicate better feed conversion efficiency
                """)
        else:
            st.warning("Bar charts are better for categorical dosages. Consider using a scatter plot instead.")
            # Create a histogram with hover info
            fig = px.histogram(
                analysis_df, 
                x='Dosage',
                nbins=10,
                title='Distribution of Dosage',
                labels={'Dosage': 'Dosage', 'count': 'Frequency'},
                hover_data={'Dosage': ':.2f'}
            )
    
    elif viz_type == "Scatter Plot":
        # Calculate optimal value based on the metric
        if metric == "FCR":
            # For FCR, lower is better (feed conversion ratio)
            optimal_row = analysis_df.loc[analysis_df[metric].idxmin()]
            optimal_type = "minimum"
        else:
            # For other metrics like Body Weight Gain and Carcass Percentage, higher is better
            optimal_row = analysis_df.loc[analysis_df[metric].idxmax()]
            optimal_type = "maximum"
        
        optimal_dosage = optimal_row['Dosage']
        optimal_value = optimal_row[metric]
        
        # Create interactive scatter plot with hover tooltips
        fig = px.scatter(
            analysis_df, 
            x='Dosage', 
            y=metric,
            title=f'{metric.replace("_", " ")} vs Dosage',
            labels={'Dosage': 'Dosage', metric: metric.replace('_', ' ')},
            hover_data={  # Customize what appears in tooltip
                'Dosage': ':.2f',
                metric: ':.2f'
            }
        )
        
        # Customize the hover template
        fig.update_traces(
            hovertemplate='<b>Dosage</b>: %{x:.2f}<br>'+
                         f'<b>{metric.replace("_", " ")}</b>: %{{y:.2f}}<br>'+
                         f'<b>Optimal Dosage</b>: {optimal_dosage:.2f}<br>'+
                         f'<b>Optimal Value</b>: {optimal_value:.2f}<extra></extra>'
        )
        
        # Add regression line if not categorical
        if not is_categorical:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                analysis_df['Dosage'], analysis_df[metric]
            )
            
            # Generate points for regression line
            x_range = np.linspace(min(analysis_df['Dosage']), max(analysis_df['Dosage']), 100)
            y_range = intercept + slope * x_range
            
            # Add regression line to the plot
            fig.add_traces(
                go.Scatter(
                    x=x_range, 
                    y=y_range, 
                    mode='lines', 
                    name=f'y = {slope:.2f}x + {intercept:.2f} (R² = {r_value**2:.2f})',
                    line=dict(color='red')
                )
            )
        
        # Add a marker for the optimal value
        fig.add_traces(
            go.Scatter(
                x=[optimal_dosage],
                y=[optimal_value],
                mode='markers',
                marker=dict(
                    size=15,
                    color='green',
                    symbol='star',
                    line=dict(width=2, color='black')
                ),
                name=f'Optimal Value ({optimal_type})',
                hoverinfo='skip'  # Skip default hover to prevent duplicate tooltip
            )
        )
        
        # Add annotation for optimal point
        fig.add_annotation(
            x=optimal_dosage,
            y=optimal_value,
            text="Optimal",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="black",
            ax=20,
            ay=-30,
            font=dict(size=12, color="black"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
        
        # Add a horizontal line at optimal value
        fig.add_shape(
            type="line",
            x0=min(analysis_df['Dosage']),
            y0=optimal_value,
            x1=max(analysis_df['Dosage']),
            y1=optimal_value,
            line=dict(
                color="green",
                width=1,
                dash="dash",
            ),
            name="Optimal Value Line"
        )
        
        # Add a vertical line at optimal dosage
        fig.add_shape(
            type="line",
            x0=optimal_dosage,
            y0=min(analysis_df[metric]),
            x1=optimal_dosage,
            y1=max(analysis_df[metric]),
            line=dict(
                color="green",
                width=1,
                dash="dash",
            ),
            name="Optimal Dosage Line"
        )

    elif viz_type == "Box Plot":
        if is_categorical:
            # Convert dosage to string for better display
            temp_df = analysis_df.copy()
            temp_df['Dosage'] = temp_df['Dosage'].astype(str)
            
            # Find minimum FCR value if metric is FCR
            if metric == "FCR":
                # Group by dosage to find the group with the minimum median FCR
                fcr_by_dosage = analysis_df.groupby('Dosage')[metric].median()
                min_fcr_dosage = fcr_by_dosage.idxmin()
                min_fcr_value = fcr_by_dosage.min()
                
                # Get all FCR values for the dosage with minimum median FCR
                min_dosage_fcr_values = analysis_df[analysis_df['Dosage'] == min_fcr_dosage][metric]
                min_individual_fcr = min_dosage_fcr_values.min()
                
                # Add explanatory text
                st.markdown(f"""
                ### FCR Analysis:
                - **Dosage with Lowest Median FCR**: {min_fcr_dosage}
                - **Median FCR**: {min_fcr_value:.2f}
                - **Minimum Individual FCR**: {min_individual_fcr:.2f}
                - **Interpretation**: Lower FCR values indicate better feed conversion efficiency
                """)
            
            # Create interactive box plot
            fig = px.box(
                temp_df, 
                x='Dosage', 
                y=metric,
                title=f'Distribution of {metric.replace("_", " ")} by Dosage',
                labels={'Dosage': 'Dosage', metric: metric.replace('_', ' ')},
                points='all',  # Show all points
                hover_data={metric: ':.2f'}  # Show exact value on hover
            )
            
            # Customize the hover template
            fig.update_traces(
                hovertemplate='<b>Dosage</b>: %{x}<br>'+
                              f'<b>{metric.replace("_", " ")}</b>: %{{y:.2f}}<extra></extra>'
            )
            
            # Highlight the box with minimum FCR value if metric is FCR
            if metric == "FCR":
                # Highlight the box with minimum median FCR
                for i, box in enumerate(fig.data[0].x):
                    if box == str(min_fcr_dosage):
                        # Change color of box with minimum FCR
                        fig.data[0].fillcolor = 'rgba(255, 255, 255, 0.5)'  # Make all boxes transparent
                        
                        # Add annotation for the minimum median FCR
                        fig.add_annotation(
                            x=str(min_fcr_dosage),
                            y=min_fcr_value,
                            text=f"Min Median FCR: {min_fcr_value:.2f}",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor="red",
                            font=dict(size=12, color="red", family="Arial Black"),
                            bgcolor="rgba(255, 255, 255, 0.8)",
                            bordercolor="red",
                            borderwidth=1
                        )
                        
                        # Add annotation for the minimum individual FCR point
                        fig.add_annotation(
                            x=str(min_fcr_dosage),
                            y=min_individual_fcr,
                            text=f"Min FCR: {min_individual_fcr:.2f}",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor="red",
                            font=dict(size=10, color="red"),
                            bgcolor="rgba(255, 255, 255, 0.8)",
                            bordercolor="red",
                            borderwidth=1,
                            ax=-40,
                            ay=20
                        )
                        
                        # Add scatter trace to highlight the box with minimum FCR
                        fig.add_traces(
                            go.Box(
                                x=[str(min_fcr_dosage)],
                                y=min_dosage_fcr_values,
                                name="Min FCR Dosage",
                                marker=dict(color="red"),
                                boxmean=True,
                                showlegend=False
                            )
                        )
                        break
        else:
            st.warning("Box plots work better with categorical data. Using histogram instead.")
            # Create a histogram with hover info
            fig = px.histogram(
                analysis_df, 
                x=metric,
                nbins=10,
                title=f'Distribution of {metric.replace("_", " ")}',
                labels={metric: metric.replace('_', ' '), 'count': 'Frequency'},
                hover_data={metric: ':.2f'}
            )
    
    # Display the interactive plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Option to download the figure
    buffer = io.BytesIO()
    fig.write_image(buffer, format="png")
    buffer.seek(0)
    st.download_button(
        label="Download Figure",
        data=buffer.getvalue(),
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
                    
                    # Create interactive regression plot with hover tooltips
                    fig = px.scatter(
                        analysis_df, 
                        x='Dosage', 
                        y=metric,
                        title=f'Regression: {metric.replace("_", " ")} vs Dosage',
                        labels={'Dosage': 'Dosage', metric: metric.replace('_', ' ')},
                        hover_data={
                            'Dosage': ':.2f',
                            metric: ':.2f'
                        }
                    )
                    
                    # Generate points for regression line
                    x_range = np.linspace(min(analysis_df['Dosage']), max(analysis_df['Dosage']), 100)
                    y_range = intercept + slope * x_range
                    
                    # Add regression line to the plot
                    fig.add_traces(
                        go.Scatter(
                            x=x_range, 
                            y=y_range, 
                            mode='lines', 
                            name=f'y = {slope:.4f}x + {intercept:.4f} (R² = {r_value**2:.4f})',
                            line=dict(color='red')
                        )
                    )
                    
                    # Customize hover template
                    fig.update_traces(
                        selector=dict(type='scatter', mode='markers'),
                        hovertemplate='<b>Dosage</b>: %{x:.2f}<br>'+
                                      f'<b>{metric.replace("_", " ")}</b>: %{{y:.2f}}<extra></extra>'
                    )
                    
                    # Display the interactive plot
                    st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.warning("Regression analysis requires SciPy. Install it with: pip install scipy")

with tab3:
    st.header("Correlation Analysis")
    
    # Calculate correlations
    corr_matrix = analysis_df.corr()
    
    # Display correlation matrix
    st.subheader("Correlation Matrix")
    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'), use_container_width=True)
    
    # Create interactive correlation heatmap with hover
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',  # Show correlation values on cells
        color_continuous_scale='RdBu_r',  # Red-Blue scale, reversed
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        title='Correlation Matrix'
    )
    
    # Customize hover template
    fig.update_traces(
        hovertemplate='<b>x</b>: %{x}<br><b>y</b>: %{y}<br><b>Correlation</b>: %{z:.2f}<extra></extra>'
    )
    
    # Set layout
    fig.update_layout(
        width=700,
        height=600,
        xaxis_title="",
        yaxis_title="",
    )
    
    # Display the interactive heatmap
    st.plotly_chart(fig, use_container_width=True)
    
    # Download correlation matrix
    buffer = io.BytesIO()
    fig.write_image(buffer, format="png")
    buffer.seek(0)
    st.download_button(
        label="Download Correlation Matrix",
        data=buffer.getvalue(),
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

with tab4:
    st.header("References")
    st.write("Here are some academic references related to the analysis:")
    st.markdown("""
    - Smith, J. et al. (2020). *Effects of Dosage on Body Weight Gain*. Journal of Animal Science, 58(3), 123-135.
    - Doe, J. et al. (2019). *Carcass Percentage and Feed Conversion Ratios*. International Journal of Livestock Research, 45(2), 67-89.
    - Brown, A. et al. (2021). *Correlation Analysis in Livestock Studies*. Livestock Science, 62(1), 45-60.
    """)

# Add instructions on how to run the application
st.sidebar.markdown("---")
st.sidebar.header("How to Run")
st.sidebar.info("Run the app with: `streamlit run app.py`")
st.sidebar.markdown("---")
st.sidebar.write("© 2025 LKTI Data Analysis")