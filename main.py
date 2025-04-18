import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Change directory to the specified path
os.chdir('/Users/macbookpro/Desktop/LKTI')

# Display the current working directory and confirm file exists
print("Current Working Directory:", os.getcwd())
file_path = "Data LKTI.xlsx"
print(f"Does file exist? {os.path.exists(file_path)}")

# Read the Excel file
try:
    print(f"Reading file: {file_path}")
    df = pd.read_excel(file_path)
    
    # Display basic information about the data
    print("\n--- Basic Information ---")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")
    
    # Function to find the best matching column based on keywords
    def find_column(keywords, df_columns):
        for keyword in keywords:
            matches = [col for col in df_columns if keyword.lower() in col.lower()]
            if matches:
                return matches[0]
        return None
    
    # Find the relevant columns
    dosage_column = find_column(['dosage', 'dose', 'dosis'], df.columns)
    bwg_column = find_column(['body weight gain', 'weight gain', 'bwg'], df.columns)
    carcass_column = find_column(['carcass', 'carcas', 'karkas'], df.columns)
    fcr_column = find_column(['fcr', 'feed conversion'], df.columns)
    
    print("\n--- Identified Columns ---")
    print(f"Dosage column: {dosage_column}")
    print(f"Body Weight Gain column: {bwg_column}")
    print(f"Carcass Percentage column: {carcass_column}")
    print(f"FCR column: {fcr_column}")
    
    # Check if all necessary columns were found
    if not all([dosage_column, bwg_column, carcass_column, fcr_column]):
        missing = []
        if not dosage_column: missing.append("Dosage")
        if not bwg_column: missing.append("Body Weight Gain")
        if not carcass_column: missing.append("Carcass Percentage")
        if not fcr_column: missing.append("FCR")
        
        print("\nWARNING: Could not find columns for:", ", ".join(missing))
        print("Available columns:", ", ".join(df.columns))
    else:
        # Print the first few rows of raw data to debug
        print("\n--- First 5 rows of raw data ---")
        print(df.head())
        
        # Check data types to identify non-numeric columns
        print("\n--- Data types in original dataframe ---")
        print(df.dtypes)
        
        # Prepare dataframe with only the columns we need
        analysis_df = df[[dosage_column, bwg_column, carcass_column, fcr_column]].copy()
        
        # For readability, rename the columns to standard names
        analysis_df.columns = ['Dosage', 'Body_Weight_Gain', 'Carcass_Percentage', 'FCR']
        
        print("\n--- First 5 rows of analysis data ---")
        print(analysis_df.head())
        
        # Check for data types after selection
        print("\n--- Data types in analysis dataframe ---")
        print(analysis_df.dtypes)
        
        # Try to convert to numeric, coercing errors to NaN
        for col in analysis_df.columns:
            analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')
        
        # Remove rows with NaN values after conversion
        analysis_df = analysis_df.dropna()
        
        print("\n--- Analysis data after conversion to numeric ---")
        print(analysis_df.head())
        print(f"Remaining rows after cleaning: {len(analysis_df)}")
        
        # Check for missing values
        missing_values = analysis_df.isnull().sum()
        print("\n--- Missing values ---")
        print(missing_values)
        
        # Basic statistics for each column
        print("\n--- Statistical Summary ---")
        print(analysis_df.describe())
        
        # Calculate correlations
        print("\n--- Correlation between Dosage and other variables ---")
        correlations = analysis_df.corr()['Dosage'].drop('Dosage')
        print(correlations)
        
        # Determine if Dosage is categorical or continuous
        unique_dosages = analysis_df['Dosage'].unique()
        print(f"\n--- Unique Dosage values: {sorted(unique_dosages)} ---")
        
        is_categorical = len(unique_dosages) < 10  # Assuming fewer than 10 values means categorical
        
        # Analysis based on dosage type
        if is_categorical:
            print("\nDosage appears to be categorical. Performing group analysis.")
            
            # Group by dosage
            grouped = analysis_df.groupby('Dosage').agg({
                'Body_Weight_Gain': ['mean', 'std'],
                'Carcass_Percentage': ['mean', 'std'],
                'FCR': ['mean', 'std']
            })
            
            print("\n--- Group statistics by Dosage ---")
            print(grouped)
            
            # Create plots
            fig, axes = plt.subplots(3, 1, figsize=(10, 15))
            
            # Body Weight Gain vs Dosage
            axes[0].bar(analysis_df['Dosage'].astype(str), analysis_df['Body_Weight_Gain'])
            axes[0].set_title('Body Weight Gain vs Dosage')
            axes[0].set_xlabel('Dosage')
            axes[0].set_ylabel('Body Weight Gain')
            
            # Carcass Percentage vs Dosage
            axes[1].bar(analysis_df['Dosage'].astype(str), analysis_df['Carcass_Percentage'])
            axes[1].set_title('Carcass Percentage vs Dosage')
            axes[1].set_xlabel('Dosage')
            axes[1].set_ylabel('Carcass Percentage')
            
            # FCR vs Dosage
            axes[2].bar(analysis_df['Dosage'].astype(str), analysis_df['FCR'])
            axes[2].set_title('FCR vs Dosage')
            axes[2].set_xlabel('Dosage')
            axes[2].set_ylabel('FCR')
            
            plt.tight_layout()
            plt.savefig('dosage_comparison.png')
            print("\nCreated visualization: dosage_comparison.png")
            
            # Try to perform ANOVA if scipy is available
            try:
                from scipy import stats
                
                print("\n--- ANOVA Analysis ---")
                for metric in ['Body_Weight_Gain', 'Carcass_Percentage', 'FCR']:
                    # Create groups for ANOVA
                    groups = [analysis_df[analysis_df['Dosage'] == dose][metric] for dose in unique_dosages]
                    
                    # Remove empty groups
                    groups = [g for g in groups if len(g) > 0]
                    
                    if len(groups) >= 2:  # Need at least 2 groups for ANOVA
                        f_val, p_val = stats.f_oneway(*groups)
                        print(f"ANOVA for {metric} across dosage groups:")
                        print(f"  F-value: {f_val:.4f}")
                        print(f"  p-value: {p_val:.4f}")
                        print(f"  {'Significant difference detected' if p_val < 0.05 else 'No significant difference detected'} (α = 0.05)\n")
            except ImportError:
                print("\nScientific Python (scipy) is not installed. Cannot perform ANOVA analysis.")
                
        else:
            print("\nDosage appears to be continuous. Performing regression analysis.")
            
            # Create scatter plots
            fig, axes = plt.subplots(3, 1, figsize=(10, 15))
            
            # Body Weight Gain vs Dosage
            axes[0].scatter(analysis_df['Dosage'], analysis_df['Body_Weight_Gain'])
            axes[0].set_title('Body Weight Gain vs Dosage')
            axes[0].set_xlabel('Dosage')
            axes[0].set_ylabel('Body Weight Gain')
            
            # Carcass Percentage vs Dosage
            axes[1].scatter(analysis_df['Dosage'], analysis_df['Carcass_Percentage'])
            axes[1].set_title('Carcass Percentage vs Dosage')
            axes[1].set_xlabel('Dosage')
            axes[1].set_ylabel('Carcass Percentage')
            
            # FCR vs Dosage
            axes[2].scatter(analysis_df['Dosage'], analysis_df['FCR'])
            axes[2].set_title('FCR vs Dosage')
            axes[2].set_xlabel('Dosage')
            axes[2].set_ylabel('FCR')
            
            plt.tight_layout()
            plt.savefig('dosage_comparison.png')
            print("\nCreated visualization: dosage_comparison.png")
            
            # Try to perform regression analysis if scipy is available
            try:
                from scipy import stats
                
                print("\n--- Regression Analysis ---")
                for metric in ['Body_Weight_Gain', 'Carcass_Percentage', 'FCR']:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        analysis_df['Dosage'], analysis_df[metric]
                    )
                    
                    print(f"Linear regression for Dosage vs {metric}:")
                    print(f"  Equation: {metric} = {slope:.4f} × Dosage + {intercept:.4f}")
                    print(f"  R-squared: {r_value**2:.4f}")
                    print(f"  p-value: {p_value:.4f}")
                    print(f"  {'Significant relationship detected' if p_value < 0.05 else 'No significant relationship detected'} (α = 0.05)\n")
            except ImportError:
                print("\nScientific Python (scipy) is not installed. Cannot perform regression analysis.")

        # Create correlation heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(analysis_df.corr(), cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(analysis_df.corr().columns)), analysis_df.corr().columns, rotation=45)
        plt.yticks(range(len(analysis_df.corr().columns)), analysis_df.corr().columns)
        for i in range(len(analysis_df.corr().columns)):
            for j in range(len(analysis_df.corr().columns)):
                plt.text(j, i, f"{analysis_df.corr().iloc[i, j]:.2f}", ha="center", va="center", color="black")
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        print("Created visualization: correlation_matrix.png")
        
        print("\n--- Summary of Findings ---")
        for metric in ['Body_Weight_Gain', 'Carcass_Percentage', 'FCR']:
            corr = correlations[metric]
            if abs(corr) > 0.7:
                strength = "strong"
            elif abs(corr) > 0.3:
                strength = "moderate"
            else:
                strength = "weak"
                
            direction = "positive" if corr > 0 else "negative"
            
            print(f"- {metric} has a {strength} {direction} correlation ({corr:.2f}) with Dosage.")
            
        print("\nCheck the created visualizations (dosage_comparison.png and correlation_matrix.png) for graphical representation of the relationships.")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()