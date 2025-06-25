import pandas as pd
import os
from flask import Flask, render_template, request
import numpy as np

# --- Configuration ---
DATA_FILE_PATH = os.path.join('data', 'esg_mock_data.csv')
app = Flask(__name__)

# --- Data Loading and Initial Cleaning ---
def load_and_clean_esg_data(file_path):
    """
    Loads ESG mock data from a CSV, performs initial cleaning,
    and handles common data quality issues.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}. Initial shape: {df.shape}")

        # Data Quality Improvement 1: Handle missing 'revenue_usd_millions_2023' (simulating "Poor - Missing Revenue")
        # Replace 'NA' string and actual NaN values with 0 or a more appropriate imputation
        # For simplicity, we'll fill with 0, but in a real scenario, you'd use more sophisticated imputation
        df['revenue_usd_millions_2023'] = pd.to_numeric(
            df['revenue_usd_millions_2023'], errors='coerce'
        ).fillna(0) # 'coerce' turns non-numeric into NaN, then fillna replaces NaN

        # Data Quality Improvement 2: Handle missing 'gender_diversity_female_perc'
        # Fill missing gender diversity with the mean for that industry or overall mean
        # For simplicity, we'll fill with the overall mean
        df['gender_diversity_female_perc'] = df['gender_diversity_female_perc'].fillna(
            df['gender_diversity_female_perc'].mean()
        )

        # Data Quality Improvement 3: Handle missing 'renewable_energy_perc'
        # 'coerce' non-numeric to NaN, then fill NaN with 0
        df['renewable_energy_perc'] = pd.to_numeric(
            df['renewable_energy_perc'], errors='coerce'
        ).fillna(0)

        # Data Quality Improvement 4: Ensure correct data types
        # Convert relevant columns to numeric, coercing errors to NaN and then filling with 0
        numeric_cols = [
            'total_employees_2023',
            'carbon_emissions_tonnes_2023',
            'water_withdrawal_m3_2023',
            'employee_turnover_rate_perc',
            'board_independence_perc',
            'waste_generated_tonnes_2023'
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Fill NaN with 0 after coercion

        print("Data cleaning complete.")
        return df

    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return pd.DataFrame()

# --- ESG Metric Prototyping ---
def calculate_esg_metrics(df):
    """
    Calculates key ESG metrics from the cleaned DataFrame.
    This demonstrates 'prototyping ESG metrics'.
    """
    if df.empty:
        print("Cannot calculate metrics: DataFrame is empty.")
        return df

    print("Calculating ESG metrics...")

    # Metric 1: Carbon Emission Intensity (tonnes per $ million revenue)
    # Handle division by zero for companies with 0 revenue
    df['carbon_intensity'] = df.apply(
        lambda row: (row['carbon_emissions_tonnes_2023'] / row['revenue_usd_millions_2023'])
        if row['revenue_usd_millions_2023'] > 0 else np.nan,
        axis=1
    )

    # Metric 2: Water Withdrawal per Employee (m3 per employee)
    df['water_per_employee'] = df.apply(
        lambda row: (row['water_withdrawal_m3_2023'] / row['total_employees_2023'])
        if row['total_employees_2023'] > 0 else np.nan,
        axis=1
    )

    # Metric 3: Gender Diversity Score (using female_perc as a proxy)
    # Higher percentage is better for diversity. Can be inverse if a "risk" score.
    df['gender_diversity_score'] = df['gender_diversity_female_perc']

    # Metric 4: Board Independence Score (just using the percentage directly)
    df['board_independence_score'] = df['board_independence_perc']

    
    

    # Metric 5: Waste Intensity (waste per employee)
    df['waste_per_employee'] = df.apply(
        lambda row: (row['waste_generated_tonnes_2023']/row['total_employees_2023'])
        if row['total_employees_2023'] > 0 else np.nan,
        axis=1
    )

    # Metric 6: Waste Intensity (waste per dollar revenue)
    df['waste_per_dollar_revenue'] = df.apply(
        lambda row: (row['waste_generated_tonnes_2023']/row['revenue_usd_millions_2023'])
        if row['revenue_usd_millions_2023'] > 0 else np.nan,
        axis=1
    )
    print("ESG metrics calculated.")
    return df

@app.route('/')
def home():
    """
    Main route for the ESG Data Explorer.
    Loads, cleans, calculates metrics, and displays data on a webpage,
    with optional filtering by industry.
    """
    # Load and clean data
    esg_df = load_and_clean_esg_data(DATA_FILE_PATH)

    if esg_df.empty:
        return "Error loading data. Check server logs."

    # Calculate ESG metrics
    esg_df_with_metrics = calculate_esg_metrics(esg_df)

    # Get unique industries for the filter dropdown
    # Convert to list and sort alphabetically for the dropdown
    unique_industries = sorted(esg_df_with_metrics['industry'].unique().tolist())

    # --- Filtering Logic ---
    # Get the selected industry from the URL query parameters (e.g., ?industry=Technology)
    selected_industry = request.args.get('industry')

    # Apply filter if an industry is selected and valid
    if selected_industry and selected_industry in unique_industries:
        filtered_df = esg_df_with_metrics[esg_df_with_metrics['industry'] == selected_industry]
    else:
        filtered_df = esg_df_with_metrics # No filter applied, show all data

    # Calculate average carbon intensity by industry (for the second table)
    # This should always be based on the FULL dataset, not the filtered one,
    # unless you specifically want avg for filtered industry. For now, keep it full.
    avg_carbon_intensity_by_industry = esg_df_with_metrics.groupby('industry')['carbon_intensity'].mean().reset_index()

    # --- Additional Industry-Level Aggregations ---
    # Add average waste metrics by industry
    avg_waste_per_employee_by_industry = esg_df_with_metrics.groupby('industry')['waste_per_employee'].mean().reset_index()
    avg_waste_per_dollar_revenue_by_industry = esg_df_with_metrics.groupby('industry')['waste_per_dollar_revenue'].mean().reset_index()

    # --- Top/Bottom Performers (e.g., Top 5) ---
    num_top_bottom = 5 # Define how many companies to show for top/bottom lists

    # Top 5 by Carbon Intensity (lower is better)
    top_carbon_intensity = esg_df_with_metrics.nsmallest(num_top_bottom, 'carbon_intensity')
    # Bottom 5 by Carbon Intensity (higher is worse)
    bottom_carbon_intensity = esg_df_with_metrics.nlargest(num_top_bottom, 'carbon_intensity')

    # Top 5 by Renewable Energy Percentage (higher is better)
    top_renewable_energy = esg_df_with_metrics.nlargest(num_top_bottom, 'renewable_energy_perc')
    # Bottom 5 by Renewable Energy Percentage (lower is worse)
    bottom_renewable_energy = esg_df_with_metrics.nsmallest(num_top_bottom, 'renewable_energy_perc')

    # Top 5 by Gender Diversity Score (higher is better)
    top_gender_diversity = esg_df_with_metrics.nlargest(num_top_bottom, 'gender_diversity_score')
    # Bottom 5 by Gender Diversity Score (lower is worse)
    bottom_gender_diversity = esg_df_with_metrics.nsmallest(num_top_bottom, 'gender_diversity_score')

    # Top 5 by Waste per Employee (lower is better)
    top_waste_per_employee = esg_df_with_metrics.nsmallest(num_top_bottom, 'waste_per_employee')
    # Bottom 5 by Waste per Employee (higher is worse)
    bottom_waste_per_employee = esg_df_with_metrics.nlargest(num_top_bottom, 'waste_per_employee')

    # Top 5 by Waste per Dollar Revenue (lower is better)
    top_waste_per_dollar_revenue = esg_df_with_metrics.nsmallest(num_top_bottom, 'waste_per_dollar_revenue')    
    # Bottom 5 by Waste per Dollar Revenue (higher is worse)
    bottom_waste_per_dollar_revenue = esg_df_with_metrics.nlargest(num_top_bottom, 'waste_per_dollar_revenue')

    # Render the HTML template, passing the filtered DataFrame and other data
    return render_template(
        'index.html',
        data=filtered_df, # Pass the filtered data to the main table
        avg_carbon_intensity=avg_carbon_intensity_by_industry,
        unique_industries=unique_industries, # Pass unique industries for the dropdown
        selected_industry=selected_industry, # Pass the currently selected industry to pre-select it in the dropdown
        avg_waste_per_employee=avg_waste_per_employee_by_industry,
        avg_waste_per_dollar_revenue=avg_waste_per_dollar_revenue_by_industry,
        top_carbon_intensity=top_carbon_intensity,
        bottom_carbon_intensity=bottom_carbon_intensity,
        top_renewable_energy=top_renewable_energy,
        bottom_renewable_energy=bottom_renewable_energy,
        top_gender_diversity=top_gender_diversity,
        top_waste_per_employee=top_waste_per_employee,
        top_waste_per_dollar_revenue=top_waste_per_dollar_revenue,
        bottom_gender_diversity=bottom_gender_diversity,
        bottom_waste_per_employee=bottom_waste_per_employee,
        bottom_waste_per_dollar_revenue=bottom_waste_per_dollar_revenue
    )

# --- Main Application Run ---
if __name__ == '__main__':
    app.run(debug=True) # debug=True allows auto-reloading and helpful error messages