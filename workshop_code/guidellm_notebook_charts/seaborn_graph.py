import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

def create_ttft_plot(
    benchmark_df_1: pd.DataFrame,
    benchmark_df_2: pd.DataFrame = None,
    name_1: str = 'Model 1',
    name_2: str = 'Model 2'
):
    """
    Generates a line plot of Median TTFT vs. Poisson Rate for one or two DataFrames.

    If two DataFrames are provided, they are plotted on the same graph for comparison.

    Args:
        benchmark_df_1 (pd.DataFrame): The primary DataFrame. Must include
            'Benchmark' and 'TTFT_median_ms' columns.
        benchmark_df_2 (pd.DataFrame, optional): The second DataFrame for comparison.
            Defaults to None.
        name_1 (str, optional): The name for the first DataFrame's data, used in the legend.
            Defaults to 'Model 1'.
        name_2 (str, optional): The name for the second DataFrame's data, used in the legend.
            Defaults to 'Model 2'.
    """
    # --- Data Processing ---

    def process_dataframe(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Helper function to process each DataFrame."""
        # Work with a copy to avoid modifying the original DataFrame
        proc_df = df.copy()

        # Extract the numeric rate from the 'Benchmark' column.
        try:
            if 'Benchmark' in proc_df.columns:
                proc_df['Poisson Rate'] = proc_df['Benchmark'].str.split('@').str[1].astype(float)
            elif proc_df.index.name == 'Benchmark':
                proc_df['Poisson Rate'] = proc_df.index.str.split('@').str[1].astype(float)
            else:
                raise ValueError("'Benchmark' column or index not found in the DataFrame.")
        except (IndexError, AttributeError) as e:
            raise ValueError(f"Error parsing 'Benchmark' column. Ensure it is in 'name@rate' format. Details: {e}")

        # Add a 'Model' column to identify the data source for plotting
        proc_df['Model'] = model_name
        return proc_df

    try:
        # Process the first DataFrame
        df1_processed = process_dataframe(benchmark_df_1, name_1)
        
        # Initialize a list to hold the dataframes to be concatenated
        all_dfs = [df1_processed]

        # Process and add the second DataFrame if it exists
        if benchmark_df_2 is not None:
            df2_processed = process_dataframe(benchmark_df_2, name_2)
            all_dfs.append(df2_processed)

        # Combine the dataframes into a single one for plotting
        plot_df = pd.concat(all_dfs, ignore_index=True)

    except ValueError as e:
        print(f"Error processing data: {e}")
        return

    # --- Graph Creation ---

    # Set the visual style and size for the plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    # Create the line plot using 'hue' to differentiate between models
    sns.lineplot(
        x='Poisson Rate',
        y='TTFT_median_ms',
        hue='Model',        # Differentiate lines by the 'Model' column
        style='Model',      # Differentiate markers by the 'Model' column
        data=plot_df,
        marker='o',         # Add markers to data points for better visibility
        linewidth=2.5,
        markersize=8
    )

    # Set the title and labels for clarity
    plt.title('Median Time to First Token (TTFT) vs. Poisson Rate', fontsize=16, weight='bold')
    plt.xlabel('Requests per Second (Poisson Rate)', fontsize=12)
    plt.ylabel('Median TTFT (ms)', fontsize=12)

    # Adjust axes to start from zero and give space for the highest point
    plt.ylim(0, plot_df['TTFT_median_ms'].max() * 1.15)
    plt.xlim(left=0)

    # Customize the legend
    plt.legend(title='Model', fontsize='large', title_fontsize='13')

    # Ensure the layout is clean and nothing is cut off
    plt.tight_layout()

    # Display the plot
    plt.show()


# --- Example Usage ---
# This block will only run when the script is executed directly.
if __name__ == '__main__':
    # --- Example 1: Plotting a single DataFrame ---
    print("--- Generating plot for a single model ---")
    data_string_1 = """
Benchmark,TTFT_median_ms
poisson@1.00,355
poisson@4.00,1631
poisson@8.00,2311
poisson@16.00,2353
"""
    # Create the first DataFrame from the string data
    source_df_1 = pd.read_csv(io.StringIO(data_string_1))

    # Call the function with a single DataFrame
    create_ttft_plot(source_df_1, name_1='Baseline Model')
    
    
    # --- Example 2: Plotting two DataFrames for comparison ---
    print("\n--- Generating plot for two competing models ---")
    # Data for a second, hypothetical model
    data_string_2 = """
Benchmark,TTFT_median_ms
poisson@1.00,320
poisson@4.00,1250
poisson@8.00,1980
poisson@16.00,2150
"""
    # Create the second DataFrame
    source_df_2 = pd.read_csv(io.StringIO(data_string_2))

    # Call the function with both DataFrames and custom names
    create_ttft_plot(source_df_1, source_df_2, name_1='Baseline Model', name_2='Optimized Model')