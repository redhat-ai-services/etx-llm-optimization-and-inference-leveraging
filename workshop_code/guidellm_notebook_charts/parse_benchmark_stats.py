import pandas as pd
import os

def extract_ttft_from_file(filepath: str) -> pd.DataFrame:
    """
    Parses a benchmark stats file to extract the 'Benchmark' and 'TTFT (ms) median' values.

    Args:
        filepath (str): The path to the text file containing the benchmark data.

    Returns:
        pd.DataFrame: A DataFrame with 'Benchmark' as the index and 'TTFT_median_ms'
                      as the only column. Returns an empty DataFrame on failure.
    """
    # --- Step 1: Read the data from the file ---
    try:
        with open(filepath, 'r') as f:
            table_string = f.read()
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading file: {e}")
        return pd.DataFrame()

    # --- Step 2: Clean and Prepare the Data String ---

    # Split the entire string into individual lines
    lines = table_string.strip().split('\n')

    # Filter out the decorative border and separator lines
    filtered_lines = [
        line for line in lines
        if not line.startswith('===') and not line.startswith('---') and 'Stats:' not in line
    ]

    if len(filtered_lines) < 3:
        print("Error: Not enough data in the file to parse.")
        return pd.DataFrame()

    # --- Step 3: Extract Data with Unique Headers ---

    # The table's multi-line header creates duplicate column names ('mean', 'median').
    # To make parsing reliable, we define a clear, unique list of headers manually.
    MANUAL_HEADERS = [
        'Benchmark', 'Per_Second', 'Concurrency',
        'Out_Tok_per_sec_mean', 'Tot_Tok_per_sec_mean',
        'Req_Latency_sec_mean', 'Req_Latency_sec_median', 'Req_Latency_sec_p99',
        'TTFT_ms_mean', 'TTFT_ms_median', 'TTFT_ms_p99',
        'ITL_ms_mean', 'ITL_ms_median', 'ITL_ms_p99',
        'TPOT_ms_mean', 'TPOT_ms_median', 'TPOT_ms_p99'
    ]
    
    # The actual data rows start after the two header lines.
    data_lines = filtered_lines[2:]

    parsed_data = []
    for line in data_lines:
        # Split each data row by the '|' separator and strip whitespace
        row = [cell.strip() for cell in line.split('|')]
        if len(row) == len(MANUAL_HEADERS):
            parsed_data.append(row)

    if not parsed_data:
        print("Error: No data rows could be parsed.")
        return pd.DataFrame()

    # --- Step 4: Create DataFrame and Extract Required Data ---

    # Create the initial DataFrame with our clean headers
    df = pd.DataFrame(parsed_data, columns=MANUAL_HEADERS)
    df = df.set_index('Benchmark')

    # Convert all data columns to a numeric type for calculations
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove any rows that have NaN values after conversion
    df.dropna(inplace=True)
    
    # Select only the column of interest: the median TTFT
    result_df = df[['TTFT_ms_median']].copy()
    result_df = result_df.rename(columns={'TTFT_ms_median': 'TTFT_median_ms'})

    return result_df

# --- Example Usage ---
# This block demonstrates how to use the function.
if __name__ == '__main__':
    # The raw text data to be written to a temporary file
    benchmark_output = """
Benchmarks Stats:
==========================================================================================================================================================
Metadata      | Request Stats         || Out Tok/sec| Tot Tok/sec| Req Latency (sec) ||| TTFT (ms)             ||| ITL (ms)             ||| TPOT (ms)            ||
   Benchmark| Per Second| Concurrency|        mean|        mean|  mean| median|   p99|   mean| median|   p99|  mean| median|  p99|  mean| median|  p99
-------------|-----------|------------|------------|------------|------|-------|------|-------|-------|-------|------|-------|------|------|-------|------
 synchronous|       0.11|        1.00|        14.5|       105.6|  8.78|   8.79|  8.79|  334.5|  344.4|  345.2|  66.5|   66.5|  66.5|  65.9|   65.9|  66.0
   throughput|       1.10|       15.96|       141.1|      1022.2| 14.48|  14.46| 14.57| 2513.9| 2310.1| 4499.6|  94.2|   92.1| 109.5|  93.5|   91.4| 108.7
constant@0.61|       0.34|        3.81|        43.1|       312.1| 11.31|  11.25| 11.96|  337.4|  340.3|  379.6|  86.4|   86.0|  91.5|  85.7|   85.4|  90.7
constant@1.10|       0.57|        7.89|        72.6|       526.5| 13.90|  14.15| 14.63|  355.0|  344.9|  564.5| 106.7|  109.0| 112.3| 105.8|  108.1| 111.4
==========================================================================================================================================================
"""
    # Create a dummy file for the demonstration
    dummy_filepath = 'benchmark_data.txt'
    with open(dummy_filepath, 'w') as f:
        f.write(benchmark_output)

    # Call the function to parse the file and extract the specific data
    ttft_data = extract_ttft_from_file(dummy_filepath)

    # Clean up the dummy file
    os.remove(dummy_filepath)

    # Print the resulting simplified DataFrame
    if not ttft_data.empty:
        print("--- Extracted Benchmark and Median TTFT (ms) ---")
        print(ttft_data)

        # You can now easily access any value, for example:
        print("\n--- Example Value Access ---")
        throughput_ttft = ttft_data.loc['throughput', 'TTFT_median_ms']
        print(f"Median TTFT for 'throughput': {throughput_ttft} ms")
