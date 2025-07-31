import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def process_data(data_path, separator, label_column, test_ratio, usage_ratio, junk_columns, default_value, rng):
    """
    Loads data, performs stratified splitting, creates solution and sample submission files,
    and saves processed dataframes to CSV files.

    Args:
        data_path (str): Path to the data CSV file.
        separator (str): Delimiter for the CSV file.
        label_column (str): Name of the target column.
        test_ratio (float): Ratio of the test set size.
        usage_ratio (float): Ratio for 'Private' usage assignment.
        junk_columns (list): List of columns to drop.
        default_value (int): Default value for the label column in sample submission.
        rng (int): Random state for reproducibility.

    Returns:
        tuple: train_df, test_df, sample_df, solution_df
    """
    # Load data.csv with header
    df = pd.read_csv(data_path, delimiter=separator)

    if junk_columns:
        # Drop the "junk_column" columns
        df = df.drop(columns=junk_columns)

    # Perform stratified split
    train_df, test_df = train_test_split(df, test_size=test_ratio, stratify=df[label_column], random_state=rng)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Keep only the "label" column
    sample_df = test_df[[label_column]].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Creating solution submission
    solution_df = sample_df.copy() # Use .copy() to avoid SettingWithCopyWarning
    sample_df[label_column] = default_value
    
    # Add a new column called "Usage"
    solution_df['Usage'] = ''

    # Perform stratified assignment of 'Usage' based on 'Credit_Score'
    usage_assignments = solution_df.groupby(label_column, group_keys=False).apply(lambda x: np.random.choice(['Private'] * int(len(x) * usage_ratio) + ['Public'] * (len(x) - int(len(x) * usage_ratio)), size=len(x), replace=False))

    # Flatten the resulting Series of arrays and assign to the 'Usage' column
    solution_df['Usage'] = np.concatenate(usage_assignments.values)

    # Delete the target columns from test_df
    test_df = test_df.drop(columns=[label_column])

    # Save the dataframes to CSV files with index
    train_df.to_csv('train.csv', index=True, index_label="id")
    test_df.to_csv('test.csv', index=True, index_label="id")
    sample_df.to_csv('sample_submission.csv', index=True, index_label="id")
    solution_df.to_csv('solution.csv', index=True, index_label="id")

    print("Data processed successfully!")

    return train_df, test_df, sample_df, solution_df

# Example usage (assuming the variables are defined elsewhere)
# train_df, test_df, sample_df, solution_df = process_data(data, separator, label_column, test_ratio, usage_ratio, junk_column, default_value, rng)