

def race_dummy_encoding(df, race_column, race_categories):
    """
    Create dummy encoding for two specified race categories in a DataFrame.

    Args:
    df (pd.DataFrame): DataFrame containing the race data.
    race_column (str): Name of the column containing race information.
    race_categories (tuple): A tuple of two strings representing the race categories to be encoded.

    Returns:
    pd.DataFrame: DataFrame with dummy encoding for the specified race categories.
    """

    if not all(category in df[race_column].unique() for category in race_categories):
        raise ValueError("Provided race categories must exist in the DataFrame")

    category_1, category_2 = race_categories

    # Vectorized operation for dummy encoding
    df_encoded = pd.DataFrame({
        category_1: (df[race_column] == category_1).astype(int),
        category_2: (df[race_column] == category_2).astype(int)
    })

    return df_encoded
