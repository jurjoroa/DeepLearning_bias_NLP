
from sklearn.model_selection import train_test_split
import random

def split_data(df, debias_train_size, standard_training_set_percentage, random_state=None):
    '''
    Splits data into testing, training, and debias estimator sets.
    
    Args:
        df (pd.DataFrame): The DataFrame to be split.
        debias_train_size (int): Size of the debias training set, at least 10 times the number of features.
        standard_training_set_percentage (float): Percentage split for train/test of df excluding debias_train_df.
        random_state (int, optional): Random state for reproducibility.
        
    Returns: 
        tuple: (debias_train_df, train_df, test_df) DataFrames.
    '''

    if not 0 < standard_training_set_percentage < 1:
        raise ValueError("standard_training_set_percentage must be between 0 and 1")

    if debias_train_size > df.shape[0]:
        raise ValueError("debias_train_size cannot be greater than the number of rows in the dataframe")

    # Sample for the debiasing training set
    debias_train_df = df.sample(n=debias_train_size, random_state=random_state)
    
    # Drop the debiasing samples from the main dataframe
    df_remaining = df.drop(debias_train_df.index)
    
    # Split the remaining data into training and testing sets
    train_df, test_df = train_test_split(df_remaining, train_size=standard_training_set_percentage, shuffle=True, random_state=random_state)
    
    # Reset index
    test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    debias_train_df = debias_train_df.reset_index(drop=True)
    
    return debias_train_df, train_df, test_df

