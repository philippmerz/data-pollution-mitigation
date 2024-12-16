import pandas as pd
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, data, random_state=4):
        """
        Data preprocessor for splitting the dataset into training, validation, and test sets.
        
        Args:
            file_path (str): Path to the CSV file containing the data.
            random_state (int): Random seed for reproducibility. Default is 42.
        """
        self.random_state = random_state
        self.df = data
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.split_data()

    def split_data(self, test_size=0.3, val_size=0.5):
        """
        Split the dataset into training, validation, and test sets
        , ensuring no overlap of author_IDs between splits.
        
        Args:
            test_size (float): Proportion of the dataset to be used for testing. Default is 0.3.
            val_size (float): Proportion of the remaining data to be used for validation. Default is 0.5.
        """

        # Group data by author_ID for stratification
        grouped = self.df.groupby('author_ID').agg({'female': 'first'}).reset_index()

        # Split into train and temp (val + test)
        train_ids, temp_ids = train_test_split(
            grouped,
            test_size=test_size,
            random_state=self.random_state,
            stratify=grouped['female']
        )

        # Split temp into val and test
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=val_size,
            random_state=self.random_state,
            stratify=temp_ids['female']
        )

        # Filter original dataframe based on splits
        self.train_df = self.df[self.df['author_ID'].isin(train_ids['author_ID'])]
        self.val_df = self.df[self.df['author_ID'].isin(val_ids['author_ID'])]
        self.test_df = self.df[self.df['author_ID'].isin(test_ids['author_ID'])]

    def get_splits(self):
        """
        Get the training, validation, and test DataFrames.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        return self.train_df, self.val_df, self.test_df

