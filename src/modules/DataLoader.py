import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        """
        DataLoader handles loading and initial preparation of the dataset.
        
        Args:
            file_path (str): Path to the CSV file containing the data.
        """
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """
        Load the data from the file path and perform any initial preprocessing.
        
        Returns:
            pd.DataFrame: Loaded DataFrame with necessary preprocessing applied.
        """
        self.df = pd.read_csv(self.file_path)
        self.df.rename(columns={"auhtor_ID": "author_ID"}, inplace=True)
        
        return self.df