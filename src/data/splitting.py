from sklearn.model_selection import train_test_split
import pandas as pd
from src.config.config import Config

class DataSplitter:
    def __init__(self, config: Config):
        self.config = config

    def median_posts_per_author(self, df: pd.DataFrame) -> float: 
        # Group by author_ID and count the number of posts per author 
        posts_per_author = df.groupby('author_ID').size() 

        median_posts = posts_per_author.median() 
        
        print(f"Median posts per author: {median_posts}")

        return median_posts

    def limit_posts(self, group: pd.DataFrame, max_posts: int) -> pd.DataFrame:
        if len(group) > max_posts:
            return group.iloc[: int(max_posts)], group.iloc[int(max_posts):]
        
        return group, pd.DataFrame(columns=group.columns)

    def undersample(self, data: pd.DataFrame, max_posts: int) -> pd.DataFrame:
        # List to store undersampled rows and discarded rows
        undersampled_rows = []
        discarded_rows = []

        # Apply the limit_posts function to each group of posts by author_ID
        for _, group in data.groupby('author_ID'):
            kept, discarded = self.limit_posts(group, max_posts)
            undersampled_rows.append(kept)

            if not discarded.empty:
                discarded_rows.append(discarded)

        # Concatenate the results
        undersampled_data = pd.concat(undersampled_rows).reset_index(drop=True)
        discarded_data = pd.concat(discarded_rows).reset_index(drop=True)
        
        # Save discarded rows to CSV
        discarded_data.to_csv(self.config.discarded_data_path, index=False)

        return undersampled_data

    def split_data(self, data: pd.DataFrame):

        data = self.undersample(data, self.median_posts_per_author(data))
        grouped = data.groupby('author_ID').agg({'female': 'first'}).reset_index()

        total_size = len(grouped)
        test_size_abs = int(total_size * self.config.test_size)
        val_size_abs = int(total_size * self.config.val_size)

        train_val, test_ids = train_test_split(
            grouped,
            test_size=test_size_abs,
            random_state=self.config.seed,
            stratify=grouped['female']
        )

        # Split temp into val and test
        train_ids, val_ids = train_test_split(
            train_val,
            test_size=val_size_abs,
            random_state=self.config.seed,
            stratify=train_val['female']
        )

        # ungroup
        train = data[data['author_ID'].isin(train_ids['author_ID'])]
        val = data[data['author_ID'].isin(val_ids['author_ID'])]
        test = data[data['author_ID'].isin(test_ids['author_ID'])]

        # verify percentages
        print(f"Train size: {len(train)/len(data):.2f}")
        print(f"Val size: {len(val)/len(data):.2f}")
        print(f"Test size: {len(test)/len(data):.2f}")

        # Save the split data

        print("Saving split data...")

        train.to_csv(self.config.train_data_path, index=False)
        val.to_csv(self.config.val_data_path, index=False)
        test.to_csv(self.config.test_data_path, index=False)

        return train, val, test