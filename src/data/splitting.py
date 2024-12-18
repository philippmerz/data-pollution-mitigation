from sklearn.model_selection import train_test_split
import pandas as pd
from src.config.config import Config

class DataSplitter:
    def __init__(self, config: Config):
        self.config = config

    def split_data(self, data: pd.DataFrame):
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
        # TODO: these are a little uneven, we should make sure the post percentages are exactly as intended
        print(f"Train size: {len(train)/len(data):.2f}")
        print(f"Val size: {len(val)/len(data):.2f}")
        print(f"Test size: {len(test)/len(data):.2f}")

        return train, val, test