from pathlib import Path
from typing import Any
import pandas as pd
from src.config.config import Config
from langdetect import detect_langs

class DataPreprocessor:
    def __init__(self, config: Config):
        self.config = config

    def process(self, data_path: Path) -> Any:
        df = pd.read_csv(data_path).rename(columns={"auhtor_ID": "author_ID"})

        if self.config.dev:
            df = df.sample(1000)

        print(f"Loaded {len(df)} rows of data")

        # Eliminate all authors with less than 300 characters
        df = df.groupby('author_ID').filter(lambda x: len(''.join(x['post'])) > self.config.min_chars_per_author)

        # Elimate all posts that are 80% another language
        df = df[df['post'].apply(self.filter_language)]

        return df
    
    def filter_language(self, post, confidence_threshold=0.8):
        predictions = detect_langs(post)

        # Extract the most probable language and its confidence score
        language = max(predictions, key=lambda x: x.prob)

        # Is true if the language is English and the confidence score is above the threshold
        is_english = language.lang == 'en' and language.prob > confidence_threshold
        
        # Print if the post is not english (only for debugging)
        if not is_english:
            print(f"Post not in English: {post}") 

        return is_english


