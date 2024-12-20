from pathlib import Path
from typing import Any
import pandas as pd
from src.config.config import Config
from langdetect import detect_langs
import unicodedata
import emoji

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

        print("Starting Filtering-non-straightforward-symbols...")

        # Clean posts to remove non-straightforward symbols
        df['post'] = df['post'].apply(self.filter_non_straightforward_symbols)
        
        print("Filtering-non-straightforward-symbols done")

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
    
    def filter_non_straightforward_symbols(self, post: str) -> str:
        """
        Filters out characters from the input string that are not considered "straightforward".
        The method retains specific Unicode characters, such as ASCII letters, digits, punctuation,
        spaces, emojis, and certain special characters like Zero Width Joiner and Variation Selector.

        Args:
            post (str): The input string to be filtered.

        Returns:
            str: A new string containing only the allowed characters.

        Allowed Characters:
            - Zero Width Joiner (`\u200d`) and Variation Selector (`\uFE0F`)
            - Emojis (checked using `emoji.is_emoji`)
            - ASCII characters that belong to the following Unicode categories:
                - `L` (Letters)
                - `N` (Numbers)
                - `P` (Punctuation)
                - `Zs` (Space separators)

        Filtering Logic:
            - Each character in the input string is evaluated by the `is_allowed_char` helper function.
            - Characters that meet the allowed criteria are included in the output string.
        """
        def is_allowed_char(ch):
            # Zero Width Joiner and Variation Selector
            if ch in ['\u200d', '\uFE0F']:
                return True
            
            # Check if it's an emoji
            if emoji.is_emoji(ch):
                return True
            
            # Check if it's an ASCII character
            if ch.isascii():
                # Get the Unicode category of the character
                cat = unicodedata.category(ch)
                # Allow letters, digits, punctuation, and spaces from ASCII set
                if cat.startswith(('L', 'N', 'P')) or cat == 'Zs':
                    return True
            
            return False

        # Filter characters in the post
        return ''.join(ch for ch in post if is_allowed_char(ch))

