from pathlib import Path
import pandas as pd
from typing import List
import torch
import warnings

# Suppress the FutureWarning about torch.load (The future warning is really longand irrelevant)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

from src.data.preprocessing import DataPreprocessor
from src.data.splitting import DataSplitter
from src.data.tokenization import Tokenizer
from src.models.embeddings import Embedder
from src.models.training import ModelTrainer
from src.evaluation.metrics import ModelEvaluator
from src.config.config import Config
import src.utils.utils as utils



class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.splitter = DataSplitter(config)
        self.tokenizer = Tokenizer(config)
        self.embedder = Embedder(config)
        self.trainer = ModelTrainer(config)
        self.evaluator = ModelEvaluator(config)

        # Define pipeline stages in order
        self.stages_order: List[utils.PipelineStage] = ['raw', 'preprocessed', 'tokenized', 'classifier_tokens']

    def run(self, start_from: utils.PipelineStage = 'raw') -> dict[str, float]:

        print('Starting from ', start_from)

        if start_from == 'raw':
            df = pd.read_csv(self.config.raw_data_path).rename(columns={"auhtor_ID": "author_ID"})
            print('Loaded raw data')
            df = self.preprocessor.process(df)

        if start_from == 'raw' or start_from == 'preprocessed':
            if start_from == 'preprocessed':
                train_data = pd.read_csv(self.config.train_data_path)
                test_data = pd.read_csv(self.config.test_data_path)
                val_data = pd.read_csv(self.config.val_data_path)

                print('Loaded preprocessed data:')

                print('train data')
                train_data.head()

                print('test data')
                test_data.head()

                print('val data')
                val_data.head()
            else:
                train_data, val_data, test_data = self.splitter.split_data(df)

            print('tokenizing datasets')
            train_tokens = self.tokenizer.tokenize(train_data)
            test_tokens = self.tokenizer.tokenize(test_data)
            val_tokens = self.tokenizer.tokenize(val_data)

        if start_from == 'raw' or start_from == 'preprocessed' or start_from == 'tokenized':
            if start_from == 'tokenized':
                print('reading tokenized data')

                train_tokens = torch.load(self.config.train_token_path)
                test_tokens = torch.load(self.config.test_token_path)
                val_tokens = torch.load(self.config.val_token_path)

                print('loaded tokenized data:')

                print('train data')
                print(train_tokens['input_ids'][:1])

                print('test data')
                print(test_tokens['input_ids'][:1])

                print('val data')
                print(val_tokens['input_ids'][:1])

            print('embedding datasets...')
            val_embeddings = self.embedder.embed(val_tokens)
            train_embeddings = self.embedder.embed(train_tokens)
            test_embeddings = self.embedder.embed(test_tokens)
            print('embedding done')

        if start_from == 'raw' or start_from == 'preprocessed' or start_from == 'tokenized' or start_from == 'classifier_tokens':
            if start_from == 'classifier_tokens':
                print('reading embedded data...')
                train_embeddings = pd.read_csv(self.config.train_embedded_path)
                test_embeddings = pd.read_csv(self.config.test_embedded_path)
                val_embeddings = pd.read_csv(self.config.val_embedded_path)
                print('loaded embedded data:')

                print('train data')
                train_embeddings.head()

                print('test data')
                test_embeddings.head()

                print('val data')
                val_embeddings.head()

        # Train with validation
        print('start training...')
        model = self.trainer.train(
            train_data=train_embeddings,
            val_data=val_embeddings
        )
        print('training done')

        # Final evaluation on test set
        metrics = self.evaluator.evaluate(model, test_embeddings)

        return metrics


def main():
    config = Config()
    pipeline = Pipeline(config)
    results = pipeline.run('tokenized')
    print(f"Model evaluation results: {results}")


if __name__ == "__main__":
    main()
