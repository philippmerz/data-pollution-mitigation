import pandas as pd
from typing import List

from src.data.preprocessing import DataPreprocessor
from src.data.splitting import DataSplitter
from src.models.embeddings import Embedder
from src.models.training import ModelTrainer
from src.evaluation.metrics import ModelEvaluator
from src.config.config import Config
import src.utils.utils as utils
from src.utils.utils import make_should_start_from


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.splitter = DataSplitter(config)
        self.embedder = Embedder(config)
        self.trainer = ModelTrainer(config)
        self.evaluator = ModelEvaluator(config)

        # Define pipeline stages in order
        self.should_start_from = make_should_start_from(['raw', 'preprocessed', 'classifier_tokens'])

    def run(self, start_from: utils.PipelineStage = 'raw') -> dict[str, float]:

        print('Starting from ', start_from)

        if self.should_start_from(start_from, 'raw'):
            df = pd.read_csv(self.config.raw_data_path).rename(columns={"auhtor_ID": "author_ID"})
            print('Loaded raw data')
            df = self.preprocessor.process(df)

        if self.should_start_from(start_from, 'preprocessed'):
            if start_from == 'preprocessed':
                train_data = pd.read_csv(self.config.train_data_path)
                test_data = pd.read_csv(self.config.test_data_path)
                val_data = pd.read_csv(self.config.val_data_path)
            else:
                train_data, val_data, test_data = self.splitter.split_data(df)

            print('Preprocessed data:')

            print('train data')
            train_data.head()

            print('test data')
            test_data.head()

            print('val data')
            val_data.head()

            train_tokens = utils.load_tokens(self.config.train_data_path)
            val_tokens = utils.load_tokens(self.config.val_data_path)
            test_tokens = utils.load_tokens(self.config.test_data_path)

            print('embedding datasets...')

            train_embeddings = self.embedder.embed(train_tokens, self.config.train_embedded_path)
            val_embeddings = self.embedder.embed(val_tokens, self.config.val_embedded_path)
            test_embeddings = self.embedder.embed(test_tokens, self.config.test_embedded_path)

            print('embedding done')

        if self.should_start_from(start_from, 'classifier_tokens'):
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
    results = pipeline.run('raw')
    print(f"Model evaluation results: {results}")

if __name__ == "__main__":
    main()
