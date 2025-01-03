import pandas as pd
import ast
import numpy as np

from src.data.preprocessing import DataPreprocessor
from src.data.splitting import DataSplitter
from src.models.classifier_tokens import ClassifierTokenizer
from src.models.training import ModelTrainer
from src.evaluation.metrics import ModelEvaluator
from src.config.config import Config
import src.utils.utils as utils
from src.utils.utils import make_should_run


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.splitter = DataSplitter(config)
        self.classifier_tokenizer = ClassifierTokenizer(config)
        self.trainer = ModelTrainer(config)
        self.evaluator = ModelEvaluator(config)

        # Define pipeline stages in order
        self.should_run = make_should_run(['raw', 'preprocessed', 'classifier_tokens'])

    def run(self, start_from: utils.PipelineStage = 'raw') -> dict[str, float]:

        print('Starting from ', start_from)

        if self.should_run(start_from, 'raw'):
            df = pd.read_csv(self.config.raw_data_path).rename(columns={"auhtor_ID": "author_ID"})
            print('Loaded raw data')
            df = self.preprocessor.process(df)

        if self.should_run(start_from, 'preprocessed'):
            if start_from == 'preprocessed':
                # post & attn_mask are stored as string, convert back to list via ast.literal_eval
                train_data = pd.read_csv(self.config.train_data_path,
                                         converters={'post': ast.literal_eval, 'attention_mask': ast.literal_eval})
                test_data = pd.read_csv(self.config.test_data_path,
                                        converters={'post': ast.literal_eval, 'attention_mask': ast.literal_eval})
                val_data = pd.read_csv(self.config.val_data_path,
                                       converters={'post': ast.literal_eval, 'attention_mask': ast.literal_eval})

            else:
                train_data, val_data, test_data = self.splitter.split_data(df)

            print('Preprocessed data:')

            print('train data')
            print(train_data.head())

            print('test data')
            print(test_data.head())

            print('val data')
            print(val_data.head())

            print('embedding datasets...')

            train_cls = self.classifier_tokenizer.get_classifier_tokens(train_data, self.config.train_cls_path)
            val_cls = self.classifier_tokenizer.get_classifier_tokens(val_data, self.config.val_cls_path)
            test_cls = self.classifier_tokenizer.get_classifier_tokens(test_data, self.config.test_cls_path)

            print('embedding done')

        if self.should_run(start_from, 'classifier_tokens'):
            if start_from == 'classifier_tokens':
                print('reading embedded data...')

                converters = {'post': ast.literal_eval, 'attention_mask': ast.literal_eval,
                              'cls': lambda x: np.fromstring(x.strip('[ ]'), sep=' ')}

                train_cls = pd.read_csv(self.config.train_cls_path, converters=converters)
                test_cls = pd.read_csv(self.config.test_cls_path, converters=converters)
                val_cls = pd.read_csv(self.config.val_cls_path, converters=converters)

                print('loaded embedded data:')

                print('train data')
                print(train_cls.head())

                print('test data')
                print(test_cls.head())

                print('val data')
                print(val_cls.head())

        # Train with validation
        print('start training...')

        model = self.trainer.train(
            train_data=train_cls,
            val_data=val_cls
        )

        print('training done')

        # Final evaluation on test set
        metrics = self.evaluator.evaluate(model, test_cls)

        return metrics


def main():
    config = Config()
    pipeline = Pipeline(config)
    results = pipeline.run('preprocessed')
    print(f"Model evaluation results: {results}")


if __name__ == "__main__":
    main()
