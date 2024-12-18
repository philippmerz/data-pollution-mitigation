from pathlib import Path

from src.data.preprocessing import DataPreprocessor
from src.data.splitting import DataSplitter
from src.data.tokenization import Tokenizer
from src.models.embeddings import Embedder
from src.models.training import ModelTrainer
from src.evaluation.metrics import ModelEvaluator
from src.config.config import Config

class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.splitter = DataSplitter(config)
        self.tokenizer = Tokenizer(config)
        self.embedder = Embedder(config)
        self.trainer = ModelTrainer(config)
        self.evaluator = ModelEvaluator(config)

    def run(self, data_path: Path) -> dict[str, float]:
        processed_data = self.preprocessor.process(data_path)

        # Split data
        train_data, val_data, test_data = self.splitter.split_data(processed_data)

        # Process each split
        train_tokens = self.tokenizer.tokenize(train_data)
        train_embeddings = self.embedder.embed(train_tokens)

        val_tokens = self.tokenizer.tokenize(val_data)
        val_embeddings = self.embedder.embed(val_tokens)

        # Train with validation
        model = self.trainer.train(
            train_data=train_embeddings,
            val_data=val_embeddings
        )

        # Final evaluation on test set
        test_tokens = self.tokenizer.tokenize(test_data)
        test_embeddings = self.embedder.embed(test_tokens)
        metrics = self.evaluator.evaluate(model, test_embeddings)

        return metrics

def main():
    config = Config()
    pipeline = Pipeline(config)
    results = pipeline.run(Path("data/raw/gender.csv"))
    print(f"Model evaluation results: {results}")

if __name__ == "__main__":
    main()