from pathlib import Path

from src.data.preprocessing import DataPreprocessor
from src.data.tokenization import Tokenizer
from src.models.embeddings import Embedder
from src.models.training import ModelTrainer
from src.evaluation.metrics import ModelEvaluator
from src.config.config import Config

class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.tokenizer = Tokenizer(config)
        self.embedder = Embedder(config)
        self.trainer = ModelTrainer(config)
        self.evaluator = ModelEvaluator(config)

    def run(self, data_path: Path) -> dict[str, float]:
        # Pipeline orchestration
        processed_data = self.preprocessor.process(data_path)
        tokenized_data = self.tokenizer.tokenize(processed_data)
        embedded_data = self.embedder.embed(tokenized_data)
        model = self.trainer.train(embedded_data)
        metrics = self.evaluator.evaluate(model, embedded_data)
        return metrics

def main():
    config = Config()
    pipeline = Pipeline(config)
    results = pipeline.run(Path("data/raw/gender.csv"))
    print(f"Model evaluation results: {results}")

if __name__ == "__main__":
    main()