## Data Pollution Mitigation

## Setup

1. Add a `data` folder to the root and put the `gender.csv` into it.
2. install poetry
3. `poetry install`
4. `poetry run python src/main.py` or click run from your IDE

## Code Structure

- `src/main.py` is the entry point of the program
  - imports all other modules and defines Pipeline class
- `/data` for all the data files
- `/notebooks` for quick and dirty exploration and testing

## Usage

- Pipeline.run (called in main.py) takes one of the following as start_from param:
  - 'raw'
  - 'preprocessed'
  - 'classifier_tokens'
- Depending on which is passed, the pipeline will run from the corresponding step, loading the data from `/data`
- The pipeline will save the data at each step to `/data` as well, so data is always up to date

## Guidelines

- put all global config constants into `src/config/config.py` for easy readability and modification
- better add too many print statements than too few. That way it's easier to keep track of what's happening when running the code
- do `poetry add [package]` when adding a new package to the project

## Running it on Different Data
- Ensure that the data has the same format as the one that was used throughout the code
  - A .csv file  with columns: auhtor_ID (str), post (str), female (int64)
- If your data has other format, then change it to the previous specified format
- Link the data with the project:
  - Option1: Add the data to /data/raw folder by giving the name "gender"
  - Option2: Navigate to config.py and change 'raw_data_path' variable value to your specific path location of the csv and your csv name

## Suggestions for Extending
- Adding more complex models like RNNs and LSTMs which can capture the sequential nature of text data. Testing on these might give different results than our approach
- Make more robust embeddings with more complex LLMs that can caputre the semantic of the text better (eg. ChatGPT, Claude)