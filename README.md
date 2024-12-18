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

## Guidelines
- put all global config constants into `src/config/config.py` for easy readability and modification
- better add too many print statements than too few. That way it's easier to keep track of what's happening when running the code