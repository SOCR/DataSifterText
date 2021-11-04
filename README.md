# DataSifterText

## Setup:
	# Set up python virtual environment	
	cd DataSifterText-package
	# Remove pre-existing env
	rm -rf env
	# Define new env
	python3 -m venv env
	# Activate virtual env
	source env/bin/activate
	# Install required package
	pip install -r requirements.txt

## Usage:

### Run the whole obfuscation model:

$ python3 total.py --model_mode <MODEL_MODE> --summarize <SUMMARIZE> --keywords_position <KEYWORDS_POSITION> --file_name <FILE_NAME>

SUMMARIZE 0: no summarize, 1: summarize

KEYWORDS/POSITION SWAP MODE 0: keywords-swap, 1: position-swap

Notice that in summarization mode, we will only do keywords-swap.
	
## Example: 
$ python3 total.py --model_mode bert --summarize 0 --keywords_position 0 --file_name <FILE_NAME>

Built-in example:
$ python3 total.py --model_mode electra --summarize 0 --keywords_position 0 --file_name processed_0_prepare_small.csv

will run the obfuscation without summarization and doing keywords-swap.
