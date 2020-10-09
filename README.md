# DataSifterText

# Setup:
## """Set up python virtual environment"""
$ cd DataSifterText-package
# remove pre-existing env
$ rm -rf env
# define new env
$ python3 -m venv env
# activate virtual env
$ source env/bin/activate
# install required package
$ pip install -r requirements.txt
$ pip install -e . 


Usage:

Run the whole obfuscation model:

$ python3 total.py <SUMMARIZATION> <KEYWORDS/POSITION SWAP MODE>

SUMMARIZATION 0: no summarize, 1: summarize

KEYWORDS/POSITION SWAP MODE 0: keywords-swap, 1: position-swap

Notice that in summarization mode, we will only do keywords-swap.
	
Example: 
$ python total.py 0 0 <filename>

Built-in example:
python3 total.py 0 0 processed_0_prepare.csv

will run the obfuscation without summarization and doing keywords-swap.
