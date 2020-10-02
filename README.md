# DataSifterText

Setup:
"""Set up python virtual environment"""
$ cd DataSifterText-package
# remove pre-existing env
$ rm -rf env
# define new env
$ python3 -m venv env
# activate virtual env
$ source env/bin/activate
# install required package
$ pip install -r requirements.txt


Usage:

Run the whole obfuscation model:

$ python3 total.py <SUMMARIZATION> <KEYWORDS/POSITION SWAP MODE>

SUMMARIZATION 0: no summarize, 1: summarize

KEYWORDS/POSITION SWAP MODE 0: keywords-swap, 1: position-swap

Notice that in summarization mode, we will only do keywords-swap.
	
Example: 
$ python total.py 0 0
will run the obfuscation without summarization and doing keywords-swap.
