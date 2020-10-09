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


To train a BERT model:
# Clone BERT Github Repository:
$ git clone https://github.com/google-research/bert.git
# Download pre-trained BERT model here (Our work uses BERT-Base, Cased):
$ https://github.com/google-research/bert#pre-trained-models
# Using run_classifier.py in this repository, replace the old run_classifier.py
# Create "./data" and "./bert_output" directory
$ mkdir data
$ mkdir bert_output
# Move train_sifter.py to the directory, run train_sifter.py inside the BERT Repository; make sure the data is in the "./data" directory
$ cp [your data] data
$ python3 train_sifter.py
# Now the data is ready. run the following command to start training:
$ python3 run_classifier.py --task_name=cdc --do_train=true --do_eval=true --do_predict=true --data_dir=./data/ --vocab_file=./cased_L-12_H-768_A-12/vocab.txt --bert_config_file=./cased_L-12_H-768_A-12/bert_config.json --max_seq_length=512 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=./bert_output/ --do_lower_case=False

The result will be shown in bert_output directory.
