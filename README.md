# DataSifterText

Usage:

First, install pytorch pretrained BERT:

```pip install pytorch_pretrained_bert==0.4.0 genism```

Then, run the whole obfuscation model:

```python3 total.py <SUMMARIZATION> <KEYWORDS/POSITION SWAP MODE>```

SUMMARIZATION 0: no summarize, 1: summarize

KEYWORDS/POSITION SWAP MODE 0: keywords-swap, 1: position-swap

Notice that in summarization mode, we will only do keywords-swap.
	
Example: 
```python total.py 0 0```
will run the obfuscation without summarization and doing keywords-swap.
