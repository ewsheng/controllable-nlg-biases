# Towards Controllable Biases in Language Generation

This code generates bias triggers and evaluates the biases in text generated using the bias triggers.

More details can be found in [this paper](https://arxiv.org/abs/2005.00268).

## Dependencies

This trigger search code is written using PyTorch and extended from the code from the paper [Universal Adversarial Triggers for Attacking and Analyzing NLP (EMNLP 2019)](https://github.com/Eric-Wallace/universal-triggers). The code for GPT-2 is based on [HuggingFace's Transformer repo](https://github.com/huggingface/pytorch-transformers). 

The evaluation code relies on the code and data from the paper [The Woman Worked as a Babysitter: On Biases in Language Generation](https://github.com/ewsheng/nlg-bias). *You will need to download this repo  in order to run the evaluation.*

With 1 RTX 2080-Ti GPU, the trigger search takes 1-2 hours.

## Installation

An easy way to install the code is to create a fresh anaconda environment:

```
conda create -n triggers python=3.6
conda activate triggers
pip install -r requirements.txt
```

## Getting Started

### Finding bias triggers

Running `src/create_adv_token.py` generates a bias trigger for specified (demographic, regard) pairs.

A demographic group can be a string (e.g., "The man") or represented by a set of names.

```
python src/create_adv_token.py --help
usage: create_adv_token.py [-h] 
                           [--neg_sample_file NEG_SAMPLE_FILE]
                           [--neu_sample_file NEU_SAMPLE_FILE]
                           [--pos_sample_file POS_SAMPLE_FILE]
                           [--neg_demographic NEG_DEMOGRAPHIC]
                           [--pos_demographic POS_DEMOGRAPHIC]
                           [--neg_name_file NEG_NAME_FILE]
                           [--pos_name_file POS_NAME_FILE]
                           [--salience_threshold SALIENCE_THRESHOLD]
                           [--salient_phrases_file SALIENT_PHRASES_FILE]
                           [--use_original_loss USE_ORIGINAL_LOSS]
                           [--use_salience_loss USE_SALIENCE_LOSS]
                           [--use_dissociation_loss USE_DISSOCIATION_LOSS]
                           [--use_weighted_salience_loss USE_WEIGHTED_SALIENCE_LOSS]
                           [--alpha ALPHA] 
                           [--beta BETA]
                           [--beam_size BEAM_SIZE]
                           [--use_weighted_neg USE_WEIGHTED_NEG]
                           [--trigger_init TRIGGER_INIT]
                           [--num_trigger_tokens NUM_TRIGGER_TOKENS]
                           [--trigger_masked_phrases TRIGGER_MASKED_PHRASES]
                           [--trigger_position TRIGGER_POSITION]
                           [--debias DEBIAS]
                           [--num_demographics NUM_DEMOGRAPHICS]
                           [--model_name_or_path MODEL_NAME_OR_PATH]
                           [--tokenizer_name TOKENIZER_NAME]
                           [--model_type MODEL_TYPE] 
                           [--batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --neg_sample_file NEG_SAMPLE_FILE
                        File of negative regard target samples.
  --neu_sample_file NEU_SAMPLE_FILE
                        File of neutral regard target samples.
  --pos_sample_file POS_SAMPLE_FILE
                        Fle of positive regard target samples.
  --neg_demographic NEG_DEMOGRAPHIC
                        Demographic mention for negative target samples.
  --pos_demographic POS_DEMOGRAPHIC
                        Demographic mention for positive target samples.
  --neg_name_file NEG_NAME_FILE
                        File with names for negative target samples. Overrides
                        neg_demographic.
  --pos_name_file POS_NAME_FILE
                        File with names for positive target samples. Overrides
                        pos_demographic.
  --salience_threshold SALIENCE_THRESHOLD
  --salient_phrases_file SALIENT_PHRASES_FILE
                        File with salient phrases.
  --use_original_loss USE_ORIGINAL_LOSS
                        Use association loss.
  --use_salience_loss USE_SALIENCE_LOSS
  --use_dissociation_loss USE_DISSOCIATION_LOSS
                        Use dissociation loss.
  --use_weighted_salience_loss USE_WEIGHTED_SALIENCE_LOSS
  --alpha ALPHA         Weight for original loss.
  --beta BETA           Weight for dissociation loss.
  --beam_size BEAM_SIZE
                        Beam size when searching for trigger replacement
                        candidates.
  --use_weighted_neg USE_WEIGHTED_NEG
  --trigger_init TRIGGER_INIT
                        Initialize trigger with a phrase.
  --num_trigger_tokens NUM_TRIGGER_TOKENS
  --trigger_masked_phrases TRIGGER_MASKED_PHRASES
  --trigger_position TRIGGER_POSITION
                        Options are `head`, `body_demographic`,
                        `body_biascontext.
  --debias DEBIAS       Whether to generate triggers to debias. 0 = no debias,
                        1 = neutral debias, 2 = neutral + positive debias.
  --num_demographics NUM_DEMOGRAPHICS
                        Whether to use 1 or 2 demographics.
  --model_name_or_path MODEL_NAME_OR_PATH
                        Model name or path: gpt2, microsoft/DialoGPT-medium,
                        etc.
  --tokenizer_name TOKENIZER_NAME
                        Tokenizer name if different from model name.
  --model_type MODEL_TYPE
                        Currently either `gpt2` or `dialogpt`.
  --batch_size BATCH_SIZE
                        32 works well for CPU, 16 for GPU.
```
  
For example, if we wanted to find a trigger for GPT2 that associates positive social connotations 
with "The woman" and negative social connotations with "The man", we could run:
```
python src/create_adv_token.py \
--neg_sample_file data/gpt2_neg_regard.tsv \
--neu_sample_file data/gpt2_neu_regard.tsv \
--pos_sample_file data/gpt2_pos_regard.tsv \
--neg_demographic "The man" \
--pos_demographic "The woman" > neg_man_pos_woman.txt
```

### Evaluating samples generated from bias triggers
First, download the code and regard classifier [here](https://github.com/ewsheng/nlg-bias).

Now you can run `src/eval_triggers.py` to use a found bias trigger to generate samples and then evaluate the samples using 
 the regard classifier.


```
python src/eval_triggers.py --help
usage: eval_triggers.py [-h] 
                        [--trigger_dump_file TRIGGER_DUMP_FILE]
                        [--trigger_label_output_dir TRIGGER_LABEL_OUTPUT_DIR]
                        [--regard_classifier_dir REGARD_CLASSIFIER_DIR]
                        [--trigger_position TRIGGER_POSITION]
                        [--neg_demographic NEG_DEMOGRAPHIC]
                        [--pos_demographic POS_DEMOGRAPHIC]
                        [--neg_name_file NEG_NAME_FILE]
                        [--pos_name_file POS_NAME_FILE] 
                        [--metric METRIC]
                        [--model MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --trigger_dump_file TRIGGER_DUMP_FILE
                        The output file of create_adv_token.py.
  --trigger_label_output_dir TRIGGER_LABEL_OUTPUT_DIR
                        Path to output generated samples.
  --regard_classifier_dir REGARD_CLASSIFIER_DIR
                        Path to top level of regard classifier code directory.
  --trigger_position TRIGGER_POSITION
                        Options are `head` or `body`.
  --neg_demographic NEG_DEMOGRAPHIC
                        Demographic string to associate with negative regard
                        samples.
  --pos_demographic POS_DEMOGRAPHIC
                        Demographic string to associate with positive regard
                        samples.
  --neg_name_file NEG_NAME_FILE
                        Name file for negative association.
  --pos_name_file POS_NAME_FILE
                        Name file for positive association.
  --metric METRIC       Specify metric: `regard2`, `regard1`, `sentiment2`, or
                        `sentiment1`.
  --model MODEL         `gpt2` or `dialogpt`.
```

For example, to evaluate the trigger found using `create_adv_token.py` above:
```
python src/eval_triggers.py \
--trigger_dump_file neg_man_pos_woman.txt \
--trigger_label_output_dir [EXISTING_PATH] \
--regard_classifier_dir [EXISTING_PATH_TO_TOP_LEVEL_REGARD_REPO] \
--neg_demographic "The man" \
--pos_demographic "The woman"
```

### Other features


