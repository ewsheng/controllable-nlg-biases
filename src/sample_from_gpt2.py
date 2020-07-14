"""Sample from LM using trigger + prompts."""

import torch
import torch.nn.functional as F
from transformers import AutoModelWithLMHead, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel
import argparse
import constants
import re
import numpy as np


# Gets the score for the top-k logits to improve quality of samples.
def top_k_logits(logits, k):
    if k == 0:
        return logits
    values = torch.topk(logits, k)[0]
    batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
    return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


# Generates from the model using optional top-k sampling
def sample_sequence(model, length, batch_size=1, context=None, temperature=1, top_k=10, sample=True, device='cuda'):
    context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in range(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output


def convert_to_bert_format(bias_context_to_generations, bert_file, generation_only=False):
    """Convert to BERT regard classifier format."""
    with open(bert_file, 'w') as f:
        sample_idx = 0
        samples = []
        for bias_context, gen in bias_context_to_generations.items():
            for sample in gen:
                if generation_only:
                    samples.append('\t'.join([str(sample_idx), sample]))
                else:
                    samples.append('\t'.join([str(sample_idx), bias_context + sample]))
                sample_idx += 1
        f.write('\n'.join(samples) + '\n')


def filter_first_sentence(text):
    """Heuristic to only keep the first `sentence` in text."""
    # Cut off the line when we see the first period.
    text = text.replace('\n', '. ').replace('\t', '. ')
    if '! ' in text:
        period_idx = text.index('! ')
    elif '? ' in text:
        period_idx = text.index('? ')
    elif '. ' in text:
        period_idx = text.index('. ')
    else:
        period_idx = len(text)
    sample_end = min(period_idx + 1, len(text))
    text = text[:sample_end]
    return text


# https://github.com/django/django/blob/master/django/utils/text.py
def get_valid_filename(s):
    """
    Return the given string converted to a string that can be used for a clean
    filename. Remove leading and trailing spaces; convert other spaces to
    underscores; and remove anything that is not an alphanumeric, dash,
    underscore, or dot.
    'johns_portrait_in_2004.jpg'
    """
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


def read_file_into_list(file_name):
    if not file_name:
        return []
    with open(file_name, 'r') as f:
        data_list = f.readlines()
        data_list = [x.strip() for x in data_list]
        return data_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trigger_label_output_dir', default='', help='Directory to output generated samples.')
    parser.add_argument('--trigger_list', default='', help='List of tokenized trigger IDs.')
    parser.add_argument('--trigger_position', default='head', help='`head` or `body`.')
    parser.add_argument('--neg_demographic', default='The man')
    parser.add_argument('--pos_demographic', default='The woman')
    parser.add_argument('--neg_name_file', default='')
    parser.add_argument('--pos_name_file', default='')
    parser.add_argument('--model', default='gpt2', help='`gpt2` or `dialogpt`.')
    params = parser.parse_args()
    print('Params', params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if params.model == constants.GPT2:
        tok = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
    elif params.model == constants.DIALOGPT:
        tok = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-medium")
    else:
        raise NotImplementedError('Model has to be gpt2 or dialogpt.')

    model.eval()
    model.to(device)

    # get samples from the model
    if params.model == constants.DIALOGPT:
        bias_contexts1 = read_file_into_list(params.neg_name_file)
        bias_contexts2 = read_file_into_list(params.pos_name_file)
        generations = 10
        respect_contexts = constants.DIALOGPT_RCONTEXTS
        occupation_contexts = constants.DIALOGPT_OCONTEXTS
    else:
        if params.neg_name_file and params.pos_name_file:
            bias_contexts1 = read_file_into_list(params.neg_name_file)
            bias_contexts2 = read_file_into_list(params.pos_name_file)
            generations = 10
        else:
            bias_contexts1 = [params.neg_demographic]
            bias_contexts2 = [params.pos_demographic]
            generations = 100
        respect_contexts = constants.GPT2_RCONTEXTS
        occupation_contexts = constants.GPT2_OCONTEXTS
    orig_trigger_tokens = np.fromstring(params.trigger_list, sep=',')
    trigger = tok.decode(orig_trigger_tokens)

    bias_context_to_generations = {}
    if params.trigger_position == constants.HEAD:
        contexts = [respect_contexts, occupation_contexts]
        for context_list in contexts:
            for context in context_list:
                if type(context) == tuple and len(context) == 2:
                    for bias_context1 in bias_contexts1:
                        bias_context_to_generations[' '.join([context[0], bias_context1, context[1] + tok.eos_token])] = []
                    for bias_context2 in bias_contexts2:
                        bias_context_to_generations[' '.join([context[0], bias_context2, context[1] + tok.eos_token])] = []
                else:
                    for bias_context1 in bias_contexts1:
                        bias_context_to_generations[bias_context1 + ' ' + context] = []
                    for bias_context2 in bias_contexts2:
                        bias_context_to_generations[bias_context2 + ' ' + context] = []
    elif params.trigger_position == constants.BODY:
        for bias_context1 in bias_contexts1:
            bias_context_to_generations[bias_context1 + ' ' + trigger] = []
        for bias_context2 in bias_contexts2:
            bias_context_to_generations[bias_context2 + ' ' + trigger] = []
    else:
        raise NotImplementedError('params.trigger_position is either head or body', params.trigger_position)

    for bias_context in bias_context_to_generations:
        if params.trigger_position == constants.HEAD:
            if trigger:
                trigger_str = trigger + bias_context
                trigger_tokens = np.concatenate((orig_trigger_tokens, tok.encode(bias_context)), axis=0)
            else:
                trigger_str = bias_context
                trigger_tokens = tok.encode(bias_context)
        elif params.trigger_position == constants.BODY:
            trigger_str = bias_context
            trigger_tokens = tok.encode(trigger_str)
        for _ in range(generations):
            out = sample_sequence(
                model=model, length=40, context=trigger_tokens, temperature=0.7, top_k=40, device=device)
            out = out[:, len(trigger_tokens):].tolist()
            for i in range(1):
                text = tok.decode(out[i])
                if text.find(tok.eos_token) > 0:
                    text = text[0:text.find(tok.eos_token)]
                text = text.strip()
                first_sentence = filter_first_sentence(text)
                print("Prompt: ", trigger_str)
                print("Output: ", text)
                print("First sentence: ", first_sentence)
                print("=" * 80)
                bias_context_to_generations[bias_context].append(first_sentence)

    fname_list = [get_valid_filename(x) for x in trigger.split()]
    bert_file = '_'.join(fname_list) + '.tsv'
    bert_file = bert_file.replace('endoftext', '')
    bert_file = params.trigger_label_output_dir + '/' + bert_file
    convert_to_bert_format(bias_context_to_generations, bert_file)

    bert_file += '.XYZ'
    if params.model == constants.GPT2:
        # For GPT2, save a version of the bert file with XYZ demographics for the regard classifier.
        all_contexts = respect_contexts + occupation_contexts
        new_bias_context_to_generations = {}
        for bias_context, generations in bias_context_to_generations.items():
            if params.trigger_position == constants.HEAD:
                found = False
                for context in all_contexts:
                    try:
                        context_idx = bias_context.index(context)
                        bias_context = bias_context[context_idx:]
                        found = True
                        break
                    except ValueError:
                        pass
                if not found:
                    print('bias_context:', bias_context)
                assert found
                bias_context = 'XYZ ' + bias_context
            elif params.trigger_position == constants.BODY:
                bias_context = 'XYZ ' + trigger
            if bias_context not in new_bias_context_to_generations:
                new_bias_context_to_generations[bias_context] = []
            new_bias_context_to_generations[bias_context].extend(generations)
        convert_to_bert_format(new_bias_context_to_generations, bert_file)
    elif params.model == constants.DIALOGPT:
        # For DialoGPT, save just responses with XYZ extension for regard classifier.
        convert_to_bert_format(bias_context_to_generations, bert_file, generation_only=True)


if __name__ == "__main__":
    main()
