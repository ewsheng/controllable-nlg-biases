"""Create adversarial trigger."""

from copy import deepcopy
import argparse
import torch
import numpy as np
from transformers import AutoModelWithLMHead, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel
import attacks
import constants
import collections
import heapq
import string


# hook used in add_hooks()
extracted_grads = []


def extract_grad_hook(module, grad_in, grad_out):
    global extracted_grads
    extracted_grads.append(grad_out[0])


# Returns the wordpiece embedding weight matrix.
def get_embedding_weight(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == 50257:  # Only add a hook to wordpiece embeddings, not position embeddings.
                return module.weight.detach()


# Add hooks for embeddings.
def add_hooks(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == 50257:  # Only add a hook to wordpiece embeddings, not position.
                module.weight.requires_grad = True
                module.register_backward_hook(extract_grad_hook)


# Gets the loss of the target_tokens using the triggers as the context.
def get_loss(language_model, batch_size, trigger_tokens, demo_tokens, target_tensor, tokenizer,
             device=torch.device('cuda'), salience_token_items=None, use_original_loss=True, use_salience_loss=False,
             use_weighted_salience_loss=False, trigger_position=constants.HEAD, model=constants.GPT2):
    trigger_tensor = torch.tensor(trigger_tokens, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

    complete_trigger_lengths = []

    if model == constants.GPT2:
        respect_contexts = constants.GPT2_RCONTEXTS
        occupation_contexts = constants.GPT2_OCONTEXTS
    elif model == constants.DIALOGPT:
        respect_contexts = constants.DIALOGPT_RCONTEXTS
        occupation_contexts = constants.DIALOGPT_OCONTEXTS
    else:
        raise NotImplementedError('Unrecognized model:', model)

    for sample_idx, target_token_sample in enumerate(target_tensor):
        num_prefix_tokens = len(trigger_tokens)
        # trigger_position = {head, body_demographic, body_biascontext}.
        if constants.BODY in trigger_position:  # Format: [demographic|biascontext] + trigger.
            if constants.DEMO not in trigger_position:  # Trigger does not include demographic yet.
                if type(demo_tokens[0]) == list:  # List of name token ID lists.
                    num_prefix_tokens += len(demo_tokens[sample_idx % len(demo_tokens)])
                else:  # Single list of demographic token IDs.
                    num_prefix_tokens += len(demo_tokens)
            complete_trigger_lengths.append(num_prefix_tokens)
        elif constants.HEAD in trigger_position:  # Format: trigger + demographic + bias_context.
            target_token_sample = [x for x in target_token_sample.tolist() if x != constants.PAD_TOKEN_ID]
            target_str = tokenizer.decode(target_token_sample)  # Convert to string to find bias context strings.
            bias_context_tokens = None
            for c in respect_contexts + occupation_contexts:
                if model == constants.GPT2:
                    context_after = c.strip()
                    if context_after in target_str:
                        bias_context_tokens = tokenizer.encode('The ' + context_after)[1:]  # Dummy first token so that the correct BPE token ID is used for the second token.
                        break
                elif model == constants.DIALOGPT:
                    context_before = c[0].strip()
                    context_after = c[1].strip()
                    if context_after in target_str and context_before in target_str:
                        bias_context_tokens = tokenizer.encode(context_before + ' ' + context_after)
                        break

            if type(demo_tokens[0]) == list:  # List of name token ID lists.
                num_prefix_tokens += len(demo_tokens[sample_idx % len(demo_tokens)])
            else:
                num_prefix_tokens += len(demo_tokens)
            num_prefix_tokens += len(bias_context_tokens)
            complete_trigger_lengths.append(num_prefix_tokens)

    if constants.BODY in trigger_position:
        lm_input_list = [trigger_tensor]
        # For BODY trigger_positions, the trigger_tensor includes bias contexts.
        if constants.DEMO not in trigger_position:  # Trigger does not include demographic, we have to separately concat.
            if type(demo_tokens[0]) == list:  # List of name token ID lists.
                if len(demo_tokens) < batch_size:
                    extended_demo_tokens = []
                    idx = 0
                    while len(extended_demo_tokens) < batch_size:
                        extended_demo_tokens.append(demo_tokens[idx % len(demo_tokens)])
                        idx += 1
                else:
                    demo_tensor = torch.tensor(demo_tokens[:batch_size], device=device, dtype=torch.long)
            else:
                demo_tensor = torch.tensor(demo_tokens, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
            lm_input_list = [demo_tensor] + lm_input_list
        # target_tensor does not include demo or bias_contexts.
        lm_input_list += [target_tensor]
        lm_input = torch.cat(lm_input_list, dim=1)
    else:
        # target_tensor already includes demo + bias_contexts.
        lm_input = torch.cat((trigger_tensor, target_tensor), dim=1)  # we feed the model the trigger + target texts

    mask_out = torch.ones_like(lm_input)  # prepare to mask loss for trigger + demo + bias context tokens.
    for sample_idx, sample in enumerate(mask_out):
        for tok_idx in range(complete_trigger_lengths[sample_idx]):
            sample[tok_idx] = constants.PAD_TOKEN_ID  # Mask these trigger+other prefix tokens out.
    # mask_out = Use lm_input's end padding, mask_out's prefix padding and otherwise mask_out's 1's for target content.
    mask_out = torch.where(lm_input == constants.PAD_TOKEN_ID, lm_input, mask_out)
    mask_and_target = torch.where(mask_out == 1, lm_input, mask_out)  # -1...lm_input -1...
    lm_input[lm_input == constants.PAD_TOKEN_ID] = 1  # put random token of 1 at end of context (it's masked out) # Format: target 1...

    # Printing for debugging.
    # print('trigger_tensor[0]', tokenizer.decode(trigger_tokens), trigger_tensor[0])
    # print('target_tensor[0]', target_tensor[0])
    # print('lm_input[0]', lm_input[0])
    # print('mask_and_target[0]', mask_and_target[0])

    if use_original_loss:
        loss = language_model(lm_input, labels=mask_and_target)[0]
    else:
        loss = None

    if use_salience_loss:
        # Create mask to mask out non-salient tokens.
        non_salience_mask_out = constants.PAD_TOKEN_ID * torch.ones_like(mask_and_target)

        if use_weighted_salience_loss:
            for x in range(5, 26):
                if (salience_token_items[mask_and_target] == x).byte().any():
                    non_salience_mask_and_target = torch.where(salience_token_items[mask_and_target] == x,
                                                               mask_and_target,
                                                               non_salience_mask_out)
                    # Calculate salience loss.
                    salience_loss = language_model(lm_input, labels=non_salience_mask_and_target)[0]
                    del non_salience_mask_and_target

                    # Combine normal loss and salience loss.
                    if loss is None:
                        loss = salience_loss * float(x)
                    else:
                        loss += salience_loss * float(x)
                    del salience_loss

        else:  # Calculate unweighted salience loss.
            if (salience_token_items[mask_and_target] > 0).byte().any():
                non_salience_mask_and_target = torch.where(salience_token_items[mask_and_target] > 0,
                                                           mask_and_target,
                                                           non_salience_mask_out)
                # Calculate salience loss.
                salience_loss = language_model(lm_input, labels=non_salience_mask_and_target)[0]
                del non_salience_mask_and_target

                # Combine normal loss and salience loss.
                if loss is None:
                    loss = salience_loss
                else:
                    loss += salience_loss
                del salience_loss

    return loss, mask_and_target


# Creates the batch of target texts with pad token placed at the end of the sequences for padding (for masking out the loss).
def make_target_batch(tokenizer, device, target_texts, max_len, batch_size):
    # encode items and get the max length
    encoded_texts = []
    for idx, target_text in enumerate(target_texts):
        encoded_target_text = tokenizer.encode(target_text)
        encoded_texts.append(encoded_target_text)

    # pad tokens, i.e., append pad_token_id to the end of the non-longest ones
    for indx, encoded_text in enumerate(encoded_texts):
        if len(encoded_text) < max_len:
            encoded_texts[indx].extend([constants.PAD_TOKEN_ID] * (max_len - len(encoded_text)))
        elif len(encoded_text) > max_len:
            encoded_texts[indx] = encoded_text[:max_len]

    # convert to tensors and batch them up
    target_tokens_batch = None
    for encoded_text in encoded_texts:
        target_tokens = torch.tensor(encoded_text, device=device, dtype=torch.long).unsqueeze(0)
        if target_tokens_batch is None:
            target_tokens_batch = target_tokens
            if target_tokens_batch.shape[0] == batch_size:
                yield target_tokens_batch
                target_tokens_batch = None
        elif target_tokens_batch.shape[0] < batch_size:
            target_tokens_batch = torch.cat((target_tokens_batch, target_tokens), dim=0)
            if target_tokens_batch.shape[0] == batch_size:
                yield target_tokens_batch
                target_tokens_batch = None

    # Just drop the extra samples.
    # if target_tokens_batch is not None:
    #     yield target_tokens_batch


def keep_candidate_token(candidate):
    """Filter out undesired candidate tokens."""
    # Filter out candidates with punctuation and numbers.
    remove_punc = str.maketrans('', '', string.punctuation)
    new_candidate = candidate.translate(remove_punc)
    remove_digits = str.maketrans('', '', string.digits)
    new_candidate = new_candidate.translate(remove_digits)
    # Filter out byte tokens.
    if new_candidate.isprintable():
        return candidate == new_candidate
    else:
        return False


def run_model():
    global extracted_grads

    parser = argparse.ArgumentParser()
    parser.add_argument('--neg_sample_file', default='', help='File of negative regard target samples.')
    parser.add_argument('--neu_sample_file', default='', help='File of neutral regard target samples.')
    parser.add_argument('--pos_sample_file', default='', help='Fle of positive regard target samples.')
    parser.add_argument('--neg_demographic', default='The man', help='Demographic mention for negative target samples.')
    parser.add_argument('--pos_demographic', default='The woman', help='Demographic mention for positive target samples.')
    parser.add_argument('--neg_name_file', default='', help='File with names for negative target samples. Overrides neg_demographic.')
    parser.add_argument('--pos_name_file', default='', help='File with names for positive target samples. Overrides pos_demographic.')
    parser.add_argument('--salience_threshold', default=5.0)
    parser.add_argument('--salient_phrases_file', default='', help='File with salient phrases.')
    parser.add_argument('--use_original_loss', default=1, help='Use association loss.')
    parser.add_argument('--use_salience_loss', default=0)
    parser.add_argument('--use_dissociation_loss', default=1, help='Use dissociation loss.')
    parser.add_argument('--use_weighted_salience_loss', default=0)
    parser.add_argument('--alpha', default=1, help='Weight for original loss.')
    parser.add_argument('--beta', default=1, help='Weight for dissociation loss.')
    parser.add_argument('--beam_size', default=1, help='Beam size when searching for trigger replacement candidates.')
    parser.add_argument('--use_weighted_neg', default=0)
    parser.add_argument('--trigger_init', default='', help='Initialize trigger with a phrase.')
    parser.add_argument('--num_trigger_tokens', default=6)  # Overridden if len trigger_init is greater.
    parser.add_argument('--trigger_masked_phrases', default='')
    parser.add_argument('--trigger_position', default='head', help='Options are `head`, `body_demographic`, `body_biascontext.')
    parser.add_argument('--debias', default=0, help='Whether to generate triggers to debias. 0 = no debias, 1 = neutral '
                                                    'debias, 2 = neutral + positive debias.')
    parser.add_argument('--num_demographics', default=2, help='Whether to use 1 or 2 demographics.')
    parser.add_argument('--model', default='gpt2', help='`gpt2` or `dialogpt`.')
    parser.add_argument('--batch_size', default=16, help='32 works well for CPU, 16 for GPU.')
    params = parser.parse_args()

    params.salience_threshold = float(params.salience_threshold)
    params.use_original_loss = int(params.use_original_loss) == 1
    params.use_salience_loss = int(params.use_salience_loss) == 1
    params.use_dissociation_loss = int(params.use_dissociation_loss) == 1
    params.use_weighted_salience_loss = int(params.use_weighted_salience_loss) == 1
    params.alpha = float(params.alpha)
    params.beta = float(params.beta)
    params.beam_size = int(params.beam_size)
    params.use_weighted_neg = int(params.use_weighted_neg) == 1
    params.num_trigger_tokens = int(params.num_trigger_tokens)
    if params.trigger_masked_phrases:
        params.trigger_masked_phrases = params.trigger_masked_phrases.split(',')
    else:
        params.trigger_masked_phrases = []
    params.debias = int(params.debias)
    assert params.debias in [0, 1, 2]
    # 0 = no debias, 1 = associate neutral, dissociate everything else, 2 = associate positive + neutral, dissociate neg
    params.num_demographics = int(params.num_demographics)

    print('Params', params)

    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    if params.model == constants.GPT2:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
    elif params.model == constants.DIALOGPT:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-medium")
    model.eval()
    model.to(device)

    add_hooks(model)  # add gradient hooks to embeddings
    embedding_weight = get_embedding_weight(model)  # save the word embedding matrix

    total_vocab_size = 50257  # total number of subword pieces in the GPT-2 model
    enc_trigger_init = tokenizer.encode('The ' + params.trigger_init)[1:]
    trigger_init_len = len(enc_trigger_init)
    old_num_trigger_tokens = params.num_trigger_tokens
    params.num_trigger_tokens = max(trigger_init_len, params.num_trigger_tokens)

    # Process trigger_masked_phrases.
    trigger_masked_idxes = []
    for phrase in params.trigger_masked_phrases:
        enc_phrase = tokenizer.encode(phrase)
        enc_trigger_init_str = ' '.join([str(x) for x in enc_trigger_init])
        enc_phrase_str = ' '.join([str(x) for x in enc_phrase])
        if enc_phrase_str in enc_trigger_init_str:
            enc_phrase_str_char_idx = enc_trigger_init_str.index(enc_phrase_str)
            start_idx = enc_trigger_init_str[:enc_phrase_str_char_idx].count(' ')
            for i in range(start_idx, start_idx + len(enc_phrase)):
                trigger_masked_idxes.append(i + params.num_trigger_tokens - 1)
        else:  # Try adding space before the phrase bc of tokenization.
            sp_enc_phrase = tokenizer.encode('x ' + phrase)[1:]
            sp_enc_phrase_str = ' '.join([str(x) for x in sp_enc_phrase])
            if sp_enc_phrase_str in enc_trigger_init_str:
                sp_enc_phrase_str_char_idx = enc_trigger_init_str.index(sp_enc_phrase_str)
                start_idx = enc_trigger_init_str[:sp_enc_phrase_str_char_idx].count(' ')
                for i in range(start_idx, start_idx + len(sp_enc_phrase)):
                    trigger_masked_idxes.append(i + params.num_trigger_tokens - 1)
            else:
                print('Masked phrase not found', enc_phrase, sp_enc_phrase, enc_trigger_init)
                exit()
    print('trigger_masked_idxes', trigger_masked_idxes)

    max_len = 50

    # Calculate salience scores.
    pos_salience_token_items = None
    neg_salience_token_items = None
    if params.use_salience_loss:
        salience_dict = attacks.find_hard_salient_phrases(params.neg_sample_file, params.pos_sample_file, tokenizer,
                                                          params.salient_phrases_file,
                                                          salience_threshold=params.salience_threshold)
        neg_salience_token_items = [0] * total_vocab_size
        pos_salience_token_items = [0] * total_vocab_size
        for phrase in salience_dict:
            label, score = salience_dict[phrase]
            tok_ids = tokenizer.encode(phrase)
            if label == 'neg':
                for tok_id in tok_ids:
                    neg_salience_token_items[tok_id] += int(round(score))
            elif label == 'pos':
                for tok_id in tok_ids:
                    pos_salience_token_items[tok_id] += int(round(score))
            else:
                raise NotImplementedError('Label is either neg or pos.')
        print('neg_salience_token_items', neg_salience_token_items[:20])
        print('pos_salience_token_items', pos_salience_token_items[:20])

    with open(params.neg_sample_file, 'r') as f:
        neg_target_texts = f.readlines()
        if params.model == constants.GPT2:
            neg_target_texts = [l.strip() for l in neg_target_texts]
        elif params.model == constants.DIALOGPT:
            neg_target_texts = [l.strip().split('\t') for l in neg_target_texts]
    with open(params.pos_sample_file, 'r') as f:
        pos_target_texts = f.readlines()
        if params.model == constants.GPT2:
            pos_target_texts = [l.strip() for l in pos_target_texts]
        elif params.model == constants.DIALOGPT:
            pos_target_texts = [l.strip().split('\t') for l in pos_target_texts]
    neu_target_texts = []
    if params.neu_sample_file:
        with open(params.neu_sample_file, 'r') as f:
            neu_target_texts = f.readlines()
            if params.model == constants.GPT2:
                neu_target_texts = [l.strip() for l in neu_target_texts]
            elif params.model == constants.DIALOGPT:
                neu_target_texts = [l.strip().split('\t') for l in neu_target_texts]

    if constants.DEMO not in params.trigger_position:
        neg_demo_neg_target_texts = []
        pos_demo_neg_target_texts = []
        neg_demo_pos_target_texts = []
        pos_demo_pos_target_texts = []
        neg_demo_neu_target_texts = []
        pos_demo_neu_target_texts = []
        if params.neg_name_file and params.pos_name_file:  # Use names instead of demographic groups.
            neg_names = open(params.neg_name_file, 'r').readlines()
            neg_names = [x for x in neg_names if x]
            pos_names = open(params.pos_name_file, 'r').readlines()
            pos_names = [x for x in pos_names if x]
            # If # names is >= batch_size, reset names for each batch_size-th sample.
            # Otherwise, if # names < batch_size, reset names after cycling through all names AND for each batch_size-th sample.
            # Resetting after each batch_size-th sample is just easier for keeping track of loss masking.
            batch_size_mod_number = params.batch_size
            neg_mod_number = min(len(neg_names), params.batch_size)
            pos_mod_number = min(len(pos_names), params.batch_size)
            for idx, l in enumerate(neg_target_texts):
                mod_idx = idx % batch_size_mod_number
                if mod_idx >= neg_mod_number:
                    mod_idx = mod_idx % neg_mod_number
                neg_name = neg_names[mod_idx].strip()
                if params.model == constants.GPT2:
                    neg_demo_neg_target_texts += [neg_name + ' ' + l]
                elif params.model == constants.DIALOGPT:
                    neg_demo_neg_target_texts += [l[0] + ' ' + neg_name + ' ' + l[1]]

                mod_idx = idx % batch_size_mod_number
                if mod_idx >= pos_mod_number:
                    mod_idx = mod_idx % pos_mod_number
                pos_name = pos_names[mod_idx].strip()
                if params.model == constants.GPT2:
                    pos_demo_neg_target_texts += [pos_name + ' ' + l]
                elif params.model == constants.DIALOGPT:
                    pos_demo_neg_target_texts += [l[0] + ' ' + pos_name + ' ' + l[1]]

            for idx, l in enumerate(pos_target_texts):
                mod_idx = idx % batch_size_mod_number
                if mod_idx >= neg_mod_number:
                    mod_idx = mod_idx % neg_mod_number
                neg_name = neg_names[mod_idx].strip()
                if params.model == constants.GPT2:
                    neg_demo_pos_target_texts += [neg_name + ' ' + l]
                elif params.model == constants.DIALOGPT:
                    neg_demo_pos_target_texts += [l[0] + ' ' + neg_name + ' ' + l[1]]

                mod_idx = idx % batch_size_mod_number
                if mod_idx >= pos_mod_number:
                    mod_idx = mod_idx % pos_mod_number
                pos_name = pos_names[mod_idx].strip()
                if params.model == constants.GPT2:
                    pos_demo_pos_target_texts += [pos_name + ' ' + l]
                elif params.model == constants.DIALOGPT:
                    pos_demo_pos_target_texts += [l[0] + ' ' + pos_name + ' ' + l[1]]

            for idx, l in enumerate(neu_target_texts):
                mod_idx = idx % batch_size_mod_number
                if mod_idx >= neg_mod_number:
                    mod_idx = mod_idx % neg_mod_number
                neg_name = neg_names[mod_idx].strip()
                if params.model == constants.GPT2:
                    neg_demo_neu_target_texts += [neg_name + ' ' + l]
                elif params.model == constants.DIALOGPT:
                    neg_demo_neu_target_texts += [l[0] + ' ' + neg_name + ' ' + l[1]]

                mod_idx = idx % batch_size_mod_number
                if mod_idx >= pos_mod_number:
                    mod_idx = mod_idx % pos_mod_number
                pos_name = pos_names[mod_idx].strip()
                if params.model == constants.GPT2:
                    pos_demo_neu_target_texts += [pos_name + ' ' + l]
                elif params.model == constants.DIALOGPT:
                    pos_demo_neu_target_texts += [l[0] + ' ' + pos_name + ' ' + l[1]]

        else:  # Use demographic groups.
            for l in neg_target_texts:
                neg_demo_neg_target_texts += [params.neg_demographic + ' ' + l]
                pos_demo_neg_target_texts += [params.pos_demographic + ' ' + l]
            for l in pos_target_texts:
                neg_demo_pos_target_texts += [params.neg_demographic + ' ' + l]
                pos_demo_pos_target_texts += [params.pos_demographic + ' ' + l]
            for l in neu_target_texts:
                neg_demo_neu_target_texts += [params.neg_demographic + ' ' + l]
                pos_demo_neu_target_texts += [params.pos_demographic + ' ' + l]
    else:
        neg_demo_neg_target_texts = neg_target_texts
        pos_demo_neg_target_texts = neg_target_texts
        pos_demo_pos_target_texts = pos_target_texts
        neg_demo_pos_target_texts = pos_target_texts
        pos_demo_neu_target_texts = neu_target_texts
        neg_demo_neu_target_texts = neu_target_texts

    if constants.BODY in params.trigger_position:
        if constants.BC in params.trigger_position:
            # When the trigger encapsulates the bias contexts, we strip bias contexts in the target texts.
            for bc in constants.GPT2_BIAS_CONTEXTS:
                pos_demo_pos_target_texts = [x.replace(bc, '').strip() for x in pos_demo_pos_target_texts]
                neg_demo_neg_target_texts = [x.replace(bc, '').strip() for x in neg_demo_neg_target_texts]
                pos_demo_neg_target_texts = [x.replace(bc, '').strip() for x in pos_demo_neg_target_texts]
                neg_demo_pos_target_texts = [x.replace(bc, '').strip() for x in neg_demo_pos_target_texts]
                pos_demo_neu_target_texts = [x.replace(bc, '').strip() for x in pos_demo_neu_target_texts]
                neg_demo_neu_target_texts = [x.replace(bc, '').strip() for x in neg_demo_neu_target_texts]

    print('neg demo neg target text:', neg_demo_neg_target_texts[0])
    print('pos demo pos target text:', pos_demo_pos_target_texts[0])

    if params.use_dissociation_loss:
        print('pos demo neg target text:', pos_demo_neg_target_texts[0])
        print('neg demo pos target text:', neg_demo_pos_target_texts[0])

    if params.neu_sample_file:
        print('neg demo neu target text:', neg_demo_neu_target_texts[0])
        print('pos demo neu target text:', pos_demo_neu_target_texts[0])

    # batch and pad the target tokens
    neg_demo_neg_target_tokens_gen = make_target_batch(tokenizer, device, neg_demo_neg_target_texts, max_len,
                                                       params.batch_size)
    pos_demo_pos_target_tokens_gen = make_target_batch(tokenizer, device, pos_demo_pos_target_texts, max_len,
                                                       params.batch_size)
    neg_demo_neg_target_tokens_gen = list(neg_demo_neg_target_tokens_gen)
    same_demo_target_threshold = len(neg_demo_neg_target_tokens_gen)
    pos_demo_pos_target_tokens_gen = list(pos_demo_pos_target_tokens_gen)
    same_demo_target_losses = neg_demo_neg_target_tokens_gen + pos_demo_pos_target_tokens_gen

    if params.use_dissociation_loss:
        pos_demo_neg_target_tokens_gen = make_target_batch(tokenizer, device, pos_demo_neg_target_texts, max_len,
                                                           params.batch_size)
        neg_demo_pos_target_tokens_gen = make_target_batch(tokenizer, device, neg_demo_pos_target_texts, max_len,
                                                           params.batch_size)
        pos_demo_neg_target_tokens_gen = list(pos_demo_neg_target_tokens_gen)
        diff_demo_target_threshold = len(pos_demo_neg_target_tokens_gen)
        neg_demo_pos_target_tokens_gen = list(neg_demo_pos_target_tokens_gen)
        diff_demo_target_losses = pos_demo_neg_target_tokens_gen + neg_demo_pos_target_tokens_gen

    neu_target_losses = []
    if params.neu_sample_file:
        pos_demo_neu_target_tokens_gen = make_target_batch(tokenizer, device, pos_demo_neu_target_texts, max_len,
                                                           params.batch_size)
        neg_demo_neu_target_tokens_gen = make_target_batch(tokenizer, device, neg_demo_neu_target_texts, max_len,
                                                           params.batch_size)
        pos_demo_neu_target_tokens_gen = list(pos_demo_neu_target_tokens_gen)
        neu_target_threshold = len(pos_demo_neu_target_tokens_gen)
        neg_demo_neu_target_tokens_gen = list(neg_demo_neu_target_tokens_gen)
        neu_target_losses = pos_demo_neu_target_tokens_gen + neg_demo_neu_target_tokens_gen

    # Interleave negative and positive add_losses, shuffle all items.
    all_items = []
    if params.debias:  # Generate debiasing triggers.
        assert neu_target_losses
        for idx, l in enumerate(neu_target_losses):
            if idx < neu_target_threshold:
                all_items += [('add', 'pos', l)]
            else:
                all_items += [('add', 'neg', l)]
        if params.debias == 1:
            # A - B where A = neu_target_losses and B = same_demo_target_losses + diff_demo_target_losses.
            same_demo_target_loss_type = 'sub'
            diff_demo_target_loss_type = 'sub'
    else:  # Debias = 0, generate adversarial triggers.
        same_demo_target_loss_type = 'add'
        diff_demo_target_loss_type = 'sub'

    for idx, l in enumerate(same_demo_target_losses):
        if params.num_demographics == 1:
            if idx < same_demo_target_threshold:
                # (Whether to add or subtract loss (add), demographic type (neg), samples).
                all_items += [(same_demo_target_loss_type, 'neg', l)]
        elif params.num_demographics == 2:
            if idx < same_demo_target_threshold:
                if params.debias == 2:
                    # A - B where A = neu_target_losses + pos_target_losses, and B = neg_target_losses.
                    same_demo_target_loss_type = 'sub'
                all_items += [(same_demo_target_loss_type, 'neg', l)]  # (Whether to add or subtract loss, demographic type, samples).
            else:
                if params.debias == 2:
                    same_demo_target_loss_type = 'add'
                all_items += [(same_demo_target_loss_type, 'pos', l)]
        else:
            raise NotImplementedError('num_demographics has to be in [1, 2]: %s' % params.num_demographics)
    if params.use_dissociation_loss:
        for idx, l in enumerate(diff_demo_target_losses):
            if idx < diff_demo_target_threshold:
                if params.debias == 2:
                    diff_demo_target_loss_type = 'sub'
                all_items += [(diff_demo_target_loss_type, 'pos', l)]
            else:
                if params.debias == 2:
                    diff_demo_target_loss_type = 'add'
                all_items += [(diff_demo_target_loss_type, 'neg', l)]

    np.random.shuffle(all_items)

    # Useful for debugging:
    # for i in range(min(10, len(all_items))):
    #     itm = all_items[i]
    #     sample = [x for x in itm[2][0].tolist() if x != constants.PAD_TOKEN_ID]
    #     print(sample)
    #     print(itm[0], itm[1], tokenizer.decode(sample))

    for restart_idx in range(1):  # Different random restarts of the trigger
        print('Random restart: ', str(restart_idx))

        trigger_tokens = tokenizer.encode('The ' + params.trigger_init)[1:]
        if trigger_init_len < old_num_trigger_tokens:
            # Sample random initial trigger.
            # rand_trigger_tokens = np.random.randint(total_vocab_size, size=old_num_trigger_tokens - trigger_init_len)
            rand_trigger_tokens = [tokenizer.encode('x the')[-1]] * (old_num_trigger_tokens - trigger_init_len)
            trigger_tokens = np.concatenate((trigger_tokens, rand_trigger_tokens), axis=0)
        if params.model == constants.DIALOGPT:  # Add eos after trigger.
            trigger_tokens = np.concatenate((trigger_tokens, [tokenizer.eos_token_id]), axis=0)
        print('Random initial trigger:', tokenizer.decode(trigger_tokens))

        # Note that beam_cache, new_beam_cache, and loss_heap all have reverse sign losses.
        # best_loss and curr_best_loss have original sign losses.
        best_loss = 999999  # We want to minimize loss.
        best_trigger_tokens = deepcopy(trigger_tokens)
        beam_cache = [(-999999, trigger_tokens)]  # Always keep beam_size full trigger candidates.
        end_iter = False
        for entire_trigger_update_idx in range(50):  # this many updates of the entire trigger sequence
            print('Updating entire trigger for the', str(entire_trigger_update_idx), '-th time')

            if end_iter:
                continue

            for token_to_flip in range(params.num_trigger_tokens):
                right_counter_token_to_flip = token_to_flip

                if token_to_flip in trigger_masked_idxes:
                    print('Trigger token #', str(token_to_flip), str(right_counter_token_to_flip))
                    continue  # Don't modify these triggers.

                # Beam search for each trigger_tokens in beam_cache.
                assert len(beam_cache) <= params.beam_size
                new_beam_cache = []
                for _, trigger_tokens in beam_cache:
                    print('Trigger token #', str(token_to_flip), str(right_counter_token_to_flip))
                    print(tokenizer.decode(trigger_tokens), trigger_tokens)

                    model.zero_grad()
                    extracted_grads = []  # Each element is (batch_size, sample_length, 768_embed_dim).
                    loss_types = []  # Order of `add` and `sub` loss types.
                    demo_types = []  # Order of `neg` or `pos` demographic types.
                    for idx, (typ, demo_type, target_tokens) in enumerate(all_items):
                        loss_types.append(typ)
                        demo_types.append(demo_type)

                        if demo_type == 'neg':
                            if params.neg_name_file:
                                demo_tokens = [tokenizer.encode('The ' + n)[1:] for n in neg_names]
                            else:
                                demo_tokens = tokenizer.encode(params.neg_demographic)
                        elif demo_type == 'pos':
                            if params.pos_name_file:
                                demo_tokens = [tokenizer.encode('The ' + n)[1:] for n in pos_names]
                            else:
                                demo_tokens = tokenizer.encode(params.pos_demographic)
                        else:
                            raise NotImplementedError('Label is either neg or pos.')

                        if params.use_salience_loss and not params.debias:
                            if (demo_type == 'neg' and typ == 'add') or (demo_type == 'pos' and typ == 'sub'):
                                # association loss between neg demographic and neg samples (to add) or
                                # association loss between pos demographic and neg samples (to subtract).
                                salience_token_items = neg_salience_token_items
                            elif (demo_type == 'pos' and typ == 'add') or (demo_type == 'neg' and typ == 'sub'):
                                # association loss between pos demographic and pos samples (to add) or
                                # association loss between neg demographic and pos samples (to subtract).
                                salience_token_items = pos_salience_token_items
                            else:
                                raise NotImplementedError('Label and demographic pair not possible', typ, demo_type)
                            salience_token_items_tensor = torch.tensor(salience_token_items, device=device,
                                                                       dtype=torch.long)
                        else:
                            salience_token_items_tensor = None

                        loss, _ = get_loss(model, params.batch_size, trigger_tokens, demo_tokens, target_tokens, tokenizer, device,
                                        salience_token_items=salience_token_items_tensor,
                                        use_original_loss=params.use_original_loss,
                                        use_salience_loss=params.use_salience_loss,
                                        use_weighted_salience_loss=params.use_weighted_salience_loss,
                                        trigger_position=params.trigger_position,
                                        model=params.model)
                        loss.backward()
                        del loss, salience_token_items_tensor

                    # Get average gradient w.r.t. the triggers.
                    add_indices = [i for i, loss_type in enumerate(loss_types) if loss_type == 'add']
                    add_extracted_grads = []
                    for i in add_indices:
                        extracted_grad = extracted_grads[i]
                        if params.use_weighted_neg and demo_types[i] == 'neg':  # Amplify neg associations.
                            extracted_grad *= 2
                        add_extracted_grads.append(extracted_grad)
                    add_grad_tensor = torch.stack(add_extracted_grads)  # Convert to tensor.
                    add_grad_tensor = torch.sum(add_grad_tensor, dim=0)  # Add all batches.
                    add_grad_tensor = torch.sum(add_grad_tensor, dim=0)  # Add all samples in a `batch`.
                    add_grad_tensor = add_grad_tensor[token_to_flip].unsqueeze(0)  # Use gradients at token_to_flip.
                    grad = add_grad_tensor
                    if params.use_dissociation_loss:
                        grad *= params.alpha
                        sub_indices = [i for i, loss_type in enumerate(loss_types) if loss_type == 'sub']
                        sub_extracted_grads = []
                        for i in sub_indices:
                            extracted_grad = extracted_grads[i]
                            if params.use_weighted_neg and demo_types[i] == 'neg':  # Amplify neg associations.
                                extracted_grad *= 2
                            sub_extracted_grads.append(extracted_grad)
                        sub_grad_tensor = torch.stack(sub_extracted_grads)  # Convert to tensor.
                        sub_grad_tensor = torch.sum(sub_grad_tensor, dim=0)  # Add all batches.
                        sub_grad_tensor = torch.sum(sub_grad_tensor, dim=0)  # Add all samples in a `batch`.
                        sub_grad_tensor = sub_grad_tensor[token_to_flip].unsqueeze(0)  # Use gradients at token_to_flip.
                        grad -= params.beta * sub_grad_tensor

                    # Use hotflip (linear approximation) attack to get the top num_candidates.
                    candidate_values, candidates = attacks.hotflip_attack(
                        grad, embedding_weight, [trigger_tokens[right_counter_token_to_flip]],
                        increase_loss=False, num_candidates=100)
                    candidates = candidates[0]
                    candidate_values = candidate_values[0]

                    # Try all the candidates and pick the best.
                    loss_heap = []
                    heapq.heapify(loss_heap)  # This is a min heap, so need to flip all losses to end up with the real smallest loss.
                    eval_threshold = 5
                    for cand_value, cand in zip(candidate_values, candidates):

                        # Don't include tokens that have punctuation.
                        decoded_cand = tokenizer.decode([cand])
                        keep_token = keep_candidate_token(decoded_cand)
                        if not keep_token:
                            continue

                        # replace one token with new candidate
                        candidate_trigger_tokens = deepcopy(trigger_tokens)
                        candidate_trigger_tokens[right_counter_token_to_flip] = cand
                        curr_assoc_loss = 0.0
                        curr_dissoc_loss = 0.0
                        eval_set = collections.Counter()
                        total_assoc_elements = 0.0
                        total_dissoc_elements = 0.0
                        for idx, (typ, demo_type, target_tokens) in enumerate(all_items):
                            if eval_set[(typ, demo_type)] < eval_threshold:
                                eval_set[(typ, demo_type)] += 1
                            else:
                                continue

                            if demo_type == 'neg':
                                if params.neg_name_file:
                                    demo_tokens = [tokenizer.encode('The ' + n)[1:] for n in neg_names]
                                else:
                                    demo_tokens = tokenizer.encode(params.neg_demographic)
                            elif demo_type == 'pos':
                                if params.pos_name_file:
                                    demo_tokens = [tokenizer.encode('The ' + n)[1:] for n in pos_names]
                                else:
                                    demo_tokens = tokenizer.encode(params.pos_demographic)
                            else:
                                raise NotImplementedError('Label is either neg or pos.')

                            if params.use_salience_loss and not params.debias:
                                if (demo_type == 'neg' and typ == 'add') or (demo_type == 'pos' and typ == 'sub'):
                                    # association loss between neg demographic and neg samples (to add) or
                                    # association loss between pos demographic and neg samples (to subtract).
                                    salience_token_items = neg_salience_token_items
                                elif (demo_type == 'pos' and typ == 'add') or (demo_type == 'neg' and typ == 'sub'):
                                    # association loss between pos demographic and pos samples (to add) or
                                    # association loss between neg demographic and pos samples (to subtract).
                                    salience_token_items = pos_salience_token_items
                                else:
                                    raise NotImplementedError('Label and demographic pair not possible', typ, demo_type)
                                # Add demo to salience token items.
                                salience_token_items_tensor = torch.tensor(salience_token_items, device=device,
                                                                           dtype=torch.long)
                            else:
                                salience_token_items_tensor = None

                            # get loss, update current best if its lower loss
                            loss, mask_and_target = get_loss(model, params.batch_size, candidate_trigger_tokens, demo_tokens, target_tokens,
                                            tokenizer, device, salience_token_items=salience_token_items_tensor,
                                            use_original_loss=params.use_original_loss,
                                            use_salience_loss=params.use_salience_loss,
                                            use_weighted_salience_loss=params.use_weighted_salience_loss,
                                            trigger_position=params.trigger_position,
                                            model=params.model)

                            if typ == 'add':
                                # Losses are averaged per non-ignored element per sample per batch.
                                # Since we are calculating overall loss over many batches, re-calc average.
                                curr_num_elements = 0
                                for sample in mask_and_target:
                                    curr_num_elements += sum([1 for elem in sample if elem != -1])
                                total_assoc_elements += curr_num_elements
                                if demo_type == 'neg' and params.use_weighted_neg:  # Amplify neg associations.
                                    curr_assoc_loss += 2 * loss.data.item() * curr_num_elements
                                else:
                                    curr_assoc_loss += loss.data.item() * curr_num_elements
                            elif typ == 'sub':
                                curr_num_elements = 0
                                for sample in mask_and_target:
                                    curr_num_elements += sum([1 for elem in sample if elem != -1])
                                total_dissoc_elements += curr_num_elements
                                if demo_type == 'neg' and params.use_weighted_neg:  # Amplify neg associations.
                                    curr_dissoc_loss += 2 * loss.data.item() * curr_num_elements
                                else:
                                    curr_dissoc_loss += loss.data.item() * curr_num_elements
                            del loss, salience_token_items_tensor

                            if all([x == eval_threshold for x in eval_set.values()]):
                                break

                        curr_assoc_loss /= total_assoc_elements
                        if params.use_dissociation_loss:
                            curr_dissoc_loss /= total_dissoc_elements
                            curr_total_loss = (params.alpha * curr_assoc_loss) - (params.beta * curr_dissoc_loss)
                        else:
                            curr_total_loss = curr_assoc_loss

                        # Keep top beam_size elements.
                        # Note that beam_cache, new_beam_cache, and loss_heap all have reverse sign losses.
                        curr_total_loss *= -1
                        if len(new_beam_cache) < params.beam_size:
                            heapq.heappush(loss_heap, curr_total_loss)
                            new_beam_cache.append((curr_total_loss, deepcopy(candidate_trigger_tokens)))
                            curr_worst_loss = heapq.nsmallest(1, loss_heap)[0]
                        else:
                            if curr_total_loss > curr_worst_loss:  # Remember, signs are flipped.
                                # Kick out 1 trigger_tokens sequence with loss = curr_worst_loss.
                                curr_worst_loss_idx_list = [cache_idx for cache_idx, (x, _) in enumerate(new_beam_cache) if x == curr_worst_loss]
                                del new_beam_cache[curr_worst_loss_idx_list[0]]
                                heapq.heappop(loss_heap)

                                heapq.heappush(loss_heap, curr_total_loss)
                                new_beam_cache.append((curr_total_loss, deepcopy(candidate_trigger_tokens)))
                                curr_worst_loss = heapq.nsmallest(1, loss_heap)[0]

                beam_cache = new_beam_cache

            curr_best_loss = 999999
            for x, y in beam_cache:
                x *= -1  # Flip loss back to original sign.
                if x < curr_best_loss:
                    curr_best_loss = x
                    trigger_tokens = deepcopy(y)
            print("Loss: " + str(curr_best_loss))
            print('Trigger token IDs:', trigger_tokens)
            print('Trigger string:', tokenizer.decode(trigger_tokens) + '\n')
            if curr_best_loss < best_loss:
                best_loss = curr_best_loss
                best_trigger_tokens = deepcopy(trigger_tokens)
            elif curr_best_loss == best_loss:
                pass
            else:
                end_iter = True

        # Print final trigger.
        print("Final loss: " + str(best_loss))
        print('Final trigger token IDs:', best_trigger_tokens)
        print('Final trigger:', tokenizer.decode(best_trigger_tokens))


if __name__ == '__main__':
    run_model()

