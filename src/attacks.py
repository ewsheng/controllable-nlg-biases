"""
Contains different methods and helper fns for attacking models. In particular, given the gradients for token
embeddings, it computes the optimal token replacements. This code runs on CPU.
"""

import collections
import constants
import torch
import numpy
import os


def hotflip_attack(averaged_grad, embedding_matrix, trigger_token_ids,
                   increase_loss=False, num_candidates=1):
    """
    The "Hotflip" attack described in Equation (2) of the paper. This code is heavily inspired by
    the nice code of Paul Michel here https://github.com/pmichel31415/translate/blob/paul/
    pytorch_translate/research/adversarial/adversaries/brute_force_adversary.py

    This function takes in the model's average_grad over a batch of examples, the model's
    token embedding matrix, and the current trigger token IDs. It returns the top token
    candidates for each position.

    If increase_loss=True, then the attack reverses the sign of the gradient and tries to increase
    the loss (decrease the model's probability of the true class). For targeted attacks, you want
    to decrease the loss of the target class (increase_loss=False).

    If salience_dict is not None, use the salience scores to weight gradients of different tokens.
    """
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()
    averaged_grad = averaged_grad.unsqueeze(0)
    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik", [averaged_grad, embedding_matrix])

    if not increase_loss:
        gradient_dot_embedding_matrix *= -1    # lower versus increase the class probability.
    if num_candidates > 1:  # get top k options
        best_k_values, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
        return best_k_values.detach().cpu().numpy()[0], best_k_ids.detach().cpu().numpy()[0]
    best_value_at_each_step, best_at_each_step = gradient_dot_embedding_matrix.max(2)
    return best_value_at_each_step[0].detach().cpu().numpy(), best_at_each_step[0].detach().cpu().numpy()


def random_attack(embedding_matrix, trigger_token_ids, num_candidates=1):
    """
    Randomly search over the vocabulary. Gets num_candidates random samples and returns all of them.
    """
    embedding_matrix = embedding_matrix.cpu()
    new_trigger_token_ids = [[None]*num_candidates for _ in range(len(trigger_token_ids))]
    for trigger_token_id in range(len(trigger_token_ids)):
        for candidate_number in range(num_candidates):
            # rand token in the embedding matrix
            rand_token = numpy.random.randint(embedding_matrix.shape[0])
            new_trigger_token_ids[trigger_token_id][candidate_number] = rand_token
    return new_trigger_token_ids


# steps in the direction of grad and gets the nearest neighbor vector.
def nearest_neighbor_grad(averaged_grad, embedding_matrix, trigger_token_ids,
                          tree, step_size, increase_loss=False, num_candidates=1):
    """
    Takes a small step in the direction of the averaged_grad and finds the nearest
    vector in the embedding matrix using a kd-tree.
    """
    new_trigger_token_ids = [[None]*num_candidates for _ in range(len(trigger_token_ids))]
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()
    if increase_loss: # reverse the sign
        step_size *= -1
    for token_pos, trigger_token_id in enumerate(trigger_token_ids):
        # take a step in the direction of the gradient
        trigger_token_embed = torch.nn.functional.embedding(torch.LongTensor([trigger_token_id]),
                                                            embedding_matrix).detach().cpu().numpy()[0]
        stepped_trigger_token_embed = trigger_token_embed + \
            averaged_grad[token_pos].detach().cpu().numpy() * step_size
        # look in the k-d tree for the nearest embedding
        _, neighbors = tree.query([stepped_trigger_token_embed], k=num_candidates)
        for candidate_number, neighbor in enumerate(neighbors[0]):
            new_trigger_token_ids[token_pos][candidate_number] = neighbor
    return new_trigger_token_ids


def find_hard_salient_phrases(neg_data_file, pos_data_file, tokenizer, output_file, use_ngrams=False, salience_threshold=5.0,
                              smooth_lambda=0.5):
    """Find keyphrases that are salient for neg and for pos samples.

    Inspired by "Delete, Retrieve, Generate" style transfer model.

    :param neg_data_file: file with negative regard samples.
    :param pos_data_file: file with positive regard samples.
    :param tokenizer: tokenizer.
    :param output_file: file to output salient phrases to.
    :param use_ngrams: whether to use salient n-grams or just tokens.
    :param salience_threshold: threshold for salience.
    :param smooth_lambda: for smoothing salience calculations.
    :return: a dictionary of phrase to salience score.
    """
    neg_phrases = collections.Counter()
    pos_phrases = collections.Counter()
    salient_phrases = {}
    if os.path.exists(output_file):
        print('Using existent salient phrases file:', output_file)
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                line_split = line.split('\t')
                phrase = line_split[0]
                label = line_split[1]
                score = float(line_split[2])
                salient_phrases[phrase] = (label, score)
    else:
        with open(neg_data_file, 'r') as n, open(pos_data_file, 'r') as p:
            for line in n:
                line = line.strip()
                line_split = tokenizer.tokenize(line)  # Use GPT-2 tokenizer.
                if use_ngrams:  # Use 1- to 3-grams as salient phrase candidates.
                    for start_idx in range(len(line_split)):
                        for n_idx in range(1, 4):
                            if start_idx + n_idx <= len(line_split):
                                ngram = ' '.join(line_split[start_idx: start_idx + n_idx])
                                neg_phrases[ngram] += 1
                else:
                    for token in line_split:
                        neg_phrases[token] += 1

            for line in p:
                line = line.strip()
                line_split = tokenizer.tokenize(line)  # Use GPT-2 tokenizer.
                if use_ngrams:  # Use 1- to 3-grams as salient phrase candidates.
                    for start_idx in range(len(line_split)):
                        for n_idx in range(1, 4):
                            if start_idx + n_idx <= len(line_split):
                                ngram = ' '.join(line_split[start_idx: start_idx + n_idx])
                                pos_phrases[ngram] += 1
                else:
                    for token in line_split:
                        pos_phrases[token] += 1

        # Calculate salience score.
        for phrase in neg_phrases:
            neg_over_pos_score = float(neg_phrases[phrase] + smooth_lambda) / (pos_phrases[phrase] + smooth_lambda)
            pos_over_neg_score = float(pos_phrases[phrase] + smooth_lambda) / (neg_phrases[phrase] + smooth_lambda)
            if neg_over_pos_score >= salience_threshold:
                salient_phrases[phrase] = ('neg', neg_over_pos_score)
            elif pos_over_neg_score >= salience_threshold:
                salient_phrases[phrase] = ('pos', pos_over_neg_score)
        for phrase in pos_phrases:
            if phrase not in neg_phrases:
                score = float(pos_phrases[phrase] + smooth_lambda) / smooth_lambda
                if score >= salience_threshold:
                    salient_phrases[phrase] = ('pos', score)

        with open(output_file, 'w') as f:
            for phrase in salient_phrases:
                has_demo = any([d in phrase.lower() for d in constants.DEMO_LIST])
                if not has_demo:
                    score_type, score = salient_phrases[phrase]
                    f.write('\t'.join([phrase, score_type, str(score)]) + '\n')

    return salient_phrases
