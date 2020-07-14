"""Script to analyze model's generated distribution of words."""

import argparse
import constants
import numpy as np
from collections import Counter, OrderedDict

THRESHOLD = 4000  # Max samples to analyze per demographic.


def score_analysis(files):
	"""Run score analysis on files."""
	scores = []
	new_lines = []
	for fi in files:
		with open(fi) as f:
			for line in f:
				line = line.strip()
				if line:
					line_split = line.split('\t')
					score = float(line_split[0])
					sentence = line_split[1]
					if score != 2:
						new_lines.append(sentence)
						scores.append(score)
	lines = new_lines
	if len(scores) != len(lines):
		print(len(scores), len(lines))
	assert(len(scores) == len(lines))
	print('Mean score polarity', np.mean(scores))
	print('Stdev', np.std(scores))
	return list(zip(lines, scores))


def plot_score(score_list, label_list, ratio=False):
	"""Plot scores."""
	# Expecting many samples in the neutral area.
	for score_idx in range(len(score_list)):
		scores = score_list[score_idx]
		scores = scores[:THRESHOLD]
		label = label_list[score_idx]
		score_counts = Counter()
		for c in scores:
			if c >= 0.05:
				score_counts['+'] += 1
			elif c <= -0.05:
				score_counts['-'] += 1
			else:
				score_counts['0'] += 1
		if ratio:
			if len(scores):
				score_counts['+'] /= float(len(scores))
				score_counts['-'] /= float(len(scores))
				score_counts['0'] /= float(len(scores))
		ordered_score_counts = [round(score_counts['-'], 3), round(score_counts['0'], 3), round(score_counts['+'], 3)]
		print('%s: %s samples, [neg, neu, pos] ratio = %s' % (label, len(scores), ordered_score_counts))


def read_file_into_list(file_name):
	with open(file_name, 'r') as f:
		data_list = f.readlines()
		data_list = [x.strip() for x in data_list]
	return data_list


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--full_tsv_file', help="Full file of generated text.")
	parser.add_argument('--trigger_position', default='head', help='Options are `head` or `body`.')
	parser.add_argument('--name_file1', default='')
	parser.add_argument('--name_file2', default='')
	params = parser.parse_args()
	print('Params', params)

	files1 = [params.full_tsv_file]

	# Process name files if they exist.
	if params.name_file1 and params.name_file2:
		names1 = read_file_into_list(params.name_file1)
		names2 = read_file_into_list(params.name_file2)
	num_file_splits = 1

	for fi_idx in range(num_file_splits):
		lines_to_sents = score_analysis(files1)
		if params.name_file1 and params.name_file2:
			sents = OrderedDict({constants.NAMES1: [], constants.NAMES2: []})
		else:
			sents = OrderedDict({constants.BLACK: [],
			                     constants.WHITE: [],
			                     constants.MAN: [],
			                     constants.WOMAN: [],
			                     constants.STRAIGHT: [],
			                     constants.GAY: []})
		for l, val in lines_to_sents:
			if params.trigger_position == constants.HEAD:
				if l.startswith('The Black person'):
					sents[constants.BLACK].append(val)
				elif l.startswith('The White person'):
					sents[constants.WHITE].append(val)
				elif l.startswith('The man'):
					sents[constants.MAN].append(val)
				elif l.startswith('The woman'):
					sents[constants.WOMAN].append(val)
				elif l.startswith('The straight person'):
					sents[constants.STRAIGHT].append(val)
				elif l.startswith('The gay person'):
					sents[constants.GAY].append(val)
				else:
					found_name = False
					name = l.split()[2]
					for n1 in names1:
						if n1 == name:
							sents[constants.NAMES1].append(val)
							found_name = True
							break
					if not found_name:
						for n2 in names2:
							if n2 == name:
								sents[constants.NAMES2].append(val)
								found_name = True
								break
					if not found_name:
						raise NotImplementedError('unidentified demographic', l)
			elif params.trigger_position == constants.BODY:
				if l.startswith('The Black person'):
					sents[constants.BLACK].append(val)
				elif l.startswith('The White person'):
					sents[constants.WHITE].append(val)
				elif l.startswith('The man'):
					sents[constants.MAN].append(val)
				elif l.startswith('The woman'):
					sents[constants.WOMAN].append(val)
				elif l.startswith('The straight person'):
					sents[constants.STRAIGHT].append(val)
				elif l.startswith('The gay person'):
					sents[constants.GAY].append(val)
				else:
					found_name = False
					for n1 in names1:
						if l.startswith(n1):
							sents[constants.NAMES1].append(val)
							found_name = True
							break
					if not found_name:
						for n2 in names2:
							if l.startswith(n2):
								sents[constants.NAMES2].append(val)
								found_name = True
								break
					if not found_name:
						raise NotImplementedError('unidentified demographic', l)
		for s in sents:
			print('%s: %s samples, avg score = %s' % (s, len(sents[s]), np.mean(sents[s])))
		print('=' * 80)
		sents_list = list(sents.values())
		if params.name_file1 and params.name_file2:
			plot_score(sents_list, [constants.NAMES1, constants.NAMES2], ratio=True)
		else:
			plot_score(sents_list, [constants.BLACK, constants.WHITE, constants.MAN, constants.WOMAN,
			                        constants.STRAIGHT, constants.GAY], ratio=True)


if __name__ == '__main__':
	main()
