"""Script to evaluate generated triggers (with regard classifier)."""

import argparse
import os
import re
import subprocess
from transformers import AutoTokenizer


def get_trigger_list(trigger_dump_file):
	"""Parse trigger_dump_file for final triggers."""
	final_trigger_id = 'Final trigger token IDs: '
	final_trigger = 'Final trigger: '
	trigger_ids = []
	triggers = []
	if trigger_dump_file:
		with open(trigger_dump_file, 'r') as f:
			for line in f:
				line = line.strip()
				if final_trigger_id in line:
					trigger_idx = line.index(final_trigger_id)
					trigger_val = line[trigger_idx + len(final_trigger_id):]
					trigger_val = trigger_val.strip()
					trigger_val = trigger_val.replace('[', '')
					trigger_val = trigger_val.replace(']', '')
					trigger_val_split = trigger_val.split('.')
					trigger_val = ','.join([v.strip() for v in trigger_val_split])
					trigger_ids.append(trigger_val)
				if final_trigger in line:
					trigger_idx = line.index(final_trigger)
					trigger_val = line[trigger_idx + len(final_trigger):]
					trigger_val = trigger_val.rstrip()
					triggers.append(trigger_val)
	# Remove eos.
	tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
	triggers = [x.replace(tokenizer.eos_token, '') for x in triggers]
	trigger_ids = [x for x in trigger_ids if x != tokenizer.eos_token_id]
	return triggers, trigger_ids


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


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--trigger_dump_file', help='The output file of create_adv_token.py.')
	parser.add_argument('--trigger_label_output_dir', help='Path to output generated samples.')
	parser.add_argument('--regard_classifier_dir', help='Path to top level of regard classifier code directory.')
	parser.add_argument('--trigger_position', default='head', help='Options are `head` or `body`.')
	parser.add_argument('--neg_demographic', default='The man', help='Demographic string to associate with negative regard samples.')
	parser.add_argument('--pos_demographic', default='The woman', help='Demographic string to associate with positive regard samples.')
	parser.add_argument('--neg_name_file', default='', help='Name file for negative association.')
	parser.add_argument('--pos_name_file', default='', help='Name file for positive association.')
	parser.add_argument('--metric', default='regard2', help='Specify metric: `regard2`, `regard1`, `sentiment2`, or `sentiment1`.')
	parser.add_argument('--model', default='gpt2', help='`gpt2` or `dialogpt`.')
	params = parser.parse_args()

	trigger_dump_file = params.trigger_dump_file

	print('Params', params)

	trigger_list, trigger_id_list = get_trigger_list(trigger_dump_file)

	trigger_list = trigger_list[:5]
	tsv_files = []
	for trigger, comma_trigger_list in zip(trigger_list, trigger_id_list):
		print('Trigger', trigger)

		# Sample with trigger.
		fname_list = [get_valid_filename(x) for x in trigger.split()]
		sample_tsv_file = '_'.join(fname_list) + '.tsv'
		sample_tsv_file = sample_tsv_file.replace('endoftext', '')
		print('tsv_file', sample_tsv_file)
		labeled_tsv_file = os.path.join(params.trigger_label_output_dir, params.metric + '_' + sample_tsv_file + '_labeled.tsv')

		if not os.path.exists(params.trigger_label_output_dir + '/' + sample_tsv_file):
			p = subprocess.Popen("python src/sample_from_gpt2.py --trigger_list $'" + comma_trigger_list +
			                     "' --trigger_label_output_dir " + params.trigger_label_output_dir +
			                     " --trigger_position $'" + params.trigger_position +
			                     "' --neg_demographic $'" + params.neg_demographic +
			                     "' --pos_demographic $'" + params.pos_demographic +
			                     "' --neg_name_file $'" + params.neg_name_file +
			                     "' --pos_name_file $'" + params.pos_name_file +
			                     "' --model $'" + params.model +
			                     "'", shell=True)
			p.communicate()

		tsv_files.append(params.trigger_label_output_dir + '/' + sample_tsv_file)

		# Use regard classifier to classify samples.
		sample_base_name = os.path.basename(sample_tsv_file).split('.')[0]
		if not os.path.exists(labeled_tsv_file):
			cwd = os.getcwd()
			os.chdir(params.regard_classifier_dir)
			run_classifier = 'bash scripts/run_ensemble.sh ' + params.metric + ' ' + \
			                 params.trigger_label_output_dir + '/' + sample_base_name
			print(run_classifier)
			p = subprocess.Popen(run_classifier, shell=True)
			p.communicate()
			os.chdir(cwd)

		# Calculate ratios of pos/neu/neg samples for evaluation.
		print('=' * 80)
		if params.trigger_position == 'head':
			p = subprocess.Popen('python src/get_model_distrib.py --full_tsv_file ' + labeled_tsv_file +
			                     ' --trigger_position $"' + params.trigger_position + '"' +
			                     ' --name_file1 $"' + params.neg_name_file + '"' +
			                     ' --name_file2 $"' + params.pos_name_file + '"', shell=True)
			p.communicate()
		elif params.trigger_position == 'body':
			p = subprocess.Popen('python src/get_model_distrib.py --full_tsv_file ' + labeled_tsv_file +
			                     ' --trigger_position $"' + params.trigger_position + '"' +
			                     ' --name_file1 $"' + params.neg_name_file + '"' +
			                     ' --name_file2 $"' + params.pos_name_file + '"', shell=True)
			p.communicate()


if __name__ == '__main__':
	main()
