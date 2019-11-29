from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import numpy as np
import json
import pickle
import os
from data_utils.csv_to_documents import docs_to_sheet
from skmultilearn.model_selection import IterativeStratification
import configargparse
from utils.argparser import get_parser
import configparser
import gc
from collections import OrderedDict
from main import main as train_eval

def write_interm_results(params, loss):
	global log_path

	params = OrderedDict(params)
	with open(log_path,'w') as f:
		header = "Loss | "
		header += " | ".join(k for k in params.keys())
		f.write(header + '\n')
		lengths = [len(head) for head in header.split('|')]


		result = loss
		vals = [v[0] for k,v in params.items()]

		to_write = ["{0:.03f} ".format(result)]
		to_write.extend(["{0:.03f}".format(v) for v in vals])
		for field, length in zip(to_write, lengths):
			f.write(field.ljust(length+1))
		f.write('\n')


def set_params(params, config_path):

	params['num_compressed_caps'] = int(params['num_compressed_caps'] )
	# reads in config file and overwrites params for optimization
	config = configparser.ConfigParser()
	config.optionxform = str
	config.read(config_path)
	for section in config.sections():
		for param in config[section]:
			if param in params.keys():
				config[section][param] = str(params[param])

	with open(config_path, 'w') as f:
		config.write(f)






def objective(params):
	# return {'loss': 1, 'status': STATUS_OK}
	# objective fn to be minimized
	global data_path, label_to_idx_path, K, config_path
	set_params(params, config_path)

	# read in data

	# get stratisfied split
	df = docs_to_sheet(data_path, 'tmp.csv', label_to_idx_path)
	df.drop(columns=['text'], inplace=True)
	df.reset_index(inplace=True)

	# hacky way to make use of SkMultiLearn
	X = df.index
	y = df[[col for col in df.columns if col != 'index']].values
	del df
	k_fold = IterativeStratification(n_splits=K, order=1)

	# get docs
	with open(data_path, 'rb') as f:
		docs = pickle.load(f)

	scores = []
	tmp_tr_path = 'temp_train.pkl'
	tmp_dev_path = 'temp_dev.pkl'
	params['train_path'] = tmp_tr_path
	params['dev_path'] = tmp_dev_path
	params['test_path'] = tmp_dev_path

	for train_idx, dev_idx in k_fold.split(X,y):
		# get split
		train_docs = [docs[i] for i in train_idx]
		dev_docs = [docs[i] for i in dev_idx]
		# save docs in temp location and free memory
		with open(tmp_tr_path, 'wb') as f:
			pickle.dump(train_docs, f)

		with open(tmp_dev_path, 'wb') as f:
			pickle.dump(dev_docs, f)

		del train_docs, dev_docs
		gc.collect()

		# call main
		r_k, p_k, rp_k, ndcg_k, avg_loss, hamming, emr, f1_micro, f1_macro = train_eval()
		scores.append(f1_micro)



	return {'loss':1-np.mean(scores), 'status':STATUS_OK}

if __name__ == '__main__':

	###########################################
	# INPUT VARIABLES
	###########################################
	# optim settings
	max_evals = 100
	preload_trials = False
	in_trials_path = 'trials.pkl'
	out_trials_path = 'trials.pkl'
	log_path = 'opt_log.txt'
	K = 3
	config_path = 'parameters.ini'

	# data settings
	data_path = r'dataset\reuters-full\train.pkl'
	label_to_idx_path = r'dataset\reuters-full\label_to_idx.json'



	###########################################



	# define search space
	space = {
		'dropout':hp.uniform('dropout', 0.25, 0.75),
		'weight_decay':hp.loguniform('weight_decay', np.log(1e-5), np.log(0.1)),
		'dropout_caps':hp.uniform('dropout_caps', 0.25, 0.75),
		'lambda_reg_caps':hp.loguniform('lambda_reg_caps', np.log(1e-7), np.log(1e-2)),
		'dropout_factor':hp.uniform('dropout_factor', 1.0, 3.0),
		'num_compressed_caps':hp.quniform('num_compressed_caps', 50, 250, 1)
	}

	# Create Trials object to log the performance
	if preload_trials:
		with open(in_trials_path, 'rb') as f:
			trials = pickle.load(f)
		max_evals = len(trials.trials) + max_evals
		print("Rerunning from {} trials to add another one.".format(
			len(trials.trials)))
	else:
		trials = Trials()

	# perform optimization
	# try: # to be sure that trials object will be saved
	best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
	# except:
	# 	pass
	# Store trials

	with open(out_trials_path, 'wb') as f:
		pickle.dump(trials, f)

	# with open(log_path,'w') as f:
	# 	header = "Loss | "
	# 	header += " | ".join(k for k in trials.trials[0]['misc']['vals'].keys())
	# 	f.write(header + '\n')
	# 	lengths = [len(head) for head in header.split('|')]
	# 	for trial in trials.trials:
	# 		result = trial['result']['loss']
	# 		vals = [v[0] for k,v in trial['misc']['vals'].items()]
	#
	# 		to_write = ["{0:.03f} ".format(result)]
	# 		to_write.extend(["{0:.03f}".format(v) for v in vals])
	# 		for field, length in zip(to_write, lengths):
	# 			f.write(field.ljust(length+1))
	# 		f.write('\n')
	# 		# f.write("{:03f}".format(result) + )
			# f.write(' '.join() + '\n')
