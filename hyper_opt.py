from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import numpy as np
import json
import pickle

def objective(params):
	# objective fn to be minimized

	return {'loss':0,'status':STATUS_OK}


if __name__ == '__main__':

	###########################################
	# INPUT VARIABLES
	###########################################
	max_evals = 100
	preload_trials = True
	in_trials_path = 'trials.json'
	out_trials_path = 'trials.json'



	###########################################



	# define search space
	space = {
		'dropout':hp.uniform('dropout', 0.25, 0.75),
		'weight_decay':hp.loguniform('weight_decay', np.log(1e-5), np.log(0.1)),
		'dropout_caps':hp.uniform('dropout_caps', 0.25, 0.75),
		'lambda_reg_caps':hp.loguniform('lambda_reg_caps', np.log(1e-7), np.log(1e-2)),
		'dropout_factor':hp.uniform('dropout_factor', 1.0, 3.0),
		'num_compressed_caps':hp.uniform('num_compressed_caps', 50, 250)
	}

	# Create Trials object to log the performance
	if preload_trials:
		with open(in_trials_path, 'rb') as f:
			trials = pickle.load(f)
	else:
		trials = Trials()

	# perform optimization
	best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

	# Store trials

	with open(out_trials_path, 'wb') as f:
		pickle.dump(trials, f)

	for trial in trials:
		print(trial)
