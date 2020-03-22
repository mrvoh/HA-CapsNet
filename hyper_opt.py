from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import numpy as np
import os
import pickle
from data_utils.csv_to_documents import docs_to_sheet
from skmultilearn.model_selection import IterativeStratification
import argparse
import configparser
import gc
from collections import OrderedDict
from main import main as train_eval
import sys


def write_interm_results(params, loss):
    global log_path

    params = OrderedDict(params)
    with open(log_path, "w") as f:
        header = "Loss | "
        header += " | ".join(k for k in params.keys())
        f.write(header + "\n")
        lengths = [len(head) for head in header.split("|")]

        result = loss
        vals = [v[0] for k, v in params.items()]

        to_write = ["{0:.03f} ".format(result)]
        to_write.extend(["{0:.03f}".format(v) for v in vals])
        for field, length in zip(to_write, lengths):
            f.write(field.ljust(length + 1))
        f.write("\n")


def set_params(params, config_path):

    for param in [
        "num_compressed_caps",
        "min_freq_word",
        "num_cycles_lr"
    ]:
        params[param] = int(params[param])
    # reads in config file and overwrites params for optimization
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_path)
    for section in config.sections():
        for param in config[section]:
            if param in params.keys():
                config[section][param] = str(params[param])

    with open(config_path, "w") as f:
        config.write(f)


def objective(params):
    # objective fn to be minimized
    global train_path, test_path, label_to_idx_path, K, config_path, trials

    # get stratisfied split
    df = docs_to_sheet(train_path, "tmp.csv", label_to_idx_path)
    df.drop(columns=["text"], inplace=True)
    df.reset_index(inplace=True)

    # hacky way to make use of SkMultiLearn
    X = df.index
    y = df[[col for col in df.columns if col != "index"]].values
    del df

    k_fold = IterativeStratification(n_splits=K, order=1)

    # get docs
    with open(train_path, "rb") as f:
        docs = pickle.load(f)

    scores = []
    tmp_tr_path = "temp_train.pkl"
    tmp_dev_path = "temp_dev.pkl"
    params["train_path"] = tmp_tr_path
    params["dev_path"] = tmp_dev_path
    params["test_path"] = test_path
    set_params(params, config_path)

    for train_idx, dev_idx in k_fold.split(X, y):
        # get split
        train_docs = [docs[i] for i in train_idx]
        dev_docs = [docs[i] for i in dev_idx]
        # save docs in temp location and free memory
        with open(tmp_tr_path, "wb") as f:
            pickle.dump(train_docs, f)

        with open(tmp_dev_path, "wb") as f:
            pickle.dump(dev_docs, f)

        del train_docs, dev_docs
        gc.collect()

        # call main
        r_k, p_k, rp_k, ndcg_k, avg_loss, hamming, emr, f1_micro, f1_macro = train_eval(
            False
        )
        scores.append(f1_micro)

    # save trials object for safety
    with open("trials_tmp.pkl", "wb") as f:
        pickle.dump(trials, f)

    return {"loss": 1 - np.mean(scores), "status": STATUS_OK}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_evals", default=40, type=int, help="Total nr of optimization steps."
    )
    parser.add_argument(
        "--K", default=2, type=int, help="Number of splits for cross validation"
    )
    parser.add_argument(
        "--preload_trials",
        action="store_true",
        help="Whether to preload an existing trials object.",
    )
    parser.add_argument(
        "--in_trials_path",
        default="trials_tmp.pkl",
        type=str,
        required=False,
        help="Path of trials object to read in.",
    )
    parser.add_argument(
        "--out_trials_path",
        default="trials.pkl",
        type=str,
        required=False,
        help="Path of trials object save.",
    )
    parser.add_argument(
        "--log_path",
        default="opt_log.txt",
        type=str,
        required=False,
        help="The path where to dump logging.",
    )
    parser.add_argument(
        "--config_path",
        default="parameters.ini",
        type=str,
        required=False,
        help="Path from where to read the config for training.",
    )
    parser.add_argument(
        "--train_path",
        default=os.path.join("dataset", "trec", "train.pkl"),
        type=str,
        required=False,
        help="The path where to dump logging.",
    )
    parser.add_argument(
        "--test_path",
        default=os.path.join("dataset", "trec", "dev.pkl"),
        type=str,
        required=False,
        help="The path where to dump logging.",
    )
    parser.add_argument(
        "--label_to_idx_path",
        default=os.path.join("dataset", "trec", "label_to_idx.json"),
        type=str,
        required=False,
        help="The path where to dump logging.",
    )

    args = parser.parse_args()
    ###########################################
    # INPUT VARIABLES
    ###########################################
    # optim settings
    max_evals = args.max_evals
    preload_trials = args.preload_trials
    in_trials_path = args.in_trials_path
    out_trials_path = args.in_trials_path
    log_path = args.log_path
    K = args.K
    config_path = args.config_path

    # data settings
    train_path = args.train_path
    test_path = args.test_path
    label_to_idx_path = args.label_to_idx_path

    sys.argv = [sys.argv[0]]  # to untangle command lind arguments of hyper_opt and main
    os.rename(config_path, 'parameters.ini')
    config_path = 'parameters.ini'
    ###########################################
    # define search space
    space = {
        "dropout": hp.uniform("dropout", 0.25, 0.75),
        "dropout_caps": hp.uniform("dropout_caps", 0.0, 0.4),
        "lambda_reg_caps": hp.loguniform("lambda_reg_caps", np.log(1e-7), np.log(0.5)),
        "dropout_factor": hp.uniform("dropout_factor", 1.0, 3.0),
        "num_compressed_caps": hp.quniform("num_compressed_caps", 5, 60, 5),
        "min_freq_word": hp.quniform("min_freq_word", 1, 50, 1),
        "num_cycles_lr": hp.quniform("num_cycles_lr", 1, 10, 1),
        "lr_div_factor": hp.uniform("lr_div_factor", 1, 20),
        # "num_head_doc":hp.quniform("num_head_doc", 1, 5, 1)
    }

    # Create Trials object to log the performance
    if preload_trials:
        with open(in_trials_path, "rb") as f:
            trials = pickle.load(f)
        max_evals = len(trials.trials) + max_evals
        print("Rerunning from {} trials to add another one.".format(len(trials.trials)))
    else:
        trials = Trials()

    # perform optimization
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    # Store trials
    with open(out_trials_path, "wb") as f:
        pickle.dump(trials, f)

    with open(log_path, "w") as f:
        header = "Loss | "
        header += " | ".join(k for k in trials.trials[0]["misc"]["vals"].keys())
        f.write(header + "\n")
        lengths = [len(head) for head in header.split("|")]
        for trial in trials.trials:
            result = trial["result"]["loss"]
            vals = [v[0] for k, v in trial["misc"]["vals"].items()]

            to_write = ["{0:.03f} ".format(result)]
            to_write.extend(["{0:.03f}".format(v) for v in vals])
            for field, length in zip(to_write, lengths):
                f.write(field.ljust(length + 1))
            f.write("\n")
            f.write("{:03f}".format(result))

