from nltk.corpus import reuters
from document_model import Document, TextPreprocessor
from random import shuffle
import os
import pickle
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from data_utils.csv_to_documents import df_to_docs


def reuters_to_df(set_name, label_to_idx):

    data = [x for x in reuters.fileids() if set_name in x]

    # collect all data to create df from
    all_texts = [
        " ".join([" ".join(sen) for sen in reuters.sents(doc_id)]) for doc_id in data
    ]

    all_labels = np.zeros((len(all_texts), len(label_to_idx)))
    all_label_indices = [
        [label_to_idx[lab] for lab in reuters.categories(doc_id)] for doc_id in data
    ]

    for i, labs in enumerate(all_label_indices):
        # binary encode the labels
        all_labels[i][labs] = 1

    all_labels = all_labels.astype(int)
    # all_labels[all_label_indices] = 1
    cols = ["text"]
    label_cols = ["topic_{}".format(lab) for lab in reuters.categories()]
    cols.extend(label_cols)
    # create df and set values
    df = pd.DataFrame(columns=cols)
    df["text"] = all_texts
    df[label_cols] = all_labels

    return df


def parse(
    out_dir, percentage_train, restructure_doc=True, max_seq_len=50, use_ulmfit=False
):

    assert (
        0 < percentage_train <= 1
    ), "The percentage of docs to be used for training should be between 0 and 1."

    # Make sure the output dir exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Get label_mapping
    label_to_idx = {lab: i for i, lab in enumerate(reuters.categories())}
    # collect all data to create df from
    train_df = reuters_to_df("train", label_to_idx)
    test_df = reuters_to_df("test", label_to_idx)

    # Store dfs and pickled doc files
    df_to_docs(
        train_df,
        label_to_idx,
        out_dir,
        do_split=True,
        dev_percentage=1 - percentage_train,
        store_df=True,
        set_name="train",
        restructure_doc=restructure_doc,
        max_seq_len=max_seq_len,
        use_ulmfit=use_ulmfit,
    )
    df_to_docs(
        test_df,
        label_to_idx,
        out_dir,
        do_split=False,
        dev_percentage=0.5,
        store_df=True,
        set_name="test",
        restructure_doc=restructure_doc,
        max_seq_len=max_seq_len,
        use_ulmfit=use_ulmfit,
    )

    label_to_idx_path = os.path.join(out_dir, "label_to_idx.json")
    with open(label_to_idx_path, "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f)


if __name__ == "__main__":
    None
    # parse('dataset\\reuters', 0.9)
