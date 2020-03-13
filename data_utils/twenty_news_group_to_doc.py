from sklearn.datasets import fetch_20newsgroups
import os
import json
import pandas as pd
import numpy as np

from data_utils.csv_to_documents import df_to_docs


def twentynews_to_df(is_train, label_to_idx=None, filter=False):

    remove = ("headers") if filter else ()
    dset = fetch_20newsgroups(
        data_home="data", subset="train" if is_train else "test", remove=remove
    )

    if label_to_idx is None:
        label_to_idx = {lab: ix for ix, lab in enumerate(dset["target_names"])}

    # create one hot encoding of labels
    num_labels = len(label_to_idx)
    all_labels = np.zeros((len(dset["data"]), num_labels))
    all_label_indices = dset["target"]

    for i, labs in enumerate(all_label_indices):
        # binary encode the labels
        all_labels[i][labs] = 1
    all_labels = all_labels.astype(int)

    cols = ["text"]
    label_cols = ["topic_{}".format(lab) for lab in label_to_idx.keys()]
    cols.extend(label_cols)
    df = pd.DataFrame(columns=cols)
    df["text"] = dset["data"]

    df[label_cols] = all_labels

    return df, label_to_idx


def parse(
    out_dir, percentage_train, restructure_doc=True, max_seq_len=50, use_ulmfit=False
):

    assert (
        0 < percentage_train <= 1
    ), "The percentage of docs to be used for training should be between 0 and 1."

    # Make sure the output dir exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # convert to dataframes
    train_df, label_to_idx = twentynews_to_df(True)
    test_df, label_to_idx = twentynews_to_df(False, label_to_idx)

    # split and process data to Documents
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
    pass
