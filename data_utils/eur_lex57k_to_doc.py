import glob
import os
import tqdm
from collections import defaultdict
from data_utils.json_loader import JSONLoader
import operator
import pickle
import json
import concurrent.futures
from joblib import Parallel, delayed

# from gensim.models import FastText


def load_dataset(
    dataset_dir,
    dataset_name,
    num_threads=6,
    restructure_doc=True,
    split_size_long_seqs=50,
):
    """
	Load dataset and return list of documents
	:param dataset_name: the name of the dataset
	:return: list of Document objects
	"""
    filenames = glob.glob(os.path.join(dataset_dir, dataset_name, "*.json"))
    loader = JSONLoader()

    # documents = []
    with concurrent.futures.ProcessPoolExecutor(num_threads) as executor:
        documents = list(
            tqdm.tqdm(executor.map(loader.read_file, filenames), total=len(filenames))
        )

    return documents


def parse(
    raw_data_dir,
    write_data_dir,
    dataset_name,
    num_tags,
    num_backtranslations=None,
    create_wordvecs=False,
    word_vec_dim=300,
    min_count=20,
    restructure_doc=True,
    split_size_long_seqs=50,
):
    out_paths = []

    label_occs = defaultdict(int)
    train_docs = load_dataset(raw_data_dir, "train")
    for doc in train_docs:
        for tag in doc.tags:
            label_occs[tag] += 1

    if num_tags:  # sort and filter
        sorted_tags = sorted(
            label_occs.items(), key=operator.itemgetter(1), reverse=True
        )
        tags_to_use = [x[0] for x in sorted_tags[:num_tags]]
    else:
        tags_to_use = [x[0] for x in label_occs.items()]

    # Dump mapping
    label_to_idx = {x: i for i, x in enumerate(tags_to_use)}
    label_to_idx_path = os.path.join(
        raw_data_dir, "label_to_idx_{}.json".format(num_tags)
    )
    with open(label_to_idx_path, "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f)
    out_paths.append(label_to_idx_path)
    # Filter tags in docs
    for doc in train_docs:
        doc.tags = [tag for tag in doc.tags if tag in tags_to_use]

    # if create_wordvecs: # Create custom word vectors based on train set
    # 	fasttext = FastText(size=word_vec_dim, window=5, min_count=min_count)
    # 	tokens = [sen for doc in train_docs for sen in doc.sentences]
    # 	fasttext.build_vocab(tokens, update=fasttext.corpus_count != 0)
    #
    # 	fasttext.train(sentences=tokens, total_examples=fasttext.corpus_count, epochs=5)

    if num_backtranslations:  # TODO: test backtranslation

        parallel = Parallel(5, backend="threading", verbose=5)
        doc_translations = parallel(
            delayed(doc.back_translate)(num_backtranslations) for doc in train_docs
        )
        # doc_translations = [doc.back_translate(num_backtranslations) for doc in train_docs]
        doc_translations = [doc for sublist in doc_translations for doc in sublist]
        print(
            "CREATED {} EXTRA TRAIN SAMPLES BY BACKTRANSLATION".format(
                len(doc_translations)
            )
        )
        train_docs.extend(doc_translations)

    train_path = os.path.join(
        write_data_dir, "{}_train_{}.pkl".format(dataset_name, num_tags)
    )
    out_paths.append(train_path)
    with open(train_path, "wb") as f:
        pickle.dump(train_docs, f)

    # Process dev and test set
    for name in ["dev", "test"]:
        docs = load_dataset(raw_data_dir, name)

        out_name = "{}_{}_{}".format(dataset_name, name, num_tags)
        for doc in docs:
            doc.tags = [tag for tag in doc.tags if tag in tags_to_use]

        eval_path = os.path.join(write_data_dir, out_name + ".pkl")
        out_paths.append(eval_path)
        with open(eval_path, "wb") as f:
            pickle.dump(docs, f)

    return out_paths
