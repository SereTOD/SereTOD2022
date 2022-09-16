# Copyright 2022 SereTOD Challenge Organizers
# Authors: Hao Peng (peng-h21@mails.tsinghua.edu.cn)
# Apache 2.0
from typing import Dict
from whitespace_tokenizer import WordLevelTokenizer


def get_bio_labels(original_labels, labels_to_exclude=["NA"]) -> Dict[str, int]:
    bio_labels = {"O": 0}
    for label in original_labels:
        if label in labels_to_exclude:
            continue
        bio_labels[f"B-{label}"] = len(bio_labels)
        bio_labels[f"I-{label}"] = len(bio_labels)
    return bio_labels


def get_word_ids(tokenizer,
                 outputs,
                 word_list):
    """Return a list mapping the tokens to their actual word in the initial sentence for a tokenizer.

    Return a list indicating the word corresponding to each token. Special tokens added by the tokenizer are mapped to
    None and other tokens are mapped to the index of their corresponding word (several tokens will be mapped to the same
    word index if they are parts of that word).

    Args:
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer that has been used for word tokenization.
        outputs (`BatchEncoding`):
            The outputs of the tokenizer.
        word_list (`List[str]`):
            A list of word strings.
    Returns:
        word_ids (`List[int]`):
            A list mapping the tokens to their actual word in the initial sentence
    """
    word_list = [w.lower() for w in word_list]
    try:
        word_ids = outputs.word_ids()
        return word_ids
    except:
        assert isinstance(tokenizer, WordLevelTokenizer)
        pass
    tokens = tokenizer.convert_ids_to_tokens(outputs["input_ids"])
    word_ids = []
    word_idx = 0

    for token in tokens:
        if token not in word_list and token != "[UNK]":
            word_ids.append(None)
        else:
            if token != "[UNK]":
                if token != word_list[word_idx]:
                    print("Warning!", token, word_list)
            word_ids.append(word_idx)
            word_idx += 1
    return word_ids