import pandas as pd
from datasets import load_dataset, DatasetDict, concatenate_datasets

# Unified labels across CoNLL-2003 and MasakhaNER (excluding MISC and DATE)
LABEL_LIST = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

def map_labels(example):
    """
    CoNLL-2003 and MasakhaNER both use standard tags for PER, ORG, LOC:
    0: O, 1: B-PER, 2: I-PER, 3: B-ORG, 4: I-ORG, 5: B-LOC, 6: I-LOC
    CoNLL-2003 has 7: B-MISC, 8: I-MISC
    MasakhaNER has 7: B-DATE, 8: I-DATE
    This function maps 7 and 8 to 0 (O) so we strictly extract PER, ORG, LOC.
    """
    new_tags = []
    for tag in example["ner_tags"]:
        if tag in [7, 8]:
            new_tags.append(0)
        else:
            new_tags.append(tag)
    example["ner_tags"] = new_tags
    return example

def load_and_prepare_datasets():
    print("Loading CoNLL-2003 (English)...")
    conll = load_dataset("conll2003")
    
    print("Loading MasakhaNER (Amharic)...")
    # For this assignment, we use Amharic as a representative African language.
    # To include all 10 languages, one would iterate over MasakhaNER configs.
    masakha = load_dataset("masakhane/masakhaner", "amh")
    
    # Map labels to drop MISC and DATE
    conll = conll.map(map_labels)
    masakha = masakha.map(map_labels)
    
    # CoNLL has extra columns we don't need
    conll = conll.remove_columns(["pos_tags", "chunk_tags"])
    
    combined_dataset = DatasetDict({
        "train": concatenate_datasets([conll["train"], masakha["train"]]),
        "validation": concatenate_datasets([conll["validation"], masakha["validation"]]),
        "test": concatenate_datasets([conll["test"], masakha["test"]]),
    })
    
    # Shuffle train set
    combined_dataset["train"] = combined_dataset["train"].shuffle(seed=42)
    return combined_dataset

def get_tokenize_and_align_labels_fn(tokenizer):
    """
    Returns a function that tokenizes text and aligns NER labels with the subword tokens.
    """
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=128
        )
        
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens get -100 so they are ignored in loss computation
                if word_idx is None:
                    label_ids.append(-100)
                # First subword of a word gets the original label
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # Subsequent subwords get -100 (or could be mapped to I- tags)
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
            
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    return tokenize_and_align_labels
