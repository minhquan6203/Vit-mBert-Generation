
from typing import List, Dict, Optional
from datasets import load_dataset
import os

def create_vocab(config: Dict):
    data_folder=config['data']['data_folder']
    train_set=config["data"]["train_dataset"]
    val_set=config["data"]["val_dataset"]
    test_set=config["data"]["test_dataset"]
    dataset = load_dataset(
        "json", 
        data_files={
            "train": os.path.join(data_folder, train_set),
            "val": os.path.join(data_folder, val_set),
            "test": os.path.join(data_folder, test_set)
        },field='annotations'
    )

    word_counts = {}

    for data_file in dataset.values():
        for ques in data_file['question']:
            for q in ques.split():
                if q not in word_counts:
                    word_counts[q] = 1
                else:
                    word_counts[q] += 1
        for ans in data_file['answers']:
            for a in ans[0].split():
                if a not in word_counts:
                    word_counts[a] = 1
                else:
                    word_counts[a] += 1

    sorted_word_counts = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))
    vocab = list(sorted_word_counts.keys())
    return vocab, sorted_word_counts


import heapq

def beam_search(word, vocab, k):
    heap = []    
    for token in vocab:
        if word in token.lower():
            score = -abs(len(word) - len(token))  # Scoring based on the length difference
            heapq.heappush(heap, (score, token))
            
            if len(heap) > k:
                heapq.heappop(heap)
    
    results = sorted(heap, key=lambda x: -x[0])  # Sort the results by score in descending order
    return [result[1] for result in results]


