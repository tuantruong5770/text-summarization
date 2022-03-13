from gensim.models import Word2Vec
from data_processing import ProcessedDataset
from utils import timer
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD = 0
UNK = 1


class Word2VecHelper:
    @staticmethod
    @timer
    def train_model(dataset, **kwargs):
        sentences = dataset.get_text(0).copy()
        for i in range(1, len(dataset)):
            sentences.extend(dataset.get_text(i))

        word2v = Word2Vec(sentences=sentences, **kwargs)
        return word2v


    @staticmethod
    def save_model(model_name, model):
        """
        Save the pretrained model in ./pretrained
        """
        model.save(f'./pretrained/word2vec_{model_name}.model')


    @staticmethod
    def load_model(model_name):
        """
        Load a pretrained model in ./pretrained
        """
        try:
            return Word2Vec.load(f'./pretrained/word2vec_{model_name}.model')
        except FileNotFoundError:
            print(f'Pretrained model word2vec_{model_name}.model not found at ./pretrained')


    @staticmethod
    def process_dataset(dataset: ProcessedDataset):
        """
        Process the dataset to extract all the list of sentences into a one list
        Used as a "sentences" param for Word2Vec
        """
        processed = []
        for i in range(len(dataset)):
            processed.extend(dataset.get_text(i))
        return processed


    @staticmethod
    def get_embedding(word2v, word_to_index):
        """
        Get weight matrix (type np.ndarray) of embeddings. Return as a torch.FloatTensor
        :param word2v: trained Word2Vec model
        :param word_to_index: dictionary of word to index
        :return: FloatTensor([vocab_size, emb_dim])
        """
        vocab_size = len(word_to_index)
        emb_dim = word2v.vector_size
        word_to_vec = word2v.wv
        embedding = nn.Embedding(vocab_size, emb_dim).weight
        with torch.no_grad():
            for word, index in word_to_index.items():
                if index != PAD and index != UNK:
                    embedding[index] = torch.Tensor(word_to_vec[word])
        return embedding


    @staticmethod
    def get_word_to_index(word2v, top_k=0):
        """
        From Chen-Bansal codebase

        Get the top k most common word in the dictionary with padding and unknown

        :param word2v: pretrained word2v
        :param top_k: k most common word
        :return: defaultdict(str:int) with default of 'unk'
        """
        vocab_size = top_k
        if vocab_size <= 0:
            vocab_size = len(word2v.wv)
        word_to_index = dict()
        word_to_index['<pad>'] = PAD
        word_to_index['<unk>'] = UNK
        for i, word in enumerate(word2v.wv.index_to_key[:vocab_size], 2):
            word_to_index[word] = i
        return defaultdict(lambda: UNK, word_to_index)


    @staticmethod
    def text_to_id(processed_text, word_to_index):
        """
        From Chen-Bansal codebase

        Convert processed text into processed word index according to given Word2Vec model
        Use for nn.Embedding indexing purposes

        :param processed_text: List(List(str)) of processed text
        :param word_to_index: dictionary of word to index
        :param pad: padding index
        :return: LongTensor(LongTensor(Long))
        """
        inputs = [[word_to_index[w] for w in sentence] for sentence in processed_text]
        num_sentence = len(inputs)
        max_len = max(len(ids) for ids in inputs)
        tensor = torch.LongTensor(num_sentence, max_len).to(device)
        tensor.fill_(PAD)
        for i, ids in enumerate(inputs):
            tensor[i, :len(ids)] = torch.LongTensor(ids).to(device)
        return tensor


if __name__ == "__main__":
    # Parameters for init
    kwargs = {
        'vector_size': 128,
        'min_count': 5,
        'workers': 16,
        'sg': 1
        }

    # Train Word2Vec
    dataset = ProcessedDataset('cnn_dailymail')
    sentences = Word2VecHelper.process_dataset(dataset)
    model = Word2VecHelper.train_model(dataset, **kwargs)
    Word2VecHelper.save_model('cnn_dailymail_128_min5', model)
    # model = Word2VecHelper.load_model('cnn_dailymail')
