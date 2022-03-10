from gensim.models import Word2Vec
from data_processing import ProcessedDataset
from utils import timer

import torch
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    def get_weights(model):
        """
        Get weight matrix (type np.ndarray) of Word2Vec. Return as a torch.FloatTensor
        :param model: trained Word2Vec model
        :return: FloatTensor([vocab_size, emb_size])
        """
        return torch.FloatTensor(model.wv.vectors)


    @staticmethod
    def text_to_id(processed_text, word2v, pad=0):
        """
        From Chen-Bansal codebase

        Convert processed text into processed word index according to given Word2Vec model
        Use for nn.Embedding indexing purposes

        :param processed_text: List(List(str)) of processed text
        :param word2v: trained Word2Vec model
        :param pad: padding index
        :return: LongTensor(LongTensor(Long))
        """
        word_to_index = word2v.wv.key_to_index
        inputs = [[word_to_index[w] for w in sentence] for sentence in processed_text]
        num_sentence = len(inputs)
        max_len = max(len(ids) for ids in inputs)
        tensor = torch.LongTensor(num_sentence, max_len).to(device)
        tensor.fill_(pad)
        for i, ids in enumerate(inputs):
            tensor[i, :len(ids)] = torch.LongTensor(ids).to(device)
        return tensor


    @staticmethod
    def text_to_vector(processed_text, word2v, pad=0):
        """
        Convert processed text into word vectors according to given Word2Vec model
        Use for predicting / testing purposes

        :param processed_text: List(List(str)) of processed text
        :param word2v: trained Word2Vec model
        :return: LongTensor(LongTensor(LongTensor(Long)))
        """
        index_to_vector = word2v.wv.vectors
        id_tensor = Word2VecHelper.text_to_id(processed_text, word2v, pad)
        return torch.FloatTensor(np.array([[index_to_vector[id] for id in word_ids] for word_ids in id_tensor])).to(device)


if __name__ == "__main__":
    # Parameters for init
    emb_dim = 10
    n_hidden = 20
    wd = 10
    sg = 0

    # Example use
    sentences = Word2VecHelper.process_dataset(ProcessedDataset(dataset='cnn_dailymail'))
    model = Word2Vec(sentences=sentences, min_count=1, vector_size=emb_dim, window=wd, sg=sg)
    Word2VecHelper.save_model('cnn_dailymail', model)
    model = Word2VecHelper.load_model('cnn_dailymail')
