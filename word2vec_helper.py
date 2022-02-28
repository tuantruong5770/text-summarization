from gensim.models import Word2Vec
from data_processing import ProcessedDataset

import torch

PADDING_INDEX = 0


class Word2VecHelper:
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
    def text_to_id(processed_text, model, pad, cuda=True):
        """
        Convert processed text into processed word index according to given Word2Vec model
        Use for nn.Embedding indexing purposes

        :param processed_text: List(List(str)) of processed text
        :param model: trained Word2Vec model
        :param pad: padding index
        :param cuda: True if built with cuda
        :return: IntTensor(IntTensor())
        """
        word_to_index = model.wv.key_to_index
        inputs = [torch.IntTensor([word_to_index[w] for w in sentence]) for sentence in processed_text]
        tensor_type = torch.cuda.IntTensor if cuda else torch.IntTensor
        num_sentence = len(inputs)
        max_len = max(len(ids) for ids in inputs)
        tensor_shape = (num_sentence, max_len)
        tensor = tensor_type(*tensor_shape)
        tensor.fill_(pad)
        for i, ids in enumerate(inputs):
            tensor[i, :len(ids)] = tensor_type(ids)
        return tensor


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

