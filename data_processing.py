import tensorflow_datasets as tfds
import nltk
import torch
from ignite.metrics import RougeL
from cytoolz import curry
import json
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ProcessData:
    @staticmethod
    def process(text):
        """
        Split text to sentence and words
        :param text: Unprocessed str
        :return: List(List())
        """
        return [sent.strip('.').split() for sent in nltk.sent_tokenize(text)]


class ProcessedDataset:
    supported_datasets = ['billsum', 'cnn_dailymail']
    key_map = {'cnn_dailymail': ('article', 'highlights'), 'billsum': ('text', 'summary')}


    def __init__(self, dataset='cnn_dailymail', data_dir='./data', processed_data_dir='./data/processed', split='train'):
        """
        Load the dataset and save into self.dataset

        The processed data has the following semantic:
        [
            {
                'text': [['sentence', 'one'], ['sentence','two'], ...],
                'summary': [['sentence', 'one'], ['sentence','two'], ...]},
                'label': [0, 3, 1, ...]
                'score': [0.0512, 0.0231, 0.0125, ...]
            },
            ...
        ]

        :param dataset: Name of dataset, if not supported default to 'cnn_dailymail'
        :param data_dir: Directory of dataset, download and save in this directory if not yet downloaded.
        The data here is of type tensor and not yet processed
        :param processed_data_dir: Directory of processed data. Save in this directory if not yet processed
        :param split: Split of the data (i.e. train, validation, test)
        """
        self._name = dataset
        if dataset not in ProcessedDataset.supported_datasets:
            self._name = 'cnn_dailymail'
            print(f'Dataset {dataset} is not supported. Defaulted to "cnn_dailymail" dataset')

        self._split = split
        if split not in ['train', 'test', 'validation']:
            self._split = 'train'

        self._data_dir = data_dir
        self._process_data_dir = processed_data_dir
        self._text_key = ProcessedDataset.key_map[self._name][0]
        self._summ_key = ProcessedDataset.key_map[self._name][1]
        self.dataset = []

        self._load_data()


    def _load_data(self):
        print(f'Loading {self._name} dataset...')
        try:
            with open(self._process_data_dir + f'/{self._name}_{self._split}.json') as f:
                print(f'Found available preprocessed {self._name} {self._split} dataset. Loading...')
                self.dataset = json.load(f)
                print(f'{self._name} {self._split} dataset loaded successfully. Dataset containing {len(self.dataset)} entries.')
        except FileNotFoundError:
            print(f'Processing dataset and saving at "{self._process_data_dir}/{self._name}_{self._split}.json"...')
            self.dataset = self._process_data()
            print(f'{self._name} {self._split} dataset loaded successfully. Dataset containing {len(self.dataset)} entries.')


    def _process_data(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = tfds.load(self._name, data_dir=self._data_dir, split=self._split)
        processed_data = []
        for data in dataset:
            # Convert from tensor string to python string
            text = data[self._text_key].numpy().decode()
            summ = data[self._summ_key].numpy().decode()
            # Split text into sentences and split each sentence in to words
            text_sent = ProcessData.process(text)
            summ_sent = ProcessData.process(summ)
            label = []
            score = []

            # Skip data point if summary has equal or more sentences than text
            if len(summ_sent) >= len(text_sent):
                continue

            # Curry to enable pre-input for the parameters to use map()
            @curry
            def evaluate(references, candidate, metric):
                metric.update(([candidate], [[references]]))
                return metric.compute()['Rouge-L-R']

            m = RougeL(device=device)
            available_index = list(range(len(text_sent)))
            for sent in summ_sent:
                rouge = list(map(evaluate(candidate=sent, metric=m), text_sent))
                text_index = max(available_index, key=lambda i: rouge[i])
                label.append(text_index)
                score.append(rouge[text_index])
                available_index.remove(text_index)

            processed_data.append(
                {self._text_key: text_sent, self._summ_key: summ_sent, 'label': label, 'score': score})

        # Save data and return
        with open(self._process_data_dir + f'/{self._name}_{self._split}.json', 'w') as outfile:
            json.dump(processed_data, outfile)
        return processed_data


    def get_dataset(self):
        return self.dataset


    def get_num_sentences(self, i):
        return len(self.dataset[i][self._text_key])


    def get_num_words(self, i, j):
        return len(self.dataset[i][self._text_key][j])


if __name__ == '__main__':
    # Process and save the 2 data into default directory './data'
    ProcessedDataset(dataset='cnn_dailymail')
    ProcessedDataset(dataset='billsum')














