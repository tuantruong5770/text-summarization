from cytoolz import curry
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from ignite.metrics import RougeL
from utils import timer

import tensorflow_datasets as tfds
import nltk
import torch
import json
import os


# Ignore misc. tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ProcessData:
    lemmatizer = WordNetLemmatizer()
    stop_word = stopwords.words('english')


    @staticmethod
    def process_tokenize(text):
        """
        Split text to sentence and words. Lower case all words

        :param text: str
        :return: List(List(str))
        """
        return [nltk.word_tokenize(sent.lower()) for sent in nltk.sent_tokenize(text)]


    @staticmethod
    def process_lemmatize(processed_text):
        """
        Lemmatize processed text

        :param processed_text: List(List(str))
        :return List(List(str))
        """
        processed_lemmatize_text = []
        for i, sent in enumerate(processed_text):
            nltk_tagged = nltk.pos_tag(sent)
            wordnet_tagged = map(lambda x: (x[0], ProcessData.nltk_pos_tagger(x[1])), nltk_tagged)
            processed_sent = []

            for j, (word, tag) in enumerate(wordnet_tagged):
                processed_sent.append(ProcessData.lemmatizer.lemmatize(word, tag) if tag else word)

            processed_lemmatize_text.append(processed_sent)

        return processed_lemmatize_text


    @staticmethod
    def process_remove_stop_word(processed_text):
        """
        Remove stop word in processed text. Might omit sentences
        Mapping is map of processed_remove_stop word_text sentence index to original processed_text sentence index

        :param processed_text: List(List(str))
        :return: List(List(str)), Dict(int:int)
        """
        mapping = dict()
        mapping_index = 0
        processed_remove_stop_word_text = []
        for i, sent in enumerate(processed_text):
            processed_sent = []

            for word in sent:
                if word not in ProcessData.stop_word and word.isalnum():
                    processed_sent.append(word)

            if processed_sent:
                processed_remove_stop_word_text.append(processed_sent)
                mapping[mapping_index] = i
                mapping_index += 1

        return processed_remove_stop_word_text, mapping


    @staticmethod
    def nltk_pos_tagger(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None


class ProcessedDataset:
    supported_datasets = ['billsum', 'cnn_dailymail']
    key_map = {'cnn_dailymail': ('article', 'highlights'), 'billsum': ('text', 'summary')}


    def __init__(self, dataset='cnn_dailymail', data_dir='./data', processed_data_dir='./data/processed',
                 raw_data_dir='./data/raw', split='train', load_processed=True, load_raw=False):
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

        self._data_dir = data_dir
        self._process_data_dir = processed_data_dir
        self._raw_data_dir = raw_data_dir

        self._file_dir = f'{self._process_data_dir}/{self._name}_{self._split}.json'
        self._raw_dir = f'{self._raw_data_dir}/{self._name}_{self._split}.json'

        self._text_key = ProcessedDataset.key_map[self._name][0]
        self._summ_key = ProcessedDataset.key_map[self._name][1]

        self.dataset = None
        self.raw_dataset = None
        # self.dataset and self.raw_dataset is initialized in self._load_data()
        self._load_data(load_processed, load_raw)


    def _load_data(self, load_processed, load_raw):
        print(f'Loading {self._name} dataset...')
        try:
            if load_processed:
                with open(self._file_dir) as f:
                    print(f'Found preprocessed {self._name} {self._split} dataset with specified options. Loading...')
                    self.dataset = json.load(f)
                    print(f'{self._file_dir} dataset loaded successfully. Dataset containing {len(self.dataset)} entries.')
            if load_raw:
                with open(self._raw_dir) as f:
                    print(f'Found raw {self._name} {self._split} dataset with specified options. Loading...')
                    self.raw_dataset = json.load(f)
                    print(f'{self._raw_dir} dataset loaded successfully. Dataset containing {len(self.raw_dataset)} entries.')
        except FileNotFoundError:
            print(f'Processing dataset and saving at "{self._file_dir}" ...')
            self.dataset, self.raw_dataset = self._process_data()
            print(f'{self._file_dir} dataset loaded successfully. Dataset containing {len(self.dataset)} entries.')
            print(f'{self._raw_dir} dataset loaded successfully. Dataset containing {len(self.raw_dataset)} entries.')


    @timer
    def _process_data(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = tfds.load(self._name, data_dir=self._data_dir, split=self._split)
        processed_data = []
        raw_data = []
        rouge_calc = RougeL(device=device)
        for i, data in enumerate(dataset):
            # Convert from tensor string to python string
            text = data[self._text_key].numpy().decode()
            summ = data[self._summ_key].numpy().decode()
            # Split text into sentences and split each sentence in to words
            text_sent = ProcessData.process_tokenize(text)
            summ_sent = ProcessData.process_tokenize(summ)

            # Skip data point if summary has equal or more sentences than text
            if len(summ_sent) >= len(text_sent) or len(summ_sent) == 0 or len(text_sent) == 0:
                continue

            # Lemmatize texts
            text_sent_lemmatize = ProcessData.process_lemmatize(text_sent)
            summ_sent_lemmatize = ProcessData.process_lemmatize(summ_sent)

            # Remove stop word
            text_sent_rem_stop_word, text_mapping = ProcessData.process_remove_stop_word(text_sent_lemmatize)
            summ_sent_rem_stop_word, summ_mapping = ProcessData.process_remove_stop_word(summ_sent_lemmatize)

            # Skip data point if processed summary has equal or more sentences than text
            if len(summ_sent_rem_stop_word) >= len(text_sent_rem_stop_word) or len(summ_sent_rem_stop_word) == 0 or len(
                    text_sent_rem_stop_word) == 0:
                continue

            label, score = self._generate_label_and_score(text_sent_rem_stop_word, summ_sent_rem_stop_word,
                                                          text_mapping, rouge_calc)

            processed_data.append(
                {self._text_key: text_sent, self._summ_key: summ_sent, 'label': label, 'score': score})

            raw_data.append(
                {self._text_key: text, self._summ_key: summ, 'label': label, 'score': score})

        with open(self._file_dir, 'w') as outfile:
            json.dump(processed_data, outfile)

        with open(self._raw_dir, 'w') as outfile:
            json.dump(raw_data, outfile)

        return processed_data, raw_data


    def _generate_label_and_score(self, processed_text_sent, processed_summ_sent, index_mapping, rouge_calc):
        label = []
        score = []
        available_index = list(range(len(processed_text_sent)))
        for sent in processed_summ_sent:
            rouge = list(map(self.evaluate(references=sent, rouge_calc=rouge_calc), processed_text_sent))
            text_index = max(available_index, key=lambda i: rouge[i])
            label.append(index_mapping[text_index])
            score.append(rouge[text_index])
            available_index.remove(text_index)
        return label, score


    @staticmethod
    @curry
    def evaluate(candidate, references, rouge_calc):
        rouge_calc.reset()
        rouge_calc.update(([candidate], [[references]]))
        return rouge_calc.compute()['Rouge-L-R']


    def get_dataset_name(self):
        return self._name


    def get_dataset(self):
        return self.dataset


    def get_raw_dataset(self):
        return self.raw_dataset


    def get_entry(self, i):
        return self.dataset[i]


    def get_raw_entry(self, i):
        return self.raw_dataset[i]


    def get_text(self, i):
        return self.dataset[i][self._text_key]


    def get_raw_text(self, i):
        return self.raw_dataset[i][self._text_key]


    def get_summary(self, i):
        return self.dataset[i][self._summ_key]


    def get_raw_summary(self, i):
        return self.raw_dataset[i][self._summ_key]


    def get_label(self, i):
        return self.dataset[i]['label']


    def get_score(self, i):
        return self.dataset[i]['score']


    def get_num_sentences(self, i):
        return len(self.dataset[i][self._text_key])


    def get_num_words(self, i, j):
        return len(self.dataset[i][self._text_key][j])


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, i):
        return self.get_text(i), self.get_summary(i), self.get_label(i), self.get_score(i)


if __name__ == '__main__':
    # Process and save the 2 data into default directory './data'
    cnn = ProcessedDataset('cnn_dailymail', split='test')
    # bill = ProcessedDataset(dataset='billsum')



