from summ_extractor_helper import load_model, train_summ_extractor, test_summ_extractor, \
    compare_reference_label_generated
from word2vec_helper import Word2VecHelper
from data_processing import ProcessedDataset
from model.SummaryExtractor import SummaryExtractor, SummaryExtractorHyperParameters

import tensorflow as tf
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_built_with_gpu():
    print("Tensorflow built with CUDA:", tf.test.is_built_with_cuda())
    print(tf.config.list_physical_devices('GPU'))


def test_data_point(model, word2v, dataset, data_index):
    text = dataset.get_text(data_index)
    summ = dataset.get_summary(data_index)
    label = dataset.get_label(data_index)
    # sent_id = Word2VecHelper.text_to_id(text, word2v)
    # preds = model(sent_id, summary_length=len(label)).to(device)
    preds = test_summ_extractor(model, word2v, text, label)
    compare_reference_label_generated(preds, text, summ, label)


def test_cnn_dailymail_datapoint(data_index):
    dataset = ProcessedDataset('cnn_dailymail')
    word2v = Word2VecHelper.load_model('cnn_dailymail_128')
    model = load_model('07-03-2022_12-23-57_cnn_dailymail', word2v)
    model._sentence_encoder.training = False
    test_data_point(model, word2v, dataset, data_index)


if __name__ == '__main__':
    # dataset = ProcessedDataset('cnn_dailymail')
    # word2v = Word2VecHelper.load_model('cnn_dailymail_128')
    # model = SummaryExtractor(SummaryExtractorHyperParameters(word2v)).to(device)
    #
    # num_epochs = 5
    # batch_size = 32
    # learning_rate = 0.001
    # num_training = len(dataset)
    # teacher_forcing_prob = 0.5
    # print_per = 500
    #
    # train_summ_extractor(model, dataset, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
    #                      num_training=num_training, teacher_forcing_prob=teacher_forcing_prob, print_per=print_per)

    # dataset = ProcessedDataset('cnn_dailymail')
    # word2v = Word2VecHelper.load_model('cnn_dailymail')
    # model = load_model('05-03-2022_09-00-12_cnn_dailymail', word2v)
    # test_data_point(model, word2v, dataset, 251277)

    test_cnn_dailymail_datapoint(0)
