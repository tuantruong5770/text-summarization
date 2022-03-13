from summ_extractor_helper import load_model
from word2vec_helper import Word2VecHelper
from data_processing import ProcessedDataset
from model.SummaryExtractor import SummaryExtractor
from model.ExtractorWrapper import ExtractorWrapper
from model.Trainer import Trainer
from evaluate import EvaluateModel, generate_evaluation_data

import tensorflow as tf
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_built_with_gpu():
    print("Tensorflow built with CUDA:", tf.test.is_built_with_cuda())
    print(tf.config.list_physical_devices('GPU'))


def test_data_point(model, word2v, dataset, data_index):
    pass
    # text = dataset.get_text(data_index)
    # summ = dataset.get_summary(data_index)
    # label = dataset.get_label(data_index)
    # # sent_id = Word2VecHelper.text_to_id(text, word2v)
    # # preds = model(sent_id, summary_length=len(label)).to(device)
    # preds = test_summ_extractor(model, word2v, text, label)
    # compare_reference_label_generated(preds, text, summ, label)


def test_cnn_dailymail_datapoint(data_index):
    dataset = ProcessedDataset('cnn_dailymail')
    word2v = Word2VecHelper.load_model('cnn_dailymail_128')
    model = load_model('10-03-2022_13-47-19_cnn_dailymail', word2v)
    model._sentence_encoder.training = False
    test_data_point(model, word2v, dataset, data_index)


if __name__ == '__main__':
    # dataset = ProcessedDataset('cnn_dailymail')
    # word2v = Word2VecHelper.load_model('cnn_dailymail_128_min5')
    # model = SummaryExtractor().to(device)
    # word_to_index = Word2VecHelper.get_word_to_index(word2v, top_k=30000)
    # model.set_embedding(Word2VecHelper.get_embedding(word2v, word_to_index))
    # model_wrapper = ExtractorWrapper(model, word_to_index)
    # trainer = Trainer(model_wrapper, dataset)
    # trainer.train_epoch()
    #
    # kwargs = {
    #     'vector_size': 128,
    #     'min_count': 5,
    #     'workers': 16,
    #     'sg': 1
    # }
    # dataset = ProcessedDataset('billsum')
    # word2v = Word2VecHelper.train_model(dataset, **kwargs)
    # Word2VecHelper.save_model('billsum_128_min5', word2v)
    # model = SummaryExtractor().to(device)
    # word_to_index = Word2VecHelper.get_word_to_index(word2v, top_k=30000)
    # model.set_embedding(Word2VecHelper.get_embedding(word2v, word_to_index))
    # model_wrapper = ExtractorWrapper(model, word_to_index)
    # trainer = Trainer(model_wrapper, dataset)
    # trainer.train_epoch()

    # num_epochs = 5
    # batch_size = 32
    # learning_rate = 0.001
    # num_training = len(dataset)
    # teacher_forcing_prob = 0.5
    # print_per = 500
    #
    # train_summ_extractor(model, dataset, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
    #                      num_training=20000, teacher_forcing_prob=teacher_forcing_prob, print_per=print_per)
    #
    # dataset = ProcessedDataset('cnn_dailymail')
    # word2v = Word2VecHelper.load_model('cnn_dailymail')
    # model = load_model('05-03-2022_09-00-12_cnn_dailymail', word2v)
    # test_data_point(model, word2v, dataset, 251277)

    # test_cnn_dailymail_datapoint(0)

    model_name = '12-03-2022_09-04-17_cnn_dailymail'
    word2v_model_name = 'cnn_dailymail_128_min5'
    dataset = ProcessedDataset('cnn_dailymail', split='test', load_raw=True)
    word2v = Word2VecHelper.load_model(word2v_model_name)
    model = load_model(model_name)
    model_wrapper = ExtractorWrapper(model, Word2VecHelper.get_word_to_index(word2v, top_k=30000))
    evaluator = EvaluateModel(model_name=model_name)
    generate_evaluation_data(model_wrapper, model_name, dataset)
    print(evaluator.evaluate())
    # data_index = 0
    # text = dataset.get_text(data_index)
    # summ = dataset.get_summary(data_index)
    # label = dataset.get_label(data_index)
    # model_wrapper.comprehensive_test(text, summ, label, data_index, print_text=True)

    # for i in range(10):
    #     data_index = 69 + i
    #     text = dataset.get_text(data_index)
    #     summ = dataset.get_summary(data_index)
    #     label = dataset.get_label(data_index)
    #     model_wrapper.comprehensive_test(text, summ, label, data_index, print_text=False)
    #
    # for i in range(10):
    #     data_index = 100000 + i
    #     text = dataset.get_text(data_index)
    #     summ = dataset.get_summary(data_index)
    #     label = dataset.get_label(data_index)
    #     model_wrapper.comprehensive_test(text, summ, label, data_index, print_text=False)
