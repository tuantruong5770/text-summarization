from word2vec_helper import Word2VecHelper
from data_processing import ProcessedDataset

from model.FFSumarryExtractor import FFSummaryExtractor
from model.FFExtractorWrapper import FFExtractorWrapper

from model.SummaryExtractor import SummaryExtractor
from model.ExtractorWrapper import ExtractorWrapper

from model.SummaryExtractorNoCoverage import SummaryExtractorNoCoverage
from model.ExtractorNoCoverageWrapper import ExtractorNoCoverageWrapper

from model.Trainer import Trainer
from evaluate import EvaluateModel, generate_evaluation_data

import argparse
import tensorflow as tf
import torch
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_built_with_gpu():
    print("Tensorflow built with CUDA:", tf.test.is_built_with_cuda())
    print(tf.config.list_physical_devices('GPU'))


def train_extractor_cnn_epoch(num_training=0, num_val=0, teacher_forcing=0.0):
    train_dataset = ProcessedDataset('cnn_dailymail')
    val_dataset = ProcessedDataset('cnn_dailymail', split='validation')
    word2v = Word2VecHelper.load_model('cnn_dailymail_128_min5')
    model = SummaryExtractor().to(device)
    word_to_index = Word2VecHelper.get_word_to_index(word2v, top_k=30000)
    model.set_embedding(Word2VecHelper.get_embedding(word2v, word_to_index))
    model_wrapper = ExtractorWrapper(model, word_to_index)
    trainer = Trainer(model_wrapper, train_dataset, val_dataset, num_training=num_training, num_val=num_val)
    trainer.teacher_forcing_prob = teacher_forcing
    trainer.train_epoch()


def train_extractor_billsum_epoch(num_training=0, num_val=0, teacher_forcing=0.0):
    train_dataset = ProcessedDataset('billsum')
    val_dataset = ProcessedDataset('billsum', split='ca_test')
    word2v = Word2VecHelper.load_model('billsum_128_min5')
    model = SummaryExtractor().to(device)
    word_to_index = Word2VecHelper.get_word_to_index(word2v, top_k=30000)
    model.set_embedding(Word2VecHelper.get_embedding(word2v, word_to_index))
    model_wrapper = ExtractorWrapper(model, word_to_index)
    trainer = Trainer(model_wrapper, train_dataset, val_dataset, num_training=num_training, num_val=num_val)
    trainer.teacher_forcing_prob = teacher_forcing
    trainer.train_epoch()


def train_extractor_no_coverage_cnn_epoch(num_training=0, num_val=0, teacher_forcing=0.0):
    train_dataset = ProcessedDataset('cnn_dailymail')
    val_dataset = ProcessedDataset('cnn_dailymail', split='validation')
    word2v = Word2VecHelper.load_model('cnn_dailymail_128_min5')
    model = SummaryExtractorNoCoverage().to(device)
    word_to_index = Word2VecHelper.get_word_to_index(word2v, top_k=30000)
    model.set_embedding(Word2VecHelper.get_embedding(word2v, word_to_index))
    model_wrapper = ExtractorNoCoverageWrapper(model, word_to_index)
    trainer = Trainer(model_wrapper, train_dataset, val_dataset, num_training=num_training, num_val=num_val)
    trainer.teacher_forcing_prob = teacher_forcing
    trainer.train_epoch()


def train_ff_extractor_cnn_epoch(num_training=0, num_val=0, teacher_forcing=0.0):
    train_dataset = ProcessedDataset('cnn_dailymail')
    val_dataset = ProcessedDataset('cnn_dailymail', split='validation')
    word2v = Word2VecHelper.load_model('cnn_dailymail_128_min5')
    model = FFSummaryExtractor().to(device)
    word_to_index = Word2VecHelper.get_word_to_index(word2v, top_k=30000)
    model.set_embedding(Word2VecHelper.get_embedding(word2v, word_to_index))
    model_wrapper = FFExtractorWrapper(model, word_to_index)
    trainer = Trainer(model_wrapper, train_dataset, val_dataset, num_training=num_training, num_val=num_val)
    trainer.teacher_forcing_prob = teacher_forcing
    trainer.train_epoch()


def train_extractor_cnn_early_stop(num_training=0, num_val=0, teacher_forcing=0.0):
    train_dataset = ProcessedDataset('cnn_dailymail')
    val_dataset = ProcessedDataset('cnn_dailymail', split='validation')
    word2v = Word2VecHelper.load_model('cnn_dailymail_128_min5')
    model = SummaryExtractor().to(device)
    word_to_index = Word2VecHelper.get_word_to_index(word2v, top_k=30000)
    model.set_embedding(Word2VecHelper.get_embedding(word2v, word_to_index))
    model_wrapper = ExtractorWrapper(model, word_to_index)
    trainer = Trainer(model_wrapper, train_dataset, val_dataset, num_training=num_training, num_val=num_val)
    trainer.teacher_forcing_prob = teacher_forcing
    trainer.check_point_frequency = 3000
    trainer.train_converge_early_stop()


def train_extractor_billsum_early_stop(num_training=0, num_val=0, teacher_forcing=0.0):
    train_dataset = ProcessedDataset('billsum')
    val_dataset = ProcessedDataset('billsum', split='ca_test')
    word2v = Word2VecHelper.load_model('billsum_128_min5')
    model = SummaryExtractor().to(device)
    word_to_index = Word2VecHelper.get_word_to_index(word2v, top_k=30000)
    model.set_embedding(Word2VecHelper.get_embedding(word2v, word_to_index))
    model_wrapper = ExtractorWrapper(model, word_to_index)
    trainer = Trainer(model_wrapper, train_dataset, val_dataset, num_training=num_training, num_val=num_val)
    trainer.teacher_forcing_prob = teacher_forcing
    trainer.check_point_frequency = 3000
    trainer.train_converge_early_stop()


def train_extractor_no_coverage_cnn_early_stop(num_training=0, num_val=0, teacher_forcing=0.0):
    train_dataset = ProcessedDataset('cnn_dailymail')
    val_dataset = ProcessedDataset('cnn_dailymail', split='validation')
    word2v = Word2VecHelper.load_model('cnn_dailymail_128_min5')
    model = SummaryExtractorNoCoverage().to(device)
    word_to_index = Word2VecHelper.get_word_to_index(word2v, top_k=30000)
    model.set_embedding(Word2VecHelper.get_embedding(word2v, word_to_index))
    model_wrapper = ExtractorNoCoverageWrapper(model, word_to_index)
    trainer = Trainer(model_wrapper, train_dataset, val_dataset, num_training=num_training, num_val=num_val)
    trainer.teacher_forcing_prob = teacher_forcing
    trainer.check_point_frequency = 3000
    trainer.train_converge_early_stop()


def train_ff_extractor_cnn_early_stop(num_training=0, num_val=0, teacher_forcing=0.0):
    train_dataset = ProcessedDataset('cnn_dailymail')
    val_dataset = ProcessedDataset('cnn_dailymail', split='validation')
    word2v = Word2VecHelper.load_model('cnn_dailymail_128_min5')
    model = FFSummaryExtractor().to(device)
    word_to_index = Word2VecHelper.get_word_to_index(word2v, top_k=30000)
    model.set_embedding(Word2VecHelper.get_embedding(word2v, word_to_index))
    model_wrapper = FFExtractorWrapper(model, word_to_index)
    trainer = Trainer(model_wrapper, train_dataset, val_dataset, num_training=num_training, num_val=num_val)
    trainer.teacher_forcing_prob = teacher_forcing
    trainer.check_point_frequency = 3000
    trainer.train_converge_early_stop()


def main_train(args):
    train_type = args.train
    train_opt = args.train_opt
    num_train = args.amt
    num_val = args.val
    teacher_forcing = args.teacher_forcing
    if train_type == 0:
        if train_opt == 0:
            train_extractor_cnn_epoch(num_train, num_val, teacher_forcing)
        else:
            train_extractor_cnn_early_stop(num_train, num_val, teacher_forcing)
    elif train_type == 1:
        if train_opt == 0:
            train_extractor_billsum_epoch(num_train, num_val, teacher_forcing)
        else:
            train_extractor_billsum_early_stop(num_train, num_val, teacher_forcing)
    elif train_type == 2:
        if train_opt == 0:
            train_extractor_no_coverage_cnn_epoch(num_train, num_val, teacher_forcing)
        else:
            train_extractor_no_coverage_cnn_early_stop(num_train, num_val, teacher_forcing)
    elif train_type == 3:
        if train_opt == 0:
            train_ff_extractor_cnn_epoch(num_train, num_val, teacher_forcing)
        else:
            train_ff_extractor_cnn_early_stop(num_train, num_val, teacher_forcing)


def main_eval(args):
    model_name = args.eval
    model_type = args.model_type
    test_data = args.test_data

    dataset_name = 'cnn_dailymail'
    if test_data == 1:
        dataset_name = 'billsum'
    test_dataset = ProcessedDataset(dataset_name, split='test', load_raw=True)

    word2v = Word2VecHelper.load_model(f'{dataset_name}_128_min5')
    word_to_index = Word2VecHelper.get_word_to_index(word2v, top_k=30000)
    model_wrapper = None
    if model_type == 0:
        model = SummaryExtractor().to(device)
        checkpoint = torch.load(f'./pretrained/{model_name}.pt')
        model.load_state_dict(checkpoint)
        model_wrapper = ExtractorWrapper(model, word_to_index)
    elif model_type == 1:
        model = SummaryExtractorNoCoverage().to(device)
        checkpoint = torch.load(f'./pretrained/{model_name}.pt')
        model.load_state_dict(checkpoint)
        model_wrapper = ExtractorNoCoverageWrapper(model, word_to_index)
    elif model_type == 2:
        model = FFSummaryExtractor().to(device)
        checkpoint = torch.load(f'./pretrained/{model_name}.pt')
        model.load_state_dict(checkpoint)
        model_wrapper = FFExtractorWrapper(model, word_to_index)

    try:
        generate_evaluation_data(model_wrapper, model_name, test_dataset)
        evaluator = EvaluateModel(model_name)
        cmd = evaluator.generate_evaluate_command()
        print("Run the following command in terminal:")
        print(cmd)
    except:
        print('Run command provided as .txt in terminal at model folder location')


if __name__ == '__main__':
    """
    The project is inspired by Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting
    Original authors: Yen-Chun Chen and Mohit Bansal
    PDF: https://arxiv.org/pdf/1805.11080.pdf
    GitHub: https://github.com/ChenRocks/fast_abs_rl
    
    Implementation is inspired by the authors
    Hyper parameters are tuned by the authors
    Training parameters are tuned by the authors
    
    Our implementation is similar in technique but is written by our team
    """

    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--train', type=int, action='store', default=0)
    parser.add_argument('--train_opt', type=int, action='store', default=0)
    parser.add_argument('--amt', type=int, action='store', default=0)
    parser.add_argument('--val', type=int, action='store', default=0)
    parser.add_argument('--teacher_forcing', type=float, action='store', default=0.0)

    parser.add_argument('--eval', type=str, action='store', default=None)
    parser.add_argument('--model_type', type=int, action='store', default=0)
    parser.add_argument('--test_data', type=int, action='store', default=0)

    # Change this to your path to ROUGE-1.5.5.pl script
    os.environ['ROUGE'] = r'D:\Python\Python39\Lib\site-packages\pyrouge\tools\ROUGE-1.5.5'
    # Change this to your desire temp directory
    os.environ['TMP_DIR_PREF'] = r'D:/temp/'

    args = parser.parse_args()
    if args.eval:
        main_eval(args)
    else:
        main_train(args)



    #
    # data_index = 69
    # model_name = '15-03-2022_04-05-17_cnn_dailymail'
    # dataset = ProcessedDataset('cnn_dailymail', split='test', load_raw=True)
    # word2v = Word2VecHelper.load_model(f'cnn_dailymail_128_min5')
    # word_to_index = Word2VecHelper.get_word_to_index(word2v, top_k=30000)
    # model = SummaryExtractor().to(device)
    # checkpoint = torch.load(f'./pretrained/{model_name}.pt')
    # model.load_state_dict(checkpoint)
    # model_wrapper = ExtractorWrapper(model, word_to_index)
    # # # text = dataset.get_text(data_index)
    # # # summ = dataset.get_summary(data_index)
    # # # label = dataset.get_label(data_index)
    # # # model_wrapper.comprehensive_test(text, summ, label, data_index, print_text=False, outfile=None)
    # #
    # generate_evaluation_data(model_wrapper, model_name, dataset)
    # evaluator = EvaluateModel(model_name)
    # cmd = evaluator.generate_evaluate_command()
    # print("Run the following command in terminal:")
    # print(cmd)
