from utils import timer, beautify_time
from word2vec_helper import Word2VecHelper
from datetime import datetime
from model.SummaryExtractor import SummaryExtractor, SummaryExtractorHyperParameters

import torch
import torch.nn as nn
import numpy as np
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_name, word2v):
    model = SummaryExtractor(SummaryExtractorHyperParameters(word2v)).to(device)
    checkpoint = torch.load(f'./pretrained/{model_name}.pt')
    model.load_state_dict(checkpoint)
    return model


@timer
def train_summ_extractor(model, dataset, num_epochs, batch_size, learning_rate, num_training, teacher_forcing_prob=1.0,
                         print_per=100, drop_last=True):
    word2v = model.hyper_params.word2vec_model
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = torch.utils.data.DataLoader(dataset=np.arange(num_training), batch_size=batch_size, shuffle=True,
                                               drop_last=drop_last)
    total_step = len(train_loader)
    total_loss = 0

    start_time = time.time()
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    print(f"TRAINING STARTED AT {dt_string}")
    for epoch in range(num_epochs):
        for i, index in enumerate(train_loader):
            batch = [dataset[j] for j in index]
            rand = iter(np.random.rand(len(batch)))
            loss = 0

            for text, summ, label, score in batch:
                sent_ids = Word2VecHelper.text_to_id(text, word2v, 0)
                # Forward pass
                # The forward process computes the loss of each iteration on each sample
                if next(rand) <= teacher_forcing_prob:
                    output = model(sent_ids, teacher_forcing=True, target=label).to(device)
                else:
                    output = model(sent_ids, summary_length=len(label)).to(device)
                # Creating label tensor
                labels = torch.zeros(output.size()).to(device)
                for j, label_index in enumerate(label):
                    labels[j, label_index] = 1
                # Calculate and tally loss
                # Cur loss is average loss of the data point
                # Averaging is automatically calculated by input in batch to the criterion
                cur_loss = criterion(output, labels)
                loss += cur_loss

            # Backward pass
            optimizer.zero_grad()
            # Average the loss
            loss /= len(batch)
            loss.backward()
            total_loss += loss.item()
            # use the optimizer to update the parameters
            optimizer.step()

            # Below, an epoch corresponds to one pass through all the samples.
            # Each training step corresponds to a parameter update using
            # a gradient computed on a minibatch of 100 samples
            if (i + 1) % print_per == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {total_loss / print_per:.4f}')
                total_loss = 0
            if (i + 1) == total_step:
                total_loss = 0

    total_time = beautify_time(time.time() - start_time)

    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    save_name = f'{dt_string}_{dataset.get_dataset_name()}'
    save_loc = f'./pretrained/{save_name}'
    torch.save(model.state_dict(), save_loc + '.pt')
    with open(save_loc + '.txt', 'w') as f:
        parameters = model.hyper_params
        lines = [
            f'MODEL NAME: {save_name}\n',
            f'\n',
            f'MODEL PARAMETERS:\n',
            f'\n',
            f'word_embedding_dim = {word2v.vector_size}\n',
            f'\n',
            f'conv_sent_encoder_n_hidden = {parameters.conv_sent_encoder_n_hidden}\n',
            f'conv_sent_encoder_output_dim = {parameters.conv_sent_encoder_output_dim}\n',
            f'conv_sent_encoder_kernel = {parameters.conv_sent_encoder_kernel}\n',
            f'conv_sent_encoder_dropout = {parameters.conv_sent_encoder_dropout}\n',
            f'conv_sent_encoder_training = {parameters.conv_sent_encoder_training}\n',
            f'\n',
            f'lstm_encoder_n_hidden = {parameters.lstm_encoder_n_hidden}\n',
            f'lstm_encoder_n_layer = {parameters.lstm_encoder_n_layer}\n',
            f'lstm_encoder_output_dim = {parameters.lstm_encoder_output_dim}\n',
            f'lstm_encoder_dropout = {parameters.lstm_encoder_dropout}\n',
            f'\n',
            f'lstm_decoder_n_hidden = {parameters.lstm_decoder_n_hidden}\n',
            f'lstm_decoder_n_layer = {parameters.lstm_decoder_n_layer}\n',
            f'lstm_decoder_context_vec_size = {parameters.lstm_decoder_context_vec_size}\n',
            f'lstm_decoder_pointer_net_n_hidden = {parameters.lstm_decoder_pointer_net_n_hidden}\n',
            f'lstm_decoder_dropout = {parameters.lstm_decoder_dropout}\n',
            f'\n',
            f'TRAINING PARAMETERS:\n',
            f'\n',
            f'num_training_data = {num_training}\n',
            f'epochs = {num_epochs}\n',
            f'batch_size = {batch_size}\n',
            f'learning_rate = {learning_rate}\n',
            f'teacher_forcing_prob = {teacher_forcing_prob}\n',
            f'training_time: {total_time}\n'
        ]
        f.writelines(lines)


def test_summ_extractor(model, word2v, text, label):
    with torch.no_grad():
        word2v.build_vocab(text, update=True)
        inp = Word2VecHelper.text_to_vector(text, word2v)
        output = model(inp, summary_length=len(label), predict=True).to(device)
    return output


def compare_reference_label_generated(prob_vector, text, summ, label):
    print("~*~*~*~*~*~*~*~*~*~*~*~*~ GROUND TRUTH ~*~*~*~*~*~*~*~*~*~*~*~*~")
    for sent in summ:
        print(' '.join(sent))

    print("*~*~*~*~*~*~*~*~*~*~*~*~ LABELED SUMMARY ~*~*~*~*~*~*~*~*~*~*~*~")
    for label_index in label:
        print(' '.join(text[label_index]))

    print("~*~*~*~*~*~*~*~*~*~*~*~ GENERATED SUMMARY ~*~*~*~*~*~*~*~*~*~*~*")
    preds = prob_vector.argmax(dim=1)
    for pred in preds:
        print(' '.join(text[pred]))




