from utils import timer, beautify_time
from torch.utils.data import DataLoader
from datetime import datetime

import torch
import numpy as np
import torch.nn as nn
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrainHyperParameters:
    def __init__(self,
                 num_training,
                 epoch=5,
                 batch_size = 32,
                 learning_rate=0.001,
                 teacher_forcing_prob=0.5,
                 print_per=500):
        """
        Hyper-parameters for training model.
        """
        self.num_training = num_training
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.teacher_forcing_prob = teacher_forcing_prob
        self.print_per = print_per


class Trainer:
    def __init__(self, model, dataset, num_training=0):
        self.model = model
        self.dataset = dataset

        self.num_training = num_training
        if num_training <= 0:
            self.num_training = len(dataset)

        self.hyper_param = TrainHyperParameters(self.num_training)

        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.Adam(model.model.parameters(), lr=self.hyper_param.learning_rate)
        self.train_loader = DataLoader(dataset=np.arange(self.num_training), batch_size=self.hyper_param.batch_size,
                                       shuffle=True, drop_last=True)


    @timer
    def train_epoch(self):
        """
        Train the model using epochs
        The model saved is the SummaryExtractor model
        """
        num_epochs = self.hyper_param.epoch
        train_loader = self.train_loader
        dataset = self.dataset
        teacher_forcing_prob = self.hyper_param.teacher_forcing_prob
        criterion = self.criterion
        optimizer = self.optimizer
        total_loss = 0

        total_step = len(train_loader)
        print_per = self.hyper_param.print_per
        start_time = time.time()
        dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        print(f"TRAINING STARTED AT {dt_string}")
        for epoch in range(num_epochs):
            for i, index in enumerate(train_loader):
                batch = [dataset[j] for j in index]
                rand = iter(np.random.rand(len(batch)))
                loss = 0

                for text, summ, label, score in batch:
                    # Forward pass
                    # The forward process computes the loss of each iteration on each sample
                    if next(rand) <= teacher_forcing_prob:
                        output = self.model(text, teacher_forcing=True, target=label).to(device)
                    else:
                        output = self.model(text, summary_length=len(label)).to(device)
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
                if (i + 1) % self.hyper_param.print_per == 0:
                    print(
                        f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {total_loss / print_per:.4f}')
                    total_loss = 0
                if (i + 1) == total_step:
                    total_loss = 0

        training_time = beautify_time(time.time() - start_time)
        self.save_model(training_time)


    def save_model(self, training_time):
        model = self.model.model
        dataset = self.dataset
        num_epochs = self.hyper_param.epoch
        teacher_forcing_prob = self.hyper_param.teacher_forcing_prob
        num_training = self.num_training
        batch_size = self.hyper_param.batch_size
        learning_rate = self.hyper_param.learning_rate

        dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        save_name = f'{dt_string}_{dataset.get_dataset_name()}'
        save_loc = f'./pretrained/{save_name}'
        torch.save(self.model.model.state_dict(), save_loc + '.pt')
        with open(save_loc + '.txt', 'w') as f:
            parameters = model.hyper_params
            lines = [
                f'MODEL NAME: {save_name}\n',
                f'\n',
                f'MODEL PARAMETERS:\n',
                f'\n',
                f'word_embedding_dim = {parameters.word_embedding_dim}\n',
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
                f'training_time: {training_time}\n'
            ]
            f.writelines(lines)
