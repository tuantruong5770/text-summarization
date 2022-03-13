from utils import timer, beautify_time
from torch.utils.data import DataLoader
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

import torch
import numpy as np
import torch.nn as nn
import time
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrainHyperParameters:
    def __init__(self,
                 num_training,
                 epoch=5,
                 batch_size=32,
                 learning_rate=0.001,
                 learning_rate_decay_ratio=0.5,
                 patience_decay=0,
                 patience_stop=5,
                 gradient_clip=2.0,
                 teacher_forcing_prob=0.0,
                 check_point_frequency=2000):
        """
        Hyper-parameters for training model.
        """
        self.num_training = num_training
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay_ratio = learning_rate_decay_ratio
        self.patience_decay = patience_decay
        self.patience_stop = patience_stop
        self.gradient_clip = gradient_clip
        self.teacher_forcing_prob = teacher_forcing_prob
        self.check_point_frequency = check_point_frequency


class Trainer:
    def __init__(self, model, train_set, val_set, num_training=0, num_val=0):
        self.model = model
        self.train_set = train_set
        self.val_set = val_set

        self.num_training = num_training
        self.num_val = num_val
        if num_training <= 0:
            self.num_training = len(train_set)
        if num_val <= 0:
            self.num_val = len(val_set)

        self.hyper_param = TrainHyperParameters(self.num_training)

        self.epoch = self.hyper_param.epoch
        self.batch_size = self.hyper_param.batch_size
        self.learning_rate = self.hyper_param.learning_rate
        self.learning_rate_decay_ratio = self.hyper_param.learning_rate_decay_ratio
        self.patience_stop = self.hyper_param.patience_stop
        self.gradient_clip = self.hyper_param.gradient_clip
        self.teacher_forcing_prob = self.hyper_param.teacher_forcing_prob
        self.check_point_frequency = self.hyper_param.check_point_frequency

        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.Adam(model.model.parameters(), lr=self.hyper_param.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', verbose=True, factor=self.learning_rate_decay_ratio,
                                           min_lr=0, patience=self.hyper_param.patience_decay)

        self.train_loader = DataLoader(dataset=np.arange(self.num_training), batch_size=self.hyper_param.batch_size,
                                       shuffle=True, drop_last=True)
        self.val_loader = DataLoader(dataset=np.arange(self.num_val))

        self.best_val = None
        self.current_step = 0
        self.current_patience = 0

        dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        save_name = f'{dt_string}_{self.train_set.get_dataset_name()}'
        self.save_dir = f'./pretrained/{save_name}/'
        self.check_point_dir = self.save_dir + 'checkpoint/'
        os.mkdir(self.save_dir)
        os.mkdir(self.check_point_dir)


    @timer
    def train_epoch(self):
        """
        Train the model using epochs
        The model saved is the SummaryExtractor model
        """
        model_wrapper = self.model
        num_epochs = self.epoch
        batch_size = self.batch_size
        train_loader = self.train_loader
        train_set = self.train_set
        teacher_forcing_prob = self.teacher_forcing_prob
        criterion = self.criterion
        optimizer = self.optimizer
        total_loss = 0

        total_step = len(train_loader)
        check_point_frequency = self.check_point_frequency
        start_time = time.time()
        dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        print(f"TRAINING STARTED AT {dt_string}")
        for epoch in range(num_epochs):
            for i, index in enumerate(train_loader):
                batch = [train_set[j] for j in index]
                rand = iter(np.random.rand(len(batch)))
                loss = 0

                for text, summ, label, score in batch:
                    # Forward pass
                    # The forward process computes the loss of each iteration on each sample
                    if next(rand) <= teacher_forcing_prob:
                        output = model_wrapper(text, teacher_forcing=True, target=label).to(device)
                    else:
                        output = model_wrapper(text, summary_length=len(label)).to(device)

                    cur_loss = model_wrapper.calculate_loss(criterion, output, label)
                    loss += cur_loss

                # Average the loss
                loss /= batch_size
                # Backward pass
                loss.backward()
                # Update total loss
                total_loss += loss.item()
                # Clip gradient
                self.clip_gradient()
                # use the optimizer to update the parameters
                optimizer.step()
                # Reset the grad
                optimizer.zero_grad()
                model_wrapper.model.zero_grad()

                # Log training progress
                if (i + 1) % check_point_frequency == 0:
                    print(
                        f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Train Loss: {total_loss / check_point_frequency:.4f}')
                    total_loss = 0
                    self.save_check_point()
                if (i + 1) == total_step:
                    total_loss = 0
                self.current_step += 1

        training_time = beautify_time(time.time() - start_time)
        train_type = 'EPOCH TRAINING'
        model_wrapper.save_model(train_set.get_dataset_name(), training_time, train_type, self.hyper_param)


    def train_converge_early_stop(self):
        """
        Train the model using early stop
        The model saved is the SummaryExtractor model
        """
        model_wrapper = self.model
        batch_size = self.batch_size
        train_loader = self.train_loader
        train_set = self.train_set
        teacher_forcing_prob = self.teacher_forcing_prob
        criterion = self.criterion
        optimizer = self.optimizer
        total_loss = 0

        check_point_frequency = self.check_point_frequency
        start_time = time.time()
        dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        print(f"TRAINING STARTED AT {dt_string}")
        try:
            it = iter(train_loader)
            while True:
                try:
                    index = next(it)
                except StopIteration:
                    it = iter(train_loader)
                    index = next(it)

                batch = [train_set[j] for j in index]
                rand = iter(np.random.rand(len(batch)))
                loss = 0

                for text, summ, label, score in batch:
                    # Forward pass
                    # The forward process computes the loss of each iteration on each sample
                    if next(rand) <= teacher_forcing_prob:
                        output = model_wrapper(text, teacher_forcing=True, target=label).to(device)
                    else:
                        output = model_wrapper(text, summary_length=len(label)).to(device)

                    cur_loss = model_wrapper.calculate_loss(criterion, output, label)
                    loss += cur_loss

                # Average the loss
                loss /= batch_size
                # Backward pass
                loss.backward()
                # Update total loss
                total_loss += loss.item()
                # Clip gradient
                self.clip_gradient()
                # use the optimizer to update the parameters
                optimizer.step()
                # Reset the grad
                optimizer.zero_grad()
                model_wrapper.model.zero_grad()

                # Log training progress
                if (self.current_step + 1) % check_point_frequency == 0:
                    print(
                        f'Step [{self.current_step + 1}], Loss: {total_loss / check_point_frequency:.4f}')
                    total_loss = 0
                    self.save_check_point()
                    if self.check_stop():
                        break
                self.current_step += 1

        finally:
            training_time = beautify_time(time.time() - start_time)
            train_type = 'EARLY STOP TRAINING'
            model_wrapper.save_model(train_set.get_dataset_name(), training_time, train_type, self.hyper_param)


    def clip_gradient(self, max_grad=1e2):
        """
        Clip the gradients to prevent over/under flow
        :param max_grad: maximum gradient norm
        :return: grad_norm
        """
        model = self.model.model
        clip_thresh = self.gradient_clip
        grad_norm = clip_grad_norm_(
            [grad for grad in model.parameters() if grad.requires_grad], clip_thresh)
        grad_norm = grad_norm.item()
        if grad_norm >= max_grad:
            print(f'Exploding Gradients {grad_norm}')
            grad_norm = max_grad
        return grad_norm


    def save_check_point(self):
        model_wrapper = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        criterion = self.criterion

        val_loss = model_wrapper.calculate_val_loss(criterion, self.val_set, self.val_loader)

        if self.best_val is None:
            self.best_val = val_loss
        elif val_loss < self.best_val:
            self.current_patience = 0
            self.best_val = val_loss
        else:
            self.current_patience += 1

        save_dict = {'val_loss': val_loss, 'state_dict': model_wrapper.model.state_dict(), 'optimizer': optimizer.state_dict()}
        name = f'check_point_{self.current_step + 1}'
        torch.save(save_dict, self.check_point_dir + name + '.ckpt')
        print(f'Val Loss: {val_loss}')
        scheduler.step(val_loss)


    def check_stop(self):
        """
        Early stopping, return true if the model is not improving after number of patience check points
        :return: bool
        """
        return self.current_patience >= self.patience_stop

