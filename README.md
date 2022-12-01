# text-summarization
An implementation of extractive text summarization model based on a model proposed by Yen-Chun Chen and Mohit Bansal. 

UCI CS-175 Project

The project is inspired by Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting
Original authors: Yen-Chun Chen and Mohit Bansal
PDF: https://arxiv.org/pdf/1805.11080.pdf
GitHub: https://github.com/ChenRocks/fast_abs_rl


Hyper parameters and Training parameters are taken from the original paper.
We implement a hierarchichal neural network for extractive text summarization based on the extractive neural network from the original paper.



# File description in ./src

main.py: Main (terminal) program use to train and generate model evaluation

data_processing: Data preprocessor and loader

utils: Timer functions

word2vec_helper: Helper function for loading, training, saving, etc... for word2vec model

evaluate: Evaluate command cript generator (Windows 10)

# File description in ./src/model

ConvSentEncoder.py: Convolutional sentence encoder model

ExtractorNoCoverageWrapper.py: Wrapper class for pointer network summary extractor with no coverage vector 

ExtractorWrapper.py: Wrapper class for pointer network summary extractor with coverage vector

FFDecoder.py: Feedforward decoder model

FFExtractorWrapper.py: Wrapper class for feedforward summary extractor

FFNetwork.py: Feedforward network  model

FFSummaryExtractor.py: Feedforward summary extractor model

LSTMDecoder.py: LSTM decoder with pointer network and coverage vector model

LSTMDecoderNoCoverage.py: LSTM decoder with pointer network with no coverage vector model

SummaryExtractor.py: Summary extractor with pointer network and coverage vector model

SummaryExtractorNoCoverage.py: Summary extractor with pointer network with no coverage vector model

Trainer.py: Generic trainer class with epoch training and early stop training
