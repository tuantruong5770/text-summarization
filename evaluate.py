from pyrouge import Rouge155

import os.path
import nltk
import tempfile
import subprocess as sp


try:
    ROUGE_PATH = os.environ['ROUGE']
except KeyError:
    print('Warning: ROUGE is not configured')
    ROUGE_PATH = None


class EvaluateModel:
    def __init__(self, model_name):
        """
        Evaluate the model
        :param model_name: name of model to be evaluated
        """
        self.model_name = model_name


    def evaluate(self):
        """
        Evaluate the model with "official" ROUGE script. The args include:
            [-c 95]: 95% confidence interval
            [-r 1000]: 1000 sampling point in bootstrap resampling
            [-n 2]: max n-gram (ROUGE-2)
            [-m]: Stem both reference and generated using Porter stemmer before computing scores
        :return: computed scores
        """
        assert os.path.isdir(f'evaluate/{self.model_name}/')

        r = Rouge155(rouge_dir=ROUGE_PATH)
        r.system_dir = f'evaluate/{self.model_name}/generated/'
        r.model_dir = f'evaluate/{self.model_name}/reference/'

        r.system_filename_pattern = r'(\d+)_generated.txt'
        r.model_filename_pattern = '#ID#_reference.txt'

        # "Official" ROUGE script
        output = r.convert_and_evaluate(rouge_args='-c 95 -r 1000 -n 2 -m')
        print(output)
        return r.output_to_dict(output)


def generate_evaluation_data(model_wrapper, model_name, dataset_test):
    """
    Generate the data to use for evaluation

    :param model_wrapper: model wrapper with method [predict()-(>prob_vector_tensor, extracted_index)] signature
    :param model_name: name of model
    :param dataset_test: test data set for input
    :return: does not return, generate files instead
    """
    generated_dir = f'./evaluate/{model_name}/generated/'
    reference_dir = f'./evaluate/{model_name}/reference/'

    os.mkdir(f'./evaluate/{model_name}')
    os.mkdir(generated_dir)
    os.mkdir(reference_dir)

    for i in range(len(dataset_test)):
        text = dataset_test.get_text(i)
        raw_text = dataset_test.get_raw_text(i)

        prob_vector_tensor, extracted_index = model_wrapper.predict(text, summary_length=len(dataset_test.get_label(i)))

        raw_text_sents = nltk.sent_tokenize(raw_text)
        generated = '\n'.join([raw_text_sents[i] for i in extracted_index])
        reference = dataset_test.get_raw_summary(i)

        with open(reference_dir + f'{i}_reference.txt', 'w', encoding='utf-8') as ref:
            ref.write(reference)

        with open(generated_dir + f'{i}_generated.txt', 'w', encoding='utf-8') as gen:
            gen.write(generated)




