from pyrouge import Rouge155

import os.path
import nltk
import tempfile
import subprocess as sp


try:
    ROUGE_PATH = os.environ['ROUGE']
    TMP_DIR_PREF = os.environ['TMP_DIR_PREF']
except KeyError:
    print('Warning: paths are not configured')
    ROUGE_PATH = None
    TMP_DIR_PREF = None


class EvaluateModel:
    def __init__(self, model_name):
        """
        Evaluate the model
        :param model_name: name of model to be evaluated
        """
        self.model_name = model_name


    def generate_evaluate_command(self):
        """
        pyrouge install issue fix https://github.com/binhna/instruction/issues/4
        Evaluate the model with "official" ROUGE script. The args include:
            [-c 95]: 95% confidence interval
            [-r 1000]: 1000 sampling point in bootstrap resampling
            [-n 2]: max n-gram (ROUGE-2)
            [-m]: Stem both reference and generated using Porter stemmer before computing scores
        :return: command to run in terminal (Windows 10)
        """
        assert os.path.isdir(f'evaluate/{self.model_name}/')
        assert ROUGE_PATH is not None
        assert TMP_DIR_PREF is not None

        directory = f'evaluate/{self.model_name}/'
        # create a temp folder at prefix, remember to delete after running cmd
        tmp_dir = tempfile.TemporaryDirectory(prefix='D:/temp/').name

        gen_dir = directory + 'generated/'
        ref_dir = directory + 'reference/'
        gen_pattern = r'(\d+)_generated.txt'
        ref_pattern = '#ID#_reference.txt'
        system_id = 1
        cmd = '-c 95 -r 1000 -n 2 -m'

        Rouge155.convert_summaries_to_rouge_format(
            gen_dir, os.path.join(tmp_dir, 'generated'))
        Rouge155.convert_summaries_to_rouge_format(
            ref_dir, os.path.join(tmp_dir, 'reference'))
        Rouge155.write_config_static(
            os.path.join(tmp_dir, 'generated'), gen_pattern,
            os.path.join(tmp_dir, 'reference'), ref_pattern,
            os.path.join(tmp_dir, 'settings.xml'), system_id
        )
        cmd = 'perl ' + (os.path.join(ROUGE_PATH, 'ROUGE-1.5.5.pl') + ' -e {} '.format(
            os.path.join(ROUGE_PATH, 'data')) + cmd + ' -a {}'.format(os.path.join(tmp_dir, 'settings.xml')))

        with open(directory + f'{self.model_name}_eval_cmd.txt', 'w') as txt:
            txt.write(cmd)

        return cmd


def generate_evaluation_data(model_wrapper, model_name, dataset_test):
    """
    Generate the data to use for evaluation

    :param model_wrapper: model wrapper with method predict(text, summary_length)-(>prob_vector_tensor, extracted_index)
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

        with open(reference_dir + f'{i + 1}_reference.txt', 'w', encoding='utf-8') as ref:
            ref.write(reference)

        with open(generated_dir + f'{i + 1}_generated.txt', 'w', encoding='utf-8') as gen:
            gen.write(generated)




