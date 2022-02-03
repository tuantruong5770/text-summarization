import tensorflow_datasets as tfds


if __name__ == '__main__':
    datasets = ['billsum', 'cnn_dailymail']
    for dataset in datasets:
        tfds.load(dataset, data_dir='./data', split='train')

