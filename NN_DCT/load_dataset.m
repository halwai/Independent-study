%load dataset
function  [images_train, images_test, labels_train, labels_test] = load_dataset(dataset_name)

if dataset_name == 'mnist'
    dataset_path = '/home/halwai/Datasets/mnist/';
    images_train = reshape(loadMNISTImages(strcat(dataset_path,'train-images.idx3-ubyte')),[28, 28, 60000]);
    labels_train = loadMNISTLabels(strcat(dataset_path, 'train-labels.idx1-ubyte'));
    images_test = reshape(loadMNISTImages(strcat(dataset_path, 't10k-images.idx3-ubyte')),[28, 28, 10000]);
    labels_test = loadMNISTLabels(strcat(dataset_path, 't10k-labels.idx1-ubyte'));

end

end