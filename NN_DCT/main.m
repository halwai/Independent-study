[images_train, images_test, labels_train, labels_test] = load_dataset('mnist');
[results_train] = compute_dct(images_train);
[results_test] = compute_dct(images_test);
