[x_train, x_test, ~, ~] = load_dataset('mnist');
[y_train] = compute_dct(x_train);
[y_test] = compute_dct(x_test);

save('data','y_train','y_test', 'x_train', 'x_test')