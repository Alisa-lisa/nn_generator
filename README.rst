nn_generator is designed for very rapid prototyping or toying around with the FC NN
==============
Example usage from the data included into package
-----------------
>>> import nn_generator.model_generator.simple_nn as generator
>>> from nn_generator.config_parser import config_parser
>>> from nn_generator.model_generator.simple_nn import SimpleNN
>>> from nn_generator.visualization.simple_plot import simple_plot_predictions
>>> t, X_train, Y_train = generator.create_input_structure('path_to_raw_training_set.csv')
>>> t_test, X_test, Y_test = generator.create_input_structure('path_to_raw_testing_set.csv')

>>> config = config_parser.read_out_config("nn_generator/examples/periodic_state_example/example_config.json")
>>> nn = SimpleNN(config)
>>> model, meta = nn.create_and_train_nn(X_train, Y_train)
>>> for k, v in meta.items():
>>>     print(k, v)
>>> depth = meta["architecture"]["depth"]
>>> predicted = nn.predict(X_test, model, depth, False)
>>> predicted2 = nn.predict(X_test, model, depth, True)
>>> accuracy_test, errors = nn.compute_accuracy(X_test, predicted, Y_test)
>>> print("test accuracy is: {}".format(accuracy_test))
>>> simple_plot_predictions(t_test,
                            predicted2.T, Y_test[0],
                            "state", "time",
                            False, "nn_generator/examples/periodic_state_example/best_data_deep.png")
>>> plt.show()
>>> nn.simple_analysis(meta["results"]["accuracy"], accuracy_test)