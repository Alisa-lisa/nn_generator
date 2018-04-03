nn_generator
============
package is designed for a rapid prototyping or toying around with a fully connected network (FC NN)

1. Example usage using data provided in the examples folder
------------------------------------------------------------
.. code-block:: python

    >>> import nn_generator.model_generator.simple_nn as generator
    >>> from nn_generator.config_parser import config_parser
    >>> from nn_generator.model_generator.simple_nn import SimpleNN
    >>> from nn_generator.visualization.simple_plot import simple_plot_predictions
    >>> # Make sure that the inout is a numpy.array with X_dimensions = ()
    >>> # and Y_dimensions = ()
    >>> t, X_train, Y_train = custom_create_input_structure('path_to_raw_training_set.csv')
    >>> t_test, X_test, Y_test = custom_create_input_structure('path_to_raw_testing_set.csv')
    >>>
    >>> config = config_parser.read_out_config("nn_generator/examples/periodic_state_example/example_config.json")
    >>> nn = SimpleNN(config)
    >>> # Training step
    >>> model, meta = nn.create_and_train_nn(X_train, Y_train)
    >>> # Let's check what is stored in meta
    >>> for k, v in meta.items():
    >>>     print(k, v)
    >>> depth = meta["architecture"]["depth"]
    >>> predicted = nn.predict(X_test, model, depth, False)
    >>> # a bit different structure is needed for plotting
    >>> predicted2 = nn.predict(X_test, model, depth, True)
    >>> accuracy_test, errors = nn.compute_accuracy(X_test, predicted, Y_test)
    >>> print("test accuracy is: {}".format(accuracy_test))
    >>> simple_plot_predictions(t_test,
                                predicted2.T, Y_test[0],
                                "state", "time",
                                False, "placeholder_name.png")
    >>> plt.show()  # this one is needed for matplotlib to stay opened
    >>> nn.simple_analysis(meta["results"]["accuracy"], accuracy_test)

2. models comparison can be executed via:
-----------------------------------------
.. code-block:: python

    >>> model1, meta1 = generator.read_out_model("PATH_TO_MODEL1")
    >>> model2, meta2 = generator.read_out_model("PATH_TO_MODEL2")
    >>> predicted1 = nn.predict(X_test, model1, meta1["architecture"]["depth"], False)
    >>> predicted2 = nn.predict(X_test, model2, meta2["architecture"]["depth"], False)
    >>> accuracy1 = nn.compute_accuracy(predicted1, Y_test)
    >>> accuracy1 = nn.compute_accuracy(predicted2, Y_test)

3. Configuration file limitations
----------------------------------
Must have keys:
    - "architecture":dict
    - "learning_rate":float
    - "iterations":int
    - "seeded":bool
    - "seed":int
Might have keys:
    - "activation":dict
    - "show_cost":bool
    - "error_analysis":bool
    - "human_expertise":float within [0,1]
    - "init_random":bool
    - "regularization":float (l2 regularization)
Additional restrictions:
    - "activation" key must have the same structure as the architecture but with AF names in values

Example configuration, data and create_input_structure function can be found in the examples folder

Current limitations (WIP):
---------------------------
1) Only Gradient Descent is available
2) RELU and sigmoid are the only AF
3) Only simple NN are implemented (no CNN, RNN, etc.)
4) More visualizations are coming
5) CPU execution only (Numpy implementation)