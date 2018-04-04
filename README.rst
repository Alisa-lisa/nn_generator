nn_generator
============
package is designed for a rapid prototyping or toying around with a fully connected network (FC NN)

.. code-block:: python

    >>> pip install nn_generator

1. Example usage using data provided in the examples folder
------------------------------------------------------------
.. code-block:: python

    >>> # you could parse the config or create an appropriate dictionary
    >>> from nn_generator.config_parser import config_parser
    >>>
    >>> from nn_generator.model_generator.simple_nn import SimpleNN
    >>> from nn_generator.visualization.simple_plot import simple_plot_predictions
    >>>
    >>>
    >>> # Make sure that the inout is a numpy.array
    >>> # X_dimensions = (features,dataset_size)
    >>> # Y_dimensions = (class, dataset_size)
    >>> t, X_train, Y_train = custom_create_input_structure('path_to_raw_training_set.csv')
    >>> t_test, X_test, Y_test = custom_create_input_structure('path_to_raw_testing_set.csv')
    >>>
    >>> # initialize the config and NN based on the config
    >>> config = config_parser.read_out_config("path_to_your_config.json")
    >>> nn = SimpleNN(config)
    >>>
    >>>
    >>> # Training step
    >>> model, meta = nn.create_and_train_nn(X_train, Y_train)
    >>>
    >>>
    >>> # Depth is the amount of hidden layers in the model - WIP: detect automatically
    >>> depth = meta["architecture"]["depth"]
    >>>
    >>> # Building predictions. The last parameter is needed for plotting due to different input types
    >>> predicted = nn.predict(X_test, model, depth, False)
    >>> accuracy_test, errors = nn.compute_accuracy(X_test, predicted, Y_test)
    >>> print("test accuracy is: {}".format(accuracy_test))
    >>>
    >>>
    >>> # Plotting our predictions vs expected classes
    >>> predicted2 = nn.predict(X_test, model, depth, True)
    >>> simple_plot_predictions(t_test,
                                predicted2.T, Y_test[0],
                                "state", "time",
                                False, "placeholder_name.png")
    >>> # this one is needed for matplotlib plots to stay opened
    >>> plt.show()
    >>>
    >>> # Simple analysis will detect bias or variance problems and will display possible solutions
    >>> nn.simple_analysis(meta["results"]["accuracy"], accuracy_test)

2. Different models can be saved and read out, thus compared to each other:
-----------------------------------------
.. code-block:: python

    >>> # read out the models. The file should be in json format and have proper structure. The example can be found in the example folder
    >>> model1, meta1 = generator.read_out_model("PATH_TO_MODEL1.json")
    >>> model2, meta2 = generator.read_out_model("PATH_TO_MODEL2.json")
    >>>
    >>> # Predict
    >>> predicted1 = nn.predict(X_test, model1, meta1["architecture"]["depth"], False)
    >>> predicted2 = nn.predict(X_test, model2, meta2["architecture"]["depth"], False)
    >>>
    >>> # Compare the accuracy
    >>> accuracy1 = nn.compute_accuracy(predicted1, Y_test)
    >>> accuracy1 = nn.compute_accuracy(predicted2, Y_test)

3. Configuration file:
----------------------
Must have keys:
    - "architecture" (dict): The last layer is the output should always be present and have 1 as the value.
                             Keys are the number of the layer, make sure there are no missing layers.
                             Values are integers for the amount of units in the layer.

    - "learning_rate" (float): A hyperparameter to provide for the Gradient Descent learning algorithm.
                               If the value is to big the GD might explode. If the value is too small the learning process
                               might take ages. Anything between 0.2 and 0.01 is considered to be a good start.

    - "iterations" (int): Also known as "epochs". The number of times the algorithm is being retrained. Very big number will slow down the learning
                          plus it might be not a very efficient hyperparameter to tune.
                          Very small number of iterations will result in less optimal results.

    - "seeded" (bool): In order for the results to be replicable this option should be used. If set to true the seed will be considered.
                       Otherwise the learning will happen with a random initialization every time the training function is called.
    - "seed" (int): If "seeded" is enabled an optimal seed should be fixed for replicable results.

Might have keys:
    - "activation" (dict): If provided the structure should be the same as in for the "architecture" key.
                           The key is the number of the layer. The value is the string-name of the activation function.
                           Available functions are: "sigmoid", "relu".

    - "show_cost" (bool): If set to true will plot the cost for each iterations thus showing how the learning was going.

    - "error_analysis" (bool): If set to true, the falsely detected classes and corresponding input vectors are collected for later error analysis.

    - "human_expertise" (float): If set somewhere between 0 and 1 (closer to 1 is probably a good idea)
                                 will be used for bias/variance detection within the error analysis. Human expertise can be seen as the desired accuracy for the model.

    - "regularization" (float): Some float used in L2-norm regularizetion to penalize the model for overfitting (using too many features).

    - "prediction_confidence" (float): The output probability of the data to be assigned a class is compared to this theshold. Default is 0.7

    - "init_random" (bool): I snot implemented yet. Is planned to be used for custom weights initialization.

Example configuration, data and create_input_structure function can be found in the examples folder.

4. Under the hood:
------------------
Simple straight-forward implementation of the forward propagation, backwards propagation, Gradient Descent and parameters update with NumPy.
The code detects the architecture and other parameters from the config file and follows simple for-loops for iterations and layers for each step.
Error analysis is a very basic test_accuracy and train_accuracy comparison with a very basic information what to do next.

The idea of the project was to have a wrapper for very fast models prototyping and being able to concentrate on the input data preparation.

5. Current limitations (WIP, TODOs):
---------------------------
1) Only Gradient Descent is available
2) RELU and sigmoid are the only AF
3) Only simple NN are implemented (no CNN, RNN, etc.)
4) More visualizations are coming
5) CPU execution only (Numpy implementation)
