# nn_generator
Allows rapid prototyping of a simple NN by providing a config file
### Current usage:
```python
  # data preparation -> feature extraction, normalization, etc.
  X_train, Y_train = generator.create_input_structure('examples/training_set.csv')
  X_test, Y_test = generator.create_input_structure('examples/test_set.csv')

  # make sure that all parameters are filled in
  config = json_config.read_out_config("examples/example_config.json")
  nn = SimpleNN(config)
  model, meta = nn.create_and_train_nn(X_train, Y_train)

  depth = meta["architecture"]["depth"]
  predicted = nn.predict(X_test, model, depth, False)
  predicted2 = nn.predict(X_test, model, depth, True)
  accuracy_test = nn.compute_accuracy(predicted, Y_test)
  print("test accuracy is: {}".format(accuracy_test))

  plot_predictions(None, predicted2.T, Y_test[0])

```

1. architecture dictionary is the description of desired NN configuration:
  keys: the number of the hidden layer
  values: number of nodes in the laye
  
2. currently only simple Gradient Descent is implemented
3. turnoff the seed to get "the best" random initialized parameters
    currently the seed is optimized for the example data

4. the model can be saved and read output with:
```python
  generator.save_model("WHERE_TO_SAVE", model, meta)
  saved_model, saved_meta = generator.read_out_model("WHERE_TO_FIND")    

```
