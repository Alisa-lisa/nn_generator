# nn_generator
Allows rapid prototyping of a simple NN by providing a config file
### Current usage:
```python
# prepare your data (here we use example data for this project)
  X_train, Y_train = generator.create_input_structure('examples/training_set.csv')
  X_test, Y_test = generator.create_input_structure('examples/test_set.csv')

  # desired NN architecture (last layer is always an output layer)
  architecture = {1:12, 2:4, 3:1}
  model, meta = nn.create_and_train_shallow_nn(X_train, Y_train, learning_rate = 0.01, iterations=5000, hidden_units=architecture, seed=345, seeded=True)

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
  
2. currently simple Gradient Descent is implemented, so learning rate and iterations might be set
3. turnoff the seed to get "the best" random initialized parameters

4. the model can be saved and read oput with
```python
  generator.save_model("WHERE_TO_SAVE", model, meta)
  saved_model, saved_meta = generator.read_out_model("WHERE_TO_FIND")    

```
