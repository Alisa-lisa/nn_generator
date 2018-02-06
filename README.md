# nn_generator
Allows rapid prototyping of a simple NN by providing a config file
### Current usage example:
```python
  from visualization.simple_plot import plot_predictions
  from config_parser import json_config
  from model_generator.simple_nn import SimpleNN
  import model_generator.simple_nn as generator

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
  
  saved_model, saved_meta = generator.read_out_model("WHERE_TO_FIND")

  plot_predictions(None, predicted2.T, Y_test[0])

```
The data in the example describes a cyclic behaviour over a certain time.
In order to build a model best capturing this time patterns there are 3 questions 
to fine-tune in the whole process:
1. What data (features) should I feed in?
  - In this case example_processor turns a timestamp into 4 numeric features
  - You can think of any other features to draw out of a time stamp
  or reduce the number of features to see how it works.
2. What model (architecture) could be suitable?
  - Would you like to start withe a shallow model?
  - Compare not so deep models with a deeper version?
  - Save the resulting parameters and use prediction function to compare those.
  example config builds a 3-layered nn, also you can find a saved shallow model for a comparison.
```python
  model1, meta1 = generator.read_out_model("PATH_TO_MODEL1")
  model2, meta2 = generator.read_out_model("PATH_TO_MODEL2")
  
  predicted1 = nn.predict(X_test, model1, meta1["architecture"]["depth"], False)
  predicted2 = nn.predict(X_test, model2, meta2["architecture"]["depth"], False)
  accuracy1 = nn.compute_accuracy(predicted1, Y_test)
  accuracy1 = nn.compute_accuracy(predicted2, Y_test)
  
  # compare accuracy -> decide on model
```
3. How do I want to train my model?
  - Depending on the amount of data this question might vary.
  - For the given example you can play around with the learning rate and iterations number

Current limitations (wip):
1. Only Gradient Descent is available
2. N-1 hidden layers use RELU as activation function, N-th layer uses sigmoid
3. Only Json config is allowed
4. Only simple NN are implemented (no CNN, RNN, etc.)
5. More visualizations are comming
6. CPU execution only (numpy implementation)
