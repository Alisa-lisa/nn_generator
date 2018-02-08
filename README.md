# nn_generator
Allows rapid prototyping of a simple NN by providing a config file
### Usage example:
```python
from visualization.simple_plot import simple_plot_predictions
from config_parser import config_parser
from model_generator.simple_nn import SimpleNN
import model_generator.simple_nn as generator
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # data preparation -> feature extraction, normalization, etc.
    t, X_train, Y_train = generator.create_input_structure('examples/training_set.csv')
    t_test, X_test, Y_test = generator.create_input_structure('examples/test_set.csv')

    config = config_parser.read_out_config("examples/example_config.json")
    nn = SimpleNN(config)
    model, meta = nn.create_and_train_nn(X_train, Y_train)
    # if proper starter weights are provided:
    # starter_weights = {...}
    # model, meta = nn.create_and_train_nn(X_train, Y_train, starter_weights)

    depth = meta["architecture"]["depth"]
    predicted = nn.predict(X_test, model, depth, False)
    predicted2 = nn.predict(X_test, model, depth, True)
    accuracy_test = nn.compute_accuracy(predicted, Y_test)
    print("test accuracy is: {}".format(accuracy_test))

    simple_plot_predictions(t_test,
                            predicted2.T, Y_test[0],
                            "state", "time",
                            True, "examples/3_layer_nn.png")
    # needed for matplotlib to keep plots opened
    plt.show()

    # if "error_analysis" is set to true and "human_expertise" is provided in the config
    # very basic analysis with hints is available
    nn.simple_analysis(meta["results"]["accuracy"], accuracy_test)

```
The data in the example describes a cyclic behaviour over a certain time.
In order to build a model best capturing these time patterns there are 3 questions
to answer:
1. What data (features) should I use?
  - In this case example_processor turns a timestamp into 4 numeric features
  - You can think of any other features to draw out of a time stamp
  or reduce the number of the features to see how it works.
2. What model (architecture) could be suitable?
  - Would you like to start withe a shallow model?
  - Compare not so deep models with a deeper version?
  - Save the resulting parameters and use prediction function to compare those.
  example config builds a 3-layered nn, also you can find a saved shallow model for a comparison.
### Models comparison example:
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

4. As a simple debugging tools cost plotting is implemented. Add: ```"show_cost": true```
to the config file, otherwise the plot will not be created

5. Configuration file must have following keys:
    - "architecture":dict
    - "learning_rate":float
    - "iterations":int
    - "seeded":bool
    - "seed":int
configuration file can have:
    - "activation":dict
    - "show_cost":bool
    - "error_analysis":bool
    - "human_expertise":float within [0,1]
    - "init_random":bool
"activation" key must have the same structure as the architecture but with AF names in values
example configuration can be found in the examples folder

Current limitations (wip):
1. Only Gradient Descent is available
2. RELU and sigmoid are the only AF
4. Only simple NN are implemented (no CNN, RNN, etc.)
5. More visualizations are comming
6. CPU execution only (numpy implementation)
