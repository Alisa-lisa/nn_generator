from visualization.simple_plot import plot_predictions
from config_parser import config_parser
from model_generator.simple_nn import SimpleNN
import model_generator.simple_nn as generator

if __name__ == '__main__':
    # data preparation -> feature extraction, normalization, etc.
    X_train, Y_train = generator.create_input_structure('examples/training_set.csv')
    X_test, Y_test = generator.create_input_structure('examples/test_set.csv')

    config = config_parser.read_out_config("examples/example_config.yml")
    nn = SimpleNN(config)
    model, meta = nn.create_and_train_nn(X_train, Y_train)

    for k, v in meta.items():
        print(k, v)
    #
    # depth = meta["architecture"]["depth"]
    # predicted = nn.predict(X_test, model, depth, False)
    # predicted2 = nn.predict(X_test, model, depth, True)
    # accuracy_test = nn.compute_accuracy(predicted, Y_test)
    # print("test accuracy is: {}".format(accuracy_test))
    #
    # plot_predictions(None, predicted2.T, Y_test[0])
