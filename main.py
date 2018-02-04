from visualization.simple_plot import plot_predictions
from generator.simple_nn import SimpleNN
import generator.simple_nn as generator

if __name__ == '__main__':
    nn = SimpleNN()

    # data preparation -> feature extraction, normalization, etc.
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
