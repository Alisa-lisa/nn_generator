import json
import logging

import numpy

from nn_generator.visualization.simple_plot import simple_cost_plot

# If you need numpy warnings comment out this line
numpy.warnings.filterwarnings('ignore')

MUST_KEYS = ["architecture", "learning_rate", "iterations", "seeded", "seed"]
MIGHT_KEYS = ["activation", "show_cost"]


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def relu(x):
    return numpy.maximum(x, 0)


def activation_function(activation_name, X):
    """ Uses proper AF
    :param activation_name: String representing the function
    :param X: array to apply AF to
    :return: calls proper operation
    """
    if activation_name.lower() == "relu":
        return relu(X)
    elif activation_name.lower() == "sigmoid":
        return sigmoid(X)
    else:
        raise NameError("No AF with the name {} "
                        "is allowed".format(activation_name))


def save_model(filename, model_dict, meta):
    """
    Saves examples weights and additional information about training
    :param model_dict: examples weights
    :param filename: where to store file
    :return: None
    """
    # ndarray is not directly serializable -> convert it first to list
    model = {}
    for k, v in model_dict.items():
        model[k] = v.tolist()

    with open("{}.json".format(filename), 'w') as model_file:
        json.dump({"model": model, "meta": meta}, model_file)


def read_out_model(filename):
    """
    Read out previously computed model and its details
    :param filename: path to teh stored model
    :return: dict with params and dict with training details
    """
    with open("{}".format(filename), "r") as input_model:
        try:
            model_and_meta = json.load(input_model)
            model = {}
            for k, v in model_and_meta["model"].items():
                model[k] = numpy.array(v)
            return model, model_and_meta["meta"]
        except Exception:
            logging.error("Either the file is malformed "
                          "or it does not contain needed information")
            return {}, {}


class SimpleNN(object):
    def __init__(self, config):
        self.config = self.init_config(config)

    @staticmethod
    def init_config(config):
        if not config:
            raise ValueError("Empty config is not allowed")
        elif set(MUST_KEYS).issubset(config.keys()):
            return config
        else:
            print(config)
            raise ValueError("The config is not complete")

    def init_params(self, X, Y):
        """
        The architecture is fixed
        :param X: input
        :param Y: output
        :param hidden_units: dict describing the architecture
        :return: dict containing params of the model, int seed
        """
        m = X.shape[0]

        if "seeded" in self.config.keys() and self.config["seeded"]:
            numpy.random.seed(self.config["seed"])

        params = {}
        for layer in list(self.config["architecture"].keys()):
            # special treatment for the first layer init
            if layer > 1:
                m = self.config["architecture"][layer-1]
            params["W"+str(layer)] = numpy.random.rand(
                self.config["architecture"][layer], m)
            params["b"+str(layer)] = numpy.zeros((
                self.config["architecture"][layer], 1))

        return params

    def forward_prop(self, X, Y, params, depth):
        """
        Compute prediction of the network
        :param X: input
        :param Y: output
        :param params: dict containing needed NN params
        :param depth: amount of layers in the NN
        :return: (float) cost and (tuple) all final params
        """
        m = X.shape[1]
        cache = {"A0": X}
        activation_provided = "activation" in self.config.keys()
        # if AF provided -> N-1 Relu, N sigmoid are used
        reg_sum = 0

        for layer in range(1, depth):
            cache["Z"+str(layer)] = (numpy.dot(params["W"+str(layer)],
                                               cache["A"+str(layer-1)])
                                     + params["b"+str(layer)])
            if activation_provided:
                activation = self.config["activation"][layer]
                cache["A"+str(layer)] = activation_function(activation,
                                                            cache["Z"+str(layer)])
            else:
                cache["A"+str(layer)] = numpy.maximum(cache["Z"+str(layer)], 0)

            if "regularization" in self.config.keys():
                reg_sum += numpy.sum(numpy.square(params["W"+str(layer)]))

        cache["Z"+str(depth)] = numpy.dot(params["W"+str(depth)],
                                          cache["A"+str(depth-1)]) + params["b"+str(depth)]
        if activation_provided:
            af = self.config["activation"][depth]
            cache["A"+str(depth)] = activation_function(af, cache["Z"+str(depth)])
        else:
            cache["A"+str(depth)] = sigmoid(cache["Z"+str(depth)])

        # compute cost
        if Y is not None:
            log1 = numpy.multiply(numpy.log(cache["A"+str(depth)]), Y)
            log2 = numpy.multiply(numpy.log(1 - cache["A"+str(depth)]), 1 - Y)
            regularization = (self.config["regularization"] * reg_sum) / 2
            cost = (-1 * numpy.sum(log1 + log2) + regularization) / m
        else:
            cost = 0
        return cost, cache

    @staticmethod
    def back_prop(X, Y, params, cache, depth):
        """
        computes the gradients of the params
        :param X: input
        :param Y: output
        :param params: tuple containing params to compute gradient of
        :return: dict params gradients
        """
        m = X.shape[1]
        grads = {"dZ{}".format(depth): cache["A" + str(depth)] - Y}
        grads["A0"] = X

        for i in list(reversed(range(1, depth+1))):
            if i < depth:
                grads["dA"+str(i)] = numpy.dot(params["W" + str(i + 1)].T,
                                               grads["dZ" + str(i + 1)])
                grads["dZ"+str(i)] = numpy.multiply(grads["dA"+str(i)],
                                                    numpy.int64(cache["A" + str(i)] > 0))
            grads["dW"+str(i)] = 1. / m * numpy.dot(grads["dZ"+str(i)],
                                                    cache["A" + str(i - 1)].T)
            grads["db"+str(i)] = 1. / m * numpy.sum(grads["dZ"+str(i)],
                                                    axis=1, keepdims=True)

        return grads

    @staticmethod
    def update_params(params, grads, learning_rate, depth):
        """
        Update weights
        :param params: dict with model parameters
        :param grads: dict with params changes
        :param learning_rate: float decrease ratio
        :return: None
        """
        for i in range(1, depth+1):
            params["W"+str(i)] -= learning_rate * grads["dW" + str(i)]
            params["b"+str(i)] -= learning_rate * grads["db"+str(i)]

    def predict(self, X, model, depth, convert=True, confidence_level=0.7):
        """
        Computes the outcome of the given NN
        :param X: input
        :param model: all computed params
        :param confidence_level: the probability threshold
        :return: prediction.shape = (X.shape[1], 1)
        """
        cost, cache = self.forward_prop(X, None, model, depth)
        prediction_prob = cache["A" + str(depth)]

        if convert:
            res = []
            for i in range(0, prediction_prob.shape[1]):
                if prediction_prob[0][i] > confidence_level:
                    res.append(1)
                else:
                    res.append(0)
            return numpy.array([res])
        else:
            return prediction_prob

    def compute_accuracy(self, X, prediction, Y, confidence_level=0.7):
        """
        Computes accuracy for a computed model on a given set
        https://en.wikipedia.org/wiki/Confusion_matrix
        :param X: input matrix
        :param Y: output vector
        :return: (float) accuracy
        """
        res = []
        for i in range(0, prediction.shape[1]):
            if prediction[0][i] > confidence_level:
                res.append(1)
            else:
                res.append(0)
        res = numpy.array([res])
        # to avoid some computational mysteries
        assert prediction.shape == Y.shape

        FN = 0
        TP = 0
        errors = {}
        for i in range(0, prediction.shape[1]):
            if res[0][i] == 0 and Y[0][i] == res[0][i]:
                FN += 1
            elif res[0][i] == 1 and Y[0][i] == res[0][i]:
                TP += 1
            else:
                if ("error_analysis" in self.config.keys()
                    and self.config["error_analysis"]):
                    errors[i] = {"detected": [res[0][i],
                                              X[:, i]],
                                 "actual": Y[0][i]}
        accuracy = (FN + TP) / Y.shape[1]

        return accuracy, errors

    def create_and_train_nn(self, X, Y, init_weights=None):
        """ Simple numpy implementation of a shallow NN training process:
         1. initialize parameters
         2. forward prop
         3. cost function
         4. backward prop
         5. update the weights

         :param X: features input vector
         :param Y: expected output
         :param iterations: int amount of recalculations
         :param architecture: dict desired structure
         {layer_number: amount of units}
         :param seed: int seed to make results reproducible
         :param seeded: bool to use seed or not

         :return: dict model parameters, meta information
         """
        depth = len(self.config["architecture"].keys())
        if "init_random" in self.config.keys() and not self.config["init_random"]:
            if init_weights is not None:
                params = init_weights
            else:
                logging.warning("Different starter weights option was enabled, "
                                "but no proper parameters were provided."
                                "Fallback - random initialization")
                params = self.init_params(X, Y)
        else:
            params = self.init_params(X, Y)
        cache = ()
        cost_dev = []
        for i in range(0, self.config["iterations"]):
            # forward propagation
            cost, cache = self.forward_prop(X, Y, params, depth)
            cost_dev.append(cost)
            # backward propagation
            gradients = self.back_prop(X, Y, params, cache, depth)
            # update weights
            self.update_params(params, gradients,
                               self.config["learning_rate"], depth)

        accuracy, errors = self.compute_accuracy(X, cache["A"+str(depth)], Y)
        if "show_cost" in self.config.keys() and self.config["show_cost"]:
            simple_cost_plot([i for i in range(0, self.config["iterations"])],
                             cost_dev, "cost function", "iterations numbers",
                             save=False, name="default")

        model = {}
        for k, v in params.items():
            model[k] = v

        if "activation" in self.config.keys():
            activation = self.config["activation"]
        else:
            activation = ["RELU x N-1", "sigmoid"]
        meta = {"seeded": [self.config["seeded"], self.config["seed"]],
                "architecture": {"arch": self.config["architecture"],
                                 "AF": activation,
                                 "depth": depth},
                "results": {"accuracy": accuracy},
                "training": {"learning algo": "GD",
                             "learning_rate": self.config["learning_rate"],
                             "iterations": self.config["iterations"],
                             "train size": X.shape[1]},
                "errors": errors
                }
        return model, meta

    def simple_analysis(self, accuracy_train, accuracy_test):
        """
        Gives a basic advice what can be done to improve the model
        :param accuracy_train: float showing accuracy of the model on a train set
        :param accuracy_test: float showing accuracy of the model on a test set
        :return: None, prints out some suggestions
        """
        if set(["human_expertise", "error_analysis"]).issubset(self.config.keys()):
            if self.config["human_expertise"] <= accuracy_train:
                print("You have surpassed human expertise level. "
                             "Most approaches won't be as effective as they used to. "
                             "Try more data with possibly different distribution maybe")
            else:
                avoidable_bias = self.config["human_expertise"] - accuracy_train
                variance = accuracy_train - accuracy_test
                if avoidable_bias > variance:
                    print("Seems like bias is a bigger problem than the variance.\n"
                                 "Try:\n"
                                 "1. adding new features to the input\n"
                                 "2. increasing size of the network (either units or architecture)\n"
                                 "3. different learning algorithm.")
                else:
                    print("Seems like variance is a bigger problem now.\n"
                                 "Try: \n"
                                 "1. increasing amount of training data\n"
                                 "2. adding regularization\n"
                                 "3. changing architecture to slimmer one.")
