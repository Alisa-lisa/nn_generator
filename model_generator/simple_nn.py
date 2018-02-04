import math
import numpy
import csv
import json
import logging
import preprocessing.example_preprocessor as process

numpy.warnings.filterwarnings('ignore')


SET_UP_KEYS = ["architecture","learning_rate","iterations","seeded","seed"]

def create_input_structure(filename):
    """
    Reads out the data from the original file
    create proper feature vector
    shape should be (features, num_examples)

    :param filename: file containing the data

    returns: X - numpy.array containing the features,
             Y - numpy vector containing the actual stand
    """
    X = []
    Y = []
    with open(filename, 'r') as raw_data:
        next(raw_data)
        reader = csv.reader(raw_data, delimiter=',')
        for row in reader:
            X.append(process.extract_features(row[0]))
            Y.append(math.floor(float(row[1])))

    return numpy.array(X).T, numpy.array([Y])


def sigmoid(x):
   return 1 / (1 + numpy.exp(-x))


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

    with open("examples/{}.json".format(filename), 'w') as model_file:
        json.dump({"model":model, "meta":meta}, model_file)


def read_out_model(filename):
    """
    Read out previously computed model and its details
    :param filename: path to teh stored model
    :return: dict with params and dict with training details
    """
    with open("examples/{}.json".format(filename), "r") as input_model:
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
        elif set(SET_UP_KEYS).issubset(config.keys()):
            return config
        else:
            print(config)
            raise ValueError("The config is not complete")


    @staticmethod
    def _init_params(X, Y, hidden_units, seeded=False, seed=345):
        """
        The architecture is fixed
        :param X: input
        :param Y: output
        :param hidden_units: dict describing the architecture
        :return: dict containing params of the model, int seed
        """
        m = X.shape[0]

        if seeded:
            numpy.random.seed(seed)

        params = {}
        for layer in list(hidden_units.keys()):
            # special treatment for the first layer init
            if layer > 1:
                m = hidden_units[layer-1]
            params["W"+str(layer)] = numpy.random.rand(hidden_units[layer], m)
            params["b"+str(layer)] = numpy.zeros((hidden_units[layer], 1))


        return params, seed

    @staticmethod
    def forward_prop(X, Y, params, depth):
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
        # here we use n-1 RELU
        for layer in range(1, depth):
            cache["Z"+str(layer)] = (numpy.dot(params["W"+str(layer)],
                                              cache["A"+str(layer-1)])
                                     + params["b"+str(layer)])
            cache["A"+str(layer)] = numpy.maximum(cache["Z"+str(layer)], 0)

        # last layer is a binary classifier -> sigmoid
        cache["Z"+str(depth)] = numpy.dot(params["W"+str(depth)], cache["A"+str(depth-1)]) + params["b"+str(depth)]
        cache["A"+str(depth)] = sigmoid(cache["Z"+str(depth)])

        # compute cost
        if Y is not None:
            log1 = numpy.multiply(numpy.log(cache["A"+str(depth)]), Y)
            log2 = numpy.multiply(numpy.log(1 - cache["A"+str(depth)]), 1 - Y)
            cost = (-1 * numpy.sum(log1 + log2)) / m
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

        for i in list(reversed(range(1,depth+1))):
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
            for i in range(0,prediction_prob.shape[1]):
                if prediction_prob[0][i] > confidence_level:
                    res.append(1)
                else:
                    res.append(0)
            return numpy.array([res])
        else:
            return prediction_prob

    @staticmethod
    def compute_accuracy(prediction, Y, confidence_level=0.7):
        """
        Computes accuracy for a computed model on a given set
        https://en.wikipedia.org/wiki/Confusion_matrix
        :param X: input matrix
        :param Y: output vector
        :return: (float) accuracy
        """
        res = []
        for i in range(0,prediction.shape[1]):
            if prediction[0][i] > 0.7:
                res.append(1)
            else:
                res.append(0)
        res = numpy.array([res])
        # to avoid some computational mysteries
        assert prediction.shape == Y.shape

        FN = 0
        TP = 0
        for i in range(0, prediction.shape[1]):
            if res[0][i] == 0 and Y[0][i] == res[0][i]:
                FN += 1
            elif res[0][i] == 1 and Y[0][i] == res[0][i]:
                TP += 1
            else:
                pass
        accuracy = (FN + TP) / Y.shape[1]

        return accuracy

    def create_and_train_nn(self, X, Y):
        """ Simple numpy implementation of a shallow NN training process:
         1. initialize parameters
         2. forward prop
         3. cost function
         4. backward prop
         5. update the weights

         :param X: features input vector
         :param Y: expected output
         :param iterations: int amount of recalculations
         :param architecture: dict desired structure {layer_number: amount of units}
         :param seed: int seed to make results reproducible
         :param seeded: bool to use seed or not

         :return: dict model parameters, meta information
         """
        # init params
        depth = len(self.config["architecture"].keys())
        params, seed = self._init_params(X, Y, self.config["architecture"],
                                         self.config["seeded"],
                                         self.config["seed"])
        cache = ()
        for i in range(0, self.config["iterations"]):
            # forward propagation
            cost, cache = self.forward_prop(X, Y, params, depth)
            # backward propagation
            gradients = self.back_prop(X, Y, params, cache, depth)
            # update weights
            self.update_params(params, gradients,
                               self.config["learning_rate"], depth)

        accuracy = self.compute_accuracy(cache["A"+str(depth)], Y)

        # examples parameters
        model = {}
        for k, v in params.items():
            model[k] = v
        # meta information describing the architecture and the training process
        # for now overlaps with the config and some hardcoded info
        # later will be completely replaced with the config
        meta = {"seeded":[self.config["seeded"], seed],
                "architecture": {"arch": self.config["architecture"],
                                 "depth": depth},
                "results": {"accuracy": accuracy},
                "training":{"backprop": "GD",
                            "learning_rate": self.config["learning_rate"],
                            "iterations": self.config["iterations"],
                            "train size":X.shape[1],
                            "AF":["RELU", "sigmoid"]}
                     }
        return model, meta
