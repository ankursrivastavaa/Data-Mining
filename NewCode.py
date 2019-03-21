# Raisa, Ankur, Neha
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import RepeatedKFold


# this function coverts the output label data (1 * n) to (C * N), where C is the number of class, this will be used
# as output data for this network
def convert_to_one_hot(y, c):
    y_inside = np.eye(c)[y.reshape(-1)]
    return y_inside.T.reshape(c, y.shape[1])


# this is an auxiliary method for debugging to check whether the gradients are correct or not, if the asert fails,
# then something is wrong, this is done only on weights not the biases
def gradient_checking(layer_dims_inside, input_dim_number, parameters_weight_inside, parameters_bias_inside,
                      gradients_inside_weight, x_inside, y_inside):
    layers_inside = list(layer_dims_inside)
    layers_inside.insert(0, input_dim_number)
    num_layers = len(layers_inside)

    for layer in range(0, num_layers - 1):
        for next_layer in range(0, layers_inside[layer + 1]):
            for prev_layer in range(0, layers_inside[layer]):
                parameters_weight_inside[layer + 1][next_layer][prev_layer] += .0000001
                output = forward_pass(x_inside, parameters_weight_inside, parameters_bias_inside, num_layers)[0]
                error_add = - np.sum(np.multiply(output, np.log(y_inside)))

                parameters_weight_inside[layer + 1][next_layer][prev_layer] -= 2 * .0000001
                output = forward_pass(x_inside, parameters_weight_inside, parameters_bias_inside, num_layers)[0]
                error_subtract = - np.sum(np.multiply(output, np.log(y_inside)))

                parameters_weight_inside[layer + 1][next_layer][prev_layer] += .0000001

                gradient = (error_add - error_subtract) / (2 * .0000001)
                diff = abs(gradient - gradients_inside_weight[layer + 1])

                assert diff < .0001


# this function initializes the weights and momentum randomly from a normal distribution
def random_initialization(layer_dims_inside, input_dim_number, num_layers):
    parameters_weight_inside = ['blank']
    parameters_bias_inside = ['blank']
    momentum_parameters_weight = ['blank']
    momentum_parameters_bias = ['blank']

    for layer in range(0, num_layers):
        if layer == 0:
            parameters_weight_inside.append(np.random.randn(layer_dims_inside[layer],
                                                            input_dim_number))
            parameters_bias_inside.append(np.random.randn(layer_dims_inside[layer], 1))
            momentum_parameters_weight.append(np.zeros([layer_dims_inside[layer], input_dim_number]))
            momentum_parameters_bias.append(np.zeros([layer_dims_inside[layer], 1]))
        else:
            parameters_weight_inside.append(np.random.randn(layer_dims_inside[layer],
                                                            layer_dims_inside[layer - 1]))
            parameters_bias_inside.append(np.random.randn(layer_dims_inside[layer], 1))

            momentum_parameters_weight.append(np.zeros([layer_dims_inside[layer], layer_dims_inside[layer - 1]]))
            momentum_parameters_bias.append(np.zeros([layer_dims_inside[layer], 1]))

    return parameters_weight_inside, parameters_bias_inside, momentum_parameters_weight, momentum_parameters_bias


# this method does a forward pass for the full dataset
def forward_pass(input_data_inside, parameters_weight_inside, parameters_bias_inside, num_layer):
    layer_variables = []
    output_layer = input_data_inside
    output_layer_preceding = output_layer

    for l in range(1, num_layer):
        input_layer = np.dot(parameters_weight_inside[l], output_layer_preceding) + parameters_bias_inside[l]
        output_layer = 1 / (1 + np.exp(-input_layer))
        layer_variables.append((output_layer_preceding, input_layer, output_layer))
        output_layer_preceding = output_layer

    input_layer = np.dot(parameters_weight_inside[num_layer], output_layer) + parameters_bias_inside[num_layer]
    a = np.exp(input_layer)
    output_layer = a / np.sum(a, axis=0)
    layer_variables.append((output_layer_preceding, input_layer, output_layer))

    return layer_variables


# this method does a backward pass for a given layer, it calculates the gradients
# the formulae has been taken from this course and verified by gradient checking
# https://www.coursera.org/learn/neural-networks-deep-learning
def backward_pass_specific_layer(gradient_input_layer, output_layer__preceding, w_current):
    gradient_w_current = np.dot(gradient_input_layer, output_layer__preceding.T)
    gradient_bias_current = np.sum(gradient_input_layer, axis=1)
    gradient_output_layer_preceding = np.dot(w_current.T, gradient_input_layer)

    return gradient_output_layer_preceding, gradient_w_current, gradient_bias_current


# this method does one backward pass for the full dataset
def backward_pass_all_layer(output_final_layer, output_inside, parameters_variables,
                            parameters_weight_inside, num_layer):
    gradients_all_layers_weight = []
    gradients_all_layers_output = []
    gradients_all_layers_bias = []

    for layer in range(num_layer + 1):
        gradients_all_layers_weight.append('blank')
        gradients_all_layers_output.append('blank')
        gradients_all_layers_bias.append('blank')

    current_variables = parameters_variables[num_layer - 1]

    gradient_input_layer = output_final_layer - output_inside
    gradients_all_layers_output[num_layer - 1], gradients_all_layers_weight[num_layer], gradients_all_layers_bias[
        num_layer] = \
        backward_pass_specific_layer(gradient_input_layer, current_variables[0][0], parameters_weight_inside[num_layer])

    for l in reversed(range(1, num_layer)):
        current_variables = parameters_variables[l - 1]

        a = 1 / (1 + np.exp(-current_variables[1]))
        sigmoid_derivative = np.multiply(a, 1 - a)
        gradient_input_layer_current = np.multiply(gradients_all_layers_output[l], sigmoid_derivative)
        gradient_output_layer_preceding_temp, gradient_w_temp, gradient_bias_temp \
            = backward_pass_specific_layer(gradient_input_layer_current,
                                           current_variables[0], parameters_weight_inside[l])

        gradients_all_layers_output[l - 1] = gradient_output_layer_preceding_temp
        gradients_all_layers_weight[l] = gradient_w_temp
        gradients_all_layers_bias[l] = gradient_bias_temp

    return gradients_all_layers_weight, gradients_all_layers_bias


# this method updates the parameters
def update_parameters(parameters_weight_inside, parameters_bias_inside, gradient_all_parameters_weight,
                      gradient_all_parameters_bias, learning_rate_inside,
                      num_layer, momentum_parameters_weight, momentum_parameters_bias, mu=0.9):
    for l in range(num_layer):
        momentum_parameters_weight[l + 1] = momentum_parameters_weight[l + 1] * mu \
                                            + learning_rate_inside * gradient_all_parameters_weight[l + 1]
        momentum_parameters_bias[l + 1] = \
            momentum_parameters_bias[l + 1] * mu \
            + learning_rate_inside * gradient_all_parameters_bias[l + 1]

        parameters_weight_inside[l + 1] = \
            parameters_weight_inside[l + 1] - momentum_parameters_weight[l + 1]
        parameters_bias_inside[l + 1] = \
            parameters_bias_inside[l + 1] - momentum_parameters_bias[l + 1]


# this is the main method for model training, forward, backward and update
def model_training(input_data_inside, output_inside, layer_dims_inside, learning_rate_inside, num_iterations_inside):
    num_layer = len(layer_dims_inside)
    parameters_weight_inside, parameters_bias_inside, momentum_parameters_weight, momentum_parameters_bias\
        = random_initialization(layer_dims_inside, input_data_inside.shape[0], num_layer)
    for i in range(0, num_iterations_inside):
        layer_variables = forward_pass(input_data_inside, parameters_weight_inside,
                                       parameters_bias_inside, num_layer)

        output_final_layer = layer_variables[len(layer_variables) - 1][2]
        error_inside = - np.sum(np.multiply(output_inside, np.log(output_final_layer)))

        gradient_all_layers_weight, gradient_all_layers_bias = backward_pass_all_layer(output_final_layer,
                                                                                       output_inside,
                                                                                       layer_variables,
                                                                                       parameters_weight_inside,
                                                                                       num_layer)

        # gradient_checking(layer_dims_inside, input_data_inside.shape[0], parameters_inside,
        #                 gradient_all_layers_weight, input_data_inside, output_inside.shape[0], output_inside)

        update_parameters(parameters_weight_inside, parameters_bias_inside, gradient_all_layers_weight,
                          gradient_all_layers_bias, learning_rate_inside,
                          num_layer, momentum_parameters_weight, momentum_parameters_bias)

        print("error Iteration umber %i: %f" % (i, error_inside))

    return parameters_weight_inside, parameters_bias_inside


# data prep

data = load_iris()
data_loaded = pd.DataFrame(data=np.c_[data['data'], data['target']],
                           columns=data['feature_names'] + ['target'])

# standard
full_X_data = np.matrix(data_loaded.iloc[:, 0:4])
full_Y_data = np.matrix(data_loaded.iloc[:, 4]).T
full_X_data -= np.mean(full_X_data, axis=0)
full_X_data /= np.std(full_X_data, axis=0)

full_X_data, full_Y_data = shuffle(full_X_data, full_Y_data, random_state=0)

K = 3
layer_dims = [64, 64, 64, K]  # two layer neural network
num_layer_outside = len(layer_dims)
learning_rate = .002
num_iterations = 15000
number_fold = 10
number_repeats = 2

errors = []
# this is cross validation
cross_validation_class = RepeatedKFold(n_splits=number_fold, n_repeats=number_repeats, random_state=0)
for train, test in cross_validation_class.split(full_X_data, full_Y_data):
    train_X, test_X, train_Y, test_Y = full_X_data[train], full_X_data[test], full_Y_data[train], full_Y_data[test]
    train_X = train_X.T
    test_X = test_X.T
    train_Y = train_Y.T
    train_Y = convert_to_one_hot(train_Y.astype(int), K)
    test_Y = test_Y.T
    test_Y = convert_to_one_hot(test_Y.astype(int), K)

    assert train_X.shape[1] == train_Y.shape[1]
    assert train_Y.shape[0] == K

    parameters_weight, parameters_bias = model_training(train_X, train_Y, layer_dims, learning_rate, num_iterations)

    outputData = forward_pass(train_X, parameters_weight, parameters_bias, len(layer_dims))[num_layer_outside - 1]
    outputClass = np.argmax(outputData[2], axis=0)
    test_outputClass = np.argmax(train_Y, axis=0)
    error = 100 * np.sum((outputClass - test_outputClass != 0).astype(int)) / train_Y.shape[1]
    # print(error)

    outputData = forward_pass(test_X, parameters_weight, parameters_bias, len(layer_dims))[num_layer_outside - 1]
    outputClass = np.argmax(outputData[2], axis=0)
    test_outputClass = np.argmax(test_Y, axis=0)
    error = 100 * np.sum((outputClass - test_outputClass != 0).astype(int)) / test_Y.shape[1]
    print('this fold error is')
    print(error)
    errors.append(error)

print('final error is')
print(np.mean(errors))
print('Accuracy is :',100-np.mean(errors))
