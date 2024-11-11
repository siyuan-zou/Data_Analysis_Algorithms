#include "perceptron.hpp"
#include "activation.hpp"
#include <iostream>
#include <assert.h>
#include <algorithm> // std::max, std::min
#include <cmath>

OneLayerPerceptron::OneLayerPerceptron(int _dim, int _size, double _rate, double _decay,
                                       std::function<double(double)> _activation,
                                       std::function<double(double)> _activation_der)
    : dim(_dim), size(_size), inputs(dim, nullptr), hidden(size, nullptr),
      rate(_rate), decay(_decay)
{
    // To complete this constructor, have a look at those
    // of the classes Node and Neuron

    // Recall:
    // dim - the dimension of the input (excluding already the regression column)
    // size - the number of neurons in the hidden layer of the perceptron

    // TODO Exercise 3: Create nodes for the inputs
    for (int i = 0; i < dim; i++)
    {
        inputs[i] = new Node();
    }

    // TODO Exercise 3: Create neurons for the hidden layer connecting them to the inputs
    for (int i = 0; i < size; i++)
    {
        hidden[i] = new Neuron(_dim, inputs, _activation, _activation_der);
    }

    // TODO Exercise 3: Create the output neuron connecting it to those of the hidden layer
    // and using identity (x |--> x) as the activation function
    output = new Neuron(_size, hidden, id, id_der);

    init_learning_rate(_rate);
}

OneLayerPerceptron::~OneLayerPerceptron()
{
    // Complete this destructor
    // (as a rule of thumb, everything that is
    //  created in a constructor must be deleted here)
    for (auto node : inputs)
    {
        // TODO Exercise 3
        delete node;
    }

    // TODO Exercise 3
    for (auto neuron : hidden)
    {
        delete neuron;
    }
    delete output;
}

/*******************************\
( *          Getters            * )
\*******************************/

int OneLayerPerceptron::get_nb_neurons()
{
    return hidden.size();
}

double OneLayerPerceptron::get_learning_rate()
{
    return rate;
}

double OneLayerPerceptron::get_decay()
{
    return decay;
}

/*******************************\
( *          Setters            * )
\*******************************/

void OneLayerPerceptron::set_learning_rate(double _rate)
{
    rate = _rate;

    for (auto neuron : hidden)
    {
        assert(neuron != nullptr);
        neuron->set_learning_rate(rate);
    }

    assert(output != nullptr);
    output->set_learning_rate(rate);
}

void OneLayerPerceptron::init_learning_rate(double _rate)
{
    epoch = 0;
    set_learning_rate(_rate);
}

void OneLayerPerceptron::decay_learning_rate()
{
    epoch++;
    set_learning_rate(rate / (1 + decay * epoch));
}

/*******************************\
( *          Execution          * )
\*******************************/

const double sqrt3 = 1.73205080757;
const double eps = 0.001;

double OneLayerPerceptron::normalize(double val, Dataset *data, int coord)
{
    // https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
    // "If the inputs are normalized to have mean 0 and standard deviation 1..."
    assert(0 <= coord && coord < data->get_dim());
    if (std::abs(data->get_std_dev(coord)) < eps)
        return val;
    else
        return (val - data->get_mean(coord)) / data->get_std_dev(coord);
}

double OneLayerPerceptron::denormalize(double val, Dataset *data, int coord)
{
    assert(0 <= coord && coord < data->get_dim());
    return val * data->get_std_dev(coord) + data->get_mean(coord);
}

void OneLayerPerceptron::prepare_inputs(Dataset *data, int row, int regr, bool print)
{
    using namespace std;

    // Initialise the input signals, skipping the regression
    // column -- do not forget to normalize the data (see above)

    if (print)
        cout << "Initialising input signals...\t";

    for (int i = 0; i < data->get_dim(); i++)
    {
        if (i < regr)
        {
            inputs[i]->set_signal(normalize(data->get_instance(row)[i], data, i));
        }
        else if (i > regr)
        {
            inputs[i - 1]->set_signal(normalize(data->get_instance(row)[i], data, i));
        }
    }

    if (print)
        cout << "done." << endl;
}

void OneLayerPerceptron::compute_hidden_step(bool print)
{
    using namespace std;

    if (print)
        cout << "Running internal layer neurons..." << endl;

    // TODO Exercise 4: Compute one step of the hidden layer
    for (auto neuron : hidden)
    {
        neuron->step();
    }

    if (print)
        cout << "done." << endl;
}

double OneLayerPerceptron::compute_output_step(Dataset *data, int row, int regr, bool print)
{
    // This method should comprise both forward and backward propagation

    using namespace std;

    double ret, error;

    if (print)
        cout << "Running output neuron..." << endl
             << *output << endl;

    // TODO Exercise 4: Compute one step of the output neuron
    output->step();
    ret = output->get_axon()->get_signal();

    if (print)
        cout << "done. Output = " << ret << endl;

    if (print)
        cout << "Setting up back propagation...\t";

    // Compute the error for back propagation
    error = -2 * (normalize(data->get_instance(row)[regr], data, regr) - ret);

    // TODO Exercise 4: Initialise the back propagation
    output->set_back_value(error);

    if (print)
    {
        cout << "done. Propagated error = " << error << endl;
        cout << "Running back propagation through the output neuron...\t";
    }

    // TODO Exercise 4: Propagate back through the output neuron
    output->step_back();
    ret = output->get_output_signal();

    if (print)
        cout << "done." << endl;

    return denormalize(ret, data, regr);
}

void OneLayerPerceptron::propagate_back_hidden(bool print)
{
    using namespace std;

    if (print)
        cout << "Running back propagation through the hidden layer...\t";

    // TODO Exercise 4: Propagate back through the hidden layer
    for (auto neuron : hidden)
    {
        neuron->step_back();
    }

    if (print)
        cout << "done." << endl;
}

double OneLayerPerceptron::run(Dataset *data, int row, int regr, bool print)
{
    assert(data->get_dim() == dim + 1);
    assert(0 <= regr && regr < data->get_dim());

    prepare_inputs(data, row, regr, print);

    compute_hidden_step(print);

    double ret = compute_output_step(data, row, regr, print);

    propagate_back_hidden(print);

    return ret;
}
