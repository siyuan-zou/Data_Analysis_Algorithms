#include "neuron.hpp"
#include "node.hpp"

#include <random>
#include <chrono> // To seed the random generator
#include <cmath>
#include <cassert>
#include <iostream>

// Vanilla constructor
Neuron::Neuron(int _nb_dendrites)
    : nb_dendrites(_nb_dendrites),
      dendrites(_nb_dendrites + 1, nullptr),
      axon(new Node()),
      activation([](double x) -> double
                 { return (x > 0) ? 1 : 0; }),
      activation_der([](double x) -> double
                     { return 0; }),
      weights(_nb_dendrites + 1, 1)
{
    // Setup "bias-related elements"
    dendrites[0] = new Node(-1);
    weights[0] = 0;
    init_weights();
    count++;
}

// Dendrites + activation function
Neuron::Neuron(int _nb_dendrites,
               std::vector<Node *> &_dendrites,
               std::function<double(double)> _activation,
               std::function<double(double)> _activation_der)
    : nb_dendrites(_nb_dendrites),
      dendrites(_dendrites),
      axon(new Node()),
      activation(_activation),
      activation_der(_activation_der),
      weights(_nb_dendrites + 1, 1)
{
    // Setup "bias-related elements"
    dendrites.insert(dendrites.begin(), new Node(-1));
    weights[0] = 0;
    init_weights();
    count++;
}

// Dendrites + activation function
Neuron::Neuron(int _nb_dendrites,
               std::vector<Neuron *> &_neurons,
               std::function<double(double)> _activation,
               std::function<double(double)> _activation_der)
    : nb_dendrites(_nb_dendrites),
      dendrites(_nb_dendrites + 1, nullptr),
      axon(new Node()),
      activation(_activation),
      activation_der(_activation_der),
      weights(_nb_dendrites + 1, 1)
{
    // Setup "bias-related elements"
    dendrites[0] = new Node(-1);
    weights[0] = 0;

    init_weights();

    for (int i = 0; i < nb_dendrites; i++)
        dendrites[i + 1] = _neurons[i]->axon;

    count++;
}

Neuron::~Neuron()
{
    delete dendrites[0];
    delete axon;
    count--;
}

void Neuron::init_weights()
{
    // We randomly initialise the weights for the dendrites,
    // not for the bias, the weight for the bias is initialised to 0
    // https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
    typedef std::chrono::high_resolution_clock myclock;
    myclock::time_point beginning = myclock::now();

    double upper_bound = 1.0 / std::sqrt(nb_dendrites);
    double lower_bound = -upper_bound;
    std::uniform_real_distribution<double>
        unif(lower_bound, upper_bound);
    std::default_random_engine re;

    // obtain a seed from the timer
    myclock::duration d = myclock::now() - beginning;
    re.seed(d.count());

    for (int i = 1; i <= nb_dendrites; i++)
        weights[i] = unif(re);
}

// Getters
int Neuron::get_nb_dendrites() const
{
    return nb_dendrites;
}

const Node *Neuron::get_dendrite(int pos) const
{
    assert(0 <= pos && pos < nb_dendrites);
    return dendrites[pos + 1];
}

const Node *Neuron::get_axon() const
{
    return axon;
}

double Neuron::get_weight(int pos) const
{
    assert(0 <= pos && pos < nb_dendrites);
    return weights[pos + 1];
}

double Neuron::get_bias() const
{
    return weights[0];
}

const Node *Neuron::get_bias_dendrite() const
{
    return dendrites[0];
}

double Neuron::get_learning_rate() const
{
    return rate;
}

double Neuron::get_output_signal() const
{
    return axon->get_signal();
}

double Neuron::get_collected_input() const
{
    return collected_input;
}

// Setters
void Neuron::set_weights(std::vector<double> &_weights)
{
    assert(_weights.size() == nb_dendrites + 1);
    weights = _weights;
}

void Neuron::set_weight(int pos, double _weight)
{
    assert(0 <= pos && pos < nb_dendrites);
    weights[pos + 1] = _weight;
}

void Neuron::set_bias(double _bias)
{
    weights[0] = _bias;
}

void Neuron::set_learning_rate(double _rate)
{
    rate = _rate;
}

void Neuron::set_collected_input(double _collected_input)
{
    collected_input = _collected_input;
}

void Neuron::set_back_value(double back_value)
{
    axon->set_back_value(back_value);
}

// Forward propagation
void Neuron::step()
{
    // TODO Exercise 1
    // Collecting signals from dendrites
    // Bias is stored in dendrites[0] and weights[0]

    collected_input = 0;
    for (int i = 0; i < get_nb_dendrites(); i++)
    {
        collected_input += get_dendrite(i)->get_signal() * get_weight(i);
    }
    collected_input += get_bias() * get_bias_dendrite()->get_signal();

    // Compute and set the axon signal
    axon->set_signal(activation(collected_input));
}

// Backward propagation (training)
void Neuron::step_back()
{
    // TODO Exercise 2
    // See page 208 from INF 442 Lecture 9
    // For each dendrite

    // Propagate the error storing it in the dendrite's back value

    // Update the associated weight

    double err = activation_der(collected_input) * axon->get_back_value();

    // For each dendrite (excluding the bias dendrite)
    for (int i = 0; i < nb_dendrites + 1; i++)
    {
        // Propagate the error storing it in the dendrite's back value
        dendrites[i]->set_back_value(err * weights[i]);
        // Update the associated weight
        weights[i] -= rate * err * dendrites[i]->get_signal();
    }
}

int Neuron::count = 0;

int Neuron::get_count()
{
    return count;
}

/********************************************************************/
/* Auxiliary function for printing */

std::ostream &operator<<(std::ostream &str, const Neuron &n)
{
    using namespace std;

    int count = n.get_nb_dendrites();

    str << "Weights:";
    for (int i = 0; i < count; i++)
        str << " " << n.get_weight(i);
    str << endl;

    str << "Bias: " << n.get_bias() << endl;

    str << "Dendrites:";
    for (int i = 0; i < count; i++)
    {
        if (n.get_dendrite(i) != nullptr)
            str << " " << *(n.get_dendrite(i));
        else
            str << " - ";
    }
    str << endl;

    str << "Bias: " << *n.get_bias_dendrite() << endl;
    str << "Axon: " << *n.get_axon() << endl;

    return str;
}
