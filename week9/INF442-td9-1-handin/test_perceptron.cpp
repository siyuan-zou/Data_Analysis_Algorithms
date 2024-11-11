#include <iostream>
#include <fstream>
#include <time.h>
#include <math.h>

#include <assert.h>
#include "dataset.hpp"
#include "perceptron.hpp"
#include "activation.hpp"

using namespace std;

void run_on_data(OneLayerPerceptron *perceptron,
                 Dataset &data,
                 int rounds,
                 int regr,
                 bool train = true,
                 bool print = false)
{
    clock_t tic, toc;
    clock_t cumulative = 0;
    double rss = 0;

    int rows = data.get_nb_samples();

    for (int round = 0; round < rounds; round++)
    {
        tic = clock();
        for (int row = 0; row < rows; row++)
        {
            if (print)
                cout << "Epoch/round = " << round
                     << ", row = " << row << endl;
            double output = perceptron->run(&data, row, regr, print);
            if (print)
                cout << "\tOutput: " << output;
            double err = (output - data.get_instance(row)[regr]);
            if (print)
                cout << "\tError: " << err << endl;
            rss += err * err;
            if (print)
                cout << "Current RSS: " << rss << endl;
        }

        if (train)
        {
            if (print)
                cout << "Updating learning rate...\t";
            perceptron->decay_learning_rate();
            if (print)
                cout << perceptron->get_learning_rate() << endl;
        }

        toc = clock();
        cumulative += toc - tic;
    }

    cout << "Mean RSS: " << rss / rounds << endl;
    cout << "Total time elapsed = "
         << cumulative / ((float)(CLOCKS_PER_SEC))
         << "s" << endl;
    cout << "Mean time per epoch/round = "
         << cumulative / ((float)(CLOCKS_PER_SEC)) / rounds
         << "s" << endl;
}

int run(int argc, char *argv[])
{
    std::ifstream ftrain(argv[1]);
    std::ifstream ftest(argv[2]);

    if (ftrain.fail())
    {
        std::cout << "Cannot read from the testing file" << std::endl;
        return 1;
    }

    if (ftest.fail())
    {
        std::cout << "Cannot read from the testing file" << std::endl;
        return 1;
    }

    Dataset training(ftrain);
    int dim = training.get_dim() - 1;
    int count = training.get_nb_samples();
    cout << "Read training data from " << argv[1] << endl;
    cout << count << " rows of dimension " << training.get_dim()
         << endl;

    Dataset testing(ftest);
    cout << "Read testing data from " << argv[2] << endl;
    cout << count << " rows of dimension " << testing.get_dim()
         << endl;

    assert(training.get_dim() == testing.get_dim());

    int regr = (argc > 3) ? std::atoi(argv[3]) : dim;
    int size = (argc > 4) ? std::atoi(argv[4]) : default_nb_neurons;
    int epochs = (argc > 5) ? std::atoi(argv[5]) : default_nb_epochs;
    double rate = (argc > 6) ? std::atof(argv[6]) : default_learning_rate;
    bool print = (argc > 7) ? std::atoi(argv[7]) == 1 : false;

    cerr << "If you see an assert failure now that probably means that" << endl
         << "\tsome of the neurons are not properly initialised or not properly connected."
         << endl;
    OneLayerPerceptron *perceptron = new OneLayerPerceptron(dim, size, rate, rate / epochs, sigma, sigma_der);

    cout << "Initialised a 1-layer perceptron" << endl;
    cout << "\tSize of the hidden layer:\t"
         << perceptron->get_nb_neurons() << endl;
    cout << "\tLearning rate:\t\t\t"
         << perceptron->get_learning_rate() << endl;
    cout << "\tDecay:\t\t\t\t" << perceptron->get_decay() << endl;

    cout << "Training the perceptron over "
         << epochs << " epochs for regression over column "
         << regr << endl;

    run_on_data(perceptron, training, epochs, regr, true, print);

    cout << "Switching off learning...\t";

    perceptron->init_learning_rate(0);

    cout << "done. Learning rate = "
         << perceptron->get_learning_rate() << endl;

    cout << "Testing the perceptron on the training data "
         << "(" << epochs << " times)" << endl;

    run_on_data(perceptron, training, epochs, regr, false, print);

    cout << "Testing the perceptron on the testing data "
         << "(" << epochs << " times)" << endl;

    run_on_data(perceptron, testing, epochs, regr, false, print);

    cout << "Deleting the perceptron...\t";
    delete perceptron;
    cout << "done." << endl;
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc > 1)
    {
        if (argc < 3)
        {
            cout << "Usage: " << endl
                 << argv[0] << " <train_file> <test_file> " << endl
                 << "\t[ <column_for_regression> [ <nuber_of_neurons> [ <nuber_of_epochs> [ <learning_rate> [ <print> ] ] ] ] ] " << endl;
            cout << argv[0] << endl
                 << endl;

            cout << "The second form is for unitary testing" << endl;
            return 1;
        }

        return run(argc, argv);
    }
}
