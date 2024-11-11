#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdarg>
#include <iterator>
#include <string>
#include <regex>
#include <numeric>
#include <cmath>
#include <fstream>
#include <random>
#include <limits>
#include <array>

#include "gradinglib.hpp"
#include "Dataset.hpp"
#include "Kernel.hpp"
#include "ConfusionMatrix.hpp"
#include "Svm.hpp"

namespace tdgrading {

using namespace testlib;
using namespace std;

const double deps = 0.001;
const std::string default_path = "./grading/tests/";    

double rel_error(double a, double b) {
    return fabs(a - b) / fabs(a);
}

template <typename T, typename... Arguments>
bool test_rel_error(std::ostream &out,
                      const std::string &function_name,
                      T result,
                      T expected,
                      T delta,
                      const Arguments&... args) 
{
    bool success = (rel_error(result, expected) <= delta);
    
    out << (success ? "[SUCCESS] " : "[FAILURE] ");

    print_tested_function(out, function_name, args...);

    out << ": got " << result
        << " expected " << expected << "  The relative error should be in [-" << delta << "," << delta << "]";
    out << std::endl;

    return success;
}

std::string exec(const char* in) {
    std::array<char, 128> buffer;
    std::string out;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(in, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        out += buffer.data();
    }
    return out;
}

int ex1_kernels(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "Ex1_Testing_kernels";
    start_test_suite(out, test_name);

    // Dataset train_dataset(dataset_file);
    Kernel kernel_linear({0, 0, 0.0, 0.0});

    Kernel kernel_poly_1({1, 1, 1.0, 1.0});
    Kernel kernel_poly_2({1, 2, 2.0, 2.0});
    Kernel kernel_poly_3({1, 3, 0.5, 1.0});
    Kernel kernel_poly_4({1, 4, 1.0, -2.0});

    Kernel kernel_rbf_1({2, 0, 0.0, 0.0});
    Kernel kernel_rbf_2({2, 0, 1.0, 0.0});
    Kernel kernel_rbf_3({2, 0, 2.0, 0.0});
    Kernel kernel_rbf_4({2, 0, -1.0, 0.0});

    Kernel kernel_sigmoid_1({3, 0, 0.0, 0.0});
    Kernel kernel_sigmoid_2({3, 0, 1.0, -2.0});
    Kernel kernel_sigmoid_3({3, 0, 2.0, -3.0});
    Kernel kernel_sigmoid_4({3, 0, -1.0, 1.0});

    Kernel kernel_ratquad_1({4, 0, 0.0, 0.0});
    Kernel kernel_ratquad_2({4, 0, 0.0, 2.0});
    Kernel kernel_ratquad_3({4, 0, 0.0, 1.0});
    Kernel kernel_ratquad_4({4, 0, 0.0, 4.0});

    std::vector<double> x1 = {1.0, 0.0};
    std::vector<double> x2 = {2.0, 1.0};

    double EPS = 0.00001;

    std::vector<int> res = {
      test_eq_approx(out, "test linear 1", kernel_linear.k(x1, x2), 2.0, EPS),
      test_eq_approx(out, "test poly 1", kernel_poly_1.k(x1, x2), 3.0, EPS),
      test_eq_approx(out, "test poly 2", kernel_poly_2.k(x1, x2), 36.0, EPS),
      test_eq_approx(out, "test poly 3", kernel_poly_3.k(x1, x2), 8.0, EPS),
      test_eq_approx(out, "test poly 4", kernel_poly_4.k(x1, x2), 0.0, EPS),
      test_eq_approx(out, "test rbf 1", kernel_rbf_1.k(x1, x2), 1.0, EPS),
      test_eq_approx(out, "test rbf 2", kernel_rbf_2.k(x1, x2), exp(-2.0), EPS),
      test_eq_approx(out, "test rbf 3", kernel_rbf_3.k(x1, x2), exp(-4.0), EPS),
      test_eq_approx(out, "test rbf 4", kernel_rbf_4.k(x1, x2), exp(2.0), EPS),
      test_eq_approx(out, "test sigmoid 1", kernel_sigmoid_1.k(x1, x2), 0.0, EPS),
      test_eq_approx(out, "test sigmoid 2", kernel_sigmoid_2.k(x1, x2), 0.0, EPS),
      test_eq_approx(out, "test sigmoid 3", kernel_sigmoid_3.k(x1, x2), tanh(1.0), EPS),
      test_eq_approx(out, "test sigmoid 4", kernel_sigmoid_4.k(x1, x2), -tanh(1.0), EPS),
      test_eq_approx(out, "test ratquad 1", kernel_ratquad_1.k(x1, x2), 0.0, EPS),
      test_eq_approx(out, "test ratquad 2", kernel_ratquad_2.k(x1, x2), 0.5, EPS),
      test_eq_approx(out, "test ratquad 3", kernel_ratquad_3.k(x1, x2), 1.0/3.0, EPS),
      test_eq_approx(out, "test ratquad 4", kernel_ratquad_4.k(x1, x2), 2.0/3.0, EPS)
    };
    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

int ex2_svm_constructor(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "[Ex2]_Testing_SVM_constructor";
    start_test_suite(out, test_name);

    const char* train_file = "csv/tests1.csv";
    int col_class = 0;

    Dataset train_dataset(train_file);
    Kernel kernel({0, 0, 0, 0.0});
    SVM svm(&train_dataset, 0, kernel);
    std::vector<int> train_labels = {1, 1, -1, -1};
    std::vector<std::vector<double>> train_features = {{0.5}, {1}, {-1}, {-0.5}};
    std::vector<std::vector<double>> computed_kernel = {{0.25, 0.5, 0.5, 0.25}, {0.5, 1, 1, 0.5}, {0.5, 1, 1, 0.5}, {0.25, 0.5, 0.5, 0.25}};

    double EPS = 0.00001;

    std::vector<int> res = {
      test_eq(out, "test col_class", svm.get_col_class(), col_class),
      test_eq(out, "test kernel", svm.get_kernel().get_kernel_type(), 0),
      test_eq(out, "test train_labels", svm.get_train_labels(), train_labels),
      test_eq_approx(out, "test train_features", svm.get_train_features(), train_features, EPS),
      test_eq_approx(out, "test computed_kernel", svm.get_computed_kernel(), computed_kernel, EPS)
    };
    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

int ex3_beta_0(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "[Ex3]_Testing_beta_0";
    start_test_suite(out, test_name);

    int col_class = 0;
    const char* train_file = "csv/tests1.csv";
    Dataset train_dataset(train_file);
    Kernel kernel({0, 0, 0, 0.0});
    SVM svm(&train_dataset, 0, kernel);
    std::vector<double> alpha = {0, 1, 0.05, 0.95};  // only 2 support vectors
    svm.set_alphas(alpha);
    svm.compute_beta_0();

    std::vector<int> res = {
      test_eq_approx(out, "test beta_0", svm.get_beta_0(), -0.14375, 0.05)
    };
    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

int ex4_train(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "[Ex4]_Train";
    start_test_suite(out, test_name);

    const char* train_file = "csv/tests1.csv";
    int col_class = 0;

    Dataset train_dataset(train_file);
    Kernel kernel({0, 0, 0, 0.0});
    SVM svm(&train_dataset, 0, kernel);
    svm.train(1.0, 0.01);

    std::vector<int> res = {
      test_eq_approx(out, "test alpha[0]", svm.get_alphas()[0], 1.0, 0.0001),
      test_eq_approx(out, "test alpha[1]", svm.get_alphas()[1], 0.000484634, 0.0001),
      test_eq_approx(out, "test alpha[2]", svm.get_alphas()[2], 0.000479787, 0.0001),
      test_eq_approx(out, "test alpha[3]", svm.get_alphas()[3], 1.0, 0.00001)
    };
    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

int ex5_predict(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "[Ex5]_Predict";
    start_test_suite(out, test_name);

    const char* train_file = "csv/tests1.csv";
    int col_class = 0;

    Dataset train_dataset(train_file);
    Kernel kernel({0, 0, 0, 0.0});
    SVM svm(&train_dataset, 0, kernel);
    svm.train(1.0, 0.01);

    std::vector<int> res = {
      // everything positive in x should result in y=1 (and conversely [...])
      test_eq(out, "test f_hat", svm.f_hat(std::vector<double> {2}), 1),
      test_eq(out, "test f_hat", svm.f_hat(std::vector<double> {-2}), -1)
    };
    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

int ex6_test(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "[Ex6]_Test";
    start_test_suite(out, test_name);

    const char* train_file = "csv/tests1.csv";
    const char* test_file = "csv/tests2.csv";
    int col_class = 0;

    Dataset train_dataset(train_file);
    Dataset test_dataset(test_file);
    Kernel kernel({0, 0, 0, 0.0});
    SVM svm(&train_dataset, 0, kernel);
    svm.train(1.0, 0.01);
    ConfusionMatrix cm = svm.test(&test_dataset);

    std::vector<int> res = {
      test_eq(out, "TP", cm.get_tp(), 2),
      test_eq(out, "TN", cm.get_tn(), 2),
      test_eq(out, "FP", cm.get_fp(), 1),
      test_eq(out, "FN", cm.get_fn(), 1),
      test_eq_approx(out, "f_score", cm.f_score(), 2./3., 0.03),
      test_eq_approx(out, "precision", cm.precision(), 2./3., 0.03),
      test_eq_approx(out, "error_rate", cm.error_rate(), 1./3., 0.03),
      test_eq_approx(out, "detection_rate", cm.detection_rate(), 2./3., 0.03),
      test_eq_approx(out, "false_alarm_rate", cm.false_alarm_rate(), 1./3., 0.03)
    };
    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

int ex7_test_train_mails(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "Test_svm::train";
    start_test_suite(out, test_name);

    const char* train_file = "csv/mail_train.csv";
    const char* test_file = "csv/mail_test.csv";
    int col_class = 0;

    Dataset train_dataset(train_file);
    Dataset test_dataset(test_file);
    Kernel kernel({2, 0, 0.0078125, 0.0});
    SVM svm(&train_dataset, 0, kernel);
    svm.train(8.0, 0.01);
    ConfusionMatrix cm = svm.test(&test_dataset);

    std::vector<int> res = {
      test_eq(out, "TP", cm.get_tp(), 304),
      test_eq(out, "TN", cm.get_tn(), 687),
      test_eq(out, "FP", cm.get_fp(), 5),
      test_eq(out, "FN", cm.get_fn(), 4),
      test_eq_approx(out, "f_score", cm.f_score(), 0.98, 0.03),
      test_eq_approx(out, "precision", cm.precision(), 0.98, 0.03),
      test_eq_approx(out, "error_rate", cm.error_rate(), 0.01, 0.005),
      test_eq_approx(out, "detection_rate", cm.detection_rate(), 0.99, 0.03),
      test_eq_approx(out, "false_alarm_rate", cm.false_alarm_rate(), 0.007, 0.002)
    };
    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

int grading(std::ostream &out, const int test_case_number)
{
/**

Annotations used for the autograder.

[START-AUTOGRADER-ANNOTATION]
{
  "total" : 6,
  "names" : [
        "Kernel.cpp::ex1_kernels",
        "Svm.cpp::ex2_svm_constructor",
        "Svm.cpp::ex3_beta_0",
        "Svm.cpp::ex4_train",
        "Svm.cpp::ex5_predict",
        "Svm.cpp::ex6_test"
        ],
  "points" : [10, 10, 10, 10, 10, 10]
}
[END-AUTOGRADER-ANNOTATION]
*/

    int const total_test_cases = 7;
    std::string const test_names[total_test_cases] = {"Ex1_kernels", "Ex2_svm_constructor", "Ex3_beta_0", "Ex4_train", "Ex5_predict", "Ex6_test", "Ex7_test_train_mails"};
    int const points[total_test_cases] = {10, 10, 10, 10, 10, 10, 0};
    int (*test_functions[total_test_cases]) (std::ostream &, const std::string) = {
      ex1_kernels, ex2_svm_constructor, ex3_beta_0, ex4_train, ex5_predict, ex6_test, ex7_test_train_mails
    };

    return run_grading(out, test_case_number, total_test_cases,
                       test_names, points,
                       test_functions);
}

} // End of namepsace tdgrading
