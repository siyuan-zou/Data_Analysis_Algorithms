#include <numeric>
#include <cmath>

#include "Kernel.hpp"

Kernel::Kernel(const kernel_parameter& param):
	// variable(value) gives you the default
	// "constructor" of the object "variable"
	// here, default constructors of ints and
	// doubles mean "copy the value"
	kernel_type(param.kernel_type),
	degree(param.degree),
	gamma(param.gamma),
	coef0(param.coef0) {}
;

double Kernel::k(const std::vector<double> &x1, const std::vector<double> &x2) const {
	// switch is a C++ keyword allowing for a shorter syntax
	// than if toto == "tata" else if [...] which we would
	// have to write e.g. in Python
	switch (kernel_type) {
		case LINEAR:
			return kernel_linear(x1, x2);
		case POLY:
			return kernel_poly(x1, x2);
		case RBF:
			return kernel_rbf(x1, x2);
		case SIGMOID:
			return kernel_sigmoid(x1, x2);
		case RATQUAD:
			return kernel_ratquad(x1, x2);
		default:
			std::cout << "Invalid kernel" << std::endl;
			return 0.0;
	}
};

double Kernel::dot(const std::vector<double> &x1, const std::vector<double> &x2) {
	// std::inner_product accumulates the sum of the element-wise
	// product of two vectors given the provided begin of x1
	// and the beginning of x2 up to the end of x1 s.t. there can
	// be edge cases if they're not the same length
	return std::inner_product(x1.begin(), x1.end(), x2.begin(), 0.0);
};

double Kernel::kernel_linear(const std::vector<double> &x1, const std::vector<double> &x2) const {
	// Exercise 1
	return 0;
};

double Kernel::kernel_poly(const std::vector<double> &x1, const std::vector<double> &x2) const {
	// Exercise 1
	return 0;
};

double Kernel::kernel_rbf(const std::vector<double> &x1, const std::vector<double> &x2) const {
	// Exercise 1
	return 0;
};

double Kernel::kernel_sigmoid(const std::vector<double> &x1, const std::vector<double> &x2) const {
	// Exercise 1
	return 0;
};

double Kernel::kernel_ratquad(const std::vector<double> &x1, const std::vector<double> &x2) const {
	// Exercise 1
	return 0;
};

int Kernel::get_kernel_type() const {
	return kernel_type;
}
