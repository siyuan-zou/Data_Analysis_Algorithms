#include <cmath> // for pow, atan, should you need them
#include <iostream> // for cerr

#include <point.hpp>
#include <cloud.hpp>
#include <gaussian.hpp>

// TODO 2.1.2: implement volume, profile and guess_bandwidth
// HINTS: pi = std::atan(1) * 4.0, e^x is std::exp(x)
double gaussian::volume() const {
	return pow((2 * M_PI), data->get_point(0).get_dim() / 2.0);
}

double gaussian::profile(double t) const {
	return std::exp(-1.0 *  t/2.0);
}

void gaussian::guess_bandwidth() {
	bandwidth = 1.06 * data->standard_deviation() / pow(data->get_n(), 1.0/5.0);
}
