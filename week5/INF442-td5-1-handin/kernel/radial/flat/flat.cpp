#include <cmath> // for pow, atan, should you need them

#include <point.hpp>
#include <flat.hpp>

// TODO 2.1.1: implement volume and profile
// HINT: pi = std::atan(1) * 4.0
double flat::volume() const {
	int d = data->get_point(0).get_dim();

	double sum=0;
	sum = pow(M_PI, d/ 2.0) / std::tgamma(d/2.0 + 1.0);

	return sum;
}

double flat::profile(double t) const {
	if (t <= 1) {
		return 1.0;
	} else {return 0.0;}
}
