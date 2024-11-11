#include <cmath> // for pow, should you need it

#include <point.hpp>
#include <cloud.hpp>
#include <radial.hpp>

// TODO 2.1: density and radial constructor
// Use profile and volume... although it will only be implemented in the "sisters" classes
// Use kernel's constructor

radial::radial(cloud* data_, double bandwidth_) : kernel(data_) {
	bandwidth = bandwidth_;
}

double radial::density(const point& p) const
{
	int d = data->get_point(0).get_dim();

	double sum=0;
	for (int i=0; i< data->get_n(); i++) {
		sum = sum + profile(pow(p.dist(data->get_point(i)), 2) / pow(bandwidth,2));
	}
	sum = 1 / (volume() * data->get_n() * pow(bandwidth,d)) * sum;

	return sum;
}