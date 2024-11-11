#include <iostream>
#include <cassert>
#include <cmath>	// for sqrt, fabs
#include <cfloat>	// for DBL_MAX
#include <cstdlib>	// for rand, srand
#include <ctime>	// for rand seed
#include <fstream>
#include <cstdio>	// for EOF
#include <string>
#include <algorithm>	// for count
#include <vector>

using std::rand;
using std::srand;
using std::time;

class point
{
    public:

        static int d;
        double *coords;
        int label;
	
	point() {
		coords = new double[d]();
		label = 0;
	}

	~point() {
		delete[] coords;
	}

	void print() const {

		for(int i = 0; i < d; i++) {
			std::cout << coords[i];
			if(i < d - 1) {
				std::cout << "\t";
			}
		}
		std::cout << "\n";
	}

	double squared_dist(const point &q) const {
		double sum = 0.0;
		for(int i = 0; i < d; i++) {
			sum += pow((coords[i] - q.coords[i]), 2);
		}
		return sum;
	}
};

int point::d;

class cloud
{
	private:

	int d;
	int n;
	int k;

	// maximum possible number of points
	int nmax;

	point *points;
	point *centers;


	public:

	cloud(int _d, int _nmax, int _k)
	{
		d = _d;
		point::d = _d;
		n = 0;
		k = _k;

		nmax = _nmax;

		points = new point[nmax];
		centers = new point[k];

		srand(time(0));
	}

	~cloud()
	{
		delete[] centers;
		delete[] points;
	}

	void add_point(const point &p, int label)
	{
		for(int m = 0; m < d; m++)
		{
			points[n].coords[m] = p.coords[m];
		}

		points[n].label = label;

		n++;
	}

	int get_d() const
	{
		return d;
	}

	int get_n() const
	{
		return n;
	}

	int get_k() const
	{
		return k;
	}

	point &get_point(int i)
	{
		return points[i];
	}

	point &get_center(int j)
	{
		return centers[j];
	}

	void set_center(const point &p, int j)
	{
		for(int m = 0; m < d; m++)
			centers[j].coords[m] = p.coords[m];
	}

	double intracluster_variance() const
	{
		double sum = 0.0;
		for (int i = 0; i < n; i++)
		{
			sum += points[i].squared_dist(centers[points[i].label]);
		}
		return sum / n;
	}

	int set_voronoi_labels()
	{
		int changes = 0;
		for (int i=0; i<n; i++) {
			double min_dist = DBL_MAX;
			int min_label = points[i].label;
			for (int j=0; j<k; j++) {
				double dist = points[i].squared_dist(centers[j]);
				if (dist < min_dist) {
					min_dist = dist;
					min_label = j;
				}
			}
			if (min_label != points[i].label) {
				changes++;
				points[i].label = min_label;
			}
		}

		return changes;
	}

	void set_centroid_centers()
	{
		int *counts = new int[k]();
		point *sums = new point[k]();
		for (int i=0; i<n; i++) {
			counts[points[i].label]++;
			for (int m=0; m<d; m++) {
				sums[points[i].label].coords[m] += points[i].coords[m];
			}
		} // complexity O(n*d)

		for (int j=0; j<k; j++) {
			if (counts[j] > 0) {
				for (int m=0; m<d; m++) {
					centers[j].coords[m] = sums[j].coords[m] / counts[j];
				}
			}
		}// complexity O(k*d)

		delete[] counts;
		delete[] sums;
	}

	void init_random_partition()
	{
		for (int i=0; i<n; i++) {
			points[i].label = rand() % k;
		}
	}

	void lloyd()
	{
		set_centroid_centers();
		int changes;
		do {
			changes = set_voronoi_labels();
			set_centroid_centers();
		} while (changes > 0);
	}

	void init_forgy()
	{
		std::vector<int> points_indices;
		for (int i=0; i<n; i++) {
			points_indices.push_back(i); // index集合
		}

		for (int j=0; j<k; j++) {
			int r;
			point* p;

			if (j == 0) {
				int r = rand() % points_indices.size(); // choose center point index
				point &p = points[points_indices[r]]; // p 是一个引用，指向同一个地址
				set_center(p, 0); // assign center point
				points_indices.erase(points_indices.begin() + r); // remove the center point index
			} 
			
			else {
				bool duplicate = true;
				// std::cout << "duplicate =" << duplicate << std::endl;

				while (duplicate) {
					// std::cout << "begin while + duplicate =" << duplicate  << std::endl;

					r = rand() % points_indices.size(); //重新选择一个index
					point &p = points[points_indices[r]]; 

					for (int m=0; m<j; m++) { // Check if the point is already a center
						// std::cout << "begin for" << std::endl;
						if (centers[m].coords != p.coords) { // Compare the coordinates of the points
							duplicate = false;
							break; //离开for循环
					}
					// 在这里，如果duplicate不等于false，说明这个点已经被选中了，while继续循环
					}
				set_center(p, j);
				points_indices.erase(points_indices.begin() + r);
				}

			}	
		}
	}

    void init_plusplus()
    {
        int c1 = rand() % n;
        set_center(points[c1], 0);

        double* distances = new double[n];
        for (int i = 0; i < n; i++)
            distances[i] = points[i].squared_dist(centers[0]); // squared distances to the first center

        for (int j = 1; j < k; j++) // add cj
        {
            double sum = 0;
            for (int i = 0; i < n; i++)
                sum += distances[i]; // sum of squared distances

            double r = (double)rand() / RAND_MAX * sum;
            int cj = 0;
            while (r > distances[cj]){
                r -= distances[cj];
                cj++;
            }
            set_center(points[cj], j);

            for(int i = 0; i < n; i++){
                distances[i] = std::min(distances[i], points[i].squared_dist(centers[j])); //更新sum = min(sum, squared distances)
            }
        }
        delete[] distances;
    }
};
