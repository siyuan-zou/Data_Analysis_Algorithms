#include <iostream>
#include <cassert>
#include "LinearRegression.hpp"
#include "Dataset.hpp"
#include "Regression.hpp"

LinearRegression::LinearRegression(Dataset *dataset, int col_regr)
	: Regression(dataset, col_regr)
{
	m_beta = NULL;
	set_coefficients();
}

LinearRegression::~LinearRegression()
{
	if (m_beta != NULL)
	{
		m_beta->resize(0);
		delete m_beta;
	}
}

Eigen::MatrixXd LinearRegression::construct_matrix()
{
	Eigen::MatrixXd X(m_dataset->get_nbr_samples(), m_dataset->get_dim());

	for (int i = 0; i < m_dataset->get_nbr_samples(); i++)
	{
		for (int j = 1; j < m_col_regr + 1; j++)
		{
			X(i, j) = m_dataset->get_instance(i)[j - 1]; // 0 to col_regr - 1
		}
		for (int j = m_col_regr + 2; j < m_dataset->get_dim() + 1; j++)
		{
			X(i, j - 1) = m_dataset->get_instance(i)[j - 1]; // col_regr + 1 to dim - 1
		}
	}

	X.col(0) = Eigen::VectorXd::Ones(m_dataset->get_nbr_samples());

	return X;
}

Eigen::VectorXd LinearRegression::construct_y()
{
	Eigen::VectorXd y(m_dataset->get_nbr_samples());
	for (int i = 0; i < m_dataset->get_nbr_samples(); i++)
	{
		y(i) = m_dataset->get_instance(i)[m_col_regr];
	}
	return y;
}

void LinearRegression::set_coefficients()
{
	m_beta = new Eigen::VectorXd(m_dataset->get_dim());
	Eigen::VectorXd y = construct_y();
	Eigen::MatrixXd X = construct_matrix();

	*m_beta = (X.transpose() * X).fullPivHouseholderQr().solve(X.transpose() * y);
}

const Eigen::VectorXd *LinearRegression::get_coefficients() const
{
	if (!m_beta)
	{
		std::cout << "Coefficients have not been allocated." << std::endl;
		return NULL;
	}
	return m_beta;
}

void LinearRegression::show_coefficients() const
{
	if (!m_beta)
	{
		std::cout << "Coefficients have not been allocated." << std::endl;
		return;
	}

	if (m_beta->size() != m_dataset->get_dim())
	{ // ( beta_0 beta_1 ... beta_{d} )
		std::cout << "Warning, unexpected size of coefficients vector: " << m_beta->size() << std::endl;
	}

	std::cout << "beta = (";
	for (int i = 0; i < m_beta->size(); i++)
	{
		std::cout << " " << (*m_beta)[i];
	}
	std::cout << " )" << std::endl;
}

void LinearRegression::print_raw_coefficients() const
{
	std::cout << "{ ";
	for (int i = 0; i < m_beta->size() - 1; i++)
	{
		std::cout << (*m_beta)[i] << ", ";
	}
	std::cout << (*m_beta)[m_beta->size() - 1];
	std::cout << " }" << std::endl;
}

void LinearRegression::sum_of_squares(Dataset *dataset, double &ess, double &rss, double &tss) const
{
	assert(dataset->get_dim() == m_dataset->get_dim());

	Eigen::MatrixXd X(dataset->get_nbr_samples(), dataset->get_dim());

	for (int i = 0; i < dataset->get_nbr_samples(); i++)
	{
		for (int j = 1; j < m_col_regr + 1; j++)
		{
			X(i, j) = dataset->get_instance(i)[j - 1]; // 0 to col_regr - 1
		}
		for (int j = m_col_regr + 2; j < dataset->get_dim() + 1; j++)
		{
			X(i, j - 1) = dataset->get_instance(i)[j - 1]; // col_regr + 1 to dim - 1
		}
	}

	X.col(0) = Eigen::VectorXd::Ones(dataset->get_nbr_samples());

	Eigen::VectorXd y(dataset->get_nbr_samples());
	for (int i = 0; i < dataset->get_nbr_samples(); i++)
	{
		y(i) = dataset->get_instance(i)[m_col_regr];
	}

	std::cout << "X: " << X.rows() << "x" << X.cols() << std::endl;
	std::cout << "y: " << y.rows() << "x" << y.cols() << std::endl;
	std::cout << "beta: " << m_beta->rows() << "x" << m_beta->cols() << std::endl;
	Eigen::VectorXd y_hat = X * *m_beta;

	double mean = y.sum() / dataset->get_nbr_samples();

	Eigen::VectorXd y_bar = Eigen::VectorXd::Ones(dataset->get_nbr_samples()) * mean;

	std::cout << "y_hat: " << y_hat.rows() << "x" << y_hat.cols() << std::endl;
	std::cout << "y_bar: " << y_bar.rows() << "x" << y_bar.cols() << std::endl;

	ess = (y_hat - y_bar).squaredNorm();
	rss = (y - y_hat).squaredNorm();
	tss = (y - y_bar).squaredNorm();
}

double LinearRegression::estimate(const Eigen::VectorXd &x) const
{

	return m_beta->tail(m_beta->size() - 1).transpose() * x + (*m_beta)[0];
}
