#include <iostream>
#include <ANN/ANN.h>
#include "KnnRegression.hpp"

KnnRegression::KnnRegression(int k, Dataset *dataset, int col_regr)
	: Regression(dataset, col_regr)
{
	m_k = k;
	int dim = dataset->get_dim() - 1; // delete the target column
	int n_samples = dataset->get_nbr_samples();

	m_dataPts = annAllocPts(n_samples, dim);

	for (int i = 0; i < n_samples; i++)
	{
		for (int j = 0; j < dim + 1 && j < col_regr; j++)
		{
			m_dataPts[i][j] = dataset->get_instance(i)[j];
		}
		for (int j = col_regr + 1; j < dim + 1; j++)
		{
			m_dataPts[i][j - 1] = dataset->get_instance(i)[j];
		}
	}
	m_kdTree = new ANNkd_tree(m_dataPts, n_samples, dim);
}

KnnRegression::~KnnRegression()
{
	annDeallocPts(m_dataPts);
	delete m_kdTree;
}

double KnnRegression::estimate(const Eigen::VectorXd &x) const
{
	assert(x.size() == m_dataset->get_dim() - 1);

	Eigen::VectorXd y(x.size());

	for (int i = 0; i < x.size(); i++)
	{

		ANNidxArray nnIdx = new ANNidx[m_k];
		ANNdistArray dists = new ANNdist[m_k];

		ANNpoint queryPt = annAllocPt(x.size());
		for (int j = 0; j < x.size(); j++)
		{
			queryPt[j] = x[j];
		}

		m_kdTree->annkSearch(queryPt, m_k, nnIdx, dists);

		double sumy = 0.0;

		for (int i = 0; i < m_k; i++)
		{
			sumy += m_dataset->get_instance(nnIdx[i])[m_col_regr];
		}

		delete[] nnIdx;
		delete[] dists;

		return sumy / m_k;
	}
}

int KnnRegression::get_k() const
{
	return m_k;
}

ANNkd_tree *KnnRegression::get_kdTree() const
{
	return m_kdTree;
}
