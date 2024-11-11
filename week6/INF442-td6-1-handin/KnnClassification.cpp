
#include "KnnClassification.hpp"
#include <iostream>
#include <ANN/ANN.h>
#include <cassert>

KnnClassification::KnnClassification(int k, Dataset *dataset, int col_class)
    : Classification(dataset, col_class)
{

    // assert(k > 0 && "k must be greater than 0");
    m_k = k;

    // assert(dataset->get_n_samples() >= k && "k must be less than the number of samples");
    // assert(dataset->get_dim() > 1 && "The dataset must have at least 2 columns");
    int dim = dataset->get_dim() - 1; // delete the target column
    int n_samples = dataset->get_n_samples();

    m_data_pts = annAllocPts(n_samples, dim);

    for (int i = 0; i < n_samples; i++)
    {
        for (int j = 0; j < dim + 1 && j < m_col_class; j++)
        {
            m_data_pts[i][j] = dataset->get_instance(i)[j];
        }
        for (int j = m_col_class + 1; j < dim + 1; j++)
        {
            m_data_pts[i][j - 1] = dataset->get_instance(i)[j];
        }
    }
    m_kd_tree = new ANNkd_tree(m_data_pts, n_samples, dim);
}

KnnClassification::~KnnClassification()
{
    delete m_kd_tree;
    annDeallocPts(m_data_pts);
}

int KnnClassification::estimate(const ANNpoint &x, double threshold) const
{
    ANNidxArray nnIdx = new ANNidx[m_k];
    ANNdistArray dists = new ANNdist[m_k];

    m_kd_tree->annkSearch(x, m_k, nnIdx, dists);

    double votes;
    for (int i = 0; i < m_k; i++)
    {
        votes += m_dataset->get_instance(nnIdx[i])[m_col_class];
    }

    delete[] nnIdx;
    delete[] dists;

    if (votes / double(m_k) > threshold)
        return 1;
    else
        return 0;
}

int KnnClassification::get_k() const
{
    return m_k;
}

ANNkd_tree *KnnClassification::get_kd_tree()
{
    return m_kd_tree;
}

const ANNpointArray KnnClassification::get_points() const
{
    return m_data_pts;
}
