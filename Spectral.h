//
//  Spectral.h
//  latenttree
/*******************************************************
* Copyright (C) 2014 {Furong Huang} <{furongh@uci.edu}>
*
* This file is part of {Intergrated latent tree structure and parameters learning project}.
*
* All rights reserved.
*******************************************************/
#ifndef __latentTree__Spectral__
#define __latentTree__Spectral__
#include "stdafx.h"

typedef struct
{
	Eigen::VectorXd singular_values;
	Eigen::MatrixXd left_singular_vectors;
	Eigen::MatrixXd right_singular_vectors;
}EigenSparseSVD; // struct interface between svdlibc and eigen

// set of exact svd functions
pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> latenttree_svd(Eigen::MatrixXd A);
pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseVector<double> >latenttree_svd(SparseMatrix<double> A);
// EigenSparseSVD sparse_svd(Eigen::SparseMatrix<double> eigen_sparse_matrix, int rank = 0);

// applications of the exact SVD
double pinv_num(double pnum);
Eigen::VectorXd pinv_vector(Eigen::VectorXd pinvvec);
Eigen::SparseVector<double> pinv_vector(Eigen::SparseVector<double> pinvvec);
Eigen::MatrixXd pinv_matrix(Eigen::MatrixXd pinvmat);
// Eigen::SparseMatrix<double> pinv_matrix(Eigen::SparseMatrix<double> pinvmat);

Eigen::MatrixXd sqrt_matrix(Eigen::MatrixXd pinvmat);

//////////////////////////////////////////////////////////////////////////////////////////
//Nystrom method
SparseMatrix<double> random_embedding_mat(long cols_num, int k);
SparseMatrix<double> random_embedding_mat_dense(long cols_num, int k);
SparseMatrix<double> orthogonalize_cols(SparseMatrix<double> Y);
pair< SparseMatrix<double>, SparseVector<double> > SVD_symNystrom_sparse(SparseMatrix<double> M);
pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseVector<double> > SVD_asymNystrom_sparse(SparseMatrix<double> X);// FIX ME!! not tested
pair< SparseMatrix<double>, SparseVector<double> > SVD_Nystrom_columnSpace_sparse(SparseMatrix<double> A, SparseMatrix<double> B);// FIX ME!!! not tested

// applications of the nystrom method
SparseMatrix<double> pinv_Nystrom_sparse(SparseMatrix<double> X); // FIX ME! not tested 
SparseMatrix<double> pinv_aNystrom_sparse(SparseMatrix<double> X);
pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseMatrix<double> > pinv_Nystrom_sparse_component(SparseMatrix<double> X); // FIX ME! not tested 

std::pair<pair<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > k_svd_observabe(Eigen::SparseMatrix<double> A);// FIX ME! not tested 

// Determinant Calculation
double  prod_sigvals(Node * a, Node * b);// FIX ME! not tested 
double singularvalue_prod(MatrixXd A);// FIX ME! not tested 
double singularvalue_prod(Node *a, Node * b);// FIX ME! not tested 
//////////////////////////////////////////////////////////////////////////////////////////
#endif

/*
std::pair<Eigen::MatrixXd, Eigen::VectorXd> k_svd(Eigen::MatrixXd A, int k);
std::pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> k_svd_observabe(Eigen::MatrixXd A, int k);
pair< SparseMatrix<double>, SparseVector<double> > SVD_symNystrom_sparse(SparseMatrix<double> A, int k);
SparseMatrix<double> sqrt_symNystrom_sparse(SparseMatrix<double> X);
SparseMatrix<double> sqrt_symNystrom_sparse(SparseMatrix<double> X, SparseMatrix<double> Y);
SparseMatrix<double> sqrt_asymNystrom_sparse(SparseMatrix<double> X);
*/
