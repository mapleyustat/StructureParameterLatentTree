//
//  Spectral.cpp
//  latenttree
/*******************************************************
* Copyright (C) 2014 {Furong Huang} <{furongh@uci.edu}>
*
* This file is part of {Intergrated latent tree structure and parameters learning project}.
*
* All rights reserved.
*******************************************************/
#pragma once
#include "stdafx.h"
#ifndef _WIN32
#define _copysign copysign
#endif
using namespace std;
using namespace Eigen;
extern int KHID;
extern int VOCA_SIZE;
/////////////////////////////////////////////////////////////////////
// set of svd functions
pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> latenttree_svd(Eigen::MatrixXd A)
{	// works with zero matrix too
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::MatrixXd u = svd.matrixU();
	Eigen::VectorXd s = svd.singularValues();
	Eigen::MatrixXd v = svd.matrixV();
	pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> mv;
	mv.first.first = u;
	mv.first.second = v;
	mv.second = s;
	return mv;
}
pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseVector<double> >latenttree_svd(SparseMatrix<double> A)
{
	MatrixXd A_dense = (MatrixXd)A;
	pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> mv_dense;
	mv_dense = latenttree_svd(A_dense);
	pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseVector<double> > mv;
	mv.first.first = mv_dense.first.first.sparseView();
	mv.first.second = mv_dense.first.second.sparseView();
	mv.second = mv_dense.second.sparseView();
	return mv;
}

//EigenSparseSVD sparse_svd(Eigen::SparseMatrix<double> eigen_sparse_matrix, int rank) // input arguments are a sparse matrix in csc format in eigen toolkit and the rank parameter which corresponds to the number of singular values required; note that this is in double precision
//{	// the results will always has #=rank such many eigenvalues no matter zero or not. 
//	// Even if rank is 0, we will always need up with rank eigenvalues, they are zero in the rank 0 case. 
//	int i, j; // loop variables
//	EigenSparseSVD eigen_svd_variable; // to be returned
//	eigen_svd_variable.singular_values.resize(rank); // allocating memory for our svd struct variable members
//	eigen_svd_variable.left_singular_vectors.resize(eigen_sparse_matrix.rows(), rank); // allocating memory for our svd struct variable members
//	eigen_svd_variable.right_singular_vectors.resize(eigen_sparse_matrix.cols(), rank); // allocating memory for our svd struct variable members
//
//	// we need to initialize them
//	eigen_svd_variable.singular_values = VectorXd::Zero(rank);
//	eigen_svd_variable.left_singular_vectors = MatrixXd::Zero(eigen_sparse_matrix.rows(), rank);
//	eigen_svd_variable.right_singular_vectors = MatrixXd::Zero(eigen_sparse_matrix.cols(), rank);
//	for (int this_id = 0; this_id < min(eigen_sparse_matrix.rows(), rank); ++this_id){
//		eigen_svd_variable.left_singular_vectors(this_id, this_id) = 1;
//	}
//	for (int this_id = 0; this_id < min(eigen_sparse_matrix.cols(), rank); ++this_id){
//		eigen_svd_variable.right_singular_vectors(this_id, this_id) = 1;
//	}
//	SMat svdlibc_sparse_matrix = svdNewSMat(eigen_sparse_matrix.rows(), eigen_sparse_matrix.cols(), eigen_sparse_matrix.nonZeros());  // allocate dynamic memory for a svdlibc sparse matrix
//	if (svdlibc_sparse_matrix == NULL)
//	{
//		printf("memory allocation for svdlibc_sparse_matrix variable in the sparse_svd() function failed\n");
//		fflush(stdout);
//		exit(3);
//	}
//	SVDRec svd_result = svdNewSVDRec(); // allocate dynamic memory for a svdlibc svd record for storing the result of applying the lanczos method on the input matrix
//	if (svd_result == NULL)
//	{
//		printf("memory allocation for svd_result variable in the sparse_svd() function failed\n");
//		fflush(stdout);
//		exit(3);
//	}
//	int iterations = 0; // number of lanczos iterations - 0 means until convergence
//	double las2end[2] = { -1.0e-30, 1.0e-30 }; // tolerance interval
//	double kappa = 1e-6; // another tolerance parameter
//	double copy_tol = 1e-6; // tolerance threshold for copying from svdlibc to eigen format
//
//	eigen_sparse_matrix.makeCompressed(); // very crucial - this ensures correct formatting with correct inner and outer indices
//	if (rank == 0) // checking for full rank svd option
//		rank = ((eigen_sparse_matrix.rows()<eigen_sparse_matrix.cols()) ? eigen_sparse_matrix.rows() : eigen_sparse_matrix.cols());
//
//	i = 0;
//	while (i < eigen_sparse_matrix.nonZeros()) // loop to assign the non-zero values
//	{
//		svdlibc_sparse_matrix->value[i] = *(eigen_sparse_matrix.valuePtr() + i);
//		i++;
//	}
//	i = 0;
//	while (i < eigen_sparse_matrix.nonZeros()) // loop to assign the inner indices
//	{
//		svdlibc_sparse_matrix->rowind[i] = *(eigen_sparse_matrix.innerIndexPtr() + i);
//		i++;
//	}
//	i = 0;
//	while (i < eigen_sparse_matrix.cols()) // loop to assign the outer indices
//	{
//		svdlibc_sparse_matrix->pointr[i + 1] = *(eigen_sparse_matrix.outerIndexPtr() + i + 1); // both must be +1 - this has been tested; refer comment in the struct in svdlib.h to verify
//		i++;
//	}
//
//	svd_result = svdLAS2(svdlibc_sparse_matrix, rank, iterations, las2end, kappa); // computing the sparse svd of the input matrix using lanczos method
//	for (i = 0; i < rank; i++) // update the rank based on zero singular value check; used for adaptively resizing the resultant
//		if (svd_result->S[i] == 0)
//		{
//			rank = i;
//			break;
//		}
//	/*
//	if (rank > 0){
//		eigen_svd_variable.singular_values.resize(rank); // allocating memory for our svd struct variable members
//		eigen_svd_variable.left_singular_vectors.resize(eigen_sparse_matrix.rows(), rank); // allocating memory for our svd struct variable members
//		eigen_svd_variable.right_singular_vectors.resize(eigen_sparse_matrix.cols(), rank); // allocating memory for our svd struct variable members
//		eigen_svd_variable.singular_values = VectorXd::Zero(rank);
//		eigen_svd_variable.left_singular_vectors = MatrixXd::Zero(eigen_sparse_matrix.rows(), rank);
//		eigen_svd_variable.right_singular_vectors = MatrixXd::Zero(eigen_sparse_matrix.cols(), rank);
//	}
//	*/
//
//	// note that efficiency can be increased by avoiding this copy from svdlib to eigen format and using blas-style function calling; but this was done to facilitate subsequent operations in other parts of the algorithm
//	for (i = 0; i < rank; i++) // loop to copy the singular values
//		eigen_svd_variable.singular_values(i) = svd_result->S[i];
//	for (i = 0; i < rank; i++) // loop to copy the left singular vectors
//		for (j = 0; j < eigen_sparse_matrix.rows(); j++)
//			if (fabs(svd_result->Ut->value[i][j]) > copy_tol) // checking numerical tolerance
//				eigen_svd_variable.left_singular_vectors(j, i) = svd_result->Ut->value[i][j]; // transpose while copying
//			else
//				eigen_svd_variable.left_singular_vectors(j, i) = 0; // transpose while copying
//
//	for (i = 0; i < rank; i++) // loop to copy the right singular vectors
//		for (j = 0; j < eigen_sparse_matrix.cols(); j++) 
//			if (fabs(svd_result->Vt->value[i][j]) > copy_tol) // checking numerical tolerance
//				eigen_svd_variable.right_singular_vectors(j, i) = svd_result->Vt->value[i][j]; // transpose while copying
//			else
//				eigen_svd_variable.right_singular_vectors(j, i) = 0; // transpose while copying
//
//	svdFreeSVDRec(svd_result); // free the dynamic memory allocated for the svdlibc svd record
//	svdFreeSMat(svdlibc_sparse_matrix); // free the dynamic memory allocated for the svdlibc sparse matrix
//
//	return eigen_svd_variable;
//}


// pseudo inverse
double pinv_num(double pnum)
{
	double pnum_inv = (fabs(pnum) > TOLERANCE) ? 1.0 / pnum : 0;
	return pnum_inv;
}

Eigen::VectorXd pinv_vector(Eigen::VectorXd pinvvec)
{
	Eigen::VectorXd singularValues_inv(pinvvec.size());
	for (int i = 0; i<pinvvec.size(); ++i) {
		singularValues_inv(i) = (fabs(pinvvec(i)) > TOLERANCE) ? 1.0 / pinvvec(i) : 0;
	}
	return singularValues_inv;
}
Eigen::SparseVector<double> pinv_vector(Eigen::SparseVector<double> pinvvec)
{
	Eigen::SparseVector<double> singularValues_inv;
	singularValues_inv.resize(pinvvec.size());

	for (int i = 0; i<pinvvec.size(); ++i) {
		singularValues_inv.coeffRef(i) = (fabs(pinvvec.coeff(i)) > TOLERANCE) ? 1.0 / pinvvec.coeff(i) : 0;
	}
	singularValues_inv.prune(TOLERANCE);
	return singularValues_inv;
}
//////////////////////////////////////////////////
Eigen::MatrixXd pinv_matrix(Eigen::MatrixXd pmat)
{
	MatrixXd pinvmat(pmat.cols(), pmat.rows());
	if (pmat.nonZeros() == 0){
		pinvmat = pmat.transpose();
	}
	else{

		pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> U_L = latenttree_svd(pmat);
		Eigen::VectorXd singularValues_inv = pinv_vector(U_L.second);
		pinvmat = (U_L.first.second*singularValues_inv.asDiagonal()*U_L.first.first.transpose());
	}
	return pinvmat;
}
/*
Eigen::SparseMatrix<double> pinv_matrix(Eigen::SparseMatrix<double> A)
{
	Eigen::SparseMatrix<double> pinvmat;
	pinvmat.resize(A.cols(), A.rows());
	//incorporate sparseSVD
	if (A.nonZeros() == 0 || A.rows() == 0 || A.cols() == 0){
		pinvmat = A.transpose();
	}
	else{
		MatrixXd A_tmp_dense = (MatrixXd)A;
		if (A_tmp_dense.hasNaN()){
			A.setZero();
			for (int i = 0; i< min(A.rows(), A.cols()); i++){
				A.coeffRef(i, i) = 1;
			}
		}

		EigenSparseSVD u_l = sparse_svd(A, KHID);
		if (u_l.left_singular_vectors.cols() == 0)
		{
			pinvmat = A.transpose();
		}
		else{
			Eigen::MatrixXd u = u_l.left_singular_vectors.leftCols(KHID);
			Eigen::MatrixXd v = u_l.right_singular_vectors.leftCols(KHID);
			Eigen::VectorXd s = u_l.singular_values.head(KHID);
			Eigen::VectorXd singularValues_inv = pinv_vector(s);
			pinvmat = (v * singularValues_inv.asDiagonal()* u.transpose()).sparseView();
			pinvmat.makeCompressed();
			pinvmat.prune(TOLERANCE);
		}
	}
	return pinvmat;

}
*/
// sqrt
Eigen::MatrixXd sqrt_matrix(Eigen::MatrixXd pinvmat)
{

	Eigen::MatrixXd sqrtmat;
	if (pinvmat.nonZeros() == 0){
		sqrtmat = pinvmat.transpose();
	}
	else{
		pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> U_L = latenttree_svd(pinvmat);
		Eigen::VectorXd singularValues_sqrt = U_L.second.head(KHID);
		Eigen::MatrixXd left_sing_vec = U_L.first.first.leftCols(KHID);
		Eigen::MatrixXd right_sing_vec = U_L.first.second.leftCols(KHID);
		for (long i = 0; i < KHID; ++i) {
			singularValues_sqrt(i) = sqrt(U_L.second(i));
		}

		sqrtmat = (left_sing_vec*singularValues_sqrt.asDiagonal()*right_sing_vec.transpose());
	}
	return sqrtmat;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Nystrom for sparseSVD
SparseMatrix<double> random_embedding_mat(long cols_num, int k)
{	// phi.transpose() * phi is always diagonal 
	srand(clock());
	SparseMatrix<double> phi(cols_num, k);
	for (int i = 0; i<cols_num; i++)
	{

		int r = rand() % k;							// randomly-------
		int r_sign = rand() % 5;
		double p_rand = (double)r_sign / (double)5.0;
		if (p_rand<0.5)
			phi.coeffRef(i, r) = -1;
	}
	phi.makeCompressed();
	phi.prune(TOLERANCE);
	return phi;
}

SparseMatrix<double> random_embedding_mat_dense(long cols_num, int k)
{	// phi.transpose() * phi is always diagonal 
	srand(clock());
	MatrixXd randMat = MatrixXd::Random((int)cols_num, k);
	SparseMatrix<double> phi = randMat.sparseView();
	phi.makeCompressed();	phi.prune(TOLERANCE);
	phi = orthogonalize_cols(phi);
	return phi;
}

SparseMatrix<double> orthogonalize_cols(SparseMatrix<double> Y)
{
	
	unsigned int c = Y.cols();;
	unsigned int r = Y.rows();
	for (unsigned int j = 0; j < c; ++j)
	{
		for (unsigned int i = 0; i < j; ++i)
		{
			double dotij = ((VectorXd)Y.block(0, i, r, 1)).dot((VectorXd)Y.block(0, j, r, 1));

			for (unsigned int k = 0; k < r; ++k)
				Y.coeffRef(k, j) -= dotij * Y.coeff(k,i);

			dotij = ((VectorXd)Y.block(0, i, r, 1)).dot((VectorXd)Y.block(0, j, r, 1));

			for (unsigned int k = 0; k < r; ++k)
				Y.coeffRef(k, j) -= dotij * Y.coeff(k, i);
		}

		double normsq = ((VectorXd)Y.block(0, j, r, 1)).dot((VectorXd)Y.block(0, j, r, 1));
		double normsq_sqrt = sqrt(normsq);
		double scale = (normsq_sqrt > TOLERANCE) ? 1.0 / normsq_sqrt : 0.0;

		for (unsigned int k = 0; k < r; ++k)
			Y.coeffRef(k, j) *= scale;
	}
	return Y;
}


pair< SparseMatrix<double>, SparseVector<double> > SVD_symNystrom_sparse(SparseMatrix<double> M)
{
	pair< SparseMatrix<double>, SparseVector<double> > USigma;
	// Generate random_mat; 
	int k_prime = 2 * KHID;
//	cout << "M rows: " << M.rows() << "M cols: " << M.cols() << endl;
	if (M.rows() > 20 * KHID)
	{
		SparseMatrix<double> random_mat = random_embedding_mat((long)M.cols(), k_prime);// : random_embedding_mat_dense((long)M.cols(), k_prime);
		//	cout << "random_mat rows: " << random_mat.rows() << "random_mat cols: " << random_mat.cols() << endl;
		SparseMatrix<double> Q = M * random_mat; //Q.makeCompressed(); Q.prune(TOLERANCE);
		//	cout << "Q rows: " << Q.rows() << "Q cols: " << Q.cols() << endl;
		Q = orthogonalize_cols(Q);
		SparseMatrix<double> C = M * Q;
		MatrixXd Z = (MatrixXd)C.transpose() * C;
		pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> svd_Z = latenttree_svd(Z);
		MatrixXd V = (svd_Z.first.second).leftCols(KHID);
		SparseMatrix<double> V_sparse = V.sparseView();
		VectorXd S = svd_Z.second.head(KHID);
		USigma.second = S.cwiseSqrt().sparseView(); // S.array().sqrt();
		MatrixXd diag_inv_S_sqrt = pinv_vector(S.cwiseSqrt()).asDiagonal();
		SparseMatrix<double> diag_inv_S_sqrt_s = diag_inv_S_sqrt.sparseView();
		USigma.first = C * (V_sparse)* diag_inv_S_sqrt_s;
	}
	else
	{
		pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseVector<double> > mv = latenttree_svd(M);
		USigma.first = mv.first.first.leftCols(KHID);
		USigma.second = mv.second.head(KHID);
	}
	return USigma;
}

pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseVector<double> > SVD_asymNystrom_sparse(SparseMatrix<double> X){
	// result.first.first is U, result.first.second is V, result.second is L.
	pair< SparseMatrix<double>, SparseVector<double> > V_L = SVD_symNystrom_sparse(X.transpose()*X);
	VectorXd L_inv_vec=pinv_vector((VectorXd) V_L.second);
	L_inv_vec = L_inv_vec.cwiseSqrt();
	MatrixXd L_inv = L_inv_vec.asDiagonal();
	SparseMatrix<double> L_inv_s = L_inv.sparseView();
	SparseMatrix<double> U = X * (V_L.first) * L_inv_s;

	pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseVector<double> > UV_L;
	UV_L.first.first = U;
	UV_L.first.second = V_L.first;
	UV_L.second = V_L.second.cwiseSqrt();
	return UV_L;
}

pair< SparseMatrix<double>, SparseVector<double> > SVD_Nystrom_columnSpace_sparse(SparseMatrix<double> A, SparseMatrix<double> B){
	// right singular vectors of SVD of A' * B, where A \in R^(n * d_A), B \in R^(n * d_B);
	double norm_denom = 1.0 / (double)A.rows(); 
	pair< SparseMatrix<double>, SparseVector<double> > USigma;
	// Generate random_mat; 
	int k_prime = 2 * KHID;
	SparseMatrix<double> random_mat = (B.cols() > 20 * KHID) ? random_embedding_mat((long)B.cols(), k_prime) : random_embedding_mat_dense((long)B.cols(), k_prime);
	SparseMatrix<double> tmp = B * random_mat; 
	SparseMatrix<double> Q = A.transpose() * tmp;  Q.makeCompressed(); Q.prune(TOLERANCE); tmp.resize(0, 0);
	Q = orthogonalize_cols(Q);
	SparseMatrix<double> tmp2 = B * Q;
	SparseMatrix<double> C = A.transpose() * tmp2;
	C = norm_denom * C;
	MatrixXd Z = (MatrixXd) C.transpose() * C;
	pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> svd_Z = latenttree_svd(Z);
	MatrixXd V = (svd_Z.first.second).leftCols(KHID);
	SparseMatrix<double> V_sparse = V.sparseView();
	VectorXd S = svd_Z.second.head(KHID);
	USigma.second = S.cwiseSqrt().sparseView(); // S.array().sqrt();
	MatrixXd diag_inv_S_sqrt = pinv_vector(S.cwiseSqrt()).asDiagonal();
	SparseMatrix<double> diag_inv_S_sqrt_s = diag_inv_S_sqrt.sparseView();
	USigma.first = C * (V_sparse)* diag_inv_S_sqrt_s;
	return USigma;

}

/////////////////////

SparseMatrix<double> pinv_Nystrom_sparse(SparseMatrix<double> X){
	pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseMatrix<double> > V_invL_Utrans = pinv_Nystrom_sparse_component(X);
	SparseMatrix<double> X_pinv; X_pinv.resize(X.cols(), X.rows());
	X_pinv = V_invL_Utrans.first.first * V_invL_Utrans.second * V_invL_Utrans.first.second;
	return X_pinv;
}

SparseMatrix<double> pinv_aNystrom_sparse(SparseMatrix<double> X){
	pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseVector<double> > UV_L = SVD_asymNystrom_sparse(X);
	// result.first.first is U, result.first.second is V, result.second is L.

	///////////
	VectorXd L_inv_vec = pinv_vector((VectorXd)UV_L.second);
	MatrixXd L_inv = L_inv_vec.asDiagonal();
	SparseMatrix<double> L_inv_s = L_inv.sparseView();
	SparseMatrix<double> result= UV_L.first.second * L_inv_s * UV_L.first.first.transpose(); 
	return result;
}


pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseMatrix<double> > pinv_Nystrom_sparse_component(SparseMatrix<double> X)
{
	// Pinv_symNystrom_sparse_component computes the components of pinv(X). 
	// result.first.first is V, result.first.second is U', result.second is the diag(inv(L));
	// The pseudo inverse of the matrix is result.first.first * result.second * result.first.second. 

	pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseVector<double> > UV_L = SVD_asymNystrom_sparse(X);
	// UV_L.first.first is U, UV_L.first.second is V, UV_L.second is L.
	VectorXd L_inv_vec = pinv_vector((VectorXd)UV_L.second);
	MatrixXd L_inv = L_inv_vec.asDiagonal();
	SparseMatrix<double> L_inv_s = L_inv.sparseView();
	pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseMatrix<double> > result;
	result.first.first = UV_L.first.second;
	result.first.second = UV_L.first.first.transpose();
	result.second = L_inv_s;
	return result;
}

////////////////////////////////////////////

std::pair<pair<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > k_svd_observabe(Eigen::SparseMatrix<double> A)
{
	pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseVector<double> > u_l = SVD_asymNystrom_sparse(A);
	Eigen::SparseMatrix<double> u = u_l.first.first;
	Eigen::SparseMatrix<double> v = u_l.first.second;
	Eigen::SparseVector<double> s = u_l.second;
	pair<pair<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > mv;
	mv.first.first.resize((int)u.rows(), KHID);
	mv.first.second.resize((int)v.rows(), KHID);
	mv.second.resize(KHID);
	mv.first.first = u.leftCols(KHID);
	mv.first.second = v.leftCols(KHID);
	mv.second = s.head(KHID);
	return mv;
}


///////////////////////////////////////////

double prod_sigvals(Node * a, Node * b)
{
	double prod_result;

		double numerator = singularvalue_prod(a, b);
		double denom1 = 1; double denom2 = 1;
		if (numerator > TOLERANCE){
			double denom1 = sqrt(fabs(singularvalue_prod(a, a)));
			double denom2 = sqrt(fabs(singularvalue_prod(b, b)));
		}
		// calculate finla results:
		prod_result = numerator / (denom1*denom2);
		prod_result = (prod_result > 1) ? 1 : prod_result;
	
	return prod_result;
}


double singularvalue_prod(Eigen::MatrixXd A)
{
	double results = 1.0;
	pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> U_L = latenttree_svd(A);
	for (long i = 0; i< A.cols(); ++i) {
		if (fabs(U_L.second(i)) > TOLERANCE)
			results = results*fabs(U_L.second(i));
	}
	return results;
}

double singularvalue_prod(Node * a, Node * b)//Eigen::SparseMatrix<double> A)
{
	double result = 1.0;
	pair< SparseMatrix<double>, SparseVector<double> > V_L = SVD_Nystrom_columnSpace_sparse(a->readsamples(), b->readsamples());
	VectorXd eigenVal = (VectorXd)V_L.second;
	for (long i = 0; i< KHID; ++i) {
		result = (fabs(eigenVal(i)) > TOLERANCE) ? result*fabs(eigenVal(i)) : result;
	}
	return result;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////       THE END !!!!!!!!!!
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
std::pair<Eigen::MatrixXd, Eigen::VectorXd> k_svd(Eigen::MatrixXd A, int k)
{
Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
Eigen::MatrixXd u = svd.matrixU();
Eigen::VectorXd s = svd.singularValues();
// cout << "singular values:\n" << s<< endl;
pair<Eigen::MatrixXd, Eigen::VectorXd> mv;
mv.first = u.leftCols(k);
mv.second = s.head(k);
return mv;
}
*/

/*
std::pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> k_svd_observabe(Eigen::MatrixXd A, int k)
{
Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
Eigen::MatrixXd u = svd.matrixU();
Eigen::MatrixXd v = svd.matrixV();
Eigen::VectorXd s = svd.singularValues();
pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> mv;
mv.first.first = u.leftCols(k);
mv.first.second = v.leftCols(k);
mv.second = s.head(k);
return mv;
}
*/


/*

pair< SparseMatrix<double>, SparseVector<double> > SVD_symNystrom_sparse(SparseMatrix<double> A, int k)
{
//  cout << "start SVD_symNystrom_sparse function" << endl;
pair< SparseMatrix<double>, SparseVector<double> > result;
//////////////////////////////////////
SparseMatrix<double> random_mat = random_embedding_mat((long)A.cols(), k);
SparseMatrix<double> C = A*random_mat;
C.makeCompressed(); C.prune(TOLERANCE);

// SparseMatrix<double> Q = orthogonalize_cols(C);	Q.makeCompressed();	Q.prune(TOLERANCE);

// QR of C
SparseQR<SparseMatrix<double>, NaturalOrdering<int> > Q_R(C);
// R
MatrixXd R = (MatrixXd)(Q_R.matrixR()).topLeftCorner(k, k);
MatrixXd R_inv = pinv_matrix(R);
SparseMatrix<double> R_inv_sparse = R_inv.sparseView();
// Q
// Anothe option which might work if sparse_svd gives errors, this might be even faster ..
SparseMatrix<double> Q_tmp = C * Q_R.colsPermutation();
SparseMatrix<double> Q = Q_tmp * R_inv_sparse;
Q.makeCompressed();
Q.prune(TOLERANCE);


// W
MatrixXd W = (MatrixXd)random_mat.transpose() * C;
MatrixXd W_sqrt = sqrt_matrix(W);
SparseMatrix<double> W_sqrt_sparse = W_sqrt.sparseView();
// getting column span u
SparseMatrix<double> u_tmp = Q * R_inv_sparse.transpose();
SparseMatrix<double> u = u_tmp * W_sqrt_sparse.transpose();
u.makeCompressed(); u.prune(TOLERANCE);
// orthogolize u
SparseQR<SparseMatrix<double>, NaturalOrdering<int> > u_Q_R(u);
SparseMatrix<double> u_Q; SparseMatrix<double> u_R_inv_sparse;
MatrixXd u_R_dense = (MatrixXd)u_Q_R.matrixR().topLeftCorner(KHID, KHID);
MatrixXd u_R_dense_inv = pinv_matrix(u_R_dense);
u_R_inv_sparse = u_R_dense_inv.sparseView();
SparseMatrix<double> u_Q_tmp = u* u_Q_R.colsPermutation();
u_Q = u_Q_tmp * u_R_inv_sparse;
u_Q.makeCompressed();
u_Q.prune(TOLERANCE);

//////
MatrixXd L = (MatrixXd)u_Q.transpose() * A * u_Q;
VectorXd L_diag = L.diagonal().cwiseAbs();
/////////////////////////////////////
// cout << "------------!!!!!!!Symmetric Nystrom:"<< L_diag << endl;

result.first = u_Q;
result.first.makeCompressed();
result.first.prune(TOLERANCE);

result.second = L_diag.sparseView();
result.second.prune(TOLERANCE);
//     cout << "end SVD_symNystrom_sparse function" << endl;
return result;
}

*/

/*
pair< SparseMatrix<double>, SparseVector<double> > SVD_symNystrom_sparse(SparseMatrix<double> A, SparseMatrix<double>B)
{

//  cout << "start SVD_symNystrom_sparse function" << endl;
pair< SparseMatrix<double>, SparseVector<double> > result;
//////////////////////////////////////
SparseMatrix<double> random_mat = random_embedding_mat((long)B.cols(), k);
SparseMatrix<double> tmp = B*random_mat;
SparseMatrix<double> C = A * tmp;
C.makeCompressed(); C.prune(TOLERANCE);
// QR of C
SparseQR<SparseMatrix<double>, NaturalOrdering<int> > Q_R(C);
// R
MatrixXd R = (MatrixXd)(Q_R.matrixR()).topLeftCorner(k, k);
MatrixXd R_inv = pinv_matrix(R);
SparseMatrix<double> R_inv_sparse = R_inv.sparseView();
// Q
// SparseMatrix<double> Q;
// First alternative


// Anothe option which might work if sparse_svd gives errors, this might be even faster ..
SparseMatrix<double> Q_tmp = C * Q_R.colsPermutation();
SparseMatrix<double> Q = Q_tmp * R_inv_sparse;
// Q.makeCompressed();
//////

Q.makeCompressed();
Q.prune(TOLERANCE);
// W
MatrixXd W = (MatrixXd)random_mat.transpose() * C;
MatrixXd W_sqrt = sqrt_matrix(W);
SparseMatrix<double> W_sqrt_sparse = W_sqrt.sparseView();
// getting column span u
SparseMatrix<double> u_tmp = Q * R_inv_sparse.transpose();
SparseMatrix<double> u = u_tmp * W_sqrt_sparse.transpose();
u.makeCompressed(); u.prune(TOLERANCE);
// orthogolize u
SparseQR<SparseMatrix<double>, NaturalOrdering<int> > u_Q_R(u);
SparseMatrix<double> u_Q; SparseMatrix<double> u_R_inv_sparse;
MatrixXd u_R_dense = (MatrixXd)u_Q_R.matrixR().topLeftCorner(KHID, KHID);
MatrixXd u_R_dense_inv = pinv_matrix(u_R_dense);
u_R_inv_sparse = u_R_dense_inv.sparseView();
SparseMatrix<double> u_Q_tmp = u* u_Q_R.colsPermutation();
u_Q = u_Q_tmp * u_R_inv_sparse;
u_Q.makeCompressed();
u_Q.prune(TOLERANCE);

//////
SparseMatrix<double> tmp_1 = u_Q.transpose() * A;
SparseMatrix<double> tmp_2 = B * u_Q;
SparseMatrix<double> tmp_1_2_sparse = tmp_1 * tmp_2;
MatrixXd L = (MatrixXd)tmp_1_2_sparse;
VectorXd L_diag = L.diagonal().cwiseAbs();
/////////////////////////////////////
// cout << "------------!!!!!!!Symmetric Nystrom:"<< L_diag << endl;

result.first = u_Q;
result.first.makeCompressed();
result.first.prune(TOLERANCE);

result.second = L_diag.sparseView();
result.second.prune(TOLERANCE);
//    cout << "end SVD_symNystrom_sparse function 2" << endl;
return result;
}
*/


/*
SparseMatrix<double> sqrt_symNystrom_sparse(SparseMatrix<double> X, int k)
{
	//  cout << "start sqrt_symNystrom_sparse function" << endl;
	//int k = min(X.cols(),X.rows());
	//  k = min(k,2*KHID);
	SparseMatrix<double> result;
	pair<SparseMatrix<double>, SparseVector<double> > my_svd;
	my_svd = SVD_symNystrom_sparse(X, k);

	SparseMatrix<double>sig;
	sig.resize(k, k);
	for (int i = 0; i < k; i++)
	{
		sig.coeffRef(i, i) = sqrt(fabs(my_svd.second.coeff(i, i)));
	}
	sig.makeCompressed();
	sig.prune(TOLERANCE);
	result = my_svd.first * sig * my_svd.first.transpose();
	result.makeCompressed();
	result.prune(TOLERANCE);
	// cout << "end sqrt_symNystrom_sparse function" << endl;
	return result;
}

SparseMatrix<double> sqrt_symNystrom_sparse(SparseMatrix<double> X, SparseMatrix<double> Y, int k)
{
	//   cout << "start sqrt_symNystrom_sparse function2" << endl;
	SparseMatrix<double> result;
	pair<SparseMatrix<double>, SparseVector<double> > my_svd;
	my_svd = SVD_symNystrom_sparse(X, Y, k);

	SparseMatrix<double>sig;
	sig.resize(k, k);
	for (int i = 0; i < k; i++)
	{
		sig.coeffRef(i, i) = sqrt(fabs(my_svd.second.coeff(i, i)));
	}
	sig.makeCompressed();
	sig.prune(TOLERANCE);
	result = my_svd.first * sig * my_svd.first.transpose();
	result.makeCompressed();
	result.prune(TOLERANCE);
	//     cout << "end sqrt_symNystrom_sparse function2" << endl;
	return result;
}
*/

/*
SparseMatrix<double> sqrt_asymNystrom_sparse(SparseMatrix<double> X)
{
int k = min(X.cols(), X.rows());
k = min(k, KHID);
SparseMatrix<double> result;
pair<SparseMatrix<double>, SparseVector<double> > my_column;
pair<SparseMatrix<double>, SparseVector<double> > my_row;
//    SparseMatrix<double> A= X*X.transpose();
//    SparseMatrix<double> B=X.transpose()*X;
my_column = SVD_symNystrom_sparse(X, X.transpose(), k);
my_row = SVD_symNystrom_sparse(X.transpose(), X, k);

SparseVector<double> sig1 = my_column.second;
SparseVector<double> sig2 = my_row.second;
SparseMatrix<double>sig;
sig.resize(k, k);
for (int i = 0; i < k; i++)
{
sig.coeffRef(i, i) = sqrt(sqrt(sqrt(fabs(sig1.coeff(i)))*sqrt(fabs(sig2.coeff(i)))));
}
sig.makeCompressed();
sig.prune(TOLERANCE);

result = my_column.first * sig * my_row.first.transpose();
result.makeCompressed();
result.prune(TOLERANCE);
return result;
}
*/