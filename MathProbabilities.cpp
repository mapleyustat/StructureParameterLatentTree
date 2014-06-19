//
//  MathProbabilities.cpp
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

using namespace Eigen;
using namespace std;
//This is for Parameter estimation
// set of basic functions
Eigen::MatrixXd normc(Eigen::MatrixXd phi)
{
	for (int i = 0; i < phi.cols(); i++)
	{
		phi.col(i).normalize();
	}

	return phi;
}
Eigen::SparseMatrix<double> normc(Eigen::SparseMatrix<double> phi)
{
	MatrixXd phi_f = normc((MatrixXd)phi);

	return phi_f.sparseView();
}
////////////////////////////////////////////////////////////

Eigen::VectorXd normProbVector(VectorXd P_vec)
{
	VectorXd P_norm = P_vec;
	if (P_vec == Eigen::VectorXd::Zero(P_vec.size())){
	}
	else{
		double P_positive = 0; double P_negative = 0;

		for (int row_idx = 0; row_idx < P_vec.size(); row_idx++){
			P_positive = (P_vec(row_idx) > 0) ? (P_positive + P_vec(row_idx)) : P_positive;
			P_negative = (P_vec(row_idx) > 0) ? P_negative : (P_negative + P_vec(row_idx));
		}
		if (fabs(P_positive) < fabs(P_negative)){
			P_norm = -P_vec / fabs(P_negative);
		}
		else{
			P_norm = P_vec / fabs(P_positive);
		}

		for (int row_idx = 0; row_idx < P_vec.size(); row_idx++){
			P_norm(row_idx) = (P_norm(row_idx)<0) ? 0 : P_norm(row_idx);
		}
	}
	return P_norm;
}

Eigen::SparseVector<double> normProbVector(Eigen::SparseVector<double> P_vec)
{
	VectorXd P_dense_vec = (VectorXd)P_vec;
	Eigen::SparseVector<double> P_norm;
	if (P_dense_vec == VectorXd::Zero(P_vec.size()))
	{
		P_norm = P_vec;
	}
	else{
		double P_positive = 0; double P_negative = 0;

		for (int row_idx = 0; row_idx < P_vec.size(); row_idx++){
			P_positive = (P_vec.coeff(row_idx) > 0) ? (P_positive + P_vec.coeff(row_idx)) : P_positive;
			P_negative = (P_vec.coeff(row_idx) > 0) ? P_negative : (P_negative + P_vec.coeff(row_idx));
		}
		if (fabs(P_positive) < fabs(P_negative)){
			P_norm = -P_vec / fabs(P_negative);
		}
		else{
			P_norm = P_vec / fabs(P_positive);
		}

		for (int row_idx = 0; row_idx < P_vec.size(); row_idx++){
			P_norm.coeffRef(row_idx) = (P_norm.coeff(row_idx)<0) ? 0 : P_norm.coeff(row_idx);
		}
	}
	P_norm.prune(TOLERANCE);
	return P_norm;
}

Eigen::MatrixXd normProbMatrix(Eigen::MatrixXd P)
{
	// each column is a probability simplex
	Eigen::MatrixXd P_norm(P.rows(), P.cols());
	for (int col = 0; col < P.cols(); col++)
	{
		Eigen::VectorXd P_vec = P.col(col);
		P_norm.col(col) = normProbVector(P_vec);
	}
	return P_norm;
}

Eigen::SparseMatrix<double> normProbMatrix(Eigen::SparseMatrix<double> P)
{
	// each column is a probability simplex
	Eigen::SparseMatrix<double> P_norm;
	P_norm.resize(P.rows(), P.cols());
	for (int col = 0; col < P.cols(); col++)
	{
		//SparseVector<double> A_col_sparse = A_sparse.block(0, i, A_sparse.rows(),1);
		SparseVector<double> P_vec = P.block(0, col, P.rows(), 1);
		SparseVector<double> P_vec_norm;
		P_vec_norm.resize(P_vec.size());
		P_vec_norm = normProbVector(P_vec);
		for (int id_row = 0; id_row < P.rows(); id_row++)
		{
			P_norm.coeffRef(id_row, col) = P_vec_norm.coeff(id_row);
		}
	}
	P_norm.makeCompressed();
	P_norm.prune(TOLERANCE);
	return P_norm;
}
///////////////////////////////////////////////////
Eigen::MatrixXd Condi2Joint(Eigen::MatrixXd Condi, Eigen::VectorXd Pa)
{	// second dimension of Condi is the parent
	Eigen::MatrixXd Joint(Condi.rows(), Condi.cols());
	for (int cols = 0; cols < Condi.cols(); cols++)
	{
		Joint.col(cols) = Condi.col(cols)*Pa(cols);
	}
	return Joint;

}
Eigen::SparseMatrix<double> Condi2Joint(Eigen::SparseMatrix<double> Condi, Eigen::SparseVector<double> Pa)
{	// second dimension of Condi is the parent
	Eigen::SparseMatrix<double> Joint;
	Joint.resize(Condi.rows(), Condi.cols());

	for (int cols = 0; cols < Condi.cols(); cols++)
	{
		Eigen::SparseVector<double> tmp_vec = Condi.block(0, cols, Condi.rows(), 1)*Pa.coeff(cols);
		for (int id_rows = 0; id_rows < tmp_vec.size(); id_rows++)
		{
			Joint.coeffRef(id_rows, cols) = tmp_vec.coeff(id_rows);
		}

	}
	Joint.prune(TOLERANCE);
	return Joint;

}
///////////////////////////////////////////////////
Eigen::MatrixXd joint2conditional(Eigen::MatrixXd edgePot)// assuming parent is the second dimension in edgepot
{	// second dimension of edgePot is the parent
	Eigen::MatrixXd Conditional(edgePot.rows(), edgePot.cols());
	Eigen::VectorXd Parent_Marginal(edgePot.cols());
	Parent_Marginal = edgePot.colwise().sum();
	Parent_Marginal = normProbVector(Parent_Marginal);
	for (int col = 0; col < edgePot.cols(); col++)
	{
		if (Parent_Marginal(col)> TOLERANCE) Conditional.col(col) = edgePot.col(col) / (Parent_Marginal(col));
		else Conditional.col(col) = Eigen::VectorXd::Zero(edgePot.rows());
	}

	return Conditional;
}
/////////////////
Eigen::SparseMatrix<double> joint2conditional(Eigen::SparseMatrix<double> edgePot)// pa is the second dimension
{	// second dimension of edgePot is the parent
	Eigen::SparseMatrix<double> Conditional;
	Conditional.resize(edgePot.rows(), edgePot.cols());

	Eigen::SparseVector<double> Parent_Marginal;
	Parent_Marginal.resize(edgePot.cols());
	for (int id_col = 0; id_col < edgePot.cols(); id_col++)
	{
		Eigen::SparseVector<double> tmp_vec = edgePot.block(0, id_col, edgePot.rows(), 1);
		Parent_Marginal.coeffRef(id_col) = tmp_vec.sum();
		if (Parent_Marginal.coeff(id_col)>TOLERANCE)
			for (int id_row = 0; id_row < edgePot.rows(); id_row++)
			{
				Conditional.coeffRef(id_row, id_col) = edgePot.coeff(id_row, id_col) / Parent_Marginal.coeff(id_col);
			}
	}
	Conditional.makeCompressed();
	Conditional.prune(TOLERANCE);
	return Conditional;
}
/////////////////////////////////////////////////
Eigen::MatrixXd condi2condi(Eigen::MatrixXd p_x_h, Eigen::VectorXd p_h)
{	// second dimension of p_x_h is the parent
	Eigen::MatrixXd p_x_h_joint = Condi2Joint(p_x_h, p_h); // child*parent, parent.
	Eigen::MatrixXd p_h_x_joint = p_x_h_joint.transpose();
	Eigen::MatrixXd p_h_x = joint2conditional(p_h_x_joint);
	return p_h_x;
}
Eigen::SparseMatrix<double> condi2condi(Eigen::SparseMatrix<double> p_x_h, Eigen::SparseVector<double> p_h)
{	// second dimension of p_x_h is the parent
	Eigen::SparseMatrix<double> p_x_h_joint = Condi2Joint(p_x_h, p_h); // child*parent, parent.
	Eigen::SparseMatrix<double> p_h_x_joint = p_x_h_joint.transpose();
	Eigen::SparseMatrix<double> p_h_x = joint2conditional(p_h_x_joint);
	p_h_x.prune(TOLERANCE);
	return p_h_x;
}