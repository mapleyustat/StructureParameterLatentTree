//
//  TensorDecom.h
//  latenttree
/*******************************************************
* Copyright (C) 2014 {Furong Huang} <{furongh@uci.edu}>
*
* This file is part of {Intergrated latent tree structure and parameters learning project}.
*
* All rights reserved.
*******************************************************/
#ifndef __latentTree__TensorDecom__
#define __latentTree__TensorDecom__
#include "stdafx.h"
using namespace std;
using namespace Eigen;
//#include "SVDLIBC/svdlib.h" // change the path
// set of whitening functions
void second_whiten(SparseMatrix<double> Gx_a, SparseMatrix<double> Gx_b, SparseMatrix<double> Gx_c, SparseMatrix<double> &W, SparseMatrix<double> &Z_B, SparseMatrix<double> &Z_C, VectorXd &mu_a, VectorXd &mu_b, VectorXd &mu_c);
void whiten_unshifted(SparseMatrix<double> Gx_a, SparseMatrix<double> Gx_b, SparseMatrix<double> Gx_c, SparseMatrix<double> &W, SparseMatrix<double> &Z_B, SparseMatrix<double> &Z_C);
// decompositon functions
pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > tensorDecom(Node * x1, Node * x2, Node * x3);

void tensorDecom_alpha0(SparseMatrix<double> D_a_mat, VectorXd D_a_mu, SparseMatrix<double> D_b_mat, VectorXd D_b_mu, SparseMatrix<double> D_c_mat, VectorXd D_c_mu, VectorXd &lambda, MatrixXd & phi_new);
void tensorDecom_alpha0_online(SparseMatrix<double> D_a_mat, VectorXd D_a_mu, SparseMatrix<double> D_b_mat, VectorXd D_b_mu, SparseMatrix<double> D_c_mat, VectorXd D_c_mu, VectorXd &lambda, MatrixXd & phi_new);

VectorXd tensor_form_main(SparseMatrix<double> D_a_mat, SparseMatrix<double> D_b_mat, SparseMatrix<double> D_c_mat, VectorXd curr_eigenvec);
VectorXd tensor_form_shift1(MatrixXd Pair_ab, MatrixXd Pair_ac, MatrixXd Pair_bc, VectorXd D_a_mu, VectorXd D_b_mu, VectorXd D_c_mu, VectorXd curr_eigenvec);
VectorXd tensor_form_shift0(VectorXd D_a_mu, VectorXd D_b_mu, VectorXd D_c_mu, VectorXd curr_eigenvec);

MatrixXd Diff_Loss(VectorXd Data_a_g, VectorXd Data_b_g, VectorXd Data_c_g, VectorXd Data_a_mu, VectorXd Data_b_mu, VectorXd Data_c_mu, MatrixXd phi, double learningrate);

VectorXd The_second_term(VectorXd Data_a_g, VectorXd Data_b_g, VectorXd Data_c_g, VectorXd Data_a_mu, VectorXd Data_b_mu, VectorXd Data_c_mu, VectorXd phi);

//Eigen::MatrixXd edgepot_observables (Node * x1, Node * x2);
Eigen::SparseMatrix<double> edgepot_observables(Node * x1, Node * x2);


#endif
