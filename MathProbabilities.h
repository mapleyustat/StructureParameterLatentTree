//
//  MathProbabilities.h
//  latenttree
/*******************************************************
* Copyright (C) 2014 {Furong Huang} <{furongh@uci.edu}>
*
* This file is part of {Intergrated latent tree structure and parameters learning project}.
*
* All rights reserved.
*******************************************************/
#ifndef __latentTree__MathProbabilities__
#define __latentTree__MathProbabilities__
#include "stdafx.h"
////////////////////////////////////////////////////////////
//This is for Parameter estimation
// set of basic functions
Eigen::MatrixXd normc(Eigen::MatrixXd phi);
Eigen::SparseMatrix<double> normc(Eigen::SparseMatrix<double> phi);
Eigen::VectorXd normProbVector(Eigen::VectorXd P);
Eigen::SparseVector<double> normProbVector(Eigen::SparseVector<double> P);
Eigen::MatrixXd normProbMatrix(Eigen::MatrixXd P);
Eigen::SparseMatrix<double> normProbMatrix(Eigen::SparseMatrix<double> P);
//
Eigen::MatrixXd Condi2Joint(Eigen::MatrixXd Condi, Eigen::VectorXd Pa);
Eigen::SparseMatrix<double> Condi2Joint(Eigen::SparseMatrix<double> Condi, Eigen::SparseVector<double> Pa);
//
Eigen::MatrixXd joint2conditional(Eigen::MatrixXd edgePot);
Eigen::SparseMatrix<double> joint2conditional(Eigen::SparseMatrix<double> edgePot);
//
Eigen::MatrixXd condi2condi(Eigen::MatrixXd p_x_h, Eigen::VectorXd p_h);
Eigen::SparseMatrix<double> condi2condi(Eigen::SparseMatrix<double> p_x_h, Eigen::SparseVector<double> p_h);
#endif