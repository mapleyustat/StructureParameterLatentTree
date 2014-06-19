//
//  IOfn.h
//  latenttree
/*******************************************************
* Copyright (C) 2014 {Furong Huang} <{furongh@uci.edu}>
*
* This file is part of {Intergrated latent tree structure and parameters learning project}.
*
* All rights reserved.
*******************************************************/
#ifndef __latentTree__IOfn__
#define __latentTree__IOfn__
#include "stdafx.h"

using namespace Eigen;
using namespace std;
//////////////////////////////////////////////////////////////////////////////////////////////////////
void read_G_categorical(char* file_name, vector<Node> * mynodelist_ptr, int start);
void read_G_vec(char *file_name, vector<Node> *mynodelist_ptr, int start);
Eigen::SparseMatrix<double> read_G_vec_scalor(char *file_name, vector<Node> *mynodelist_ptr, int start);
Eigen::MatrixXd read_G_dense(char *file_name, char *G_name, int N1, int N2, int start);
/////////////////////////////////////////////////////////////////////////////////////////////////////
void write_vector(char *file_name, vector<int> vector_write);
int write_sparseMat(char *filename, SparseMatrix<double> spmat);
#endif
/*
Eigen::SparseMatrix<double> G_sparse(vector<Edge>& EV, int N);
double** generateRandomGraph(long long int &, int);
double** createAdjacencyMatrix(int N);
void displayAdjacencyMatrix(int** adjMatrix, int N);
void displayAdjacencyMatrix(double** adjMatrix, int N);
vector<Edge> generateRandomEdges(int N, vector<Node*> nodelist, int &count);
*/
