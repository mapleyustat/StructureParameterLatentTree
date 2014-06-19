//
//  Edge.h
//  latenttree
/*******************************************************
* Copyright (C) 2014 {Furong Huang} <{furongh@uci.edu}>
*
* This file is part of {Intergrated latent tree structure and parameters learning project}.
*
* All rights reserved.
*******************************************************/
#ifndef __latentTree__Edge__
#define __latentTree__Edge__
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/Core"
#include "Eigen/OrderingMethods"
#include "Eigen/SparseQR"
#include "Node.h"
using namespace Eigen;
using namespace std;

class Edge
{
public:
	// a few constructors
	Edge(Node *a, Node *b, double w = 0.0);
	~Edge();
	// display
	Node *readv1() const;
	Node *readv2() const;
	double readw() const;
	Eigen::SparseMatrix<double> readedgepot() const;
	//set
	bool setv1(Node *v);
	bool setv2(Node *v);
	//    bool setedgepot(vector<vector<double> > p);
	bool setedgepot(Eigen::SparseMatrix<double> p);
	bool setweight(double w);

private:
	Node *v1;
	Node *v2;
	//    vector<vector<double> > edgepot;
	Eigen::SparseMatrix<double> edgepot;
	double weight;
};
#endif