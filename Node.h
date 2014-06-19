//
//  Node.h
//  latenttree
/*******************************************************
* Copyright (C) 2014 {Furong Huang} <{furongh@uci.edu}>
*
* This file is part of {Intergrated latent tree structure and parameters learning project}.
*
* All rights reserved.
*******************************************************/
#ifndef __latentTree__Node__
#define __latentTree__Node__
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/Core"
#include "Eigen/OrderingMethods"
#include "Eigen/SparseQR"

using namespace Eigen;
using namespace std;

class Node
{
public:
	// a few constructors
	Node();
	//    Node(Node &n);
	Node(int i, int m = -1, char c = '0');
	~Node();
	// display
	int readi() const;
	char readc() const;
	int readindex() const;
	Eigen::SparseVector<double> readpot() const;
	//
	Node *readp() const;
	int readrank() const;
	int readmark() const;
	Eigen::SparseMatrix<double> readsamples() const;
	//    int readlocal_index() const;//
	// set
	bool set(int i);
	bool set(char c);
	bool setindex(int i);
	bool set(Eigen::SparseVector<double> p);
	//
	bool setp(Node *pa);
	bool setrank(int r);
	bool setmark(Node *n);
	bool makeset();
	bool setsamples(Eigen::SparseMatrix<double> sample_estimated);
	//    bool setlocal_index(int l_i);//
	bool operator<(const Node& rhs);
	bool operator==(const Node& rhs);

private:
	// members
	int idata;
	char cdata;
	int index;
	Eigen::SparseVector<double> pot;
	// members for union find
	int rank;
	int mark;
	//    int local_index;
	Node *parent;
	Eigen::SparseMatrix<double> samples;

};
bool short_Node_is_less(Node const & a, Node const & b);
#endif
