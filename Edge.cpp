
//
//  Edge.cpp
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
Edge::Edge(Node* a, Node* b, double w)
{
	//    cout << "Edge constructor is running... "<< endl;

	v1 = a;
	v2 = b;
	weight = w;

}

// destructor
Edge::~Edge()
{
	//    cout << "Edge destructor is running... "<<endl;
}
///////////////
// display functions

Node* Edge::readv1() const{
	return v1;
}

Node* Edge::readv2() const{
	return v2;
}

Eigen::SparseMatrix<double> Edge::readedgepot() const{
	return edgepot;
}

double Edge::readw() const{
	return weight;
}
///////////////////////////////////////////////////////////////////////////////////

// set functions

bool Edge::setv1(Node *v)//child
{
	v1 = v;
	return true;
}

bool Edge::setv2(Node *v)//parent
{
	v2 = v;
	return true;
}

bool Edge::setedgepot(Eigen::SparseMatrix<double> p)
{
	edgepot.resize(p.rows(), p.cols());
	edgepot.reserve(p.nonZeros());
	edgepot = p;
	edgepot.makeCompressed();
	return true;
}

bool Edge::setweight(double w)
{
	weight = w;
	return true;
}