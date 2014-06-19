//
//  Node.cpp
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
extern int NVAR;
extern int KHID;
extern int VOCA_SIZE;
///////////////////////////////////////////////////////////////////////////////////
typedef bool(*Node_compare)(Node const &, Node const &);

bool short_Node_is_less(Node const & a,
	Node const & b) {
	if (a.readi() < b.readi()) {
		return (true);
	}
	if (b.readi() < a.readi()) {
		return (false);
	}
	else {
		if (a.readc() < b.readc())
		{
			return (true);
		}
		else
		{
			return (false);
		}
	}

}
///////////////////////////////////////////////////////////////////////////////////

// display functions
int Node::readi() const{
	return idata;
}
char Node::readc() const{
	return cdata;
}
int Node::readindex() const{
	return index;
}
Eigen::SparseVector<double> Node::readpot() const{
	return pot;
}
Node* Node::readp() const{
	return parent;
}
int Node::readrank() const{
	return rank;
}
int Node::readmark() const{
	return mark;
}
///////////////////////////////////////////////////////////////////////////////////


// set functions
bool Node::set(int i)
{
	idata = i;
	return true;
}
bool Node::set(char c)
{
	cdata = c;

	return true;
}
bool Node::setindex(int i)
{
	index = i;
	return true;
}
bool Node::set(Eigen::SparseVector<double> p)
{
	pot.resize(p.size());
	pot.reserve(p.nonZeros());
	pot = p;
	return true;
}
bool Node::setp(Node *pa)
{
	parent = pa;
	return true;
}
bool Node::setrank(int r)
{
	rank = r;
	return true;
}
bool Node::setmark(Node *n) // this is the SetMark(Forest_Node* node)
{
	mark = n->readmark();
	return true;
}
///////////////////////////////////////////////////////////////////////////////////
// compare functions

bool Node::operator<(const Node& rhs)
{
	if (idata == rhs.readi())
		return cdata < rhs.readc();
	else
		return idata < rhs.readi();
}
bool Node::operator==(const Node& rhs)
{
	return (idata == rhs.readi() && cdata == rhs.readc());
}
//////

bool Node::makeset() // MakeSet before using unionfind
{
	mark = idata;
	return true;
}

bool Node::setsamples(Eigen::SparseMatrix<double> sample_estimated)
{
	samples.resize(sample_estimated.rows(), sample_estimated.cols());
	samples.reserve(sample_estimated.nonZeros());
	samples = sample_estimated;
	samples.makeCompressed();
	return true;
}

///////////////////////////////////////////////////////////////////////////////////

Eigen::SparseMatrix<double> Node::readsamples() const
{
	return samples;
}
///////////////////////////////////////////////////////////////////////////////////

// constructor
//Node::Node()
//{
//    //    cout << "Node constructor is running... "<< endl;
//
//    idata=0;
//    cdata='0';
//    //pot.assign(d,0.0);
//    pot=Eigen::VectorXd::Zero(VOCA_SIZE);
//    parent =NULL;
//    rank =0;
//    mark =0;
//    //    local_index=0;
//
//}
Node::Node(int i, int m, char c)
{
	//    cout << "Node construtor is running... "<< endl;

	idata = i;
	cdata = c;

	if (cdata == '0')
	{
		Eigen::VectorXd tmp = Eigen::VectorXd::Zero(VOCA_SIZE);
		pot.resize((int)tmp.size());
		pot.reserve((int)tmp.nonZeros());
		pot = tmp.sparseView();
	}
	else
	{
		Eigen::VectorXd tmp = Eigen::VectorXd::Zero(KHID);
		pot.resize((int)tmp.size());
		pot.reserve((int)tmp.nonZeros());
		pot = tmp.sparseView();
	}
	parent = NULL;
	rank = 0;
	mark = m;
	//samples = s;
}
//Node::Node(Node &n)
//{
//    idata = n.idata;
//    cdata = n.cdata;
//    pot   = n.pot;
//    parent= n.parent;
//    rank  = n.rank;
//    mark  = n.mark;
//}

// destructor
Node::~Node()
{
	//    cout << "Node destructor is running... "<<endl;
}
