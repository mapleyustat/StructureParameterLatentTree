//
//  Graph.h
//  latenttree
/*******************************************************
* Copyright (C) 2014 {Furong Huang} <{furongh@uci.edu}>
*
* This file is part of {Intergrated latent tree structure and parameters learning project}.
*
* All rights reserved.
*******************************************************/
#ifndef __latentTree__Graph__
#define __latentTree__Graph__
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/Core"
#include "Eigen/OrderingMethods"
#include "Eigen/SparseQR"
#include "Node.h"


using namespace Eigen;
using namespace std;



class Graph
{
public:
	// constructor
	Graph(vector<Edge *> neighbors);
	//Graph(vector<Edge> neighbors);
	Graph(vector<Node *> neighbors, Node * surNode);
	// destructor
	~Graph();
	// display
	void displayadj_edgeD(char * filename1, char * filename2, char * filename3, char * filename4);

	// get values
	int  readnum_N() const; // numner of observable variables.
	int readnum_H() const; // number of hidden varaibles; dynamically update num_H;
	Node * readsur() const;
	vector<vector<Node *> > readadj() const;
	vector<vector<double> > readedgeD() const;
	vector<Node*> readnodeset() const;
	vector<vector<Eigen::SparseMatrix<double > > > readedgePot() const;
	double readedgeD_oneval(Node *a, Node *b) const;

	//
	Eigen::SparseMatrix<double> estimateCategory();


	// dynamically add Edge
	void addEdge(Edge *myedge);
	// select the shortestPath
	vector<Node *> shortestPath(Node *v, Node *w);
	void deleteEdge(Node *a, Node *b);


private:
	Node * surrogate;
	vector<Node *> nodeset;
	vector<vector<Node *> > adj;
	vector<vector<double> > edgeD;
	vector<vector<Eigen::SparseMatrix<double > > > edgePot;
	int num_N;
	int num_H;
};
#endif