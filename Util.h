//
//  Util.h
//  latenttree
/*******************************************************
* Copyright (C) 2014 {Furong Huang} <{furongh@uci.edu}>
*
* This file is part of {Intergrated latent tree structure and parameters learning project}.
*
* All rights reserved.
*******************************************************/
#ifndef __latentTree__Util__
#define __latentTree__Util__
#include "stdafx.h"
// set of alignment functions

int furong_atoi(string word);
double furong_atof(string word);
double** zeros(unsigned int r, unsigned int c);
void furongfree_matrix(double** M, unsigned int r, unsigned int c);

double pdist_pairwise(Eigen::VectorXd a, Eigen::VectorXd b);
double pdist_pairwise(SparseVector<double> a, SparseVector<double> b);
//
std::pair<vector<Eigen::MatrixXd>, Eigen::VectorXd> alignPara(std::pair<vector<Eigen::MatrixXd>, Eigen::VectorXd> para_out, Eigen::MatrixXd ref_mat);
std::pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > alignPara(std::pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > para_out, Eigen::SparseMatrix<double> ref_mat);
Eigen::SparseMatrix<double> alignTwoFamily(Node * global_ref, Node * ref_family_other, Node * ref, Node * this_family_other, Node * pa_ref, Node * pa);
//
std::pair<vector<Eigen::MatrixXd>, Eigen::VectorXd> alignPara(std::pair<vector<Eigen::MatrixXd>, Eigen::VectorXd> para_out, std::pair<vector<Eigen::MatrixXd>, Eigen::VectorXd> para_ref);
std::pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > alignPara(std::pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > para_out, std::pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > para_ref);

// set of concatenation functions
Eigen::VectorXd concatenation_vector(Eigen::VectorXd A, Eigen::VectorXd B);
Eigen::MatrixXd concatenation_matrix(Eigen::MatrixXd A, Eigen::MatrixXd B);
void concatenation_matrix(vector<Node*> childset, vector<Edge *> output, Node *pa, SparseMatrix<double> &AB, SparseMatrix<double> &concat_samples);


/////////// This is for RG
pair<Eigen::MatrixXd, Eigen::MatrixXd> testnode(Eigen::MatrixXd Dist, vector<Node* > nodeset);
void MakeSet(vector<Node> nodelist);
pair<vector<vector<Node*> >, vector<Node *> > FamilyUnion(Eigen::MatrixXd *Dist_ptr, vector<Node*> V, pair<Eigen::MatrixXd, Eigen::MatrixXd > *Phi_ptr);
Node* FindParent(vector<Node*> family, vector<long> *id_this_family_ptr, Eigen::MatrixXd *Dist_ptr, Eigen::MatrixXd *Phi_mean_ptr);
// set of combination between RG and Param Est
pair<vector<Edge *>, double> computeHiddenChild_g2members_inRefFamily(Eigen::MatrixXd dist, Eigen::MatrixXd phi_second, vector<Node *>  childset, vector<Node *> currNodeset, Node *pa, Node *ref);
pair<vector<Edge *>, double> computeHiddenChild_2members_inRefFamily(Eigen::MatrixXd dist, Eigen::MatrixXd phi_second, vector<Node *>  childset, vector<Node *> anotherset, vector<Node *> currNodeset, Node *pa);
//
pair<vector<Edge *>, double> computeHiddenChild_g2members(Eigen::MatrixXd dist, Eigen::MatrixXd phi_second, vector<Node *>  childset, vector<Node *> currNodeset, Node *pa, Node *ref, vector<Node *> ref_childset, Node *glob_ref, Node * pa_ref);
pair<vector<Edge *>, double> computeHiddenChild_2members(Eigen::MatrixXd dist, Eigen::MatrixXd phi_second, vector<Node *>  childset, vector<Node *> currNodeset, Node *pa, vector<Node *> ref_childset, Node *glob_ref, Node * pa_ref);
//////////////////

Eigen::MatrixXd  computeNewDistance(Eigen::MatrixXd Dist, vector<vector<Node *> > adjNewOld, vector<Node *> currNodeset, vector<Node *>  nextNodeset, vector<double> edge_distance_sum_set);
//
void RG(Graph *g, SparseMatrix<double> *AJM);
//////////////////merging graph
bool Graph_merge(Graph * g_merged_old, Graph * g_rg_new);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// This is for MST
Node* Find(Node* node);
void Union(Node* node1, Node* node2);
bool compareByWeight(const Edge &a, const Edge &b);
///////////////////Boruvka
bool compareByWeightObj(const Edge &a, const Edge &b);
bool compareByWeightPtr(const Edge* a, const Edge* b);
#endif

