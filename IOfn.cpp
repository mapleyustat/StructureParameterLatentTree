//
//  IOfn.cpp
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
extern int NX;
extern int NVAR;
extern int VOCA_SIZE;

using namespace Eigen;
using namespace std;

// returns a vector of length number of observable nodes
void read_G_categorical(char* file_name, vector<Node> * mynodelist_ptr, int start)
{
	// file format: rows are samples, columns are categories, we expand the categorical variables to be basis vectors. 
	// For example, if VOCA_SIZE=3, my category is 0, then the true sample is [1 0 0]; my category is 1, the true sample is [0 1 0]; max(my category is VOCA_SIZE-1).
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//	node[0].sample[0].category			node[1].sample[0].category ...			node[NVAR].sample[0].category
	//	node[0].sample[1].category			node[1].sample[1].category ...			node[NVAR].sample[1].category
	//	...
	//	node[0].sample[NX-1].category		node[1].sample[NX-1].category ...		node[NVAR].sample[NX-1].category
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	FILE* file_ptr = fopen(file_name, "r"); // opening G_name
	if (file_ptr == NULL) // exception handling if reading G_name fails
	{
		printf("reading samples failed\n"); 
		exit(1);
	}
	// read files 
	printf("reading file\n"); 
	double ** samples = zeros(NVAR,NX);
	for (int nx = 0; nx < NX; ++nx){
		for (int nvar = 0; nvar < NVAR; ++nvar){
			double val;
			fscanf(file_ptr, "%lf", &val);
			samples[nvar][nx] = val;
		}
	}
	// load samples for each variable
	printf("setting samples\n");
	for (int nvar = 0; nvar < NVAR; ++nvar){
		Eigen::SparseMatrix<double> G_mat; G_mat.resize(NX, VOCA_SIZE); G_mat.makeCompressed();
		vector<Triplet<double> > triplets_sparse;
		for (int nx = 0; nx < NX; ++nx){
			triplets_sparse.push_back(Triplet<double>(nx,(int) samples[nvar][nx] - start, 1.0));
			
		}
		G_mat.setFromTriplets(triplets_sparse.begin(), triplets_sparse.end());
		G_mat.prune(TOLERANCE);
		(*mynodelist_ptr)[nvar].setsamples(G_mat);
	}

}
void read_G_vec(char *file_name, vector<Node> *mynodelist_ptr, int start)
{
	// file format: Nodes are seperated by -1
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//	node[0].sample[0].value(0)			node[0].sample[0].value(1) ...			node[0].sample[0].value(VOCA_SIZE)
	//	node[0].sample[1].value(0)			node[0].sample[1].value(1) ...			node[0].sample[1].value(VOCA_SIZE)
	//	...
	//	node[0].sample[NX-1].value(0)		node[0].sample[NX-1].value(1) ...		node[0].sample[NX-1].value(VOCA_SIZE)
	//	-1
	//	node[1].sample[0].value(0)			node[1].sample[0].value(1) ...			node[1].sample[0].value(VOCA_SIZE)
	//	node[1].sample[1].value(0)			node[1].sample[1].value(1) ...			node[1].sample[1].value(VOCA_SIZE)
	//	...
	//	node[1].sample[NX-1].value(0)		node[1].sample[NX-1].value(1) ...		node[1].sample[NX-1].value(VOCA_SIZE)
	//	-1
	//	...
	//	...
	//	node[NAR-1].sample[NX-1].value(0)	node[NAR-1].sample[NX-1].value(1) ...	node[NAR-1].sample[NX-1].value(VOCA_SIZE)
	//	-1
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	Eigen::SparseMatrix<double> G_mat;
	G_mat.resize(NX, VOCA_SIZE);
	G_mat.makeCompressed();
	vector<Triplet<double> > triplets_sparse;
	// number of samples \times number of states
	double r_idx, c_idx, val;// row and column indices - matlab style

	FILE* file_ptr = fopen(file_name, "r"); // opening G_name
	if (file_ptr == NULL) // exception handling if reading G_name fails
	{
		printf("reading samples failed\n"); fflush(stdout);
		exit(1);
	}

	printf("reading file\n"); 
	int node_index = 0;
	while (!feof(file_ptr)) // reading G_name
	{
		fscanf(file_ptr, "%lf", &r_idx);
		if (r_idx == -1) // the samples of 2 nodes are assumed to be separated by a line of -1 in the input file
		{
			G_mat.setFromTriplets(triplets_sparse.begin(), triplets_sparse.end());	G_mat.prune(TOLERANCE);
			(*mynodelist_ptr)[node_index].setsamples(G_mat);
			node_index++;
			G_mat.resize(NX, VOCA_SIZE); G_mat.setZero(); G_mat.makeCompressed();// reinitialize
			triplets_sparse.clear();//reinitialize vector<Triplet<double> >
			continue;
		}
		
		fscanf(file_ptr, "%lf", &c_idx); // before it was, NROWS = (NA or NB or NC) (y-axis), NCOLS = NX (x-axis); now it is changed
		
		fscanf(file_ptr, "%lf", &val);
		triplets_sparse.push_back(Triplet<double>((int)(r_idx - start), (int)(c_idx - start), val));
	}
	fclose(file_ptr);
}
// returns a vector of length number of observable nodes
Eigen::SparseMatrix<double> read_G_vec_scalor(char *file_name, vector<Node> *mynodelist_ptr, int start)
{
	//FIXME:  We might not need this.  Topic modeling can be binned into say 5 bins. Then each word becomes a categorical variable,
	// which fits into the read_G_categorical framework. 
	vector<Triplet<double> > triplets_sparse;
	long trip_len = (long) 0.1 * NX * NVAR;
	triplets_sparse.reserve(trip_len); // set large enough estimate for efficiency

	Eigen::SparseMatrix<double> G_mat;
	G_mat.resize(NX, NVAR);
	G_mat.makeCompressed();
	double r_idx, c_idx, val; // row and column indices 
	FILE* file_ptr = fopen(file_name, "r"); // opening G_name
	if (file_ptr == NULL) // exception handling if reading G_name fails
	{
		printf("reading samples failed\n"); 
		exit(1);
	}
	printf("reading file\n");
	int node_index = 0;
	while (!feof(file_ptr)) // reading G_name
	{
		fscanf(file_ptr, "%lf", &r_idx);
		if (r_idx == -1)
		{
			node_index++;
			continue;
		}
		fscanf(file_ptr, "%lf", &c_idx);
		fscanf(file_ptr, "%lf", &val);
		triplets_sparse.push_back(Triplet<double>((int)(r_idx - start), (int)node_index, val));
	}
	fclose(file_ptr);
	cout << "successfully get the triplets! " << endl;
	G_mat.setFromTriplets(triplets_sparse.begin(), triplets_sparse.end()); G_mat.prune(TOLERANCE);
	cout << "successfully get the G_mat!" << endl;
	return G_mat;
}
// function to read the adjacency submatrices from file (when stored in two-column or three-column sparse matrix format with doubleing point entries); 
// note: this reads G_XA
Eigen::MatrixXd read_G_dense(char *file_name, char *G_name, int N1, int N2, int start) // input file name, matrix name, pointer to matrix buffer, (NA or NB or NC)
{
	Eigen::MatrixXd G_mat(N1, N2);
	double r_idx, c_idx, val; // row and column indices - matlab style
	printf("reading %s\n", G_name); fflush(stdout);
	cout << file_name << endl;

	FILE* file_ptr = fopen(file_name, "r"); // opening G_name
	if (file_ptr == NULL) // exception handling if reading G_name fails
	{
		printf("reading %s adjacency submatrix failed\n", G_name);
		exit(1);
	}
	while (!feof(file_ptr)) // reading G_name
	{
		fscanf(file_ptr, "%lf", &r_idx); // note: since we need (NA or NB or NC) \times NX, we read the column index first, then usual column-major
		fscanf(file_ptr, "%lf", &c_idx); // now, NROWS = (NA or NB or NC) (y-axis), NCOLS = NX (x-axis)
		fscanf(file_ptr, "%lf", &val);
		G_mat((int)(r_idx - start), (int)(c_idx - start)) = val;
	}
	fclose(file_ptr);
	return G_mat;
}


///////////////////////////////////////////////////////////////////////////////////
void write_vector(char *file_name, vector<int> vector_write) // input arguments: file name string and vector<int> to be written into file
{
	ofstream fout(file_name, ios::binary); // open file in binary mode
	for (vector<int>::iterator it = vector_write.begin(); it != vector_write.end(); it++) // using iterator
		fout << *it << endl;
	fout.close();
}

int write_sparseMat(char *filename, SparseMatrix<double> mat)
{
	fstream f(filename, ios::out);
	for (long k = 0; k<mat.outerSize(); ++k)
		for (SparseMatrix<double>::InnerIterator it(mat, k); it; ++it)
		{
			f << it.row()  << "\t" << it.col() << "\t" << it.value() << endl;
		}
	f.close();
	return 0;
}

/*
Eigen::SparseMatrix<double> G_sparse(vector<Edge>& EV, int N) // input: file name, adjacent matrix name, NA/NB/NC, output: sparse matrix
{
	//    printf("reading %s\n", G_name); fflush(stdout);
	Eigen::SparseMatrix<double> G_mat(N, N); // NX \times (NA or NB or NC)
	G_mat.makeCompressed();
	//    double r_idx, c_idx; // row and column indices - matlab style
	for (int i = 0; i < EV.size(); i++)
	{
		G_mat.coeffRef(EV[i].readv1()->readi(), EV[i].readv2()->readi()) = EV[i].readw(); // this is now modified in r and c idx; reads in weighted also;
		G_mat.coeffRef(EV[i].readv2()->readi(), EV[i].readv1()->readi()) = EV[i].readw();
	}
	return G_mat;
}
*/

/*
double** generateRandomGraph(long long int &count, int N) {         //count = 0, after calling this function count = 2 * actual # edges (cycles exist)
	double** adjMatrix = createAdjacencyMatrix(N);

	srand((unsigned int)time(NULL));
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (i == j) continue;
			adjMatrix[i][j] = ((rand() % MAX_WEIGHT) + 2.1)*1.2;
			adjMatrix[j][i] = adjMatrix[i][j];

			if (adjMatrix[i][j] > 0)
				count++;
		}
	}

	return adjMatrix;
}
*/


/*
double** createAdjacencyMatrix(int N) {
double** adjMatrix = new double*[N];

for (int i = 0; i < N; i++)
adjMatrix[i] = new double[N];

// Initializes all nodes to ZERO 
for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++)
		adjMatrix[i][j] = 0.0;

return adjMatrix;
}
*/

/*
void displayAdjacencyMatrix(int** adjMatrix, int N) {
	cout << endl << "Adjacency Matrix [" << endl;
	for (int i = 0; i < N; i++) {
		cout << "\t{";
		for (int j = 0; j < N; j++) {
			cout << " " << adjMatrix[i][j] << " ";
		}
		cout << "}" << endl;
	}
	cout << "]" << endl;
}
*/

/*
void displayAdjacencyMatrix(double** adjMatrix, int N) {
	cout << endl << "Adjacency Matrix [" << endl;
	for (int i = 0; i < N; i++) {
		cout << "\t{";
		for (int j = 0; j < N; j++) {
			cout << " " << adjMatrix[i][j] << " ";
		}
		cout << "}" << endl;
	}
	cout << "]" << endl;
}
*/

/*
vector<Edge> generateRandomEdges(int N, vector<Node*> nodelist, int &count) {         //count = 0, after calling this function count = # edges
//void generateRandomEdges(int N, vector<Node*> nodelist, int &count, vector<Edge>* EdgeList) {         //count = 0, after calling this function count = # edges
srand((unsigned int)time(NULL));
int user_defined_edges = 0;
double randomN;       //simulated SVD result for the distance between i and j
double threshold=4;
vector<Edge> EdgeList;
// cout << "nodelist size = " << nodelist.size()<<endl;

for (int i = 0; i < N; i++) {
for (int j = 0; j < i; j++) {

if (user_defined_edges != 0){
if (count >= user_defined_edges * NVAR) break;
randomN = ((rand() % (MAX_WEIGHT * 1000)) / 1000.0 + 1.01)*1.02;
if (randomN <= threshold) {
Edge new_edge(nodelist[i], nodelist[j], randomN);
// cout << "i -> " << nodelist[i]->readi() <<endl;
// cout << "j -> " << nodelist[j]->readi() <<endl;
EdgeList.push_back(new_edge);
count++;
}
}

else{
randomN = ((rand() % (MAX_WEIGHT * 1000)) / 1000.0 + 1.01)*1.02;
Edge new_edge(nodelist[i], nodelist[j], randomN);
EdgeList.push_back(new_edge);
count++;
}

}
}
return EdgeList;
}
*/