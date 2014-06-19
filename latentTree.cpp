//
//  latentTree.cpp
//  latenttree
/*******************************************************
* Copyright (C) 2014 {Furong Huang} <{furongh@uci.edu}>
*
* This file is part of {Intergrated latent tree structure and parameters learning project}.
*
* All rights reserved.
*******************************************************/
// latentTree.cpp : Defines the entry point for the console application.
//  main.cpp


#include "stdafx.h"
#define _CRT_SECURE_NO_WARNINGS
using namespace Eigen;
using namespace std;

int NVAR;
int NX;
int VOCA_SIZE;
int KHID ;
double alpha0;
double edgeD_MAX;
bool useDistance;
bool WITH_PARA;
double time_readfile, time_svd_dis, time_mst, time_mst_graph, time_rg, time_merge;  // Time taken 
clock_t TIME_start, TIME_end;


int main(int argc, const char * argv[])
{
	/*
	MatrixXd test(4,4);
	test << 1,	-1,	2,	0,
	3,	1,	-2,	0,
	2,	-2,	1,	0,
	0,	0,	0,	0;
	SparseMatrix<double> test_s = test.sparseView();
	KHID = 3;
	pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseMatrix<double> > result_pinv = pinv_Nystrom_sparse_component(test_s);
	pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseVector<double> > result_asySVD = SVD_asymNystrom_sparse(test_s);
	pair< SparseMatrix<double>, SparseVector<double> > result_sySVD = SVD_symNystrom_sparse(test_s.transpose()*test_s.transpose());
	cout << "result_asySVD.first.first : " << endl << (MatrixXd) result_pinv.first.first << endl;
	cout << "result_asySVD.first.second : " << endl << (MatrixXd)result_pinv.first.second << endl;
	cout << "result_asySVD.second: " << endl << (MatrixXd)result_pinv.second << endl;
//	cout << "pinv: " << (MatrixXd)(result_pinv.first.first* result_pinv.second * result_pinv.first.second) << endl;
	exit(0);
	*/
	/*MatrixXd test(5,4); 
	test << 0, 3, 0, 0, 0,
		22, 0, 0, 0, 17,
		7, 5, 0, 1, 0,
		0, 0, 0, 0, 0,
		0, 0, 14, 0, 8;
	cout << test << endl;
			
	MatrixXd a = MatrixXd::Random(100,4);
	MatrixXd A = a*a.transpose();
	SparseMatrix<double> A_s = A.sparseView();

	
	

	pair< SparseMatrix<double>, SparseVector<double> > result = \
		SVD_symNystrom_sparse(A_s);
	cout << "U: " << (MatrixXd) result.first << endl;
	cout << "Sigma: " << (VectorXd)result.second << endl;


	pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> result_exact = \
		latenttree_svd(A);
	cout << "U exact : " << (result_exact.first.first).leftCols(KHID) << endl;
	cout << "Sigma exact : " << result_exact.second.head(KHID) << endl;
	exit(0);*/
	
	// synthetic input arguments:
	// 9 10000 50 2 0 0.8 0 1 $(SolutionDir)\..\..\dataset\synthetic\samples.txt $(SolutionDir)\..\..\dataset\synthetic\result\sample_matlab.txt $(SolutionDir)\..\..\dataset\synthetic\result\DistanceMatrix.txt $(SolutionDir)\..\..\dataset\synthetic\result\category.txt $(SolutionDir)\..\..\dataset\synthetic\result\nodelist.txt $(SolutionDir)\..\..\dataset\synthetic\result\adjsparse.txt $(SolutionDir)\..\..\dataset\synthetic\result\neighborhood.txt $(SolutionDir)\..\..\dataset\synthetic\result\edgePot.txt
	// 3 4 3 2 0  0.8 0 1 C:\Users\t-fuhuan\Documents\latenttreecode\dataset\synthetic\samples.txt C:\Users\t-fuhuan\Documents\latenttreecode\dataset\synthetic\result\sample_matlab.txt C:\Users\t-fuhuan\Documents\latenttreecode\dataset\synthetic\result\DistanceMatrix.txt C:\Users\t-fuhuan\Documents\latenttreecode\dataset\synthetic\result\category.txt C:\Users\t-fuhuan\Documents\latenttreecode\dataset\synthetic\result\nodelist.txt C:\Users\t-fuhuan\Documents\latenttreecode\dataset\synthetic\result\adjsparse.txt C:\Users\t-fuhuan\Documents\latenttreecode\dataset\synthetic\result\neighborhood.txt C:\Users\t-fuhuan\Documents\latenttreecode\dataset\synthetic\result\edgePot.txt

	NVAR = furong_atoi(argv[1]);				// 9
	NX = furong_atoi(argv[2]);					// 10000 
	VOCA_SIZE = furong_atoi(argv[3]);			// 50
	KHID = furong_atoi(argv[4]);				// 2
	alpha0 = furong_atof(argv[5]);				// 0
	edgeD_MAX = -log(furong_atof(argv[6]));		// -log(0.80);
	useDistance = (furong_atoi(argv[7]) == 1);	// default is 0, which means calculate the distance
	WITH_PARA =   (furong_atoi(argv[8]) == 1);	// default is 1, which means we calculate the parameters as well
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	const char* DataPath = argv[9];				// "C:\\Users\\t - fuhuan\\Documents\\latenttreecode\\dataset\\synthetic\\"
	const char* FILE_G = argv[9];				// "samples.txt" C:\Users\t - fuhuan\Documents\latenttreecode\dataset\synthetic\samples.txt
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	const char* ResultPath = argv[11];			//  "C:\\Users\\t - fuhuan\\Documents\\latenttreecode\\dataset\\synthetic\\result\\"
	const char*  matlabsample_name = argv[10];	//  C:\Users\t-fuhuan\Documents\latenttreecode\dataset\synthetic\result\sample_matlab.txt
	const char*  adjMatrix_name = argv[11];		//  C:\Users\t-fuhuan\Documents\latenttreecode\dataset\synthetic\result\DistanceMatrix.txt
	const char*  category_name = argv[12];		//  C:\Users\t-fuhuan\Documents\latenttreecode\dataset\synthetic\result\category.txt
	const char*  nodelist_name = argv[13];		//  C:\Users\t-fuhuan\Documents\latenttreecode\dataset\synthetic\result\nodelist.txt
	const char*  adjSparse_name = argv[14];		//  C:\Users\t-fuhuan\Documents\latenttreecode\dataset\synthetic\result\adjsparse.txt
	const char*  neighborhood_name = argv[15];	//  C:\Users\t-fuhuan\Documents\latenttreecode\dataset\synthetic\result\neighborhood.txt
	const char*  edgepot_name = argv[16];		//  C:\Users\t-fuhuan\Documents\latenttreecode\dataset\synthetic\result\edgePot.txt
	
//	int syntheticSampleGen(string folder_name, int number_of_levels, );
	//////////////////////////////////////////////////////////////////////////////////////////
	TIME_start = clock();
	std::vector<int> umark;
	vector<Node> mynodelist;
	for (int i = 0; i<NVAR; i++)
	{	Node tmpnode(i, i, '0');// constructor for node: whose id is i and whose mark is i.
		mynodelist.push_back(tmpnode);
		umark.push_back(i);
	}
	vector<Node> *mynodelist_ptr = &mynodelist;
	/* categorical 
	*/
	read_G_categorical((char *)FILE_G, mynodelist_ptr, 0);
	/* vector 
	read_G_vec((char *)FILE_G, mynodelist_ptr, 0);
	*/

	TIME_end = clock();
	time_readfile = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("Exec Time reading matrices before preproc = %5.25e (Seconds)\n", time_readfile);
	//////////////////////////////////////////////////////////////////////////////////////////

	TIME_start = clock();
	MatrixXd adjMatrix = MatrixXd::Zero(NVAR, NVAR);
	SparseMatrix<double> adjMatrix_sparse;	adjMatrix_sparse.resize(NVAR, NVAR);

	if (useDistance){
		adjMatrix = read_G_dense((char*) adjMatrix_name, "adj_G_name", NVAR, NVAR,0);
		/*
		cout << "adjMatrix.coeff(0,0):" << adjMatrix.coeff(0, 0) << endl;
		cout << "adjMatrix.coeff(0,1):" << adjMatrix.coeff(0, 1) << endl;
		*/
	}
	else{
//		cout << "=======Begin distance calculation!=======" << endl;
		vector<int> iter_row;
		vector<int> iter_col;
		vector<int> num_coreviews;
#pragma omp parallel
		{
#pragma omp for
			for (int row_id = 0; row_id < NVAR; row_id++)
			{
				for (int col_id = 0; col_id < row_id; col_id++)
				{
#pragma omp critical
					{
						iter_row.push_back(row_id);
						iter_col.push_back(col_id);
						//                    double value_prod = prod_sigvals(&(mynodelist[row_id]), &(mynodelist[col_id]));
						double value_prod = prod_sigvals(&(mynodelist[row_id]), &(mynodelist[col_id]));

						if (value_prod < TOLERANCE)
						{
							adjMatrix(row_id, col_id) = MAXDISTANCE;
							adjMatrix(col_id, row_id) = MAXDISTANCE;
						}
						else
						{
							adjMatrix(row_id, col_id) = -log(value_prod);
							adjMatrix(col_id, row_id) = adjMatrix(row_id, col_id);
						}
					}
				}
			}
		}
		adjMatrix_sparse = adjMatrix.sparseView();
		write_sparseMat((char*)adjMatrix_name, adjMatrix_sparse);

	}

	




	adjMatrix_sparse = adjMatrix.sparseView();
	
	TIME_end = clock();
	time_svd_dis = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("Exec Time SVD distance  = %5.25e (Seconds)\n", time_svd_dis);


	//////////////////////////////////////////////////////////////////////////////////////////
	TIME_start = clock();
	int c = 0;
	std::vector<Edge> EV;    //All edges

	long long int NUM_EDGES = NVAR*(NVAR - 1);
	for (int i = 0; i < NVAR; i++) {         //All edges have cycles,
		for (int j = 0; j < i; j++) {
			if (i == j) continue;
			if (adjMatrix(i, j) > 0 && c < NUM_EDGES) {
				Edge new_edge(&mynodelist[i], &mynodelist[j], adjMatrix(i, j));
				EV.push_back(new_edge);
				c++;                    //c = NUM_EDGES after loop
			}
		}
	}
//	cout << "==================8888888888888888============================" << endl;
//	cout << "All Vertices: #" << NVAR << endl; cout << "All Edges: #" << c << endl;


	

	std::vector<Edge *> mst;   //mst
	cout << "-----------beginning of kruskal-----------------" << endl;
	//call Kruskal function
	std::sort(EV.begin(), EV.end(), compareByWeightObj);
	mst = KruskalMST(EV, mynodelist);


	TIME_end = clock();
	time_mst = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("Exec Time MST  = %5.25e (Seconds)\n", time_mst);


	/////////////////////////////////////////////////////////////////
	TIME_start = clock();
	//Build a MST graph
	
	cout << "------------------End of Kruskal: starting to build g_mst-------------" << endl;
	Graph g_mst(mst);
	//free the EV memory!!!!!!!!!!!!!!!!
	//////////////////////////////////////////////////////////////
	//split neighborhood to each RG algorithm
	vector<int> internal_nodes;
	for (int num = 0; num < g_mst.readadj().size(); num++)
	{
		if (g_mst.readadj()[num].size()>1)
		{
			internal_nodes.push_back(num);
			//  cout << "Internal Node :" << num << endl;
		}
	}
	cout << "internal_nodes.size():  " << internal_nodes.size() << endl;


	TIME_end = clock();
	time_mst_graph = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("Exec Time MST graph  = %5.25e (Seconds)\n", time_mst_graph);

	// write the mst to file
	// cout << "====================MST results===================="<<endl;
	// g_mst.displayadj_edgeD();
	// cout << "====================END of MST results===================="<<endl;



	/////////////////////////////////////////////////////////////////
	TIME_start = clock();
	cout << "=======================RG starts==========================" << endl;
	vector<Graph *> g_RG;
	for (int ind_rg = 0; ind_rg < internal_nodes.size(); ind_rg++)
	{
		cout << "start to build new neighborhood graph" << endl;
		Graph * g_rg = new Graph(g_mst.readadj()[internal_nodes[ind_rg]], &mynodelist[internal_nodes[ind_rg]]);
		cout << "end of this new neighborhood graph" << endl;
		cout << "start this RG" << endl;
		cout << "number of nodes: " << g_rg->readnum_N() << endl;
		RG(g_rg, &adjMatrix_sparse);
		cout << "end of this rg" << endl;	
		g_rg->displayadj_edgeD("C:\\Users\\leo_s_000\\1FurongHuang\\latenttreecode\\dataset\\synthetic\\result\\test_nodelist.txt", "C:\\Users\\leo_s_000\\1FurongHuang\\latenttreecode\\dataset\\synthetic\\result\\test_adjSparse.txt", "C:\\Users\\leo_s_000\\1FurongHuang\\latenttreecode\\dataset\\synthetic\\result\\test_neighborhood.txt", "C:\\Users\\leo_s_000\\1FurongHuang\\latenttreecode\\dataset\\synthetic\\result\\test_edgepot.txt");
		g_RG.push_back(g_rg);
		cout << "=====!!!!Index of RG completed:" << ind_rg << endl;
		//////////////////////////////////////////////////////////////
		// check the edgePot;
		int num_nodes_thisRG_x = g_rg->readnum_N();
		int num_nodes_thisRG_h = g_rg->readnum_H();

	}
	adjMatrix.resize(0, 0);

	TIME_end = clock();
	time_rg = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("Exec Time RG  = %5.25e (Seconds)\n", time_rg);

	/////////////////////////////////////////////////////////////////


	// merging step

	TIME_start = clock();
	cout << "===start merging step!!!====" << endl;
	Graph * g_Merged = g_RG[0];
	g_RG.erase(g_RG.begin());

	int curr_id = 0;
	while (g_RG.size()>0)
	{
		bool merged = Graph_merge(g_Merged, g_RG[curr_id]);
		if (merged == false)
		{
			curr_id++;
		}
		else{
			g_RG.erase(g_RG.begin() + curr_id);// check this!
			curr_id = 0;
		}

	}
//	cout << "The final RG forest: Number of observable nodes: " << g_Merged->readnum_N() << endl;
//	cout << "The final RG forest: Number of hidden nodes: " << g_Merged->readnum_H() << endl;

	SparseMatrix<double> category;
	category = g_Merged->estimateCategory();
	write_sparseMat((char*)category_name, category);

	TIME_end = clock();
	time_merge = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("Exec Time SVD distance  = %5.25e (Seconds)\n", time_svd_dis);
	printf("Exec Time MST  = %5.25e (Seconds)\n", time_mst);
	printf("Exec Time MST graph  = %5.25e (Seconds)\n", time_mst_graph);
	printf("Exec Time RG  = %5.25e (Seconds)\n", time_rg);
	printf("Exec Time MERGE  = %5.25e (Seconds)\n", time_merge);
	printf("Exec Time Total = %5.2e (Seconds)\n", time_svd_dis + time_mst + time_mst_graph + time_rg+ time_merge);

	//cout << "====================RG results====================" << endl;
	g_Merged->displayadj_edgeD((char*)nodelist_name, (char*)adjSparse_name, (char*)neighborhood_name, (char*)edgepot_name);
	//cout << "====================END of RG results====================" << endl;

 	return 0;
}

