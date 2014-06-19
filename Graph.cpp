//
//  Graph.cpp
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
extern int VOCA_SIZE;
extern int KHID;
extern bool WITH_PARA;
using namespace Eigen;
using namespace std;
typedef bool(*Node_compare)(Node const &, Node const &);

bool short_Node_is_less_(Node const & a,
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
//Graph
//////////////////////////////////////////////////////////////////////////////////////////////////////////
// constructor
Graph::Graph(vector<Edge *> neighbors) // this is only used for mst
{
	//    cout <<"Graph constructor for MST... "<< endl;
	surrogate = NULL;
	//nodeset =NULL;
	set<Node *> my_tmp_nodeset;
	for (int i = 0; i < neighbors.size(); i++)
	{
		my_tmp_nodeset.insert(neighbors[i]->readv1());
		my_tmp_nodeset.insert(neighbors[i]->readv2());
	}
	for (int i = 0; i < my_tmp_nodeset.size(); i++)
	{
		vector<Node * > tmpadj;
		adj.push_back(tmpadj);
	}

	for (int i = 0; i<neighbors.size(); i++)
	{
		adj[neighbors[i]->readv1()->readi()].push_back(neighbors[i]->readv2());// push node * to the neighbor of node v1;
		adj[neighbors[i]->readv2()->readi()].push_back(neighbors[i]->readv1());// push node * to the neighbor of node v2;
	}
	num_N = (int)adj.size();

}

// constructor
Graph::Graph(vector<Node *> neighbors, Node * surNode) // this is not used for mst;
{
	//    cout <<"Graph constructor for Local latent tree... "<< endl;
	// initialize my surrogate
	surrogate = surNode;
	// initialize my nodeset
	nodeset.push_back(surrogate);
	for (int i = 0; i<neighbors.size(); i++)
	{
		nodeset.push_back(neighbors[i]);
	}
	// I don't intialize the adj since we are going to learn the local latent tree structure
	num_N = (int)nodeset.size();
	num_H = 0;
	// initialize adj with empty
	vector<Node *> tmp_adj_row;
	vector<double> tmp_edgeD_row;
	vector<Eigen::SparseMatrix<double > > tmp_edgePot_row;
	for (int i = 0; i < num_N; i++)
	{
		adj.push_back(tmp_adj_row);
		edgeD.push_back(tmp_edgeD_row);
		edgePot.push_back(tmp_edgePot_row);
	}

}

// destructor
Graph::~Graph()
{
	//    cout << "Graph destructor is running... "<< endl;
}
///////////////////////////////////////////////////////////////////////////////////
// display adj


void Graph::displayadj_edgeD(char * filename1, char * filename2, char * filename3, char * filename4)
{
//	cout << "a list of nodeset is written:" << endl;

	// first output a list of vector<Node *> nodeset; mark observable with [1 : num_N], and hidden with [num_N+1 : num_N+num_H]
	int index_obs = 0;
	int index_hid = num_N;
	vector<int> index_nodeset;
	fstream f1(filename1, ios::out);
	for (int i = 0; i<nodeset.size(); i++){
		if (nodeset[i]->readc() == '0')
		{
			f1 << index_obs << endl;
			//	cout << index_obs << endl;
			nodeset[i]->setindex(index_obs);
			index_obs++;
		}
		else
		{
			f1 << index_hid << endl;
			//	cout << index_hid << endl;
			nodeset[i]->setindex(index_hid);
			index_hid++;
		}

	}
	f1.close();
//	std::cout << "End of nodeset listing." << endl;


	//std::find(vector.begin(), vector.end(), item)!=vector.end()
//	std::cout << "begin of sparse adjcency matrix" << endl;
	fstream f2(filename2, ios::out);
	fstream f4(filename4, ios::out);
	int del = -1;

	
	for (int i = 0; i < adj.size(); i++){
		int tmpindex = nodeset[i]->readindex();// << "\t" <<;
		for (int j = 0; j < adj[i].size(); j++){
			f2 << tmpindex << "\t" << adj[i][j]->readindex()  << "\t" << edgeD[i][j] << endl;
			//      cout << tmpindex << "\t" << adj[i][j]->readindex()+1<< endl;
			if (WITH_PARA)
			{
				if (adj[i][j]->readc() != '0'){
					SparseMatrix<double> mat = edgePot[i][j];
					for (long k = 0; k < mat.outerSize(); ++k)
						for (SparseMatrix<double>::InnerIterator it(mat, k); it; ++it){
							f4 << it.row()  << "\t" << it.col()  << "\t" << it.value() << endl;
						}
				}

				f4 << del << "\t" << del << "\t" << del << endl;
			}
		}
	}
	f2.close();
	f4.close();
//	cout << "end of sparse adjcency matrix" << endl;

	fstream f3(filename3, ios::out);
	for (int i = 0; i<adj.size(); i++){
		f3 << "(" << nodeset[i]->readi() << "," << nodeset[i]->readc() << ")'s neighbors:" << endl;
		//        cout << "(" <<nodeset[i]->readi() << "," << nodeset[i]->readc()<< ")'s neighbors:" << endl;
		for (int j = 0; j<adj[i].size(); j++)
		{
			f3 << "(" << adj[i][j]->readi() << "," << adj[i][j]->readc() << "(" << edgeD[i][j] << "))\t";
			//            cout<<"(" <<adj[i][j]->readi() << "," << adj[i][j]->readc()<< "("<< edgeD[i][j]<<"))\t";
		}
		f3 << endl;
		//        cout<<endl;
	}
	f3.close();
}

// void Graph::displayadj_edgeD()
// {
//     for(int i=0;i<adj.size();i++){
//       // if (nodeset[i]->readc()=='0')
//       // 	cout << "V" << nodeset[i]->readi()<<":\t";
//       // else
//       // 	cout << "H" << nodeset[i]->readi()<<"_"<<nodeset[i]->readc()<<":\t";
//       //        cout << "(" <<nodeset[i]->readi() << "," << nodeset[i]->readc()<< ")'s neighbors:" << endl;
//         for(int j=0;j<adj[i].size();j++)
//         {
// 	  if (nodeset[i]->readc()=='0')
// 	    cout << "V" << nodeset[i]->readi()<<":\t";
// 	  else
// 	    cout << "H" << nodeset[i]->readi()<<"_"<<nodeset[i]->readc()<<":\t";

// 	  if (adj[i][j]->readc()=='0')
// 	    {
// 	      cout << "V" << adj[i][j]->readi()<<endl;
// 	    }
// 	  else
// 	    {
// 	      cout << "H" << adj[i][j]->readi()<<"_"<<adj[i][j]->readc()<<endl;
// 	    }
// 	  //            cout<<"(" <<adj[i][j]->readi() << "," << adj[i][j]->readc()<< "("<< edgeD[i][j]<<"))\t";
//         }
// 	//     cout<<endl;
//     }
// }
///////////////////////////////////////////////////////////////////////////////////
// get values
int  Graph::readnum_N() const
{
	return num_N;
}
int Graph::readnum_H() const
{
	return num_H;
}
Node * Graph::readsur() const
{
	return surrogate;
}
vector<vector<Node *> > Graph::readadj() const
{
	return adj;
}
vector<vector<double> > Graph::readedgeD() const
{
	return edgeD;
}
vector<Node*> Graph::readnodeset() const
{
	return nodeset;
}
vector<vector<Eigen::SparseMatrix<double > > > Graph::readedgePot() const
{
	return edgePot;
}

double Graph::readedgeD_oneval(Node *a, Node *b) const
{	// find index for Node *a;
	
	int indexa; int indexb;
	indexa = distance(nodeset.begin(), find(nodeset.begin(), nodeset.end(), a));
	indexb = distance(adj[indexa].begin(), find(adj[indexa].begin(), adj[indexa].end(), b));
	return edgeD[indexa][indexb];
}
///////////////////////////////////////
Eigen::SparseMatrix<double> Graph::estimateCategory()
{
	// int num_N is known
	set<Node *> leaves;
	vector<set<Node *> > out_cate;
	SparseMatrix<double> out_mat;
	for (int node_iter = 0; node_iter<num_N + num_H; node_iter++)
	{
		// find the leaves, for the leaves, we know that we need to get the group info, or nodes that are only connected to one hidden node
		// find the leaves between whose path, there is no hidden node
		set<Node *> curr_set;
		vector<Node *> im_nbs = adj[node_iter];
		if (nodeset[node_iter]->readc() == '0')
		{
			if (im_nbs.size()<2){ leaves.insert(nodeset[node_iter]); }
		}
		else
		{
			for (int inner_node_iter = 0; inner_node_iter<im_nbs.size(); inner_node_iter++)
			{
				vector<Node *>::iterator it = find(nodeset.begin(), nodeset.end(), im_nbs[inner_node_iter]);
				int index = (int) distance(nodeset.begin(), it);
				if (adj[index].size()<2)//leave
				{
					curr_set.insert(im_nbs[inner_node_iter]);
				}
			}
		}
		out_cate.push_back(curr_set);
	}
	// number of leaves output
	//    cout <<"number of leaves:" << leaves.size()<<endl;
	// output the matrix
	int rows = (int) out_cate.size();
	out_mat.resize(rows, num_N);
	for (int cate_iter = 0; cate_iter<rows; cate_iter++)
	{
		for (std::set<Node *>::iterator it = out_cate[cate_iter].begin(); it != out_cate[cate_iter].end(); ++it)
		{
			int ind = (*it)->readi();
			out_mat.coeffRef(cate_iter, ind) = 1;
		}
		// std::cout << ' ' << *it;
		//   for(int node_iter =0; node_iter<out_cate[cate_iter].size();node_iter++)
		// 	{
		// 	 int  ind=out_cate[cate_iter][node_iter]->readi();
		// 	  out_mat.coeffRef(cate_iter,ind)=1;
		// 	}
	}
	return out_mat;
}

///////////////////////////////////////////////////////////////////////////////////
// insert edge
void Graph::addEdge(Edge *myedge)
{
	// declare some iterators as index
	vector<Node *>::iterator it1;
	vector<Node *>::iterator it2;
	it1 = find(nodeset.begin(), nodeset.end(), myedge->readv1());
	it2 = find(nodeset.begin(), nodeset.end(), myedge->readv2());
	long index1; long index2;

	if (it1 != nodeset.end() && it2 != nodeset.end())
	{
		index1 = (long) distance(nodeset.begin(), it1);
		index2 = (long) distance(nodeset.begin(), it2);

		if (find(adj[index1].begin(), adj[index1].end(), myedge->readv2()) == adj[index1].end())
		{
			adj[index1].push_back(myedge->readv2());// Add v2 to v1’s list.
			edgeD[index1].push_back(myedge->readw());
			adj[index2].push_back(myedge->readv1());// Add v1 to v2’s list.
			edgeD[index2].push_back(myedge->readw());
			if (WITH_PARA)
			{
				edgePot[index1].push_back(myedge->readedgepot());
			//            cout << edgePot[index1][edgePot[index1].size()-1] << endl;
			//            cout << "myedge: (" << myedge->readv1()->readi()<<"," << myedge->readv1()->readc()<<") and "<<"(" << myedge->readv2()->readi()<<"," << myedge->readv2()->readc()<<")"<< ": \n"<<myedge->readedgepot()<<endl;
#ifdef joint_prob
			edgePot[index2].push_back(myedge->readedgepot().transpose());
#endif

#ifdef condi_prob
			edgePot[index2].push_back(condi2condi(myedge->readedgepot(), myedge->readv2()->readpot()));
			//            cout << "myedge->readedgepot():\n " << myedge->readedgepot() << endl;
			//            cout << "myedge: (" << myedge->readv1()->readi()<<"," << myedge->readv1()->readc()<<") and "<<"(" << myedge->readv2()->readi()<<"," << myedge->readv2()->readc()<<")"<< ": \n"<<condi2condi(myedge->readedgepot(),myedge->readv2()->readpot())<<endl;
#endif

			}
		}
	}
	else if (it1 != nodeset.end() && it2 == nodeset.end())
	{
		index1 = (long) distance(nodeset.begin(), it1);
		nodeset.push_back(myedge->readv2());
		it2 = find(nodeset.begin(), nodeset.end(), myedge->readv2());
		index2 = (long) distance(nodeset.begin(), it2);

		if (myedge->readv2()->readc() != '0') num_H++;
		else num_N++;

		vector<Node *> tmp_adj_row;
		vector<double> tmp_edgeD_row;
		vector<Eigen::SparseMatrix<double> > tmp_edgePot_row;
		adj.push_back(tmp_adj_row);
		edgeD.push_back(tmp_edgeD_row);
		edgePot.push_back(tmp_edgePot_row);

		if (find(adj[index1].begin(), adj[index1].end(), myedge->readv2()) == adj[index1].end())
		{
			adj[index1].push_back(myedge->readv2());// Add v2 to v1's list.
			edgeD[index1].push_back(myedge->readw());
			adj[index2].push_back(myedge->readv1());// Add v1 to v2's list.
			edgeD[index2].push_back(myedge->readw());
			if (WITH_PARA){
				edgePot[index1].push_back(myedge->readedgepot());
				//            cout << "myedge: (" << myedge->readv1()->readi()<<"," << myedge->readv1()->readi()<<"): \n"<<myedge->readedgepot()<<endl;
#ifdef joint_prob
				edgePot[index2].push_back(myedge->readedgepot().transpose());

#endif

#ifdef condi_prob
				edgePot[index2].push_back(condi2condi(myedge->readedgepot(), myedge->readv2()->readpot()));
				//            cout << "myedgepot:\n" << myedge->readedgepot()<< endl;
				//            cout << "end\n";
				//            cout << "myedge: (" << myedge->readv1()->readi()<<"," << myedge->readv1()->readi()<<") and "<<"(" << myedge->readv2()->readi()<<"," << myedge->readv2()->readi()<<")"<< ": \n"<<condi2condi(myedge->readedgepot(),myedge->readv2()->readpot())<<endl;
#endif
			}
		}
	}
	else if (it1 == nodeset.end() && it2 != nodeset.end())
	{
		index2 = (long) distance(nodeset.begin(), it2);
		nodeset.push_back(myedge->readv1());
		it1 = find(nodeset.begin(), nodeset.end(), myedge->readv1());
		index1 = (long) distance(nodeset.begin(), it1);
		if (myedge->readv1()->readc() != '0') num_H++;
		else    num_N++;

		vector<Node *> tmp_adj_row;
		vector<double> tmp_edgeD_row;
		vector<Eigen::SparseMatrix<double> > tmp_edgePot_row;
		adj.push_back(tmp_adj_row);
		edgeD.push_back(tmp_edgeD_row);
		edgePot.push_back(tmp_edgePot_row);

		if (find(adj[index1].begin(), adj[index1].end(), myedge->readv2()) == adj[index1].end())
		{
			adj[index1].push_back(myedge->readv2());// Add v2 to v1's list.
			edgeD[index1].push_back(myedge->readw());
			adj[index2].push_back(myedge->readv1());// Add v1 to v2's list.
			edgeD[index2].push_back(myedge->readw());
			if (WITH_PARA){
				edgePot[index1].push_back(myedge->readedgepot());
				//            cout << "myedge: (" << myedge->readv1()->readi()<<"," << myedge->readv1()->readi()<<"): \n"<<myedge->readedgepot()<<endl;
#ifdef joint_prob
				edgePot[index2].push_back(myedge->readedgepot().transpose());
#endif

#ifdef condi_prob
				edgePot[index2].push_back(condi2condi(myedge->readedgepot(), myedge->readv2()->readpot()));
				//            cout << "myedge: (" << myedge->readv1()->readi()<<"," << myedge->readv1()->readi()<<") and "<<"(" << myedge->readv2()->readi()<<"," << myedge->readv2()->readi()<<")"<< ": \n"<<condi2condi(myedge->readedgepot(),myedge->readv2()->readpot())<<endl;
#endif

			}



		}
	}
	else
	{
		nodeset.push_back(myedge->readv1());
		nodeset.push_back(myedge->readv2());
		it1 = find(nodeset.begin(), nodeset.end(), myedge->readv1());
		it2 = find(nodeset.begin(), nodeset.end(), myedge->readv2());
		index1 = (long) distance(nodeset.begin(), it1);
		index2 = (long) distance(nodeset.begin(), it2);

		if (myedge->readv1()->readc() != '0') num_H++;
		else num_N++;

		if (myedge->readv2()->readc() != '0') num_H++;
		else    num_N++;


		vector<Node *> tmp_adj_row;
		vector<double> tmp_edgeD_row;
		vector<Eigen::SparseMatrix<double> > tmp_edgePot_row;
		adj.push_back(tmp_adj_row);
		edgeD.push_back(tmp_edgeD_row);
		edgePot.push_back(tmp_edgePot_row);
		adj.push_back(tmp_adj_row);
		edgeD.push_back(tmp_edgeD_row);
		edgePot.push_back(tmp_edgePot_row);


		if (find(adj[index1].begin(), adj[index1].end(), myedge->readv2()) == adj[index1].end())
		{
			adj[index1].push_back(myedge->readv2());// Add v2 to v1's list.
			edgeD[index1].push_back(myedge->readw());
			adj[index2].push_back(myedge->readv1());// Add v1 to v2's list.
			edgeD[index2].push_back(myedge->readw());
			if (WITH_PARA){
				edgePot[index1].push_back(myedge->readedgepot());
				//            cout << "myedge: (" << myedge->readv1()->readi()<<"," << myedge->readv1()->readi()<<"): \n"<<myedge->readedgepot()<<endl;
#ifdef joint_prob
				edgePot[index2].push_back(myedge->readedgepot().transpose());
#endif

#ifdef condi_prob
				edgePot[index2].push_back(condi2condi(myedge->readedgepot(), myedge->readv2()->readpot()));
				//            cout << "myedge: (" << myedge->readv1()->readi()<<"," << myedge->readv1()->readi()<<") and "<<"(" << myedge->readv2()->readi()<<"," << myedge->readv2()->readi()<<")"<< ": \n"<<condi2condi(myedge->readedgepot(),myedge->readv2()->readpot())<<endl;
#endif
			}


		}

	}

}
////////////////////////////////////////////////////////////////////////
void Graph::deleteEdge(Node *a, Node *b)
{

	long it_a = (long) distance(nodeset.begin(), find(nodeset.begin(), nodeset.end(), a));
	long it_b = (long) distance(nodeset.begin(), find(nodeset.begin(), nodeset.end(), b));
	vector<Node *>::iterator iterator_a = find(adj[it_b].begin(), adj[it_b].end(), a);
	vector<Node *>::iterator iterator_b = find(adj[it_a].begin(), adj[it_a].end(), b);
	
	if (WITH_PARA){
		vector<SparseMatrix<double> >::iterator iterator_pot_a = edgePot[it_b].begin();
		advance(iterator_pot_a, distance(adj[it_b].begin(), iterator_a));

		vector<SparseMatrix<double> >::iterator iterator_pot_b = edgePot[it_a].begin();
		advance(iterator_pot_b, distance(adj[it_a].begin(), iterator_b));

		// erase entries
		edgePot[it_a].erase(iterator_pot_b); 
		edgePot[it_b].erase(iterator_pot_a);
	}
	vector<double>::iterator iterator_dis_a = edgeD[it_b].begin();
	vector<double>::iterator iterator_dis_b = edgeD[it_a].begin();
	advance(iterator_dis_a, distance(adj[it_b].begin(), iterator_a));
	advance(iterator_dis_b, distance(adj[it_a].begin(), iterator_b));
	adj[it_a].erase(iterator_b);
	adj[it_b].erase(iterator_a);
	edgeD[it_a].erase(iterator_dis_b);
	edgeD[it_b].erase(iterator_dis_a);
}

//////////////////////////////////////////////////
vector<Node *> Graph::shortestPath(Node *s, Node *v)
{
	/////////////////
	queue<Node *> q;   // Create a queue for BFS
	std::map<Node *, vector<Node * >, Node_compare > mymap(short_Node_is_less);

	map<Node *, vector<Node *> > path;
	long index_s, index_v;
	q.push(s);
	path[s].push_back(s);

	vector<Node*>::iterator i;

	while (!q.empty())
	{
		// Dequeue a vertex from queue and print it
		s = q.front();
		q.pop();

		index_s = (long) distance(nodeset.begin(), find(nodeset.begin(), nodeset.end(), s));
		index_v = (long) distance(nodeset.begin(), find(nodeset.begin(), nodeset.end(), v));
		for (i = adj[index_s].begin(); i != adj[index_s].end(); ++i)
		{
			if (path.find(*i) == path.end())
			{
				path[*i] = path[s];
				path[*i].push_back(*i);
			}
			if (v == *i)
				return path[v];
			else
				q.push(*i);
		}
	}

	return path[v];
}