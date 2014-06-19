//
//  MST.cpp
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
extern int NVAR;
using namespace Eigen;
using namespace std;
///////////////////////////////////////////////////////////////////////////////////
/////////////////MST
Node* Find(Node* node) {          //FINDING THE REPRESENTATIVE FOR EACH SET!
	Node* temp;
	Node* root = node;

	while (root->readp() != NULL)
		root = root->readp();

	/* Updates the parent pointers */
	while (node->readp() != NULL) {
		temp = node->readp();
		node->setp(root);
		node = temp;
	}

	return root;
}
///////////////////////////////////////////////////////////////////////////////////
/* Merges two nodes based on their rank */
void Union(Node* node1, Node* node2) {
	Node* root1 = Find(node1);
	Node* root2 = Find(node2);

	if (root1->readrank() > root2->readrank()) {                //WHO'S RANK IS LARGER, WHO'S DADDY. RANK IS THE NUMBER OF CHILDREN FOR THE VERTEX
		root2->setp(root1);
	}
	else if (root2->readrank() > root1->readrank()) {
		root1->setp(root2);
	}
	else {                                        //EQUAL, ROOT1'S RANK INCREMENTS AND IS DADDY
		//what if the root1 == root2? They are the same node
		if (root1->readi() != root2->readi() || root1->readc() != root2->readc())
		{
			root2->setp(root1);
			root1->setrank((root1->readrank()) + 1);
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

bool compareByWeight(const Edge &a, const Edge &b){     //comparator for std::sort into ascending order
	return a.readw() < b.readw();
}


bool compareByWeightObj(const Edge &a, const Edge &b){     //comparator for std::sort into ascending order
	return a.readw() < b.readw();
}

bool compareByWeightPtr(const Edge* a, const Edge* b){     //comparator for std::sort into ascending order
	return a->readw() < b->readw();
}


//Boruvka function, needs USER_DEFINED_EDGES = 0
vector<Edge*> BoruvkaMST(vector<Edge> EV, vector<Node> mynodelist){
	int NUM_EDGES = 0;
	//int NUM_EDGES_MST = 0;
	int c = 0;          //reset c=0
	long setcount = NVAR;
	vector<Edge*> mst;   //mst
	vector<int> umark;         //setmark
	vector<Edge*> Epass;
	//int t=0;
	int pass = 0;

	vector<int> minedgeindex;       //store the index of the min edge for each supervertex (index in EV) for each pass
	for (int i = 0; i<NVAR; i++)
	{
		umark.push_back(i);
	}

	while (setcount>1){      //each iteration here is a pass

		minedgeindex.clear();

		// int index[NVAR];
		// std::fill_n(index, NVAR, -1);    //checked, all -1

#pragma omp parallel
		{
			vector<int> minedgeindex_private;

#pragma omp for schedule(dynamic) nowait
			for (int k = 0; k < setcount; k++){  //iterate over supervertices, omp for here

				double min = 9999.0;
				int index_min = -1;
				/* Gets minimum edge for each supervertex */
				for (int i = 0; i < NVAR; i++) {
					if (mynodelist[i].readmark() == umark[k]){    //find vertices with mark k
						for (int j = 0; j < NUM_EDGES; j++) {    //check min edge for each vertex in the supervertex k
							if (EV[j].readv1()->readi() == i || EV[j].readv2()->readi() == i){
								if (Find(&mynodelist[EV[j].readv1()->readi()])->readi() != Find(&mynodelist[EV[j].readv2()->readi()])->readi()){    //easy to have err: only compare Find()->readi(), not Find()!
									if (EV[j].readw() <= min){
										min = EV[j].readw();
										index_min = j;

										//cout << "pass " << pass << ", index["<< k << "] = " << j << ", weight = " << EV[j].readw() << endl;
										//cout << "pass " << pass << ", {" << EV[j].readv1() << ", " << EV[j].readv2() << "}" << endl;

										break;  //break looping over edges for current vertex i, go to next vertex i+1
									}
								}
							}
						}
					}

				}   //end finding min disjoint-connecting edge for the supervertex with mark k

				if (index_min != -1){
					minedgeindex_private.push_back(index_min);
				}

			}       //omp for end

#pragma omp critical
			minedgeindex.insert(minedgeindex.end(), minedgeindex_private.begin(), minedgeindex_private.end());

			// if (tid == 0)
			// {
			// nthreads = omp_get_num_threads();
			// cout<<"Number of threads = "<<nthreads<<endl;
			// }

		}        //parallel end

		long NUM_EDGES_PASS =(long) minedgeindex.size();      //e is the number of edges in E, which is the min edges for each vertex
		/* Displays all min edges */
		// cout << "Min Edges: #" << NUM_EDGES_PASS << endl;

		c = 0;
		Epass.clear();
		for (int i = 0; i < NUM_EDGES_PASS; i++){
			Epass.push_back(&EV[minedgeindex[i]]);
		}

		/* sort the edges by weight to prevent when triangle or bigger cycle exists,

		mst is wrong because of the order or the union() */
		std::sort(Epass.begin(), Epass.end(), compareByWeightPtr);
		for (int i = 0; i < NUM_EDGES_PASS; i++){
			if (Find(&mynodelist[Epass[i]->readv1()->readi()])->readi() != Find(&mynodelist[Epass[i]->readv2()->readi()])->readi()) {    //only compare Find()->readi(), not Find()
				mst.push_back(Epass[i]);                  //add this edge to mst
				Union(&mynodelist[Epass[i]->readv1()->readi()], &mynodelist[Epass[i]->readv2()->readi()]);          //
				//	  t++;         //t is the edge index in mst
			}
			else{
				// cout << "Cycle!" << endl;
				c++;
			}
		}
		// cout << endl;
		cout << "# MST Edges in pass " << pass << ": " << NUM_EDGES_PASS - c << endl;
		cout << endl;

		/*SetMark for sets with the representatives (roots) */
		std::list<int> listmark;
		for (int i = 0; i<NVAR; i++){
			mynodelist[i].setmark(Find(&mynodelist[i]));
			// cout<<"parent i -> "<< mynodelist[i]->readmark()<<endl;
			listmark.push_back(mynodelist[i].readmark());
		}
		listmark.sort();
		listmark.unique();
		setcount = (long) listmark.size();

		umark.clear();
		umark.insert(umark.begin(), listmark.begin(), listmark.end());   //assign list to vector (clear first)

		// std::cout << "set (umark) contains:";
		// for (std::vector<int>::iterator it=umark.begin(); it!=umark.end(); ++it)
		// 	std::cout << ' ' << *it;
		// std::cout << '\n';


		cout << "# disjoint components: " << setcount << endl;
		cout << endl;
		pass++;
	}       //end of Boruvka while loop

	return mst;
}

///////////////
vector<Edge*> KruskalMST(vector<Edge> &EV, vector<Node> &mynodelist)
{
	int user_defined_edges = 0;
	long setcount = NVAR;
	vector<Edge*> mst;   //mst
	vector<int> umark;         //setmark
	for (long i = 0; i<NVAR; i++)
	{
		umark.push_back((int)i);
	}
	cout << "start union find !" << endl;
	if (user_defined_edges == 0)
	{
		while (setcount>1){
			//            cout<<"# USER DEFINED EDGES = "<<user_defined_edges<<endl;
			//            cout<<"Threshold = "<<THRESHOLD<<endl;
			//            cout<<"# edges = "<<EV.size()<<endl;

			for (long long j = 0; j < (long long)EV.size(); j++){
				//    		cout<<"(v1) = "<<mynodelist[EV[j].readv1()->readi()].readi()<<endl;
				//    		cout<<"(v2) = "<<mynodelist[EV[j].readv2()->readi()].readi()<<endl;
				//		cout<<"Find(v1) = "<<Find(&mynodelist[EV[j].readv1()->readi()])->readi()<<endl;
				//		cout<<"Find(v2) = "<<Find(&mynodelist[EV[j].readv2()->readi()])->readi()<<endl;
				//cout << "Find(&mynodelist[EV[j].readv1()->readi()])->readi():\n" << Find(&mynodelist[EV[j].readv1()->readi()])->readi() << endl;
				//cout << "Find(&mynodelist[EV[j].readv1()->readi()])->readi():\n" << Find(&mynodelist[EV[j].readv2()->readi()])->readi() << endl;
				//	  cout << "Iterator EV:" << j << "\t";
				if (Find(&mynodelist[EV[j].readv1()->readi()])->readi() != Find(&mynodelist[EV[j].readv2()->readi()])->readi()){

					mst.push_back(&EV[j]);
					//	    cout<<"add edge to mst;";                  //add this edge to mst
					//	  cout<<"mst.size() = "<<mst.size()<<endl;
					Union(&mynodelist[EV[j].readv1()->readi()], &mynodelist[EV[j].readv2()->readi()]);          //
					//	  cout<<"after Find(v1) = "<<Find(&mynodelist[EV[j].readv1()->readi()])->readi()<<endl;
					//	  cout<<"after Find(v2) = "<<Find(&mynodelist[EV[j].readv2()->readi()])->readi()<<endl;
				}
			}
			//	cout<<"22222222222222"<<endl;
			std::list<int> listmark;
			for (long i = 0; i<NVAR; i++){
				mynodelist[i].setmark(Find(&mynodelist[i]));
				listmark.push_back(mynodelist[i].readmark());
			}
			listmark.sort();
			listmark.unique();
			setcount = (long)listmark.size();

			umark.clear();
			umark.insert(umark.begin(), listmark.begin(), listmark.end());   //assign list to vector (clear first)

			//std::cout << "set (umark) contains:";
			//for (std::vector<int>::iterator it=umark.begin(); it!=umark.end(); ++it)
			//std::cout << ' ' << *it;
			//std::cout << '\n';


			cout << "# disjoint components: " << setcount << endl;
			cout << endl;

		}
		return mst;
	}
	//Kruskal for thresholding

	else
	{
		//    if(user_defined_edges > NUM_EDGES) user_defined_edges = NUM_EDGES;
		//    int Threshold = EV[user_defined_edges].readw();
		//      cout<<"# USER DEFINED EDGES = "<<user_defined_edges<<endl;
		//      cout<<"Threshold = "<<THRESHOLD<<endl;
		//      cout<<"# edges = "<<EV.size()<<endl;

		for (long j = 0; j < EV.size(); j++){
			// cout << "First value = " << Find(&mynodelist[EV[j].readv1()->readi()])->readi()<<endl;
			// cout << "Second value = " << Find(&mynodelist[EV[j].readv2()->readi()])->readi()<<endl;
			// cout << "-----------------------------------"<<endl;
			if (Find(&mynodelist[EV[j].readv1()->readi()])->readi() != Find(&mynodelist[EV[j].readv2()->readi()])->readi()){
				mst.push_back(&EV[j]);                  //add this edge to mst
				Union(&mynodelist[EV[j].readv1()->readi()], &mynodelist[EV[j].readv2()->readi()]);          //
				//	t++;         //t is the edge index in mst
			}
		}

		// std::list<int> listmark;
		// for(int i=0;i<NVAR;i++){
		// 	mynodelist[i]->setmark(Find(&mynodelist[i]));
		// 	listmark.push_back(mynodelist[i]->readmark());
		// }
		// listmark.sort();
		// listmark.unique();
		// setcount = listmark.size();

		// umark.clear();
		// umark.insert(umark.begin(), listmark.begin(),listmark.end());   //assign list to vector (clear first)

		// //std::cout << "set (umark) contains:";
		// //for (std::vector<int>::iterator it=umark.begin(); it!=umark.end(); ++it)
		// //std::cout << ' ' << *it;
		// //std::cout << '\n';


		cout << "# disjoint components: " << setcount << endl;
		cout << endl;
		return mst;

	}

}

