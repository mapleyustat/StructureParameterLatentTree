//
//  Util.cpp
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

extern int KHID;
extern int NX;
extern int NVAR;
extern int VOCA_SIZE;
extern double edgeD_MAX;
extern bool WITH_PARA;
using namespace Eigen;
using namespace std;

///////////////////////////////////////////////////////////////////////////////////
int furong_atoi(string word)
{
	int lol = atoi(word.c_str()); /*c_str is needed to convert string to const char*
								  previously (the function requires it)*/
	return lol;
}
double furong_atof(string word)
{
	double lol = atof(word.c_str()); /*c_str is needed to convert string to const char*
									 previously (the function requires it)*/
	return lol;
}
//////////////////////////////////////////
double** zeros(unsigned int r, unsigned int c)
{
	double** rv = (double**)malloc(r * sizeof(double*));
	assert(rv != NULL);

	for (unsigned int i = 0; i < r; ++i)
	{
		rv[i] = (double*)calloc(c, sizeof(double));
		assert(rv[i] != NULL);
	}

	return rv;
}

void furongfree_matrix(double** M, unsigned int r, unsigned int c)
{
	(void)c;

	for (unsigned int i = 0; i < r; ++i)
		free(M[i]);

	free(M);
}

// pdist_pairwise tesetd
double pdist_pairwise(Eigen::VectorXd a, Eigen::VectorXd b)
{
	Eigen::MatrixXd D(2, a.size());
	//    cout << "a.size(): " << a.size() << endl;
	//a= (Eigen::Matrix<double, 1, Eigen::Dynamic>) a;
	//b= (Eigen::Matrix<double, 1, Eigen::Dynamic>) b;
	Eigen::Matrix<double, 1, Eigen::Dynamic> RowVector1 = a;
	Eigen::Matrix<double, 1, Eigen::Dynamic> RowVector2 = b;
	//    cout << "RowVector1 :" << RowVector1 << endl;
	//    cout << "RowVector2 :" << RowVector2 << endl;
	//D.row(0)=a;
	//D.row(1)=b;
	D.block(0, 0, 1, a.size()) = RowVector1;
	D.block(1, 0, 1, b.size()) = RowVector2;
	int n = (int)D.rows();
	Eigen::VectorXd N = D.rowwise().squaredNorm();
	Eigen::MatrixXd S = N.replicate(1, n);
	S = S + (N.transpose()).replicate(n, 1);
	S.noalias() -= 2. * D * D.transpose();
	S = S.array().sqrt();
	//cout << "S :\n" << S(0,1)<< endl;
	return S(0, 1);

}

double pdist_pairwise(Eigen::SparseVector<double> a, Eigen::SparseVector<double> b)
{
	//  cout << "start pdist_pairwise for sparse : "<< endl;
	Eigen::VectorXd a_dense;
	a_dense.resize(a.size());
	a_dense = (VectorXd)a;
	a.resize(0);

	Eigen::VectorXd b_dense;
	b_dense.resize(b.size());
	b_dense = (VectorXd)b;
	b.resize(0);

	Eigen::MatrixXd D(2, a_dense.size());
	Eigen::Matrix<double, 1, Eigen::Dynamic> RowVector1 = a_dense;
	Eigen::Matrix<double, 1, Eigen::Dynamic> RowVector2 = b_dense;
	D.block(0, 0, 1, a_dense.size()) = RowVector1;
	D.block(1, 0, 1, b_dense.size()) = RowVector2;
	int n = (int)D.rows();
	Eigen::VectorXd N = D.rowwise().squaredNorm();
	Eigen::MatrixXd S = N.replicate(1, n);
	S = S + (N.transpose()).replicate(n, 1);
	S.noalias() -= 2. * D * D.transpose();
	S = S.array().sqrt();
	//cout << "S :\n" << S(0,1)<< endl;
	return S(0, 1);

}
//////////////////////////////
// alignPara tested
std::pair<vector<Eigen::MatrixXd>, Eigen::VectorXd> alignPara(std::pair<vector<Eigen::MatrixXd>, Eigen::VectorXd> para_out, Eigen::MatrixXd ref_mat)
{
	//    cout << "start alignPara , override # 1" << endl;
	int ref_id = 2;
	Eigen::MatrixXd perm_mat = para_out.first[ref_id];
	vector<int> permu_index(ref_mat.cols());
	//    cout << "ref_mat.cols():" << ref_mat.cols()<< endl;
	set<int> index_set;
	for (int i = 0; i < ref_mat.cols(); i++)
	{
		vector<double> Dis(ref_mat.cols());
		// find the column of ref_mat which matches with the i-th column of perm_mat
		for (int ii = 0; ii < ref_mat.cols(); ii++)
		{
			//	  cout << "i: " << i << ", ii:" << ii << endl;
			if (index_set.find(ii) != index_set.end())
			{
				Dis[ii] = std::numeric_limits<double>::infinity();
				//		cout << "Dis[ii]: "<< Dis[ii]<< endl;
			}
			else
			{
				Eigen::VectorXd tmp_perm = perm_mat.col(i);

				Eigen::VectorXd tmp_ref = ref_mat.col(ii);
				//Dis[ii]= pdist_pairwise(perm_mat.col(i),ref_mat.col(ii));
				Dis[ii] = pdist_pairwise(tmp_perm, tmp_ref);
				//		cout << "Dis[ii]: "<< Dis[ii] << endl;
			}
		}
		permu_index[i] = (int)(min_element(Dis.begin(), Dis.end()) - Dis.begin());
		//[~,permu_index[i]]=min(Dis);
		//	cout << "permu_index[i]: "<< permu_index[i] << endl;
		index_set.insert(permu_index[i]);
	}

	Eigen::MatrixXd PermutationMatrix = Eigen::MatrixXd::Zero(ref_mat.cols(), ref_mat.cols());
	//PermutationMatrix.Zero(ref_mat.cols(),ref_mat.cols());
	for (int rr = 0; rr< ref_mat.cols(); rr++) PermutationMatrix(rr, permu_index[rr]) = 1;
	//    cout << "start to return the result"<< endl;
	pair<vector<Eigen::MatrixXd>, Eigen::VectorXd> alligned;
	// permute all the matrix using PermutationMatrix
	alligned.first.push_back(para_out.first[0] * PermutationMatrix);
	alligned.first.push_back(para_out.first[1] * PermutationMatrix);
	alligned.first.push_back(para_out.first[2] * PermutationMatrix);
	Eigen::Matrix<double, 1, Eigen::Dynamic> RowVector = para_out.second;
	alligned.second = RowVector*PermutationMatrix;
	//    cout << "end alignPara, override # 1"<< endl;
	return alligned;
}
std::pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > alignPara(std::pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > para_out, Eigen::SparseMatrix<double> ref_mat)
{
	//  cout << "start alignPara , override # 2" << endl;
	int ref_id = 2;
	Eigen::SparseMatrix<double> perm_mat = para_out.first[ref_id];
	vector<int> permu_index(ref_mat.cols());
	//        cout << "ref_mat.rows():" << ref_mat.rows() <<  ", ref_mat.cols(): " << ref_mat.cols()<< endl;
	//        cout << "perm_mat.rows(): "<< perm_mat.rows() <<  ", perm_mat.cols() :" << perm_mat.cols() << endl;
	set<int> index_set;
	for (int i = 0; i < ref_mat.cols(); i++)
	{
		vector<double> Dis(ref_mat.cols());
		// find the column of ref_mat which matches with the i-th column of perm_mat
		for (int ii = 0; ii < ref_mat.cols(); ii++)
		{
			//	  cout << "i: " << i << ", ii: "<< ii << endl;
			if (index_set.find(ii) != index_set.end())
			{
				Dis[ii] = MAXDISTANCE;
				//	      Dis[ii]=std::numeric_limits<double>::infinity();
				//		cout << "Dis[ii]: "<< Dis[ii] << endl;
			}
			else
			{
				//                Eigen::SparseVector<double> tmp_perm;
				//                tmp_perm.resize(perm_mat.rows());
				//tmp_perm = perm_mat.col(i);
				//		cout << "start perm_mat column extraction" << endl;
				//		cout << perm_mat.block(0,i,perm_mat.rows(),1) << endl;
				SparseVector<double>  tmp_perm = perm_mat.block(0, i, perm_mat.rows(), 1);

				//               Eigen::SparseVector<double> tmp_ref;
				//                tmp_ref.resize(ref_mat.rows());
				//		cout << "start ref_mat column extraction" << endl;
				//		cout << ref_mat.block(0,ii,ref_mat.rows(),1) << endl;
				SparseVector<double>  tmp_ref = ref_mat.block(0, ii, ref_mat.rows(), 1);
				//tmp_ref = ref_mat.col(ii);
				//Dis[ii]= pdist_pairwise(perm_mat.col(i),ref_mat.col(ii));
				Dis[ii] = pdist_pairwise(tmp_perm, tmp_ref);
				//		cout << "Dis[ii]: "<< Dis[ii] << endl;
			}
		}
		permu_index[i] = (int)(min_element(Dis.begin(), Dis.end()) - Dis.begin());
		//[~,permu_index[i]]=min(Dis);
		index_set.insert(permu_index[i]);
	}

	Eigen::SparseMatrix<double> PermutationMatrix;
	PermutationMatrix.resize(ref_mat.cols(), ref_mat.cols());
	PermutationMatrix.setZero();
	//PermutationMatrix.Zero(ref_mat.cols(),ref_mat.cols());
	for (int rr = 0; rr< ref_mat.cols(); rr++) PermutationMatrix.coeffRef(rr, permu_index[rr]) = 1;

	//    cout << "permutationmatrix is out" << endl;
	pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > alligned;
	// permute all the matrix using PermutationMatrix
	alligned.first.push_back((para_out.first[0] * PermutationMatrix).pruned(TOLERANCE));
	alligned.first.push_back((para_out.first[1] * PermutationMatrix).pruned(TOLERANCE));
	alligned.first.push_back((para_out.first[2] * PermutationMatrix).pruned(TOLERANCE));
	Eigen::SparseVector<double> RowVector = para_out.second;
	
	//Eigen::Matrix<double, 1, Eigen::Dynamic> RowVector=para_out.second;
	alligned.second = (PermutationMatrix.transpose()*RowVector).pruned(TOLERANCE);
	//    cout << "end alignPara , override # 2" << endl;
	return alligned;
}

//////////////////////////////
// alignPara tested
std::pair<vector<Eigen::MatrixXd>, Eigen::VectorXd> alignPara(std::pair<vector<Eigen::MatrixXd>, Eigen::VectorXd> para_out, std::pair<vector<Eigen::MatrixXd>, Eigen::VectorXd> para_ref) // always put reference  as the third node
{
	//  cout << "start alignPara , override # 3" << endl;
	int ref_id = 2;
	Eigen::MatrixXd ref_mat = para_ref.first[ref_id];
	Eigen::MatrixXd perm_mat = para_out.first[ref_id];
	std::pair<vector<Eigen::MatrixXd>, Eigen::VectorXd> alligned = alignPara(para_out, ref_mat);
	//    cout << "end alignPara , override #3" << endl;
	return alligned;
}
std::pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > alignPara(std::pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > para_out, std::pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > para_ref) // always put reference  as the third node
{
	//  cout << "start alignPara , override # 4"<< endl;
	int ref_id = 2;
	Eigen::SparseMatrix<double> ref_mat = para_ref.first[ref_id];
	Eigen::SparseMatrix<double> perm_mat = para_out.first[ref_id];
	std::pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > alligned = alignPara(para_out, ref_mat);
	//    cout << "end alignPara , override # 4" << endl;
	return alligned;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
Eigen::SparseMatrix<double> alignTwoFamily(Node * global_ref, Node * ref_family_other, Node * ref, Node * this_family_other, Node * pa, Node * pa_ref)
{
	//  cout << "start alignTwoFamily function" << endl;
	//////////////////////////////////////////////////////////////////////////////////////////////////
	// triplet_cross_1 <glob_ref, ref_childset[0],ref>
	// triplet_cross_2 <childset[0],ref,glob_ref>

	Node * a = global_ref;
	Node * b = ref_family_other;

	Node * c_ = this_family_other;
	Node * d_ = ref;

	// triplet 1: triplet1=(a,b,d);
	// vector<Node *> triplet1;
	// triplet1.push_back(a); triplet1.push_back(b); triplet1.push_back(d_); // parent is pa_ref;
	// get parameters for A*, DE----------------- parent is pa_1
	pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > prob_trip1 = tensorDecom(a, b, d_);
#ifdef condi_prob
#endif
#ifdef joint_prob
	prob_trip1.first[0] = joint2conditional(prob_trip1.first[0]);
	prob_trip1.first[1] = joint2conditional(prob_trip1.first[1]);
	prob_trip1.first[2] = joint2conditional(prob_trip1.first[2]);
#endif

	// A is prob_trip1.first[0]<----------> A* is condi2condi(prob_trip1.first[0],pa_ref->readpot());
	// DE is prob_trip1.first[2]


	// triplet 2: triplet2=(c_,d_,a);
	//vector<Node *> triplet2;
	//triplet2.push_back(c_); triplet2.push_back(d_); triplet2.push_back(a); // parent is pa_2;
	// get parameters for CPi, DPi and PiEA*------------------- parent is pa_2
	pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > prob_trip2 = tensorDecom(c_, d_, a);
#ifdef condi_prob
#endif
#ifdef joint_prob
	prob_trip2.first[0] = joint2conditional(prob_trip2.first[0]);
	prob_trip2.first[1] = joint2conditional(prob_trip2.first[1]);
	prob_trip2.first[2] = joint2conditional(prob_trip2.first[2]);
#endif
	// CPi is prob_trip2.first[0]
	// DPi is prob_trip2.first[1]
	// <----------------->   PiEA* is condi2condi(prob_trip2.first[2],pa->readpot())


	// permutation function between groups
	// Eigen::SparseMatrix<double> Pi = sqrt_matrix((PiEA*) * pinv_mat(A*) * pinv_mat(DE) * (DPi));
	// Eigen::SparseMatrix<double> D = DPi * pinv_matrix(Pi);
	Eigen::SparseMatrix<double> Pi_square, Pi;
	Pi_square.resize(KHID, KHID); Pi.resize(KHID, KHID);
	Pi_square = (condi2condi(prob_trip2.first[2],pa->readpot())) * pinv_Nystrom_sparse(condi2condi(prob_trip1.first[0],pa_ref->readpot())) * pinv_Nystrom_sparse(prob_trip1.first[2]) * (prob_trip2.first[1]);
	/*SparseMatrix<double> Pi_square_tmp1 = condi2condi(prob_trip2.first[2], pa->readpot());
	pair<SparseMatrix<double>, SparseMatrix<double> > pair_pinv_tmp = pinv_asymNystrom_sparse(condi2condi(prob_trip1.first[0], pa_ref->readpot()));
	SparseMatrix<double> Pi_square_tmp2 = Pi_square_tmp1 *  pair_pinv_tmp.first.transpose();
	SparseMatrix<double> Pi_square_tmp3 = Pi_square_tmp2 * pair_pinv_tmp.second;
	SparseMatrix<double> Pi_square_tmp4 = Pi_square_tmp3 * pair_pinv_tmp.first * condi2condi(prob_trip1.first[0], pa_ref->readpot()).transpose();
	pair<SparseMatrix<double>, SparseMatrix<double> > pair_pinv_tmp_tmp = pinv_asymNystrom_sparse(prob_trip1.first[2]);
	SparseMatrix<double> Pi_square_tmp5 = Pi_square_tmp4 * pair_pinv_tmp_tmp.first.transpose();
	SparseMatrix<double> Pi_square_tmp6 = Pi_square_tmp5 * pair_pinv_tmp_tmp.second;
	SparseMatrix<double> Pi_square_tmp7 = Pi_square_tmp6 * pair_pinv_tmp_tmp.first * prob_trip1.first[2].transpose();
	Pi_square = Pi_square_tmp7 * prob_trip2.first[1];*/

	MatrixXd Pi_square_dense = (MatrixXd)Pi_square;
	Pi = (sqrt_matrix(Pi_square_dense)).sparseView();
	Pi_square.resize(0, 0);

	Eigen::SparseMatrix<double> D;
	D.resize(d_->readsamples().cols(), KHID);

	//pair<SparseMatrix<double>, SparseMatrix<double> > pair_pinv_pi = pinv_asymNystrom_sparse(Pi);
	//SparseMatrix<double> D_tmp1 = pair_pinv_pi.first.transpose()* pair_pinv_pi.second;
	//SparseMatrix<double> D_tmp2 = prob_trip2.first[1] * D_tmp1;
	//D = D_tmp2 * pair_pinv_pi.first * Pi.transpose();
	D = prob_trip2.first[1] * pinv_Nystrom_sparse(Pi);
	D.makeCompressed();
	D.prune(TOLERANCE);
	//D is the reference in this family

	Eigen::SparseVector<double> ref_nodepot;
	ref_nodepot.resize(prob_trip2.second.size());
	ref_nodepot = prob_trip2.second;
	ref_nodepot = Pi * ref_nodepot;
	ref_nodepot.prune(TOLERANCE);
	ref->set(ref_nodepot);
#ifdef condi_prob
#endif

#ifdef joint_prob
	D = Condi2Joint(D, ref_nodepot);
	D.makeCompressed();
	D.prune(TOLERANCE);
#endif
	//    cout << "start alignTwoFamily function" << endl;
	return D;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd concatenation_vector(Eigen::VectorXd A, Eigen::VectorXd B)
{
	Eigen::VectorXd C(A.size() + B.size());
	// C.resize(A.size()+B.size());
	C << A, B;
	return C;
}
///////////////////////////////////////////////////////////////////////////////////
Eigen::MatrixXd concatenation_matrix(Eigen::MatrixXd A, Eigen::MatrixXd B)
{
	Eigen::MatrixXd C(A.rows() + B.rows(), A.cols());
	//C.resize(A.rows()+B.rows(),A.cols());
	C << A, B;
	return C;
}
////////////////////////////////////////////////////////////////////
void concatenation_matrix(vector<Node*> childset, vector<Edge *> output, Node *pa, SparseMatrix<double> &AB, SparseMatrix<double> &concat_samples)
{
	//  cout << "start concatenation_matrix function " << endl;
	//  cout << "pa node:" << pa->readc() << endl;
	for (int inner_id = 0; inner_id < output.size(); inner_id++){
		SparseMatrix<double> curredgepot = output[inner_id]->readedgepot();
		//    cout << "i:" << inner_id << ",childnode:"<<childset[inner_id]->readc() << ", edgepot.rows():" << curredgepot.rows() << ", edgepot.cols(): " << curredgepot.cols() << endl;
	}
	//  cout << "childset.size()"<< childset.size()<< endl;
	//  cout << "output.size()"<< output.size()<< endl;
	//     AB.resize((int)(output.size())*VOCA_SIZE,KHID);
	//     concat_samples.resize((int)(output.size())*VOCA_SIZE,NX);
	//     cout << "concat_samples.rows(): "<< output.size() * VOCA_SIZE << ", concat_samples.cols(): "<< NX << endl;
	//     for (int i=0; i< output.size(); i++)
	//     {
	// #ifdef condi_prob
	//         Eigen::SparseMatrix<double> curr_edgepot;
	//         curr_edgepot.resize(output[i]->readedgepot().rows(),output[i]->readedgepot().cols());
	//         curr_edgepot = output[i]->readedgepot();
	//         for (int id_row = 0 ; id_row < curr_edgepot.rows(); id_row++)
	//         {
	//             for (int id_col =0; id_col < curr_edgepot.cols(); id_col++)
	//             {
	//                 AB.coeffRef(i*VOCA_SIZE+id_row,id_col)=curr_edgepot.coeff(id_row,id_col);
	//             }
	//         }
	// #endif
	// #ifdef joint_prob
	//         Eigen::SparseMatrix<double> curr_edgepot;
	//         curr_edgepot.resize(output[i]->readedgepot().rows(),output[i]->readedgepot().cols());
	//         curr_edgepot =joint2conditional(output[i]->readedgepot());
	//         for (int id_row = 0 ; id_row < curr_edgepot.rows(); id_row++)
	//         {
	//             for (int id_col =0; id_col < curr_edgepot.cols(); id_col++)
	//             {
	//                 AB.coeffRef(i*VOCA_SIZE+id_row,id_col)=curr_edgepot.coeff(id_row,id_col);
	//             }
	//         }
	// #endif

	//         Eigen::SparseMatrix<double> curr_sample;
	//         curr_sample.resize(childset[i]->readsamples().cols(),childset[i]->readsamples().rows());
	//         curr_sample = (childset[i]->readsamples()).transpose();
	//         for (int id_row = 0 ; id_row < curr_sample.rows(); id_row++)
	//         {
	//             for (int id_col =0; id_col < curr_sample.cols(); id_col++)
	//             {
	//                 concat_samples.coeffRef(i*VOCA_SIZE+id_row,id_col)=curr_sample.coeff(id_row,id_col);
	//             }
	//         }
	//     }

	//     AB.makeCompressed(); concat_samples.makeCompressed();
	//     AB.prune(TOLERANCE);
	//     concat_samples.prune(TOLERANCE);
	///////////////
	////Another Option: Instead of using all children, just use a random one.
	//////////////

	//    AB.resize((int)VOCA_SIZE,KHID);
	//   concat_samples.resize((int)VOCA_SIZE,NX);
	//    cout << "concat_samples.rows(): "<< VOCA_SIZE << ", concat_samples.cols(): "<< NX << endl;
	//    for (int i=0; i< output.size(); i++)
	srand((unsigned int)time(NULL));
	int i = rand() % (childset.size());
	//        cout << "childset[i] node:" << childset[i]->readc() << endl;
	// int i=2;
	//    cout << "random children ID: "<< i << endl;
	{
#ifdef condi_prob
		Eigen::SparseMatrix<double> curr_edgepot;
		//        curr_edgepot.resize(output[i]->readedgepot().rows(),output[i]->readedgepot().cols());
		curr_edgepot = output[i]->readedgepot();
		//	cout << "curr_edgepot.rows(): "<< curr_edgepot.rows() << ", curr_edgepot.cols(): "<< curr_edgepot.cols() << endl;
		AB.resize(curr_edgepot.rows(), curr_edgepot.cols());
		AB = output[i]->readedgepot();//curr_edgepot;
#endif
#ifdef joint_prob
		Eigen::SparseMatrix<double> curr_edgepot;
		curr_edgepot.resize(output[i]->readedgepot().rows(), output[i]->readedgepot().cols());
		curr_edgepot = joint2conditional(output[i]->readedgepot());
		AB = curr_edgepot;
#endif
		//        cout << "1"<< endl;
		SparseMatrix<double> aaaaa = (childset[i]->readsamples()).transpose();
		//	cout << "child's rows(): " << aaaaa.rows() << ",child's cols(): " << aaaaa.cols() << endl;
		concat_samples.resize(aaaaa.rows(), aaaaa.cols());
		concat_samples = (childset[i]->readsamples()).transpose();
		//	cout << "2"<< endl;
	}

	AB.makeCompressed(); concat_samples.makeCompressed();
	AB.prune(TOLERANCE);
	concat_samples.prune(TOLERANCE);



	///////////////////////////////////////

	Eigen::SparseMatrix<double> new_samples;
	new_samples.resize(NX, KHID);
	////    cout << "start of pair_pinv_AB " << endl;
	////    cout << "pinv AB,,,AB.rows()" << AB.rows() << ",AB.cols()" << AB.cols()<<endl;
	////    cout << "concat_samples.rows():" << concat_samples.rows() << "concat_samples.cols()"<< concat_samples.cols() << endl;
	//pair<SparseMatrix<double>, SparseMatrix<double> > pair_pinv_AB = pinv_asymNystrom_sparse(AB);
	////    cout << "end of pair_pinv_AB " << endl;
	//SparseMatrix<double> tmp_inv0 = pair_pinv_AB.first.transpose() * pair_pinv_AB.second;
	//SparseMatrix<double> tmp_inv1 = tmp_inv0 * pair_pinv_AB.first;
	////    cout << "tmp_inv0. and tmp_inv1 done" << endl;
	//// AB.transpose();
	//SparseMatrix<double> tmp_inv2 = AB.transpose() * concat_samples;
	//new_samples = (tmp_inv1*tmp_inv2).transpose();

	new_samples =(pinv_Nystrom_sparse(AB)*concat_samples).transpose();
	
	new_samples.makeCompressed();
	new_samples.prune(TOLERANCE);
	pa->setsamples(new_samples);
	//    cout << "end of concatenation_matrix function "<< endl;
}





/////////// This is for RG
pair<Eigen::MatrixXd, Eigen::MatrixXd> testnode(Eigen::MatrixXd Dist, vector<Node* > nodeset){
	int num_nodes = (int)nodeset.size();
	// cout << "num_nodes :" << num_nodes << endl;
	pair<Eigen::MatrixXd, Eigen::MatrixXd> phi;
	phi.first = Eigen::MatrixXd::Zero(num_nodes, num_nodes);
	phi.second = Eigen::MatrixXd::Zero(num_nodes, num_nodes);
	// cout << "Dist : \n " << Dist << endl;
	// cout << "num_nodes : \n " << num_nodes << endl;
	for (int i = 0; i< num_nodes; i++)
	{
		for (int j = i + 1; j < num_nodes; j++)
		{
			double sum = 0; double min = 999; double max = -999;

			//phi.first[i][j]
			for (int k = 0; k < num_nodes; k++)
			{
				if (k == i || k == j)
					continue;
				double phi_tmp = Dist(i, k) - Dist(j, k);
				if (phi_tmp < min) min = phi_tmp;
				if (phi_tmp > max) max = phi_tmp;
				sum = sum + phi_tmp;
			}
			phi.first(i, j) = max - min;              // first one is a fluctuation
			phi.first(j, i) = phi.first(i, j);              // CHECK THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			phi.second(i, j) = sum / (num_nodes - 2);   // second one is a mean
			phi.second(j, i) = -phi.second(i, j);     // CHECK THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		}
	}
	// cout << "phi.first: fluctuation :\n" << phi.first << endl;
	// cout << "phi.second: fluctuation :\n" << phi.second << endl;
	return phi;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//MakeSet before using Union-find. Set each node to be a separate set, reset rank
void MakeSet(vector<Node> nodelist){
	for (int i = 0; i < NVAR; i++){
		nodelist[i].makeset();
		nodelist[i].setrank(0);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pair<vector<vector<Node*> >, vector<Node *> > FamilyUnion(Eigen::MatrixXd *Dist_ptr, vector<Node*> V, pair<Eigen::MatrixXd, Eigen::MatrixXd > *Phi_ptr) //union all nodes belong to the same family, and return the set names in vector
{
	int familycount = 0;
	long index_existingfamily; vector<int> uniquemarkset; vector<vector<Node* > > families;
	// unique mark set ? We shouldn't use set, because the order is now messed up, just use vector
	// cout << "V size: " << V.size() << endl;
	for (int i = 0; i < V.size(); i++){
		for (int j = i + 1; j < V.size(); j++){
			//if(testnode(V[i],V[j],V).first <= TOLERANCE) Union(V[i], V[j]);
			if ((*Phi_ptr).first(i, j) <= TOLERANCE)
			{
				//cout << "union " << V[i]->readi()<< " and " << V[j]->readi()<< endl;
				Union(V[i], V[j]);
			}
		}
	}
	for (int i = 0; i < V.size(); i++){
		//listmark.push_back(V[i]->readmark());
		V[i]->setmark(Find(V[i]));

		if (find(uniquemarkset.begin(), uniquemarkset.end(), V[i]->readmark()) == uniquemarkset.end()){
			uniquemarkset.push_back(Find(V[i])->readmark());
			//create a new family
			vector<Node*> tempfamily;
			families.push_back(tempfamily);
			families[familycount].push_back(V[i]);
			familycount++;
		}
		else{
			index_existingfamily = (long) distance(uniquemarkset.begin(), find(uniquemarkset.begin(), uniquemarkset.end(), V[i]->readmark()));
			families[index_existingfamily].push_back(V[i]);

		}
	}
	////////////
	// find parents here
	vector<Node *> parents(families.size());
	for (int i = 0; i < families.size(); i++)
	{
		vector<long> id_this_family(families[i].size());
		// assign Dist_this_family
		for (int j = 0; j < families[i].size(); j++)
		{
			id_this_family[j] = (long) distance(V.begin(), find(V.begin(), V.end(), families[i][j]));
		}
		//
		vector<long> * id_this_family_ptr = &id_this_family;
		parents[i] = FindParent(families[i], id_this_family_ptr, Dist_ptr, &((*Phi_ptr).second));


	}

	////////////
	// make sure there is at least one family with two siblings

	bool atleast_onefam_2sibls = false;
	while (atleast_onefam_2sibls == false)
	{
		for (int i = 0; i < families.size(); i++)
		{
			if (families[i].size()>2)
			{
				atleast_onefam_2sibls = true;
			}
			if (families[i].size() == 2 && parents[i] == NULL)
			{
				atleast_onefam_2sibls = true;
			}
		}

		if (atleast_onefam_2sibls == false)
		{
			// union two single
			for (int id = 0; id < families[families.size() - 1].size(); id++)
			{
				families[families.size() - 2].push_back(families[families.size() - 1][id]);
			}
			//cout <<"original family size: " <<  families.size() << endl;
			families.erase(families.begin() + families.size() - 1);//???????delete the last row of this vector of vector
			parents.erase(parents.begin() + parents.size() - 1);//???????delete the last row of parents
			//cout <<"after deleting the last element,  family size: " <<  families.size() << endl;
		}
	}
	pair<vector<vector<Node*> >, vector<Node *> > families_parents;
	if (families.size() == 1 && families[0].size() == 3){
		parents[0] = NULL;
	}
	families_parents.first = families;
	families_parents.second = parents;
	return families_parents;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Node* FindParent(vector<Node*> family, vector<long> *id_this_family_ptr, Eigen::MatrixXd *Dist_ptr, Eigen::MatrixXd *Phi_mean_ptr)
{
	const int size = (int)family.size();
	long ind;
	vector<double> score(size);
	if (size == 1) return family[0];     //single node => parent
	for (int i = 0; i < size; i++){
		score.push_back(0.0);
		for (int j = 0; j < size; j++){
			if (j != i)
			{
				score[i] = score[i] + abs((*Phi_mean_ptr)((*id_this_family_ptr)[i], (*id_this_family_ptr)[j]) + (*Dist_ptr)((*id_this_family_ptr)[i], (*id_this_family_ptr)[j]));
			}
		}
	}
	double * score_array = &score[0];
	ind = (long) distance(score_array, min_element(score_array, score_array + size));
	if (score[ind] < 2 * edgeD_MAX*(size - 1))
		return family[ind];
	//if no parent, return NULL
	else
		return NULL;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// set of combination between RG and Param Est
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// function computeHiddenChild:used to when no parent is present. all family members are children.
pair<vector<Edge *>, double> computeHiddenChild_g2members_inRefFamily(Eigen::MatrixXd dist, Eigen::MatrixXd phi_second, vector<Node *>  childset, vector<Node *> currNodeset, Node *pa, Node *ref)
{
	//  cout << "start computeHiddenChild_g2members_inRefFamily function" << endl;
	// edge_distance is a vector to return & edge_distance_sum is a double to return
	pair<vector<Edge *>, double> output;
	output.second = 0;
	//
	VectorXd edge_distance(childset.size());
	// child_index for all the children
	vector<long> child_index;

	for (int i = 0; i < childset.size(); i++)
	{
		long tmp_index = (long)distance(currNodeset.begin(), find(currNodeset.begin(), currNodeset.end(), childset[i]));
		child_index.push_back(tmp_index);
	}
	//    long ref_index = distance(currNodeset.begin(),find(currNodeset.begin(),currNodeset.end(),ref));
	long ref_index = (long)distance(childset.begin(), find(childset.begin(), childset.end(), ref));

	Eigen::MatrixXd child_pair_distance = MatrixXd::Zero(childset.size(), childset.size());
	for (int i = 0; i < childset.size(); i++)
	{
		for (int j = i + 1; j < childset.size(); j++)
		{
			child_pair_distance(i, j) = dist(child_index[i], child_index[j]);
			child_pair_distance(j, i) = dist(child_index[j], child_index[i]);
		}
	}
//	cout << "child_pair_distance: " <<  endl << child_pair_distance << endl;

	edge_distance = child_pair_distance.rowwise().sum();
	output.second = edge_distance.sum() / (2 * (childset.size() - 1));
	//
	for (int k = 0; k < childset.size(); k++)
	{
		edge_distance(k) = (edge_distance(k) - output.second) / (childset.size() - 2);
		Edge *tmp_edge = new Edge(childset[k], pa, edge_distance(k));
		output.first.push_back(tmp_edge);
		//delete tmp_edge;
	}
	if(WITH_PARA){
	//    cout << "----START of computeHiddenChild_g2members_inRefFamily" << endl;
	// make sure index before ref_index is remained the same, the index after that is changed to index-1;
	vector<Node *> childset_cpy = childset;
	childset.erase(find(childset.begin(), childset.end(), ref));
	pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > ref_triplet;
	// this childset is the reduced one
	// if the childset.size() is odd
	//    cout << "childset.size(): " << childset.size()<< endl;
	if (childset.size() % 2)// odd case
	{
		//	cout << "odd case"<< endl;
		int k;
		for (k = 0; k < childset.size() - 1; k += 2)
		{
			//	    cout << "k: " << k << endl;
			pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > my_triplet = tensorDecom(childset[k], childset[k + 1], ref);
			if (k == 0)
			{
				ref_triplet = my_triplet;
				pa->set(ref_triplet.second);
				// set edge
				SparseMatrix<double> thisedgepot = ref_triplet.first[2];
				//		cout << "pa node is:" << pa->readc() << ", child node is:" << childset_cpy[ref_index]->readc() << ", edgepot.rows():" << thisedgepot.rows() << ", edgepot.cols():" << thisedgepot.cols() << endl;
				output.first[ref_index]->setedgepot(ref_triplet.first[2]);
			}
			if (k< ref_index)
			{
				//output.first[k]->setedgepot(my_triplet.first[0]);
				// allign this with ref_triplet
				SparseMatrix<double> thisedgepot = my_triplet.first[0];
				SparseMatrix<double> thatedgepot = (alignPara(my_triplet, ref_triplet)).first[0];
				//		cout << "pa node is:" << pa->readc() << ", child node is:" << childset_cpy[k]->readc();
				//		cout << ",orig edgepot.rows():" << thisedgepot.rows() << ",orig edgepot.cols():" << thisedgepot.cols();
				//		cout << ", aligned edgepot.rows():" << thatedgepot.rows() << ", aligned edgepot.cols():" << thatedgepot.cols() << endl;

				output.first[k]->setedgepot((alignPara(my_triplet, ref_triplet)).first[0]);

			}
			else
			{
				SparseMatrix<double> thisedgepot = my_triplet.first[0];
				SparseMatrix<double> thatedgepot = (alignPara(my_triplet, ref_triplet)).first[0];
				//		cout << "pa node is:" << pa->readc() << ", child node is:" << childset_cpy[k+1]->readc();
				//		cout << ",orig edgepot.rows():" << thisedgepot.rows() << ",orig edgepot.cols():" << thisedgepot.cols();
				//		cout << ", aligned edgepot.rows():" << thatedgepot.rows() << ", aligned edgepot.cols():" << thatedgepot.cols() << endl;


				// output.first[k+1]->setedgepot(this_triplet.first[0]);
				output.first[k + 1]->setedgepot((alignPara(my_triplet, ref_triplet)).first[0]);
			}
			if (k + 1 < ref_index)
			{
				SparseMatrix<double> thisedgepot = my_triplet.first[1];
				SparseMatrix<double> thatedgepot = (alignPara(my_triplet, ref_triplet)).first[1];
				//		cout << "pa node is:" << pa->readc() << ", child node is:" << childset_cpy[k+1]->readc();
				//		cout << ",orig edgepot.rows():" << thisedgepot.rows() << ",orig edgepot.cols():" << thisedgepot.cols();
				//		cout << ", aligned edgepot.rows():" << thatedgepot.rows() << ", aligned edgepot.cols():" << thatedgepot.cols() << endl;
				// output.first[k+1]=edgepot_k+1;
				output.first[k + 1]->setedgepot((alignPara(my_triplet, ref_triplet)).first[1]);
			}
			else
			{
				SparseMatrix<double> thisedgepot = my_triplet.first[1];
				SparseMatrix<double> thatedgepot = (alignPara(my_triplet, ref_triplet)).first[1];
				//		cout << "pa node is:" << pa->readc() << ", child node is:" << childset_cpy[k+2]->readc();
				//		cout << ",orig edgepot.rows():" << thisedgepot.rows() << ",orig edgepot.cols():" << thisedgepot.cols();
				//		cout << ", aligned edgepot.rows():" << thatedgepot.rows() << ", aligned edgepot.cols():" << thatedgepot.cols() << endl;
				// output.first[k+2]=edgepot_k+1;
				output.first[k + 2]->setedgepot((alignPara(my_triplet, ref_triplet)).first[1]);
			}
		}
		// deal with the last one

		pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > my_triplet = tensorDecom(childset[k], childset[k - 1], ref);
		if (k< ref_index)
		{
			SparseMatrix<double> thisedgepot = my_triplet.first[0];
			SparseMatrix<double> thatedgepot = (alignPara(my_triplet, ref_triplet)).first[0];
			//	    cout << "pa node is:" << pa->readc() << ", child node is:" << childset_cpy[k]->readc();
			//	    cout << ",orig edgepot.rows():" << thisedgepot.rows() << ",orig edgepot.cols():" << thisedgepot.cols();
			//	    cout << ", aligned edgepot.rows():" << thatedgepot.rows() << ", aligned edgepot.cols():" << thatedgepot.cols() << endl;
			//output.first[k]->setedgepot(my_triplet.first[0]);
			// allign this with ref_triplet
			output.first[k]->setedgepot((alignPara(my_triplet, ref_triplet)).first[0]);

		}
		else
		{
			SparseMatrix<double> thisedgepot = my_triplet.first[0];
			SparseMatrix<double> thatedgepot = (alignPara(my_triplet, ref_triplet)).first[0];
			//	    cout << "pa node is:" << pa->readc() << ", child node is:" << childset_cpy[k+1]->readc();
			//	    cout << ",orig edgepot.rows():" << thisedgepot.rows() << ",orig edgepot.cols():" << thisedgepot.cols();
			//	    cout << ", aligned edgepot.rows():" << thatedgepot.rows() << ", aligned edgepot.cols():" << thatedgepot.cols() << endl;
			// output.first[k+1]->setedgepot(this_triplet.first[0]);
			output.first[k + 1]->setedgepot((alignPara(my_triplet, ref_triplet)).first[0]);
		}


	}
	else // even case
	{
		//	cout << "even case "<< endl;
		for (int k = 0; k < childset.size() - 1; k += 2)
		{
			//	    cout << "k: " << k << endl;
			pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > my_triplet = tensorDecom(childset[k], childset[k + 1], ref);
			if (k == 0)
			{
				//		cout << "k==0" << endl;
				ref_triplet = my_triplet;
				pa->set(ref_triplet.second);
				SparseMatrix<double> thisedgepot = ref_triplet.first[2];
				//		cout << "pa node is:" << pa->readc() << ", child node is:" << childset_cpy[ref_index]->readc() << ", edgepot.rows():" << thisedgepot.rows() << ", edgepot.cols():" << thisedgepot.cols() << endl;
				//??????
				output.first[ref_index]->setedgepot(ref_triplet.first[2]);
			}
			if (k< ref_index)
			{
				//		cout << "k < ref_index" << endl;
				//output.first[k]->setedgepot(my_triplet.first[0]);
				// allign this with ref_triplet
				SparseMatrix<double> thisedgepot = my_triplet.first[0];
				SparseMatrix<double> thatedgepot = (alignPara(my_triplet, ref_triplet)).first[0];
				//		cout << "pa node is:" << pa->readc() << ", child node is:" << childset_cpy[k]->readc();
				//		cout << ",orig edgepot.rows():" << thisedgepot.rows() << ",orig edgepot.cols():" << thisedgepot.cols();
				//		cout << ", aligned edgepot.rows():" << thatedgepot.rows() << ", aligned edgepot.cols():" << thatedgepot.cols() << endl;

				output.first[k]->setedgepot((alignPara(my_triplet, ref_triplet)).first[0]);

			}
			else
				//	      cout << "k >= ref_index" << endl;
			{

				SparseMatrix<double> thisedgepot = my_triplet.first[0];
				SparseMatrix<double> thatedgepot = (alignPara(my_triplet, ref_triplet)).first[0];
				//		cout << "pa node is:" << pa->readc() << ", child node is:" << childset_cpy[k+1]->readc();
				//		cout << ",orig edgepot.rows():" << thisedgepot.rows() << ",orig edgepot.cols():" << thisedgepot.cols();
				//		cout << ", aligned edgepot.rows():" << thatedgepot.rows() << ", aligned edgepot.cols():" << thatedgepot.cols() << endl;
				// output.first[k+1]->setedgepot(this_triplet.first[0]);
				output.first[k + 1]->setedgepot((alignPara(my_triplet, ref_triplet)).first[0]);
			}
			if (k + 1 < ref_index)
			{
				//	cout << "k+1 < ref_index "<< endl;
				SparseMatrix<double> thisedgepot = my_triplet.first[1];
				SparseMatrix<double> thatedgepot = (alignPara(my_triplet, ref_triplet)).first[1];
				//		cout << "pa node is:" << pa->readc() << ", child node is:" << childset_cpy[k+1]->readc();
				//		cout << ",orig edgepot.rows():" << thisedgepot.rows() << ",orig edgepot.cols():" << thisedgepot.cols();
				//		cout << ", aligned edgepot.rows():" << thatedgepot.rows() << ", aligned edgepot.cols():" << thatedgepot.cols() << endl;
				// output.first[k+1]=edgepot_k+1;
				output.first[k + 1]->setedgepot((alignPara(my_triplet, ref_triplet)).first[1]);
			}
			else
			{
				SparseMatrix<double> thisedgepot = my_triplet.first[1];
				SparseMatrix<double> thatedgepot = (alignPara(my_triplet, ref_triplet)).first[1];
				//		cout << "pa node is:" << pa->readc() << ", child node is:" << childset_cpy[k+2]->readc();
				//		cout << ",orig edgepot.rows():" << thisedgepot.rows() << ",orig edgepot.cols():" << thisedgepot.cols();
				//		cout << ", aligned edgepot.rows():" << thatedgepot.rows() << ", aligned edgepot.cols():" << thatedgepot.cols() << endl;
				//		cout << "k+1 >= ref_index "<< endl;
				// output.first[k+2]=edgepot_k+1;
				output.first[k + 2]->setedgepot((alignPara(my_triplet, ref_triplet)).first[1]);
			}
		}


	}
	pa->set(ref_triplet.second);
	// estimate the samples of the node
	// p(x1|h), p(x2|h),
	// childset_cpy
	Eigen::SparseMatrix<double> AB;
	AB.resize((int)(childset.size() + 1)*VOCA_SIZE, KHID);
	Eigen::SparseMatrix<double>  concat_samples;
	//    cout << "to enter concatenation_matrix :" << endl;
	concat_samples.resize((int)(childset.size() + 1)*VOCA_SIZE, NX);
	concatenation_matrix(childset_cpy, output.first, pa, AB, concat_samples);

}
	//    cout << "end computeHiddenChild_g2members_inRefFamily function" << endl;
	return output;
}


///////////////////////////////////////////////////////////////////////////////////

pair<vector<Edge *>, double> computeHiddenChild_2members_inRefFamily(Eigen::MatrixXd dist, Eigen::MatrixXd phi_second, vector<Node *>  childset, vector<Node *> anotherset, vector<Node *> currNodeset, Node *pa)
{
	//  cout << "start  computeHiddenChild_2members_inRefFamily function "<< endl;
	// edge_distance is a vector to return & edge_distance_sum is a double to return
	pair<vector<Edge *>, double> output;
	output.second = 0;
	//
	VectorXd edge_distance(childset.size());

	// child_index for all the children
	vector<long> child_index;

	for (int i = 0; i < childset.size(); i++)
	{
		long tmp_index =(long) distance(currNodeset.begin(), find(currNodeset.begin(), currNodeset.end(), childset[i]));
		child_index.push_back(tmp_index);
	}

	Eigen::MatrixXd child_pair_distance = MatrixXd::Zero(childset.size(), childset.size());
	for (int i = 0; i < childset.size(); i++)
	{
		for (int j = i + 1; j< childset.size(); j++)
		{
			child_pair_distance(i, j) = dist(child_index[i], child_index[j]);
			child_pair_distance(j, i) = dist(child_index[j], child_index[i]);
		}
	}

	output.second = child_pair_distance(0, 1);

	edge_distance(0) = 0.5*(output.second + phi_second(0, 1));
	Edge *tmp_edge = new Edge(childset[0], pa, edge_distance(0));


	edge_distance(1) = 0.5*(output.second - phi_second(0, 1));
	Edge *tmp_edge2 = new Edge(childset[1], pa, edge_distance(1));

	pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > this_triplet;
	if (WITH_PARA){
		//    cout << "----START of computeHiddenChild_2members_inRefFamily" << endl;
		this_triplet = tensorDecom(childset[0], childset[1], anotherset[0]);
		tmp_edge->setedgepot(this_triplet.first[0]);
		tmp_edge2->setedgepot(this_triplet.first[1]);
		pa->set(this_triplet.second);

	}


	output.first.push_back(tmp_edge);
	output.first.push_back(tmp_edge2);

	if (WITH_PARA){
		// estimate the samples of the node
		// p(x1|h), p(x2|h),
		// childset_cpy
		Eigen::SparseMatrix<double>  AB;
		AB.resize((int)(childset.size())*VOCA_SIZE, KHID);
		Eigen::SparseMatrix<double>  concat_samples;
		concat_samples.resize((int)(childset.size())*VOCA_SIZE, NX);
		concatenation_matrix(childset, output.first, pa, AB, concat_samples);
	}

	//////////////////////////////////////////////
	//    cout << "end  computeHiddenChild_2members_inRefFamily function "<< endl;
	return output;
}


///////////////////////////////////////////////////////////////////////////////////
pair<vector<Edge *>, double> computeHiddenChild_g2members(Eigen::MatrixXd dist, Eigen::MatrixXd phi_second, vector<Node *>  childset, vector<Node *> currNodeset, Node *pa, Node *ref, vector<Node *> ref_childset, Node *glob_ref, Node * pa_ref)
{
	//  cout << "start computeHiddenChild_g2members function"<< endl;
	// edge_distance is a vector to return & edge_distance_sum is a double to return
	pair<vector<Edge *>, double> output;
	output.second = 0;
	//
	VectorXd edge_distance(childset.size());
	// child_index for all the children
	vector<long> child_index;

	for (int i = 0; i < childset.size(); i++)
	{
		long tmp_index = (long) distance(currNodeset.begin(), find(currNodeset.begin(), currNodeset.end(), childset[i]));
		child_index.push_back(tmp_index);
	}
	//    long ref_index =distance(currNodeset.begin(),find(currNodeset.begin(),currNodeset.end(),ref));
	long ref_index = (long) distance(childset.begin(), find(childset.begin(), childset.end(), ref));
	Eigen::MatrixXd child_pair_distance = MatrixXd::Zero(childset.size(), childset.size());
	for (int i = 0; i < childset.size(); i++)
	{
		for (int j = i + 1; j < childset.size(); j++)
		{
			child_pair_distance(i, j) = dist(child_index[i], child_index[j]);
			child_pair_distance(j, i) = dist(child_index[j], child_index[i]);
		}
	}

	edge_distance = child_pair_distance.rowwise().sum();
	output.second = edge_distance.sum() / (2 * (childset.size() - 1));
	//
	for (int k = 0; k < childset.size(); k++)
	{
		edge_distance(k) = (edge_distance(k) - output.second) / (childset.size() - 2);
		Edge *tmp_edge = new Edge(childset[k], pa, edge_distance(k));
		output.first.push_back(tmp_edge);
		//delete tmp_edge;
	}


	if (WITH_PARA){
	//    cout << " computeHiddenChild_g2members" << endl;
	// make sure index before ref_index is remained the same, the index after that is changed to index-1;
	vector<Node *> childset_cpy = childset;
	if (find(childset.begin(), childset.end(), ref) != childset.end())
	{
		childset.erase(find(childset.begin(), childset.end(), ref));
	}

	if (find(ref_childset.begin(), ref_childset.end(), glob_ref) != ref_childset.end())
	{
		ref_childset.erase(find(ref_childset.begin(), ref_childset.end(), glob_ref));
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////
	// align with ref_family
	Eigen::SparseMatrix<double> D = alignTwoFamily(glob_ref, ref_childset[0], ref, childset[0], pa, pa_ref);

	//    pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > ref_triplet;
	pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > my_triplet;
	// this childset is the reduced one
	// if the childset.size() is odd
	if (childset.size() % 2)// odd case
	{
		int k;
		for (k = 0; k < childset.size() - 1; k += 2)
		{

			my_triplet = tensorDecom(childset[k], childset[k + 1], ref);
			if (k == 0)  output.first[ref_index]->setedgepot((alignPara(my_triplet, D)).first[2]);
			if (k< ref_index)
			{
				//output.first[k]->setedgepot(my_triplet.first[0]);
				// allign this with ref_triplet
				output.first[k]->setedgepot((alignPara(my_triplet, D)).first[0]);

			}
			else
			{
				// output.first[k+1]->setedgepot(this_triplet.first[0]);
				output.first[k + 1]->setedgepot((alignPara(my_triplet, D)).first[0]);
			}
			if (k + 1 < ref_index)
			{
				// output.first[k+1]=edgepot_k+1;
				output.first[k + 1]->setedgepot((alignPara(my_triplet, D)).first[1]);
			}
			else
			{
				// output.first[k+2]=edgepot_k+1;
				output.first[k + 2]->setedgepot((alignPara(my_triplet, D)).first[1]);
			}
		}
		// deal with the last one

		my_triplet = tensorDecom(childset[k], childset[k - 1], ref);
		if (k< ref_index)
		{
			//output.first[k]->setedgepot(my_triplet.first[0]);
			// allign this with ref_triplet
			output.first[k]->setedgepot((alignPara(my_triplet, D)).first[0]);

		}
		else
		{
			// output.first[k+1]->setedgepot(this_triplet.first[0]);
			output.first[k + 1]->setedgepot((alignPara(my_triplet, D)).first[0]);
		}


	}
	else // even case
	{
		for (int k = 0; k < childset.size() - 1; k += 2)
		{

			my_triplet = tensorDecom(childset[k], childset[k + 1], ref);
			if (k == 0) output.first[ref_index]->setedgepot((alignPara(my_triplet, D)).first[2]);
			if (k< ref_index)
			{
				//output.first[k]->setedgepot(my_triplet.first[0]);
				// allign this with ref_triplet
				output.first[k]->setedgepot((alignPara(my_triplet, D)).first[0]);

			}
			else
			{
				// output.first[k+1]->setedgepot(this_triplet.first[0]);
				output.first[k + 1]->setedgepot((alignPara(my_triplet, D)).first[0]);
			}
			if (k + 1 < ref_index)
			{
				// output.first[k+1]=edgepot_k+1;
				output.first[k + 1]->setedgepot((alignPara(my_triplet, D)).first[1]);
			}
			else
			{
				// output.first[k+2]=edgepot_k+1;
				output.first[k + 2]->setedgepot((alignPara(my_triplet, D)).first[1]);
			}
		}


	}

	// set node potential
	pa->set((alignPara(my_triplet, D)).second);
	// estimate the samples of the node
	// p(x1|h), p(x2|h),
	// childset_cpy

	Eigen::SparseMatrix<double> AB;
	AB.resize((int)(childset.size() + 1)*VOCA_SIZE, KHID);
	Eigen::SparseMatrix<double>  concat_samples;
	concat_samples.resize((int)(childset.size() + 1)*VOCA_SIZE, NX);
	concatenation_matrix(childset_cpy, output.first, pa, AB, concat_samples);


}
	//    cout << "end computeHiddenChild_g2members function"<< endl;
	return output;
}
///////////////////////////////////////////////////////////////////////////////////

//
pair<vector<Edge *>, double> computeHiddenChild_2members(Eigen::MatrixXd dist, Eigen::MatrixXd phi_second, vector<Node *>  childset, vector<Node *> currNodeset, Node *pa, vector<Node *> ref_childset, Node *glob_ref, Node * pa_ref)
{
	//  cout << "start computeHiddenChild_2members function" << endl;
	// edge_distance is a vector to return & edge_distance_sum is a double to return
	pair<vector<Edge *>, double> output;
	output.second = 0;
	//
	VectorXd edge_distance(childset.size());

	// child_index for all the children
	vector<long> child_index;

	for (int i = 0; i < childset.size(); i++)
	{
		long tmp_index = (long) distance(currNodeset.begin(), find(currNodeset.begin(), currNodeset.end(), childset[i]));
		child_index.push_back(tmp_index);
	}

	Eigen::MatrixXd child_pair_distance = MatrixXd::Zero(childset.size(), childset.size());
	for (int i = 0; i < childset.size(); i++)
	{
		for (int j = i + 1; j< childset.size(); j++)
		{
			child_pair_distance(i, j) = dist(child_index[i], child_index[j]);
			child_pair_distance(j, i) = dist(child_index[j], child_index[i]);
		}
	}

	output.second = child_pair_distance(0, 1);

	edge_distance(0) = 0.5*(output.second + phi_second(0, 1));
	Edge *tmp_edge = new Edge(childset[0], pa, edge_distance(0));


	edge_distance(1) = 0.5*(output.second - phi_second(0, 1));
	Edge *tmp_edge2 = new Edge(childset[1], pa, edge_distance(1));
	///////
	//////////////////////////////////////////////////////////////////////////////////////////////////
	if (WITH_PARA){
		//    cout << " computeHiddenChild_2members" << endl;
		// align with ref_family
		Node *ref = childset[1];
		if (find(ref_childset.begin(), ref_childset.end(), glob_ref) != ref_childset.end())
		{
			ref_childset.erase(find(ref_childset.begin(), ref_childset.end(), glob_ref));
		}

		Eigen::SparseMatrix<double> D = alignTwoFamily(glob_ref, ref_childset[0], ref, childset[0], pa, pa_ref);



		////
		pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > this_triplet;
		this_triplet = tensorDecom(childset[0], childset[1], glob_ref);
		this_triplet = alignPara(this_triplet, D);

		tmp_edge->setedgepot(this_triplet.first[0]);
		tmp_edge2->setedgepot(this_triplet.first[1]);
		pa->set(this_triplet.second);

	}
	output.first.push_back(tmp_edge);
	output.first.push_back(tmp_edge2);

	if (WITH_PARA){
		Eigen::SparseMatrix<double>  AB;
		AB.resize((int)(childset.size())*VOCA_SIZE, KHID);
		Eigen::SparseMatrix<double>  concat_samples;
		concat_samples.resize((int)(childset.size())*VOCA_SIZE, NX);
		concatenation_matrix(childset, output.first, pa, AB, concat_samples);
	}

	//////////////////////////////////////////////

	//delete tmp_edge2;
	//delete tmp_edge;

	//    cout << "end computeHiddenChild_2members function" << endl;
	return output;
}

///////////////////////////////////////////////////////////////////////////////////

Eigen::MatrixXd  computeNewDistance(Eigen::MatrixXd Dist, vector<vector<Node *> > adjNewOld, vector<Node *> currNodeset, vector<Node *>  nextNodeset, vector<double> edge_distance_sum_set)
{
	int M = (int)edge_distance_sum_set.size();
	Eigen::MatrixXd currDist(M, M);

	for (int i = 0; i< M; i++)
	{
		for (int j = 0; j < M; j++)
		{
			Eigen::MatrixXd inter_fam_dist_mat(adjNewOld[i].size(), adjNewOld[j].size());
			double inter_fam_dist_sum;
			for (int rows = 0; rows < adjNewOld[i].size(); rows++)
			{
				for (int cols = 0; cols < adjNewOld[j].size(); cols++)
				{
					long id_row = (long) distance(currNodeset.begin(), find(currNodeset.begin(), currNodeset.end(), adjNewOld[i][rows]));
					long id_col = (long) distance(currNodeset.begin(), find(currNodeset.begin(), currNodeset.end(), adjNewOld[j][cols]));
					inter_fam_dist_mat(rows, cols) = Dist(id_row, id_col);
				}

			}
			inter_fam_dist_sum = inter_fam_dist_mat.sum(); // FURONG: check syntax!!!!!!

			currDist(i, j) = (inter_fam_dist_sum - edge_distance_sum_set[i] * adjNewOld[j].size() - edge_distance_sum_set[j] * adjNewOld[i].size()) / (adjNewOld[j].size()* adjNewOld[j].size()); // FURONG: Do I have to convert int to double to do those operations???????????????
		}
	}


	return (currDist + currDist.transpose());
}


///////////////////////////////////////////
void RG(Graph *g, SparseMatrix<double> *AJM)
{
	//  cout << "start RG function "<< endl;
	//Initial currNodeset;
	vector<Node*> currNodeset = g->readnodeset();
	//Initial currDist;
	Eigen::MatrixXd currDist = MatrixXd::Zero(currNodeset.size(), currNodeset.size());
	Eigen::VectorXd NodeID(currNodeset.size());
	for (int i = 0; i < currNodeset.size(); i++){ NodeID(i) = currNodeset[i]->readi(); }
	for (int i = 0; i < currNodeset.size(); i++)
	{
		for (int j = i + 1; j < currNodeset.size(); j++)
		{
			currDist(i, j) = (*AJM).coeffRef((int)NodeID(i), (int)NodeID(j));
			if (currDist(i, j) == 0)
				currDist(i, j) = MAXDISTANCE; // this is because I make sure the zero are maxdistance
			currDist(j, i) = currDist(i, j);
		}
		currDist(i, i) = MAXDISTANCE;
	}
//	cout << "NodeID: " << endl << NodeID << endl;
//	cout << "currDist: " << endl << currDist << endl;
	/////////////////////////////////////
	// Iterative stTOLERANCE for RG
	//    cout <<"iterative steps for RG"<< endl;
	while (currNodeset.size() > 2)
	{
		// Find currPhi for currNodeset; currPhi.first is an Eigen matrix of fluctuation, currPhi.second is an Eigen matrix of phi_mean.
		pair<Eigen::MatrixXd, Eigen::MatrixXd> currPhi = testnode(currDist, currNodeset);
		//	cout<<"endl of testnode"<<endl;
		//        cout << "currPhi.fluctuation:\n" << currPhi.first << endl;
		//        cout << "currPhi.mean:\n " << currPhi.second << endl;
		// Find families
		pair< vector<vector<Node*> >, vector<Node* > > families_parents = FamilyUnion(&currDist, currNodeset, &currPhi);
		//	cout <<"end of family union"<<endl;
		//        cout << "number of families: " <<families_parents.first.size() << endl;
		vector<vector<Node*> > families = families_parents.first;
		vector<Node* > parents = families_parents.second;// Find parents
		//       cout << "parents.size()"<<parents.size()<<endl;
		//	cout <<"parents[0]"<<parents[0]->readi()<<","<<parents[0]->readc()<<endl;
		////////////////////////////////////////////////////////////////////////////////// tested
		// Find the ref_node_families, and the ref node in this RG (surrogate)
		vector<Node *> ref_nodes; Node *surr = g->readsur();
		Node * ref_node = NULL; int ref_family = 0;
		vector<Node *> candidate_ref_node;
		vector<int> candidate_ref_family;
		for (int i = 0; i < families.size(); i++)
		{
			if (find(families[i].begin(), families[i].end(), surr) != families[i].end())//surr is in the families[i]
			{
				if (surr != parents[i] && families[i].size()>1)
				{
					ref_node = surr;
					ref_family = i;
					//		    cout <<"ref_family"<< ref_family<<endl;
					ref_nodes.push_back(surr);
				}
				else
				{
					for (int j = 0; j<families[i].size(); j++)
					{
						if (families[i][j] != parents[i])
						{
							ref_nodes.push_back(families[i][j]);
							if (families[i].size() - ((int)(parents[i] != NULL))>1)
							{
								candidate_ref_node.push_back(families[i][j]);
								candidate_ref_family.push_back(i);
							}
							break;
						}
					}
				}

			}
			else//surr is not in the families
			{
				for (int j = 0; j<families[i].size(); j++)
				{
					if (families[i][j] != parents[i])
					{
						ref_nodes.push_back(families[i][j]);
						if (families[i].size() - ((int)(parents[i] != NULL))>1)
						{
							//                        cout << "families[i][j]->readi(): " << families[i][j]->readi() << endl;

							candidate_ref_node.push_back(families[i][j]);
							candidate_ref_family.push_back(i);
						}
						break;
					}
				}

			}

		}
		if (ref_node == NULL)
		{
			ref_node = candidate_ref_node[0];
			ref_family = candidate_ref_family[0];
		}
		//	cout <<"families's size"<<families.size()<<endl;
		//	cout <<"candidate_ref_family"<< candidate_ref_family[0]<<endl;
		//	cout << "ref_family"<<ref_family<<endl;
		//        cout << " end of finding the ref_node, ref_family and ref_nodes;"<<endl;
		/////////////////////////////////////////////////////////////////////////////////////////

		//     update newNodeset, nextNodeset;
		vector<Node *> newNodeset;
		vector<Node *> nextNodeset;
		vector<vector<Node *> > adjNewOld(families.size());
		//	cout<<"families.size():"<<families.size()<<endl;
		vector<double > edge_distance_sum_set(families.size());
		// first of all I do this for ref_family
		int count_NULLpa = 0;
		// introduce the hidden node when there is no parent for that family
		int i = ref_family;
		Node * pa_ref;
		double edge_distance_sum = 0;// this is the biproduct from function computeHiddenChild
		if (parents[i] == NULL)// we don't have a parent in Ref family
		{
			//	  cout <<"start of we don't have a parent in Ref family"<<endl;
			//adjNewOld.push_back(families[i]);
			adjNewOld[i] = families[i];
			count_NULLpa++;
			// introduce a new hidden node
			//Node tmpnode = Node(g.readsur()->readi(),itoa(count_NULLpa));
			Node * tmpnode = new Node(g->readsur()->readi(), g->readsur()->readi(), (char)(((int)'0') + count_NULLpa));
			//           cout << "((int)'0')+count_NULLpa :" << ((int)'0')+count_NULLpa << endl;
			//           cout << "(char) (((int)'0')+count_NULLpa) :" << (char) (((int)'0')+count_NULLpa) << endl;
			//Node* tmpnodep = &tmpnode;
			// update nextNodeset; Note that num_next_nodes = family.size();
			nextNodeset.push_back(tmpnode);
			// update newNodeset;  Note that num_new_nodes  = number of NULL parents in all families;
			newNodeset.push_back(tmpnode);
			pa_ref = tmpnode;// new added !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			//  distance estimation step 1: (1) calucalte the distance between all currset and hiddennode.(2)Byproduct:output a edge_distance_sum.(3)addEdge(newhiddenparent, allChildren).
			// edge_edgesum.first is a vector of Edge* ; edge_edgesum.second is a double
			pair<vector<Edge *>, double> edge_edgesum;

			if (families[i].size() == 2) // Ref families[i].size ==2
			{
				if (families.size()>1)
				{
					int idx_anotherfamily;
					for (idx_anotherfamily = 0; idx_anotherfamily < families.size(); idx_anotherfamily++)
					{
						if (idx_anotherfamily != ref_family)
						{
							break;
						}
					}
					edge_edgesum = computeHiddenChild_2members_inRefFamily(currDist, currPhi.second, families[i], families[idx_anotherfamily], currNodeset, tmpnode);
				}
				else
				{
					edge_edgesum = computeHiddenChild_2members_inRefFamily(currDist, currPhi.second, families[i], families[i], currNodeset, tmpnode);// TODO: think more on this !!!!!!!!!
				}
			}
			else //Ref families[i].size > 2
			{

				edge_edgesum = computeHiddenChild_g2members_inRefFamily(currDist, currPhi.second, families[i], currNodeset, tmpnode, ref_node);
			}
			edge_distance_sum = edge_edgesum.second;
			for (int local_iedge = 0; local_iedge<edge_edgesum.first.size(); local_iedge++)
			{
				g->addEdge(edge_edgesum.first[local_iedge]);
			}
			//	    cout <<"end of if we don't have a parent in Ref Family"<<endl;
		}

		else// we have a parent in Ref Family, no need for a new node
		{
			//	  cout <<"start of we have a parent in Ref family"<<endl;
			// update nextNodeset; Note that num_next_nodes = family.size();
			nextNodeset.push_back(parents[i]);
			//	    cout <<"mark_fh0"<<endl;
			// distance estimation step 1:
			// addEdge (parent, OtherChildren);//weight is already added this step, it is just currDist(parent, OtherChildren))
			//
			//	    cout << "i:"<<i<<endl;
			//	    cout << parents[i]->readi()<<endl;
			adjNewOld[i].push_back(parents[i]);
			//           cout << "mark_fh01"<<endl;
			long parentid = (long) distance(currNodeset.begin(), find(currNodeset.begin(), currNodeset.end(), parents[i]));
			//	    cout << "mark_fh02"<<endl;
			pa_ref = parents[i];// new added !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			//	    cout<<"mark_fh1"<<endl;
			for (int j = 0; j< families[i].size(); j++)
			{
				if (families[i][j]->readi() != parents[i]->readi() || families[i][j]->readc() != parents[i]->readc())
				{
					// we think of them as children
					long childid = (long)distance(currNodeset.begin(), find(currNodeset.begin(), currNodeset.end(), families[i][j]));
					//		    cout <<"mark_fh2"<<endl;
					Edge * myedge_tmp = new Edge(families[i][j], parents[i], currDist(parentid, childid));
					//		    cout <<"mark_fh3"<<endl;
					if (WITH_PARA){
						myedge_tmp->setedgepot(edgepot_observables(families[i][j], parents[i]));
						// there is setpot function within edgepot_observables
					}

					g->addEdge(myedge_tmp);
					//		    cout <<"mark_fh4"<<endl;
				}

			}

			//	    cout <<"end of if we have a parent in Ref Family"<<endl;
		}
		edge_distance_sum_set[i] = edge_distance_sum;
		//TODO: families.size()==1 will stop here
		// For families other than Ref family
		//       cout <<"mark here...."<<endl;
		if (families.size()>1)
		{
			for (int i = 0; i < families.size(); i++)
			{
				if (i != ref_family)
				{
					double edge_distance_sum = 0;// this is the biproduct from function computeHiddenChild
					if (parents[i] == NULL)// we don't have a parent in this family
					{
						adjNewOld[i] = families[i];
						count_NULLpa++;
						Node *tmpnode = new Node(g->readsur()->readi(), g->readsur()->readi(), (char)(((int)'0') + count_NULLpa));
						nextNodeset.push_back(tmpnode);
						newNodeset.push_back(tmpnode);
						pair<vector<Edge *>, double> edge_edgesum;
						if (families[i].size() == 2)// family size ==2
						{
							edge_edgesum = computeHiddenChild_2members(currDist, currPhi.second, families[i], currNodeset, tmpnode, families[ref_family], ref_node, pa_ref);
						}
						else // family size >2
						{
							edge_edgesum = computeHiddenChild_g2members(currDist, currPhi.second, families[i], currNodeset, tmpnode, ref_nodes[i], families[ref_family], ref_node, pa_ref);
						}
						edge_distance_sum = edge_edgesum.second;
						for (int local_iedge = 0; local_iedge<edge_edgesum.first.size(); local_iedge++)
						{
							g->addEdge(edge_edgesum.first[local_iedge]);
						}

					}

					else// we have a parent, no need for a new node
					{
						nextNodeset.push_back(parents[i]);
						adjNewOld[i].push_back(parents[i]);

						long parentid = (long) distance(currNodeset.begin(), find(currNodeset.begin(), currNodeset.end(), parents[i]));
						for (int j = 0; j< families[i].size(); j++)
						{

							if (families[i][j]->readi() != parents[i]->readi() || families[i][j]->readc() != parents[i]->readc())
							{
								// we think of them as children
								long childid = (long) distance(currNodeset.begin(), find(currNodeset.begin(), currNodeset.end(), families[i][j]));
								Edge * myedge_tmp = new Edge(parents[i], families[i][j], currDist(parentid, childid));
								if (WITH_PARA){
									myedge_tmp->setedgepot(edgepot_observables(families[i][j], parents[i]));
								}
								
								g->addEdge(myedge_tmp);
							}

						}


					}
					edge_distance_sum_set[i] = edge_distance_sum;
				}

			}
		}
		// End of finding parents and calculate in-group edge distances.


		///////////////////
		//         step 2:  calcualte distances between Y, which is updated to be equal to Y_new
		// currDist is updated here.  // currDist=computeNewDistance(currDist(which is now actually oldDist), adjNewOld(Y_old, Y), edge_distance_sum );
		// currNodeset is updated here.
		// adjNewOld keeps track of nextNodeset and currNodeset;
		// adjNewOld is a vector<Node *>; there is parent, only the parent node is in the vector<Node *>; if there is no parent, all children is in the vector<Node *>.
		currDist = computeNewDistance(currDist, adjNewOld, currNodeset, nextNodeset, edge_distance_sum_set);
//		cout << "currDist" << endl << currDist << endl;
		currNodeset = nextNodeset;

	}
	//    cout<< "end of iterative steps for RG"<<endl;
	///////////////////////////////////////////
	// connect this two nodes, add an edge as well. compute the edge weight!

	if (currNodeset.size() == 2)
	{
		//       cout << "start of currNodeset.size ==2 " << endl;
		/////////parameter estimation

		if (currNodeset[0]->readc() != '0')
		{
			vector<Node *> tmp_nodeset = (g->readnodeset());
			vector<vector<Node *> > tmp_adj = (g->readadj());
			long id_a = (long) distance(tmp_nodeset.begin(), find(tmp_nodeset.begin(), tmp_nodeset.end(), currNodeset[0]));
			Node * a_child1 = tmp_adj[id_a][0];
			Node * a_child2 = tmp_adj[id_a][1];
			// ref edgePot
			long id_a_child1 = (long) distance(tmp_nodeset.begin(), find(tmp_nodeset.begin(), tmp_nodeset.end(), a_child1));
			// in adj[id_a_child1], find currNodeset[0]
			vector<Node *> tmp_parent_set = tmp_adj[id_a_child1];
			long id_a_child1_parent = (long) distance(tmp_parent_set.begin(), find(tmp_parent_set.begin(), tmp_parent_set.end(), currNodeset[0]));
			// ref_edgePot = g->readedgePot[id_a][id_a_child1];
			pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > para_last;
			if (WITH_PARA){
				para_last = alignPara(tensorDecom(currNodeset[1], a_child2, a_child1), (g->readedgePot())[id_a_child1][id_a_child1_parent]);
				currNodeset[0]->set(para_last.second);
#ifdef condi_prob
				// node * x = currNodeset[1]; node * y =currNodeset[0];
				// para_last.first[0] is p_x_y
				Eigen::SparseMatrix<double> p_x_y = para_last.first[0];
				// p_x_y_joint = Condi2Joint(p_x_y,currNodeset[0]->readnodepot());
				Eigen::SparseMatrix<double> p_x_y_joint = Condi2Joint(p_x_y, para_last.second);
				// p_x is the row sum.
				Eigen::SparseVector<double> p_x; p_x.resize(p_x_y.rows());
				for (int id_row = 0; id_row < p_x_y.rows(); id_row++)
				{
					Eigen::SparseVector<double> tmp_vec;
					tmp_vec.resize(p_x_y.cols());
					tmp_vec = p_x_y.block(id_row, 0, 1, p_x_y.cols());
					p_x.coeffRef(id_row) = tmp_vec.sum();
				}
				p_x.prune(TOLERANCE);
				currNodeset[1]->set(p_x);

#endif
#ifdef joint_prob
				// node * x = currNodeset[1]; node * y =currNodeset[0];
				// para_last.first[0] is p_x_y_joint
				Eigen::SparseMatrix<double> p_x_y_joint = para_last.first[0];
				// p_x is the row sum.
				Eigen::SparseVector<double> p_x; p_x.resize(p_x_y.rows());
				for (int id_row = 0; id_row < p_x_y.rows(); id_row++)
				{
					Eigen::SparseVector<double> tmp_vec;
					tmp_vec.resize(p_x_y.cols());
					tmp_vec = p_x_y.block(id_row, 0, 1, p_x_y.cols());
					p_x.coeffRef(id_row) = tmp_vec.sum();
				}
				p_x.prune(TOLERANCE);
				currNodeset[1]->set(p_x);

#endif

			}
			Edge *myedge_tmp_ptr = new Edge(currNodeset[1], currNodeset[0], currDist(0, 1));
			if (WITH_PARA){
				//            cout << para_last.first[0]<< endl;
				myedge_tmp_ptr->setedgepot(para_last.first[0]);
				//            cout << "edgepot:\n" << myedge_tmp_ptr->readedgepot()<< endl;


			}
			g->addEdge(myedge_tmp_ptr);///
			vector<Node *> tmp_nodeset_tmp = g->readnodeset();

			//            long iddddd_a = distance(tmp_nodeset_tmp.begin(),find(tmp_nodeset_tmp.begin(),tmp_nodeset_tmp.end(),currNodeset[0]));
			//            vector<Node *> tmp_currparents_tmp =g->readadj()[iddddd_a];
			//            long iddddd_b = distance(tmp_currparents_tmp.begin(),find(tmp_currparents_tmp.begin(),tmp_currparents_tmp.end(),currNodeset[1]));
			//            cout << g->readedgePot()[iddddd_a][iddddd_b] << endl;

		}
		else
		{
			vector<Node *> tmp_nodeset = (g->readnodeset());
			vector<vector<Node *> > tmp_adj = (g->readadj());

			long id_a = (long) distance(tmp_nodeset.begin(), find(tmp_nodeset.begin(), tmp_nodeset.end(), currNodeset[1]));
			Node * a_child1 = tmp_adj[id_a][0];
			Node * a_child2 = tmp_adj[id_a][1];

			long id_a_child1 = (long) distance((g->readnodeset()).begin(), find((g->readnodeset()).begin(), (g->readnodeset()).end(), a_child1));
			// in adj[id_a_child1], find currNodeset[0]
			vector<Node *> tmp_parent_set = tmp_adj[id_a_child1];
			long id_a_child1_parent = (long)distance(tmp_parent_set.begin(), find(tmp_parent_set.begin(), tmp_parent_set.end(), currNodeset[1]));
			// ref edgePot

			// ref_edgePot = g->readedgePot[id_a][id_a_child1];
			pair<vector<Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > para_last;
			if (WITH_PARA){
				para_last = alignPara(tensorDecom(currNodeset[0], a_child2, a_child1), (g->readedgePot())[id_a_child1][id_a_child1_parent]);

				// assign node potential to currNodeset[0];
				currNodeset[1]->set(para_last.second);
				/////////////////
#ifdef condi_prob
				// node * x = currNodeset[0]; node * y =currNodeset[1];
				// para_last.first[0] is p_x_y
				Eigen::SparseMatrix<double> p_x_y = para_last.first[0];
				// p_x_y_joint = Condi2Joint(p_x_y,currNodeset[0]->readnodepot());
				Eigen::SparseMatrix<double> p_x_y_joint = Condi2Joint(p_x_y, para_last.second);
				// p_x is the row sum.
				Eigen::SparseVector<double> p_x; p_x.resize(p_x_y.rows());
				for (int id_row = 0; id_row < p_x_y.rows(); id_row++)
				{
					Eigen::SparseVector<double> tmp_vec;
					tmp_vec.resize(p_x_y.cols());
					tmp_vec = p_x_y.block(id_row, 0, 1, p_x_y.cols());
					p_x.coeffRef(id_row) = tmp_vec.sum();
				}
				p_x.prune(TOLERANCE);
				currNodeset[0]->set(p_x);

#endif
#ifdef joint_prob
				// node * x = currNodeset[0]; node * y =currNodeset[1];
				// para_last.first[0] is p_x_y_joint
				Eigen::SparseMatrix<double> p_x_y_joint = para_last.first[0];
				// p_x is the row sum.
				Eigen::SparseVector<double> p_x; p_x.resize(p_x_y.rows());
				for (int id_row = 0; id_row < p_x_y.rows(); id_row++)
				{
					Eigen::SparseVector<double> tmp_vec;
					tmp_vec.resize(p_x_y.cols());
					tmp_vec = p_x_y.block(id_row, 0, 1, p_x_y.cols());
					p_x.coeffRef(id_row) = tmp_vec.sum();
				}
				p_x.prune(TOLERANCE);
				currNodeset[0]->set(p_x);

#endif

			}
			
			// add new edge
			Edge myedge_tmp = Edge(currNodeset[0], currNodeset[1], currDist(0, 1));
			if (WITH_PARA){
				//            cout << para_last.first[0]<< endl;
				myedge_tmp.setedgepot(para_last.first[0]);
			}
			
			g->addEdge(&myedge_tmp);

		}

	}
	//        cout<<"end of RG function"<<endl;
}
//////////////////////////
////////////////////////////////////////////////////////////
bool Graph_merge(Graph * g_merged_old, Graph * g_rg_new)
{
	//  cout << "start Graph_merge function" << endl;
	// to g_merged_old, the intersection with g_merged_old->readnodeset() and g_rg_new->readnodeset();
	vector<Node *> tmp_merged_old_nodeset = g_merged_old->readnodeset();
	vector<Node *> tmp_rg_new_nodeset = g_rg_new->readnodeset();
	std::sort(tmp_merged_old_nodeset.begin(), tmp_merged_old_nodeset.end());     //  5 10 15 20 25
	std::sort(tmp_rg_new_nodeset.begin(), tmp_rg_new_nodeset.end());   // 10 20 30 40 50
	std::vector<Node *> mynewsurrset; //Intersection of V1 and V2
	set_intersection(tmp_merged_old_nodeset.begin(), tmp_merged_old_nodeset.end(), tmp_rg_new_nodeset.begin(), tmp_rg_new_nodeset.end(), std::back_inserter(mynewsurrset));
	//    tmp_merged_old_nodeset.erase(std::unique(tmp_merged_old_nodeset.begin(), tmp_merged_old_nodeset.end()), tmp_merged_old_nodeset.end());
	//    tmp_rg_new_nodeset.erase(std::unique(tmp_rg_new_nodeset.begin(), tmp_rg_new_nodeset.end()), tmp_rg_new_nodeset.end());
	if (mynewsurrset.size() == 2)
	{


		Node * surr1 = g_rg_new->readsur();
		// erase surr1 in
		mynewsurrset.erase(find(mynewsurrset.begin(), mynewsurrset.end(), surr1));// this should be of element 1
		cout << "mynewsurrset.size(): " << mynewsurrset.size() << endl;

		/////////////////////////////////////
		Node * surr0 = mynewsurrset[0];

		vector<Node *> this_path = g_merged_old->shortestPath(surr0, surr1);
		double pathLen_merged_old = 0;
		for (int indexPath = 0; indexPath < this_path.size() - 2; ++indexPath)
		{
			pathLen_merged_old = pathLen_merged_old + g_merged_old->readedgeD_oneval(this_path[indexPath], this_path[indexPath + 1]);
		}

		double pathLen1 = pathLen_merged_old + g_merged_old->readedgeD_oneval(this_path[this_path.size() - 2], this_path[this_path.size() - 1]);

		vector<Node *> that_path = g_rg_new->shortestPath(surr1, surr0);
		double pathLen_rg_new = 0; 
		
		
		for (int indexPath = 0; indexPath < that_path.size() - 2; ++indexPath)
		{
			pathLen_rg_new = pathLen_rg_new + g_rg_new->readedgeD_oneval(that_path[indexPath], that_path[indexPath + 1]);
		}
		double pathLen2 = pathLen_rg_new + g_rg_new->readedgeD_oneval(that_path[that_path.size() - 2], that_path[that_path.size() - 1]);
		double pathLen = (pathLen1 + pathLen2) / 2.0;
		//////////////////////////
		// this_path : 2 -> h_2(1) -> 4
		// that_path : 4 -> h_4(1) -> h_4(2) -> 2
		// merged path would be 2 <-> h_2(1) <-> h_4(2) <-> h_4(1) <-> 4
		// new graph:
		//(1) still the surrogate doesn't matter, let's just remain it as the merged_old,
		//(2) vector<Node *> nodeset;
		//(3) vector<vector<Node *> > adj;
		//(4) vector<vector<double> > edgeD; NO need for this
		//(5) vector<vector<Eigen::Matrixxd> >edgePot;
		// delete an edge between this_path[this_path.size()-1] and this_path[this_path.size()-2]

		// need to align those two. Write a graph member function to allign given a permutation matrix
		// need to estimate para h_2(1) <-> h_4(2)??????????????
		Edge * myedge = new Edge(that_path[that_path.size() - 2], this_path[this_path.size() - 2]);// this_path[this_path.size()-2] is the parent, and it is
		Eigen::SparseMatrix<double> Pi;
		if (WITH_PARA){
			// from merged_old, we know:
			//(1) A  = P(sur0 | this_path[this_path.size()-2]);//propogate the conditional probability matrices
			//(2) BE = P(sur1 | this_path[this_path.size()-2]);

			Eigen::SparseMatrix<double> A;
			A.resize(VOCA_SIZE, VOCA_SIZE);
			A.setIdentity();

			Eigen::SparseMatrix<double> BE;//??????????resize??????
			BE.resize(VOCA_SIZE, KHID);
			for (int index_this = 0; index_this < this_path.size() - 2; index_this++)
			{
				vector<Node *> nodeset = g_merged_old->readnodeset();

				long id_left = (long) distance(nodeset.begin(), find(nodeset.begin(), nodeset.end(), this_path[index_this]));

				vector<Node *> currparents = ((g_merged_old->readadj())[id_left]);
				//            cout <<distance(currparents.begin(),find(currparents.begin(),currparents.end(),this_path[index_this+1]))<< endl;
				long id_right = (long) distance(currparents.begin(), find(currparents.begin(), currparents.end(), this_path[index_this + 1]));// this is the parent
				// g_merged_old->readedgePot()[id_left][id_right] , id_right is the second dimension
#ifdef joint_prob
				A = A *joint2conditional(g_merged_old->readedgePot()[id_left][id_right], this_path[index_this + 1]);
#endif

#ifdef condi_prob
				A = A *g_merged_old->readedgePot()[id_left][id_right];
				vector<vector<Eigen::SparseMatrix<double> > > test_temp = g_merged_old->readedgePot();

				//            cout <<"test_temp[id_left][id_right]:\n" << test_temp[id_left][id_right]<<endl;
				//            cout <<"g_merged_old->readedgePot()[id_left][id_right]: \n" << g_merged_old->readedgePot()[id_left][id_right] << endl;

#endif

			}
			Eigen::SparseMatrix<double> A_star = condi2condi(A, this_path[this_path.size() - 2]->readpot());
			//        cout << "A_star: \n " << A_star << endl;
			///////
			vector<Node *> nodeset = g_merged_old->readnodeset();

			long id_left = (long) distance(nodeset.begin(), find(nodeset.begin(), nodeset.end(), this_path[this_path.size() - 1]));// this is the child
			vector<Node *> currparents = ((g_merged_old->readadj())[id_left]);

			long id_right = (long) distance(currparents.begin(), find(currparents.begin(), currparents.end(), this_path[this_path.size() - 2]));// this is the parent
#ifdef joint_prob
			BE = joint2conditional((g_merged_old->readedgePot())[id_left][id_right], this_path[this_path.size() - 2]);
#endif

#ifdef condi_prob
			BE = g_merged_old->readedgePot()[id_left][id_right];
			//        cout << "BE: \n " << BE << endl;
#endif

			// from rg_new, we know:
			//(1) (AE_Pi) = P(sur0 | that_path[that_path.size()-2]);
			//(2) B_Pi    = P(sur1 | that_path[that_path.size()-2]);//propogate
			////////
			Eigen::SparseMatrix<double> AE_Pi;//??????????resize??????
			AE_Pi.resize(VOCA_SIZE, KHID);
			Eigen::SparseMatrix<double> B_Pi;
			B_Pi.resize(VOCA_SIZE, VOCA_SIZE);
			B_Pi.setIdentity();

			vector<Node *> nodeset_2 = g_rg_new->readnodeset();

			id_left = (long) distance(nodeset_2.begin(), find(nodeset_2.begin(), nodeset_2.end(), that_path[that_path.size() - 1]));//child
			vector<Node *> currparents_2 = ((g_rg_new->readadj())[id_left]);

			id_right = (long) distance(currparents_2.begin(), find(currparents_2.begin(), currparents_2.end(), that_path[that_path.size() - 2]));//parent
#ifdef joint_prob
			AE_Pi = joint2conditional((g_rg_new->readedgePot())[id_left][id_right], that_path[that_path.size() - 2]);
#endif

#ifdef condi_prob
			AE_Pi = (g_rg_new->readedgePot())[id_left][id_right];
			//        cout << "AE_Pi: \n " << AE_Pi << endl;
#endif
			Eigen::SparseMatrix<double>  Pi_EA_star = condi2condi(AE_Pi, that_path[that_path.size() - 2]->readpot());


			for (int index_that = 0; index_that < that_path.size() - 2; index_that++)
			{
				vector<Node *> nodeset_2 = g_rg_new->readnodeset();
				long id_left = (long)distance(nodeset_2.begin(), find(nodeset_2.begin(), nodeset_2.end(), that_path[index_that]));// child

				vector<Node *> currparents_2 = ((g_rg_new->readadj())[id_left]);

				long id_right = (long)distance(currparents_2.begin(), find(currparents_2.begin(), currparents_2.end(), that_path[index_that + 1]));// parent
#ifdef joint_prob
				B_Pi = B_Pi * joint2conditional((g_rg_new->readedgePot())[id_left][id_right], that_path[index_that + 1]);
#endif

#ifdef condi_prob
				B_Pi = B_Pi * ((g_rg_new->readedgePot())[id_left][id_right]);
				//            cout << "B_Pi: \n " << B_Pi << endl;
#endif

			}

			// Pi_square
			Eigen::SparseMatrix<double>  Pi_square = (Pi_EA_star * pinv_Nystrom_sparse(A_star)) * (pinv_Nystrom_sparse(BE) * B_Pi);
			//pair<SparseMatrix<double>, SparseMatrix<double> > pair_pinv_A_star = pinv_asymNystrom_sparse(A_star);
			//pair<SparseMatrix<double>, SparseMatrix<double> > pair_pinv_BE = pinv_asymNystrom_sparse(BE);
			//// pinv_asymNystrom_sparse(A_star) = pair_pinv_A_star.first.transpose() * pair_pinv_A_star.second * pair_pinv_A_star.first  * A_star.transpose();
			//// pinv_asymNystrom_sparse(BE)     = pair_pinv_BE.first.transpose() * pair_pinv_BE.second * pair_pinv_BE.first * BE.transpose();
			//SparseMatrix<double> Pi_square_tmp1 = Pi_EA_star * pair_pinv_A_star.first.transpose();
			//SparseMatrix<double> Pi_square_tmp2 = Pi_square_tmp1* pair_pinv_A_star.second;
			//SparseMatrix<double> Pi_square_tmp3 = pair_pinv_A_star.first * A_star.transpose() *  pair_pinv_BE.first.transpose();
			//SparseMatrix<double> Pi_square_tmp4 = Pi_square_tmp2*Pi_square_tmp3;
			//SparseMatrix<double> Pi_square_tmp5 = Pi_square_tmp4 * pair_pinv_BE.second;
			//SparseMatrix<double> Pi_square_tmp6 = pair_pinv_BE.first * BE.transpose() * B_Pi;
			//SparseMatrix<double> Pi_square = Pi_square_tmp5 * Pi_square_tmp6;
			//



			// since Pi is KHID by KHID, we do sqrt_matrix((MatrixXd)Pi_square)
			MatrixXd Pi_square_dense = (MatrixXd)Pi_square;
			Pi = (sqrt_matrix(Pi_square_dense)).sparseView();
			//        cout << "Pi_square:\n" << Pi_square << endl;
			// find E, E = Pi * pinv_matrix(B_Pi) * BE;
			Eigen::SparseMatrix<double>  E_merge = Pi * pinv_Nystrom_sparse(B_Pi) * BE; // parent is this_path[this_path.size()-2];
			//pair<SparseMatrix<double>, SparseMatrix<double> > pair_pinv_B_Pi = pinv_asymNystrom_sparse(B_Pi);
			//SparseMatrix<double> E_merge_tmp1 = Pi * pair_pinv_B_Pi.first.transpose();
			//SparseMatrix<double> E_merge_tmp2 = pair_pinv_B_Pi.second * pair_pinv_B_Pi.first;
			//SparseMatrix<double> E_merge_tmp3 = E_merge_tmp2 * B_Pi.transpose() *BE;
			//SparseMatrix<double> E_merge = E_merge_tmp1 * E_merge_tmp3;



			//


#ifdef condi_prob
			myedge->setedgepot(E_merge);
			double edgeWeight = pathLen - pathLen_merged_old - pathLen_rg_new;
			myedge->setweight(edgeWeight);
#endif
#ifdef joint_prob
			myedge->setedgepot(Condi2Joint(E_merge, this_path[this_path.size() - 2].readpot()));
			double edgeWeight = pathLen - pathLen_merged_old - pathLen_rg_new;
			myedge->setweight(edgeWeight);
#endif
		}


		g_merged_old->deleteEdge(this_path[this_path.size() - 1], this_path[this_path.size() - 2]);
		// add an new edge between this_path[this_path.size()-2] and that_path[that_path.size()-2]
		g_rg_new->deleteEdge(that_path[that_path.size() - 1], that_path[that_path.size() - 2]);


		g_merged_old->addEdge(myedge);// myedge->setedgepot();
//		cout << "g_rg_new->readnodeset().size()" << g_rg_new->readnodeset().size() << endl;
		for (int rows = 0; rows < g_rg_new->readnodeset().size(); rows++)
		{
//			cout << "g_rg_new->readadj()[rows].size()" << g_rg_new->readadj()[rows].size() << endl;
			for (int cols = 0; cols < g_rg_new->readadj()[rows].size(); cols++)
			{
//				cout << "node 1: " << "(" << (g_rg_new->readnodeset())[rows]->readi() << "," << (g_rg_new->readnodeset())[rows]->readc() << ")" << endl;
//				cout << "node 1: " << "(" << (g_rg_new->readadj())[rows][cols]->readi() << "," << (g_rg_new->readadj())[rows][cols]->readc() << ")" << endl;
				Edge *tmp_edge_ptr = new Edge((g_rg_new->readnodeset())[rows], (g_rg_new->readadj())[rows][cols]);// which one is child, which one is parent? This is correct, the elements are parents, and rows are children
				double tmpWeight = (g_rg_new->readedgeD())[rows][cols];
				tmp_edge_ptr->setweight(tmpWeight);
//				cout << "weight: " << tmp_edge_ptr << endl;
				if (WITH_PARA){
					if ((g_rg_new->readnodeset())[rows]->readc() == '0' && (g_rg_new->readadj())[rows][cols]->readc() == '0')
					{
						tmp_edge_ptr->setedgepot((g_rg_new->readedgePot())[rows][cols]);// which one is child, which one is parent??????????????
						
					}
					if ((g_rg_new->readnodeset())[rows]->readc() != '0' && (g_rg_new->readadj())[rows][cols]->readc() == '0')
					{
						// node potential alignment
						Eigen::SparseVector<double> Original_pot = (g_rg_new->readnodeset())[rows]->readpot();
						Eigen::SparseVector<double>  New_pot = Pi * Original_pot;
						(g_rg_new->readnodeset())[rows]->set(New_pot);
						tmp_edge_ptr->setedgepot(Pi*((g_rg_new->readedgePot())[rows][cols]));//
						
					}
					if ((g_rg_new->readnodeset())[rows]->readc() == '0' && (g_rg_new->readadj())[rows][cols]->readc() != '0')
					{
						// node potential alignment
						Eigen::SparseVector<double> Original_pot = g_rg_new->readadj()[rows][cols]->readpot();
						Eigen::SparseVector<double> New_pot = Pi * Original_pot;
						g_rg_new->readadj()[rows][cols]->set(New_pot);
						tmp_edge_ptr->setedgepot(((g_rg_new->readedgePot())[rows][cols])*Pi);//
						
					}
					if ((g_rg_new->readnodeset())[rows]->readc() != '0' && (g_rg_new->readadj())[rows][cols]->readc() != '0')
					{
						// node potential alignment
						Eigen::SparseVector<double>  Original_pot = (g_rg_new->readnodeset())[rows]->readpot();
						Eigen::SparseVector<double>  New_pot = Pi * Original_pot;
						(g_rg_new->readnodeset())[rows]->set(New_pot);
						// node potential alignment
						Eigen::SparseVector<double>  Original_pot2 = g_rg_new->readadj()[rows][cols]->readpot();
						Eigen::SparseVector<double>  New_pot2 = Pi * Original_pot2;
						g_rg_new->readadj()[rows][cols]->set(New_pot2);
						tmp_edge_ptr->setedgepot(Pi*((g_rg_new->readedgePot())[rows][cols])*Pi);//
						
					}
				}

				g_merged_old->addEdge(tmp_edge_ptr);

			}
		}
		//return g_merged_old;
		return true;
	}
	else
	{
		return false;
	}
	//    cout << "end Graph_merge function" << endl;
}