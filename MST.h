//
//  MST.h
//  latenttree
/*******************************************************
* Copyright (C) 2014 {Furong Huang} <{furongh@uci.edu}>
*
* This file is part of {Intergrated latent tree structure and parameters learning project}.
*
* All rights reserved.
*******************************************************/
#ifndef __latentTree__MST__
#define __latentTree__MST__
#include "stdafx.h"

vector<Edge*> BoruvkaMST(vector<Edge> EV, vector<Node> mynodelist);
vector<Edge*> KruskalMST(vector<Edge> &EV, vector<Node> &mynodelist);
#endif