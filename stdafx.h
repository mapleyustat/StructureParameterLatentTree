//
//  stdafx.h
//  latenttree
/*******************************************************
* Copyright (C) 2014 {Furong Huang} <{furongh@uci.edu}>
*
* This file is part of {Intergrated latent tree structure and parameters learning project}.
*
* All rights reserved.
*******************************************************/
// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#ifndef __latentTree__stdafx__
#define __latentTree__stdafx__
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <list>
#include <vector>
#include <set>
#include <queue>
#include <map>

#include <iterator>

#include <math.h>
#include <algorithm>

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/Core"
#include "Eigen/OrderingMethods"
#include "Eigen/SparseQR"

#include "omp.h"
#include <ctime>
//#include <sys/time.h>   // this is for linux machine
#include <windows.h> // this is for windows machine



//reference additional headers your program requires here
#include "Node.h"
#include "Edge.h"
#include "Graph.h"

#include "IOfn.h"
#include "Util.h"

#include "MathProbabilities.h"
#include "MST.h"
#include "Spectral.h"
#include "TensorDecom.h"




////////////////////////////////////////////////////////////////

#define NORMALIZE
#define THRE_COREVIEW 3

// #define useDistance
// #define nouseDistance
#define condi_prob
//////////////////////////////////////////////////////////////

#define MAX_WEIGHT 150

#define CENTERED	//asymmetric
#define UNCENTERED	//symmetric


#define LEARNRATE 1e-9
#define MINITER 20
#define MAXITER 1000
#define MAXDISTANCE 1000 

#define TOLERANCE 1e-6



#endif