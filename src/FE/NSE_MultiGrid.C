/** ==========================================================================
#    This file is part of the finite element software ParMooN.
# 
#    ParMooN (cmg.cds.iisc.ac.in/parmoon) is a free finite element software  
#    developed by the research groups of Prof. Sashikumaar Ganesan (IISc, Bangalore),
#    Prof. Volker John (WIAS Berlin) and Prof. Gunar Matthies (TU-Dresden):
#
#    ParMooN is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as
#    published by the Free Software Foundation, either version 3 of the
#    License, or (at your option) any later version.
#
#    ParMooN is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with ParMooN.  If not, see <http://www.gnu.org/licenses/>.
#
#    If your company is selling a software using ParMooN, please consider 
#    the option to obtain a commercial license for a fee. Please send 
#    corresponding requests to sashi@iisc.ac.in

# =========================================================================*/ 
   
// =======================================================================
// @(#)NSE_MultiGrid.C        1.24 06/27/00
//
// Class:       TNSE_MultiGrid
// Purpose:     store all data for a multi grid method fo
//              Stokes or Navier-Stokes problems
//
// Author:      Volker John 25.07.2000
//
// History:      25.07.2000 start of implementation
//
// =======================================================================

#include <NSE_MultiGrid.h>
#include <Database.h>
#include <LinAlg.h>
#include <MooNMD_Io.h>
#include <Constants.h>

#include <NSE_MGLevel5.h>
#include <NSE_MGLevel4.h>
#include <mkl.h>
#ifdef __2D__
  #include <FEDatabase2D.h>
#endif  
#ifdef __3D__
  #include <FEDatabase3D.h>
#endif  

#ifdef _MPI
#include <ParFECommunicator3D.h>
#include <ParDirectSolver.h>
#endif

#include <stdlib.h>
#include <string.h>

#define PreSmooth -1
#define CoarseSmooth 0
#define PostSmooth 1

#define CUDA_CHECK(call) \
do {\
            cudaError_t err = call;\
            if (cudaSuccess != err) \
            {\
                std::cerr << "CUDA error in " << __FILE__ << "(" << __LINE__ << "): " \
                    << cudaGetErrorString(err) << std::endl;\
                exit(EXIT_FAILURE);\
            }\
        } while(0)

double tSmoother2=0.0;
/** constructor */
TNSE_MultiGrid::TNSE_MultiGrid(int n_problems, int n_parameters, 
                                 double *parameters)
{
  N_Levels = 0;
  
  N_Problems = n_problems;

  N_Parameters = n_parameters;
  
  Parameters = parameters;
}

/** add new level as finest */
void TNSE_MultiGrid::AddLevel(TNSE_MGLevel *MGLevel)
{ 
  MultiGridLevels[N_Levels] = MGLevel;

  USpaces[N_Levels] = MGLevel->GetUSpace();
  PSpaces[N_Levels] = MGLevel->GetPSpace();

  N_Levels++;
}

#ifdef _CUDA
void TNSE_MultiGrid::InitGPU(){

  TNSE_MGLevel *finest_level = MultiGridLevels[N_Levels-1];
  
  TSquareMatrix3D *A11 = static_cast<TNSE_MGLevel4*>(finest_level)->GetA11Matrix();
  TMatrix3D *B1 = static_cast<TNSE_MGLevel4*>(finest_level)->GetB1Matrix();
  TMatrix3D *B1T = static_cast<TNSE_MGLevel4*>(finest_level)->GetB1TMatrix();

  nStreams = 2;

  stream_smooth = new cudaStream_t[nStreams];

  for (int i = 0; i < nStreams; ++i)
        CUDA_CHECK( cudaStreamCreate(&stream_smooth[i]) );

  CUDA_CHECK( cudaStreamCreate(&stream_transfer));

  int nz_A = A11->GetN_Entries();
  int n_A = A11->GetN_Rows(); 
  
  int nz_B = B1->GetN_Entries();
  int n_B = B1->GetN_Rows(); 
  
  int nz_BT = B1T->GetN_Entries();
  int n_BT = B1T->GetN_Rows(); 
  
  // cout<<nz<<endl;
  // cout<<n<<endl;

  #ifdef _CUDA
    CUDA_CHECK(cudaMalloc((void**)&d_ARowPtr1, (n_A+1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_AKCol1, nz_A * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_A11Entries1, nz_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_A12Entries1, nz_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_A13Entries1, nz_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_A21Entries1, nz_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_A22Entries1, nz_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_A23Entries1, nz_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_A31Entries1, nz_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_A32Entries1, nz_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_A33Entries1, nz_A * sizeof(double)));

    CUDA_CHECK(cudaMalloc((void**)&d_ARowPtr2, (n_A+1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_AKCol2, nz_A * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_A11Entries2, nz_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_A12Entries2, nz_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_A13Entries2, nz_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_A21Entries2, nz_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_A22Entries2, nz_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_A23Entries2, nz_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_A31Entries2, nz_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_A32Entries2, nz_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_A33Entries2, nz_A * sizeof(double)));

    CUDA_CHECK(cudaMalloc((void**)&d_BTRowPtr1, (n_BT+1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_BTKCol1, nz_BT * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_B1TEntries1, nz_BT * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_B2TEntries1, nz_BT * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_B3TEntries1, nz_BT * sizeof(double)));

    CUDA_CHECK(cudaMalloc((void**)&d_BTRowPtr2, (n_BT+1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_BTKCol2, nz_BT * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_B1TEntries2, nz_BT * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_B2TEntries2, nz_BT * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_B3TEntries2, nz_BT * sizeof(double)));

    CUDA_CHECK(cudaMalloc((void**)&d_BRowPtr1, (n_B+1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_BKCol1, nz_B * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_B1Entries1, nz_B * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_B2Entries1, nz_B * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_B3Entries1, nz_B * sizeof(double)));

    CUDA_CHECK(cudaMalloc((void**)&d_BRowPtr2, (n_B+1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_BKCol2, nz_B * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_B1Entries2, nz_B * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_B2Entries2, nz_B * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_B3Entries2, nz_B * sizeof(double)));

  #endif

}

void SwapDPointers(double** t1, double** t2){
  double* t = *t1;
  *t1 = *t2;
  *t2 = t;
}

void SwapIPointers(int** t1, int** t2){
  int* t = *t1;
  *t1 = *t2;
  *t2 = t;
}

void TNSE_MultiGrid::Swap_Matrices(){

// int rank;
//   #ifdef _MPI
//     MPI_Comm_rank(TDatabase::ParamDB->Comm,&rank);
//     #endif
  
//   if(rank==0){
//         cout<<"BIT before:"<<cblas_dnrm2(5,d_B1TEntries2,1)<<"\t"<<endl;
//   }


  SwapIPointers(&d_ARowPtr1, &d_ARowPtr2);
  SwapIPointers(&d_AKCol1, &d_AKCol2);

  SwapDPointers(&d_A11Entries1, &d_A11Entries2);
  SwapDPointers(&d_A12Entries1, &d_A12Entries2);
  SwapDPointers(&d_A13Entries1, &d_A13Entries2);

  SwapDPointers(&d_A21Entries1, &d_A21Entries2);
  SwapDPointers(&d_A22Entries1, &d_A22Entries2);
  SwapDPointers(&d_A23Entries1, &d_A23Entries2);

  SwapDPointers(&d_A31Entries1, &d_A31Entries2);
  SwapDPointers(&d_A32Entries1, &d_A32Entries2);
  SwapDPointers(&d_A33Entries1, &d_A33Entries2);

  SwapIPointers(&d_BTRowPtr1, &d_BTRowPtr2);
  SwapIPointers(&d_BTKCol1, &d_BTKCol2);

  SwapDPointers(&d_B1TEntries1, &d_B1TEntries2);
  SwapDPointers(&d_B2TEntries1, &d_B2TEntries2);
  SwapDPointers(&d_B3TEntries1, &d_B3TEntries2);

  SwapIPointers(&d_BRowPtr1, &d_BRowPtr2);
  SwapIPointers(&d_BKCol1, &d_BKCol2);

  SwapDPointers(&d_B1Entries1, &d_B1Entries2);
  SwapDPointers(&d_B2Entries1, &d_B2Entries2);
  SwapDPointers(&d_B3Entries1, &d_B3Entries2);

  // if(rank==0){
  //       cout<<"BIT after:"<<cblas_dnrm2(100,d_B1TEntries1,1)<<"\t"<<endl;
  // }
  
}

void TNSE_MultiGrid::DeviceDataTransfer(int next_level, int smooth){

  TNSE_MGLevel *next_level1 = MultiGridLevels[next_level];

  TSquareMatrix3D *A11 = static_cast<TNSE_MGLevel4*>(next_level1)->GetA11Matrix();
  TSquareMatrix3D *A12 = static_cast<TNSE_MGLevel4*>(next_level1)->GetA12Matrix();
  TSquareMatrix3D *A13 = static_cast<TNSE_MGLevel4*>(next_level1)->GetA13Matrix();

  TSquareMatrix3D *A21 = static_cast<TNSE_MGLevel4*>(next_level1)->GetA21Matrix();
  TSquareMatrix3D *A22 = static_cast<TNSE_MGLevel4*>(next_level1)->GetA22Matrix();
  TSquareMatrix3D *A23 = static_cast<TNSE_MGLevel4*>(next_level1)->GetA23Matrix();

  TSquareMatrix3D *A31 = static_cast<TNSE_MGLevel4*>(next_level1)->GetA31Matrix();
  TSquareMatrix3D *A32 = static_cast<TNSE_MGLevel4*>(next_level1)->GetA32Matrix();
  TSquareMatrix3D *A33 = static_cast<TNSE_MGLevel4*>(next_level1)->GetA33Matrix();

  TMatrix3D *B1 = static_cast<TNSE_MGLevel4*>(next_level1)->GetB1Matrix();
  TMatrix3D *B2 = static_cast<TNSE_MGLevel4*>(next_level1)->GetB2Matrix();
  TMatrix3D *B3 = static_cast<TNSE_MGLevel4*>(next_level1)->GetB3Matrix();

  TMatrix3D *B1T = static_cast<TNSE_MGLevel4*>(next_level1)->GetB1TMatrix();
  TMatrix3D *B2T = static_cast<TNSE_MGLevel4*>(next_level1)->GetB2TMatrix();
  TMatrix3D *B3T = static_cast<TNSE_MGLevel4*>(next_level1)->GetB3TMatrix();

  TStructure3D *StructureBT = B1T->GetStructure();
  TStructure3D *StructureB = B1->GetStructure();

  int nz_A = A11->GetN_Entries();
  int n_A = A11->GetN_Rows(); 
  
  int nz_B = B1->GetN_Entries();
  int n_B = B1->GetN_Rows(); 
  
  int nz_BT = B1T->GetN_Entries();
  int n_BT = B1T->GetN_Rows(); 

  // cout<<"nzA:"<<nz_A<<endl;
  // cout<<"nA:"<<n_A<<endl;

  //   cout<<"nzB:"<<nz_B<<endl;
  // cout<<"nB:"<<n_B<<endl;

  //   cout<<"nzA:"<<nz_BT<<endl;
  // cout<<"nA:"<<n_BT<<endl;


  CUDA_CHECK(cudaMemcpyAsync(d_ARowPtr2, A11->GetRowPtr(), (n_A+1) * sizeof(int), cudaMemcpyHostToDevice,stream_transfer));
    
  CUDA_CHECK(cudaMemcpyAsync(d_AKCol2, A11->GetKCol(), nz_A * sizeof(int), cudaMemcpyHostToDevice,stream_transfer));
  
  CUDA_CHECK(cudaMemcpyAsync(d_A11Entries2, A11->GetEntries(), nz_A * sizeof(double), cudaMemcpyHostToDevice,stream_transfer));
  
  CUDA_CHECK(cudaMemcpyAsync(d_A12Entries2, A12->GetEntries(), nz_A * sizeof(double), cudaMemcpyHostToDevice,stream_transfer));
  
  CUDA_CHECK(cudaMemcpyAsync(d_A13Entries2, A13->GetEntries(), nz_A * sizeof(double), cudaMemcpyHostToDevice,stream_transfer));
  
  CUDA_CHECK(cudaMemcpyAsync(d_A21Entries2, A21->GetEntries(), nz_A * sizeof(double), cudaMemcpyHostToDevice,stream_transfer));
  
  CUDA_CHECK(cudaMemcpyAsync(d_A22Entries2, A22->GetEntries(), nz_A * sizeof(double), cudaMemcpyHostToDevice,stream_transfer));
  
  CUDA_CHECK(cudaMemcpyAsync(d_A23Entries2, A23->GetEntries(), nz_A * sizeof(double), cudaMemcpyHostToDevice,stream_transfer));
  
  CUDA_CHECK(cudaMemcpyAsync(d_A31Entries2, A31->GetEntries(), nz_A * sizeof(double), cudaMemcpyHostToDevice,stream_transfer));
  
  CUDA_CHECK(cudaMemcpyAsync(d_A32Entries2, A32->GetEntries(), nz_A * sizeof(double), cudaMemcpyHostToDevice,stream_transfer));
  
  CUDA_CHECK(cudaMemcpyAsync(d_A33Entries2, A33->GetEntries(), nz_A * sizeof(double), cudaMemcpyHostToDevice,stream_transfer));

  // int rank=0;

  // int* rowptr=A11->GetRowPtr();
  
    
  //   #ifdef _MPI
  //   MPI_Comm_rank(TDatabase::ParamDB->Comm,&rank);
  //   #endif
  
  // if(rank==0){
  //       cout<<"BIT:"<<cblas_dnrm2(nz_BT,B1T->GetEntries(),1)<<"\t"<<rowptr[3]<<"\t"<<endl;
  // }
  CUDA_CHECK(cudaMemcpyAsync(d_BTRowPtr2, StructureBT->GetRowPtr(), (n_BT+1) * sizeof(int), cudaMemcpyHostToDevice,stream_transfer));
    
  CUDA_CHECK(cudaMemcpyAsync(d_BTKCol2, StructureBT->GetKCol(), nz_BT * sizeof(int), cudaMemcpyHostToDevice,stream_transfer));
  
  CUDA_CHECK(cudaMemcpyAsync(d_B1TEntries2, B1T->GetEntries(), nz_BT * sizeof(double), cudaMemcpyHostToDevice,stream_transfer));
  
  CUDA_CHECK(cudaMemcpyAsync(d_B2TEntries2, B2T->GetEntries(), nz_BT * sizeof(double), cudaMemcpyHostToDevice,stream_transfer));
  
  CUDA_CHECK(cudaMemcpyAsync(d_B3TEntries2, B3T->GetEntries(), nz_BT * sizeof(double), cudaMemcpyHostToDevice,stream_transfer));


  CUDA_CHECK(cudaMemcpyAsync(d_BRowPtr2, StructureB->GetRowPtr(), (n_B+1) * sizeof(int), cudaMemcpyHostToDevice,stream_transfer));
      
  CUDA_CHECK(cudaMemcpyAsync(d_BKCol2, StructureB->GetKCol(), nz_B * sizeof(int), cudaMemcpyHostToDevice,stream_transfer));
  
  CUDA_CHECK(cudaMemcpyAsync(d_B1Entries2, B1->GetEntries(), nz_B * sizeof(double), cudaMemcpyHostToDevice,stream_transfer));
  
  CUDA_CHECK(cudaMemcpyAsync(d_B2Entries2, B2->GetEntries(), nz_B * sizeof(double), cudaMemcpyHostToDevice,stream_transfer));
  
  CUDA_CHECK(cudaMemcpyAsync(d_B3Entries2, B3->GetEntries(), nz_B * sizeof(double), cudaMemcpyHostToDevice,stream_transfer));

}

#endif

/** replace level i by given MGLevel and return old level */
TNSE_MGLevel *TNSE_MultiGrid::ReplaceLevel(int i, 
                                           TNSE_MGLevel *MGLevel)
{
  TNSE_MGLevel *ret;

  ret = MultiGridLevels[i];

  MultiGridLevels[i] = MGLevel;

  USpaces[i] = MGLevel->GetUSpace();
  PSpaces[i] = MGLevel->GetPSpace();

  if (i>=N_Levels)
    N_Levels = i+1;
  return ret;
}

/** get parameter */
double TNSE_MultiGrid::GetParam(int i)
{
    if (i<N_Parameters)
	return Parameters[i];
    else
	return -4711;
}

/** get parameter */
void TNSE_MultiGrid::SetParam(int i, double a)
{
    if (i<N_Parameters)
	Parameters[i] = a;
    else
    {
	OutPut("TNSE_MultiGrid::SetParam: not enough parameters " << endl);
	exit(4711);
    }
}

/** restrict u1, u2 from finest grid to all coarser grids */
void TNSE_MultiGrid::RestrictToAllGrids()
{
  TNSE_MGLevel *CurrentLevel, *CoarserLevel; 
  int lev, j;
  double *CurrentU1, *CoarserU1, *CoarserAux;
#ifdef _MPI
  TParFECommunicator3D *parComm_U;
#endif
int N_U,N_P;
  for(lev=N_Levels-1;lev>0;lev--)
  {
    CurrentLevel = MultiGridLevels[lev];
    CurrentLevel->GetSolutionVector(CurrentU1);

    CoarserLevel = MultiGridLevels[lev-1];
    CoarserLevel->GetSolutionVector(CoarserU1);
    CoarserAux = CoarserLevel->GetAuxVector(1);

    #ifdef _MPI
    TDatabase::ParamDB->time_restriction_start = MPI_Wtime();
    #else
    TDatabase::ParamDB->time_restriction_start = GetTime();
    #endif
    
    RestrictFunction(USpaces[lev-1], USpaces[lev], GEO_DIM,
                     CoarserU1, CurrentU1, CoarserAux);
#ifdef _MPI
    parComm_U = CoarserLevel->GetParCommU();
    parComm_U->CommUpdateReduce(CoarserU1);
    parComm_U->CommUpdate(CoarserU1);
#endif
    
    #ifdef _MPI
    TDatabase::ParamDB->time_restriction_end =  MPI_Wtime();
    #else
    TDatabase::ParamDB->time_restriction_end =  GetTime();    
    #endif
    TDatabase::ParamDB->time_restriction = TDatabase::ParamDB->time_restriction_end - TDatabase::ParamDB->time_restriction_start;

  } // endfor lev
} // RestrictToAllGrids

/** one cycle on level i */
void TNSE_MultiGrid::Cycle(int i, double &res)
{
  int j, N_UDOF, N_PDOF, N_DOF, maxit, smoother, NLevels=N_Levels;
  int slc, gam, defect_calc, umfpack_flag, ii;
  double oldres, res2, s;
  
  TNSE_MGLevel *CurrentLevel, *CoarserLevel;
  double *CurrentU, *CurrentP;
  double *OldU, *OldP;
  double *CoarserU, *CoarserP;
  double *CoarserRhsU, *CoarserRhsP;
  double *CurrentRhsU, *CurrentRhsP;
  double *CurrentOldDefU, *CurrentOldDefP;
  double *CurrentDefectU, *CurrentDefectP;
  double *CurrentAux, *CurrentCounter;
  double *defect, alpha;
  double divfactor = TDatabase::ParamDB->SC_DIV_FACTOR;
  int rank=0;
  double t1,t2;
  
#ifdef _MPI
  MPI_Comm_rank(TDatabase::ParamDB->Comm,&rank);
#endif
// =================== THE VARIABLE i INDICATES THE multigrid level ====================
  // ======================	OBTAIN INFO. ABOUT THAT LEVEL	==================
  
  CurrentLevel = MultiGridLevels[i];
  N_UDOF = CurrentLevel->GetN_UDOF();
  N_PDOF = CurrentLevel->GetN_PDOF();
  N_DOF = GEO_DIM*N_UDOF + N_PDOF;
  
// =====================	OBTAIN : SOL, RHS, DEFECT-VECTOR, AUX	===============  
  
  CurrentLevel->GetSolutionVector(CurrentU);
  CurrentP = CurrentU + GEO_DIM*N_UDOF; 
  CurrentLevel->GetRhsVector(CurrentRhsU);
  CurrentRhsP = CurrentRhsU + GEO_DIM*N_UDOF; 
  CurrentDefectU = CurrentLevel->GetAuxVector(0);
  CurrentDefectP = CurrentDefectU+GEO_DIM*N_UDOF;
  CurrentAux = CurrentLevel->GetAuxVector(1);
  
// =================================================================
    
  slc =0;
  if ((TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_ALL_SADDLE)||
      (TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_FINE_SADDLE))
    slc = 1;
  if (slc)
  {
    CurrentCounter = CurrentLevel->GetAuxVector(2);
    CurrentOldDefU = CurrentCounter;
    CurrentOldDefP = CurrentOldDefU+GEO_DIM*N_UDOF;
    CurrentCounter = CurrentLevel->GetAuxVector(3);
    OldU = CurrentCounter;
    OldP = OldU+GEO_DIM*N_UDOF;
  }

#ifdef _MPI

  TParFECommunicator3D *ParCommU,*ParCommP;
  ParCommU = CurrentLevel->GetParCommU();
  ParCommP = CurrentLevel->GetParCommP();
 
#endif
  
// ====================		===============		===========================  

  CurrentLevel->CorrectNodes(CurrentU); // for hanging nodes
  SetDirichletNodes(i);
  CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentAux, res);// get initial residual

  
// =============================	START MULTIGRID ROUTINE=====================================
  //===============================================================================================//
  //============		=============		=====================		===========//


if(TDatabase::ParamDB->MG_DEBUG && rank==0)
{
  cout << "// =============================	START MULTIGRID LEVEL:: " << i << " RESIDUAL :: " << res << "====================================="<< endl; 
  cout << "//===============================================================================================//" << endl;
  cout << "//============		=============		=====================		===========//" << endl;
}

// if(rank==0)
//     cout<<"norm:"<<res<<"\t"<<i<<"\t"<<mg_recursions[i]<<endl;


  
// ===============	HANDLE THE SPECIAL CASE OF COARSEST LEVEL	==========================//
  if(i==0)                     
  {

    // if(rank==0)
    // cout<<"norm base:"<<res<<"\t"<<i<<endl;
        // if(rank==0)
        // cout<<"norm:"<<res<<endl;
    // ===========		SET OF DAT VARIABLES USED	::::::::::::
        // =====	SC_COARSE_SMOOTHER_SADDLE::	TO SET THE SOLVER AT THE COARSE LEVEL
        //=======	SC_COARSE_MAXIT_SADDLE::	SET MAX ITERATIONS WITH ITERATIVE SOLVER
        //=======	SC_COARSE_RED_FACTOR_SADDLE::	factor to check reduction in residual
        
        smoother = TDatabase::ParamDB->SC_COARSE_SMOOTHER_SADDLE;
        
        #ifdef _MPI
          t1 = MPI_Wtime();
        #else
          t1 = GetTime();
        #endif
        
        switch(smoother)
        {
          case 17 : 
            CurrentLevel->SolveExact(CurrentU, CurrentRhsU);
            CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, res);
            if (TDatabase::ParamDB->SC_VERBOSE>=2)
      {
          if (res>1e-12)
          {
        OutPut("MESSAGE: residual not zero !! ("<<res); 
        OutPut(") Check boundary conditions !! "<< endl);
          }
      }
          break;
          
          case 18 : // Direct Solver with UMFPACK
            //OutPut("This should be done by UMFPACK in the future" << endl);

        umfpack_flag = (int) Parameters[9];
            CurrentLevel->SolveExactUMFPACK(CurrentU, CurrentRhsU, umfpack_flag);
      Parameters[9] = umfpack_flag;
            CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, res);
            if (TDatabase::ParamDB->SC_VERBOSE>=2)
      {
          if (res>1e-12)
          {
        OutPut("MESSAGE: residual not zero !! ("<<res); 
        OutPut(") Check boundary conditions !! "<< endl);
          }
      }
          break;
      
        //	===============		1,2 - CELL-VANKA ITERATIVE SOLVERS 	====================
          
          case 1 :
          case 2 :   
      
      //	==============		STEP-LENGTH-CONTROL	===============
      
            slc = 0;
            if (TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_ALL_SADDLE)
              slc = 1;
            else 
              if ((TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_FINE_SADDLE)
                    &&(i==N_Levels-1))
                slc = 1;
      
        
            maxit = TDatabase::ParamDB->SC_COARSE_MAXIT_SADDLE; // max # of iterations
            j = 0;

            // ==================		START OF COARSE GRID COMPUTATION 	===============
      
          // ======================	CHECK DEFECT	===============
            res2 = oldres = res ;
            if (TDatabase::ParamDB->SC_VERBOSE>=2)
            {
              OutPut("smoother " <<smoother <<" , level 0 res before smoothing: " << oldres << endl);
            }
            
          // ======================	END OF CHECK DEFECT	===============
          
            res2 = res = oldres;
            divfactor *= res;
            if (slc)
      {
          ii = N_DOF*SizeOfDouble;
              memcpy(OldU, CurrentU, ii);
              memcpy(CurrentOldDefU, CurrentAux, ii);
            }
            
            
            //	============	loop starts	=========================
            
            while (((res>TDatabase::ParamDB->SC_COARSE_RED_FACTOR_SADDLE*oldres)
                || (j==0)))
            { 
              // apply smother
              CurrentLevel->CellVanka(CurrentU, CurrentRhsU, CurrentAux,
                                      N_Parameters, Parameters, smoother, N_Levels);         
        
              // compute defect
        // =================	DEFECT ROUTINE WILL UPDATE SOLUTION IN PARALLEL ======
        
              CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, res);// Required for stopping criteria
              if (res > divfactor)
              {
                OutPut("mesh cell Vanka : coarse grid solver diverged " <<  res  <<  " j :: " << j << endl);
                exit(4711);
              }
              res2 = res;
              // cout << "residual " << j << " " << res << endl;
        
        j++;	  
              // maxit reached             
              if(j>=maxit) break;
            
            }
            
            //===================	END OF OBTAINING SOLUTION	===============
            
            if (TDatabase::ParamDB->SC_VERBOSE >=2 )
            {
              OutPut("smoother " << smoother <<", level 0 res after smoothing: "<< res << " reduction " <<  res/oldres );
              OutPut(" number of Vanka iters: " << j << endl);
            }
            
            //=================	GET DEFECT	===============================
            
            if (slc)
            { 
              alpha = CurrentLevel->StepLengthControl(CurrentU, OldU, CurrentOldDefU,
                                  N_Parameters,Parameters);       
          
              for (j=0;j<N_DOF;j++)
                CurrentU[j] = OldU[j] + alpha *( CurrentU[j]-OldU[j]);
            }
            //	==============	MAKE CHANGES AS PER STEP CONTROL	================
            
      // ==================		END OF COARSE GRID COMPUTATION 	===============       
        
        break;
        
        case 10:
        case 20:
        case 11:
        case 21:
        case 12:
        case 22:
        case 13:
        case 23:
            
              //	==============		STEP-LENGTH-CONTROL	===============
      
            slc = 0;
            if (TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_ALL_SADDLE)
              slc = 1;
            else 
              if ((TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_FINE_SADDLE)
                    &&(i==N_Levels-1))
                slc = 1;
      
        
            maxit = TDatabase::ParamDB->SC_COARSE_MAXIT_SADDLE; // max # of iterations
            j = 0;

            // ==================		START OF COARSE GRID COMPUTATION 	===============
      
          // ======================	CHECK DEFECT	===============
            res2 = oldres = res ;
            if (TDatabase::ParamDB->SC_VERBOSE>=2)
            {
              OutPut("smoother " <<smoother <<" , level 0 res before smoothing: " << oldres << endl);
            }
            
          // ======================	END OF CHECK DEFECT	===============
          
            res2 = res = oldres;
            divfactor *= res;
            if (slc)
      {
          ii = N_DOF*SizeOfDouble;
              memcpy(OldU, CurrentU, ii);
              memcpy(CurrentOldDefU, CurrentAux, ii);
            }
            
            
            //	============	loop starts	=========================
            
            while (((res>TDatabase::ParamDB->SC_COARSE_RED_FACTOR_SADDLE*oldres)
                || (j==0)))
            { 
                switch(smoother)
                {
        #ifdef _CUDA
              #ifdef _MPI
                #ifdef _HYBRID
        case 10:
        case 20:
            cout<<"CellVanka_GPU"<<endl;
            CurrentLevel->CellVanka_GPU(CurrentU, CurrentRhsU, CurrentAux,
                                      N_Parameters, Parameters,smoother,CoarseSmooth, stream_smooth, d_ARowPtr1, d_AKCol1,
                                      d_A11Entries1,  d_A12Entries1,  d_A13Entries1,
                                      d_A21Entries1,  d_A22Entries1,  d_A23Entries1,
                                      d_A31Entries1,  d_A32Entries1,  d_A33Entries1,
                                      d_BTRowPtr1,   d_BTKCol1,
                                      d_B1TEntries1,  d_B2TEntries1,  d_B3TEntries1,
                                      d_BRowPtr1,   d_BKCol1,
                                      d_B1Entries1,  d_B2Entries1,  d_B3Entries1);
            j=maxit;
            break;
            
        case 11:
        case 21:
    //         cout<<"CellVanka_CPU_GPU"<<endl;
            CurrentLevel->CellVanka_CPU_GPU(CurrentU, CurrentRhsU, CurrentAux,
                                      N_Parameters, Parameters,smoother,CoarseSmooth, NULL, NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL,
         NULL, NULL, NULL);
            j=maxit;
            break;
            
        case 12:
        case 22:
    //         cout<<"CellVanka_Level_Split"<<endl;
            CurrentLevel->CellVanka_Level_Split(CurrentU, CurrentRhsU, CurrentAux,
                                      N_Parameters, Parameters,smoother,CoarseSmooth);
            
            j=maxit;
            break;
            
        case 13:
        case 23:
    //         cout<<"CellVanka_Level_Split"<<endl;
            CurrentLevel->CellVanka_Combo(CurrentU, CurrentRhsU, CurrentAux,
                                      N_Parameters, Parameters,smoother,CoarseSmooth, NULL, NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL,
         NULL, NULL, NULL);
            
            j=maxit;
            break;
        
                #endif
            #endif
        #endif      
                }
        
              // compute defect
        // =================	DEFECT ROUTINE WILL UPDATE SOLUTION IN PARALLEL ======
        
              CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, res);// Required for stopping criteria
              if (res > divfactor)
              {
                OutPut("mesh cell Vanka : coarse grid solver diverged " <<  res  <<  " j :: " << j << endl);
                exit(4711);
              }
              res2 = res;
              // cout << "residual " << j << " " << res << endl;
        
        j++;	  
              // maxit reached             
              if(j>=maxit) break;
            
            }
            
            //===================	END OF OBTAINING SOLUTION	===============
            
            if (TDatabase::ParamDB->SC_VERBOSE >=2 )
            {
              OutPut("smoother " << smoother <<", level 0 res after smoothing: "<< res << " reduction " <<  res/oldres );
              OutPut(" number of Vanka iters: " << j << endl);
            }
            
            //=================	GET DEFECT	===============================
            
            if (slc)
            { 
              alpha = CurrentLevel->StepLengthControl(CurrentU, OldU, CurrentOldDefU,
                                  N_Parameters,Parameters);       
          
              for (j=0;j<N_DOF;j++)
                CurrentU[j] = OldU[j] + alpha *( CurrentU[j]-OldU[j]);
            }
            //	==============	MAKE CHANGES AS PER STEP CONTROL	================
            
      // ==================		END OF COARSE GRID COMPUTATION 	=============== 
      
            
        break;      
        
      //================	3,4 - NODAL-VANKA ITERATIVE SOLVERS	==========================
      
          case 3 :
          case 4 :
              cout<<"coarse smooth"<<endl;
            slc =0;
            if (TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_ALL_SADDLE)
              slc = 1;
            else
              if ((TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_FINE_SADDLE)
                    &&(i==N_Levels-1))
                slc = 1;
                
            maxit = TDatabase::ParamDB->SC_COARSE_MAXIT_SADDLE; // max # of iterations
            j = 0;
        
            oldres = res;
            
      if (TDatabase::ParamDB->SC_VERBOSE>=2)
              OutPut("smoother " << smoother <<"level 0 res before smoothing: " << oldres << endl);
      
            if (slc)
      {
              memcpy(OldU, CurrentU, N_DOF*SizeOfDouble);
              memcpy(CurrentOldDefU, CurrentAux, N_DOF*SizeOfDouble); // required for slc
            }        	
            // iterate
            while (((res>TDatabase::ParamDB->SC_COARSE_RED_FACTOR_SADDLE*oldres)
                || (j==0)))
      { 
        CurrentLevel->NodalVanka(CurrentU, CurrentRhsU, CurrentAux, 
                                      N_Parameters, Parameters,smoother,N_Levels);	
        // make sure CuurentU and Rhs are updated while returning from nodal vanka
        CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, res);               	  
        
        if (TDatabase::ParamDB->SC_VERBOSE >=2)          
                OutPut("level 0: Vanka ite: " << j << " " << res << endl);

              if (res > divfactor)
        {
                OutPut("nodal Vanka : coarse grid solver diverged " <<  res <<  " j :: " << j << endl);
                exit(4711);
              }
              
              res2 = res;
              // cout << "residual " << j << " " << res << endl;
        
        j++;
              // maxit reached             
              if(j>=maxit) break;
            }
            
            if (TDatabase::ParamDB->SC_VERBOSE >=2)
      {
        
        OutPut("smoother " << smoother <<"level 0 res after smoothing: "<< res 
        << " reduction " <<  res/oldres );
              OutPut(" number of Vanka iters: " << j << endl);
            }
            
            if (slc)
      { 
              alpha = CurrentLevel->StepLengthControl(CurrentU, OldU, CurrentOldDefU, 
                                  N_Parameters,Parameters);       
            
              for (j=0;j<N_DOF;j++)
                CurrentU[j] = OldU[j] + alpha *( CurrentU[j]-OldU[j]);
            }
            
          break;
          
        case 30:
        case 40:
        case 31:
        case 41:
        case 32:
        case 42:
            
              //	==============		STEP-LENGTH-CONTROL	===============
      
            slc = 0;
            if (TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_ALL_SADDLE)
              slc = 1;
            else 
              if ((TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_FINE_SADDLE)
                    &&(i==N_Levels-1))
                slc = 1;
      
        
            maxit = TDatabase::ParamDB->SC_COARSE_MAXIT_SADDLE; // max # of iterations
            j = 0;

            // ==================		START OF COARSE GRID COMPUTATION 	===============
      
          // ======================	CHECK DEFECT	===============
            res2 = oldres = res ;
            if (TDatabase::ParamDB->SC_VERBOSE>=2)
            {
              OutPut("smoother " <<smoother <<" , level 0 res before smoothing: " << oldres << endl);
            }
            
          // ======================	END OF CHECK DEFECT	===============
          
            res2 = res = oldres;
            divfactor *= res;
            if (slc)
      {
          ii = N_DOF*SizeOfDouble;
              memcpy(OldU, CurrentU, ii);
              memcpy(CurrentOldDefU, CurrentAux, ii);
            }
            
            
            //	============	loop starts	=========================
            
            while (((res>TDatabase::ParamDB->SC_COARSE_RED_FACTOR_SADDLE*oldres)
                || (j==0)))
            { 
                switch(smoother)
                {
        #ifdef _CUDA
              #ifdef _MPI
                #ifdef _HYBRID
        case 30:
        case 40:
    //         cout<<"CellVanka_GPU"<<endl;
            CurrentLevel->NodalVanka_GPU(CurrentU, CurrentRhsU, CurrentAux,
                                      N_Parameters, Parameters,smoother,CoarseSmooth);
            j=maxit;
            break;
            
        case 31:
        case 41:
    //         cout<<"CellVanka_CPU_GPU"<<endl;
            CurrentLevel->NodalVanka_CPU_GPU(CurrentU, CurrentRhsU, CurrentAux,
                                      N_Parameters, Parameters,smoother,CoarseSmooth);
            j=maxit;
            break;
            
        case 32:
        case 42:
    //         cout<<"CellVanka_Level_Split"<<endl;
            CurrentLevel->NodalVanka_Level_Split(CurrentU, CurrentRhsU, CurrentAux,
                                      N_Parameters, Parameters,smoother,CoarseSmooth);
            
            j=maxit;
            break;
        
                #endif
            #endif
        #endif      
                }
        
              // compute defect
        // =================	DEFECT ROUTINE WILL UPDATE SOLUTION IN PARALLEL ======
        
              CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, res);// Required for stopping criteria
              if (res > divfactor)
              {
                OutPut("mesh cell Vanka : coarse grid solver diverged " <<  res  <<  " j :: " << j << endl);
                exit(4711);
              }
              res2 = res;
              // cout << "residual " << j << " " << res << endl;
        
        j++;	  
              // maxit reached             
              if(j>=maxit) break;
            
            }
            
            //===================	END OF OBTAINING SOLUTION	===============
            
            if (TDatabase::ParamDB->SC_VERBOSE >=2 )
            {
              OutPut("smoother " << smoother <<", level 0 res after smoothing: "<< res << " reduction " <<  res/oldres );
              OutPut(" number of Vanka iters: " << j << endl);
            }
            
            //=================	GET DEFECT	===============================
            
            if (slc)
            { 
              alpha = CurrentLevel->StepLengthControl(CurrentU, OldU, CurrentOldDefU,
                                  N_Parameters,Parameters);       
          
              for (j=0;j<N_DOF;j++)
                CurrentU[j] = OldU[j] + alpha *( CurrentU[j]-OldU[j]);
            }
            //	==============	MAKE CHANGES AS PER STEP CONTROL	================
            
      // ==================		END OF COARSE GRID COMPUTATION 	=============== 
      
            
        break;   
        

          case 5 :
            slc =0;
            if (TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_ALL_SADDLE)
              slc = 1;
            else
              if ((TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_FINE_SADDLE)
                    &&(i==N_Levels-1))
                slc = 1;
      
            maxit = TDatabase::ParamDB->SC_COARSE_MAXIT_SADDLE; // max # of iterations
            
            j = 0;

            res2 = oldres = res;
            
      if (slc)
      {
              memcpy(OldU, CurrentU, N_DOF*SizeOfDouble);
              memcpy(CurrentOldDefU, CurrentAux, N_DOF*SizeOfDouble);
            }

            // iterate
            while (((res>TDatabase::ParamDB->SC_COARSE_RED_FACTOR_SADDLE*oldres)
                    && (res>0.1*TDatabase::ParamDB->SC_LIN_RES_NORM_MIN_SADDLE))
                  || (j==0))
      {
        
              // apply smother
              CurrentLevel->BraessSarazin(CurrentU, CurrentDefectU, CurrentAux,
                                          N_Parameters, Parameters,N_Levels);
              j++;
              // maxit reached             
              if(j>=maxit) break;
              // compute defect
              CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, res);
              if (res > divfactor)
              {
                OutPut("Braess-Sarazin : coarse grid solver diverged " <<  res  << endl);
                exit(4711);
              }
              res2 = res;
              // cout << "residual " << j << " " << res << endl;
            }
            
            if (TDatabase::ParamDB->SC_VERBOSE >=2){
              
        // compute defect
              CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, res);
              OutPut("level 0: number of Braess Sarazin iters: " << j << " " << res/oldres << endl);
            }
            
            if (slc)
      { 
              alpha = CurrentLevel->StepLengthControl(CurrentU, 
                                                      OldU, CurrentOldDefU,
                                                      N_Parameters,Parameters);       
          
              for (j=0;j<N_DOF;j++)
                CurrentU[j] = OldU[j] + alpha *( CurrentU[j]-OldU[j]);
            }
      #ifdef _MPI        
        break; 
        case 25:
        
        ((TNSE_MGLevel4*)CurrentLevel)->Par_Directsolve(CurrentU,CurrentRhsU);
        CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, res);
        
          break;
            case 15:
        
        ((TNSE_MGLevel4*)CurrentLevel)->gmres_solve(CurrentU,CurrentRhsU);
        CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, res);
        #endif
          break;

          default :
            OutPut("coarse smoother not found !! Set SC_COARSE_SMOOTHER properly !! " << smoother);
            OutPut(endl);
            exit(4711);
        }// end iterative solver

        #ifdef _MPI
          t2 = MPI_Wtime();
        #else
          t2 = GetTime();
        #endif

        tSmoother2 += t2-t1 ;
        
          if(TDatabase::ParamDB->MG_DEBUG && rank==0)
          {	 
            printf("end of reached cycle %d %d \n",i,j);
          }
  } // end coarsest grid
 
 // ===============	END OF - HANDLE THE SPECIAL CASE OF COARSEST LEVEL	===================//  
 // not the coarsest grid *******************************************************
  else                        
  {                            // pre smoothing
                               // check if step length control is required
    slc = 0;
    if (TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_ALL_SADDLE)
      slc = 1;
    else if ((TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_FINE_SADDLE)&&(i==N_Levels-1))
      slc = 1;
    
    defect_calc = 0;

    //==============	BEGIN WITH OBTAINING THE COARSER LEVEL INFORMATION	=========================
    
    CoarserLevel = MultiGridLevels[i-1];
    ii = GEO_DIM*CoarserLevel->GetN_UDOF();
    int N_coarse_DOF = ii;
    CoarserLevel->GetSolutionVector(CoarserU);
    CoarserP = CoarserU + ii;
    CoarserLevel->GetRhsVector(CoarserRhsU);
    CoarserRhsP = CoarserRhsU + ii;

    //==============	=======================	=========================	=====================//
    
    #ifdef _MPI
    TParFECommunicator3D *CoarseParCommU,*CoarseParCommP;
    CoarseParCommU = CoarserLevel->GetParCommU();
    CoarseParCommP = CoarserLevel->GetParCommP();
    #endif
    
    // if step length control
    if (slc)
    {
      oldres = res;
      CurrentLevel->CorrectNodes(CurrentU);
      CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, oldres);
      ii = N_DOF*SizeOfDouble;
      memcpy(OldU, CurrentU, ii);
      memcpy(CurrentOldDefU, CurrentDefectU, ii);
    }
    
    
    if (TDatabase::ParamDB->SC_VERBOSE>=2)
    {
      if (!defect_calc)
      {      
        defect_calc = 1;
      }
      
      #ifndef _MPI      
      OutPut("level " << i << " ");
      OutPut("res before presmoothing: " << " " << oldres << endl);       
      #endif
    }

    
    // ====================	FINE GRID COMPUTATION STARTS	=============================
if(TDatabase::ParamDB->MG_DEBUG && rank==0)
{   
  cout << "---------------	start of pre-smoothing LEVEL :: "<< i << "---------  res  " << oldres << "---------- "<<endl;
}
 

    // ===============		PRE-SMOOTHING		============== //
    
    smoother = TDatabase::ParamDB->SC_SMOOTHER_SADDLE;
//     if(i!=N_Levels-1)
//       smoother = 4;

    #ifdef _MPI
          t1 = MPI_Wtime();
        #else
          t1 = GetTime();
        #endif

        
    switch(smoother)
    {
      case 1 :
      case 2 :
        for(j=0;j<TDatabase::ParamDB->SC_PRE_SMOOTH_SADDLE;j++)
	{
          
          CurrentLevel->CellVanka(CurrentU, CurrentRhsU, CurrentAux,
                                  N_Parameters, Parameters,smoother,N_Levels);  
	}
	break;
    
    #ifdef _CUDA
          #ifdef _MPI
            #ifdef _HYBRID
    case 10:
    case 20:
         cout<<"CellVanka_GPU"<<endl;
        CurrentLevel->CellVanka_GPU(CurrentU, CurrentRhsU, CurrentAux,
                                  N_Parameters, Parameters,smoother,PreSmooth, stream_smooth, d_ARowPtr1, d_AKCol1,
                                  d_A11Entries1,  d_A12Entries1,  d_A13Entries1,
                                  d_A21Entries1,  d_A22Entries1,  d_A23Entries1,
                                  d_A31Entries1,  d_A32Entries1,  d_A33Entries1,
                                  d_BTRowPtr1,   d_BTKCol1,
                                  d_B1TEntries1,  d_B2TEntries1,  d_B3TEntries1,
                                  d_BRowPtr1,   d_BKCol1,
                                  d_B1Entries1,  d_B2Entries1,  d_B3Entries1);
        break;
        
    case 11:
    case 21:
//         cout<<"CellVanka_CPU_GPU"<<endl;
        CurrentLevel->CellVanka_CPU_GPU(CurrentU, CurrentRhsU, CurrentAux,
                                  N_Parameters, Parameters,smoother,PreSmooth, NULL, NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL,
         NULL, NULL, NULL);
        break;
        
    case 12:
    case 22:
//         cout<<"CellVanka_Level_Split"<<endl;
        CurrentLevel->CellVanka_Level_Split(CurrentU, CurrentRhsU, CurrentAux,
                                  N_Parameters, Parameters,smoother,PreSmooth);
        break;
        
    case 13:
    case 23:
//         cout<<"CellVanka_Level_Split"<<endl;
        CurrentLevel->CellVanka_Combo(CurrentU, CurrentRhsU, CurrentAux,
                                  N_Parameters, Parameters,smoother,PreSmooth, NULL, NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL,
         NULL, NULL, NULL);
        break;
    
            #endif
        #endif
    #endif
	
        
      case 3 :
      case 4 :  	
	cout<<"pre smooth"<<endl;
        for(j=0;j<TDatabase::ParamDB->SC_PRE_SMOOTH_SADDLE;j++)
	{	  
	  CurrentLevel->NodalVanka(CurrentU, CurrentRhsU, CurrentAux, 
                                   N_Parameters, Parameters,smoother,N_Levels);
	}
	
         break;
         
#ifdef _CUDA
    #ifdef _MPI
        #ifdef _HYBRID
    case 30:
    case 40:
//         cout<<"CellVanka_GPU"<<endl;
        CurrentLevel->NodalVanka_GPU(CurrentU, CurrentRhsU, CurrentAux,
                                  N_Parameters, Parameters,smoother,PreSmooth);
        break;
        
    case 31:
    case 41:
//         cout<<"CellVanka_CPU_GPU"<<endl;
        CurrentLevel->NodalVanka_CPU_GPU(CurrentU, CurrentRhsU, CurrentAux,
                                  N_Parameters, Parameters,smoother,PreSmooth);
        break;
        
    case 32:
    case 42:
//         cout<<"CellVanka_Level_Split"<<endl;
        CurrentLevel->NodalVanka_Level_Split(CurrentU, CurrentRhsU, CurrentAux,
                                  N_Parameters, Parameters,smoother,PreSmooth);
        break;
    
        #endif
    #endif
#endif
        
         
      case 5 :         
        if (!defect_calc)
          CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, oldres);        
        oldres = sqrt(Ddot(2*N_UDOF, CurrentU,CurrentU));
        for(j=0;j<TDatabase::ParamDB->SC_PRE_SMOOTH_SADDLE;j++)
        {
          CurrentLevel->BraessSarazin(CurrentU, CurrentDefectU, CurrentAux, 
                                      N_Parameters, Parameters, N_Levels);
        }
       
	#ifdef _MPI        
       break; 
      case 25:
	  
	   ((TNSE_MGLevel4*)CurrentLevel)->Par_Directsolve(CurrentU,CurrentRhsU);
	   CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, res);
	break;
	    case 15:
	  
	   ((TNSE_MGLevel4*)CurrentLevel)->gmres_solve(CurrentU,CurrentRhsU);
	   CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, res);
	   #endif
        break;
      default :
        OutPut("smoother not found !! Set SC_SMOOTHER properly !!"<< endl);
        Error("smoother not found !! Set SC_SMOOTHER properly !!"<< endl);
        exit(4711);
    }

    #ifdef _MPI
          t2 = MPI_Wtime();
    #else
          t2 = GetTime();
    #endif

        tSmoother2 += t2-t1 ;
    
    //==========	END OF PRE-SMOOTHING 	================================// 
    CurrentLevel->CorrectNodes(CurrentU); // Required only for hanging case
    CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, res);
    if(TDatabase::ParamDB->MG_DEBUG && rank==0)
    {	    
      cout << "======	END OF VANKA SMOOTHING RESIDUAL	::" << res << endl;
      cout << "---------------	END of pre-smoothing LEVEL :: "<< i << "------------------------" <<endl;
    }
  
  // ========		CALCULATE DEFECT BASED ON SLC 		=================== //
   
    if (!slc)
    {
      oldres = res;
    } 
    else
    {
      alpha = CurrentLevel->StepLengthControl(CurrentU, OldU, CurrentOldDefU, N_Parameters,Parameters);       
      
      for (j=0;j< N_DOF;j++)
        CurrentU[j] = OldU[j] + alpha *( CurrentU[j]-OldU[j]);
      CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, oldres);
     // oldres = res;
      ii = N_DOF*SizeOfDouble;
      memcpy(OldU, CurrentU, ii);
      memcpy(CurrentOldDefU, CurrentDefectU, ii);
    }

    if (TDatabase::ParamDB->SC_VERBOSE>=2 
#ifdef _MPI
   &&    rank==0
#endif
    )
    {
        OutPut("level " << i << " ");
        OutPut("res after presmoothing: " << " " << oldres << endl);
    }

    // ========		END OF CALCULATE DEFECT BASED ON SLC 		=================== //
        
    //	=========	UPDATE DEFECT IN FINER LEVEL ACROSS PROCESSES		===============//
    //	=================	UPDATE RHS ACROSS COARSER PROCESSES	================ //

        
    // =========	END OF UPDATE DEFECT ACROSS PROCESSES		===============// 
    
    // calculate u-defect from u*-defect
    
    if(CurrentLevel->GetType() == 5)
    {
      memcpy(CurrentAux, CurrentDefectU, GEO_DIM*N_UDOF*SizeOfDouble);
      ((TNSE_MGLevel5 *)CurrentLevel)->GetDFromDstar(CurrentAux, CurrentDefectU);
    }

    
  if(TDatabase::ParamDB->MG_DEBUG && rank==0)
  {	   
      printf("==============	start of restriction level ::%d  residual :: %lf======\n",i,oldres);
  }
									  
    // ===============		RESTRICT DEFECT TO COARSER LEVEL	=============== //

    #ifdef _MPI
    TDatabase::ParamDB->time_restriction_start = MPI_Wtime();
    #else
    TDatabase::ParamDB->time_restriction_start = GetTime();
    #endif
    
    DefectRestriction(USpaces[i-1], USpaces[i], GEO_DIM,CoarserRhsU, CurrentDefectU, CurrentAux);
    DefectRestriction(PSpaces[i-1], PSpaces[i], CoarserRhsP, CurrentDefectP, CurrentAux);
    
    
    #ifdef _MPI
    CoarseParCommU->CommUpdateReduce(CoarserRhsU);
    CoarseParCommP->CommUpdateReduce(CoarserRhsP);	
    #endif  
    #ifdef _MPI
    CoarseParCommU->CommUpdate(CoarserRhsU);
    CoarseParCommP->CommUpdate(CoarserRhsP);	
    #endif
       
    
    #ifdef _MPI
    TDatabase::ParamDB->time_restriction_end =  MPI_Wtime();
    #else
    TDatabase::ParamDB->time_restriction_end =  GetTime();    
    #endif
    TDatabase::ParamDB->time_restriction = TDatabase::ParamDB->time_restriction_end - TDatabase::ParamDB->time_restriction_start;
    
    
    
    // ===============		END OF RESTRICT DEFECT TO COARSER LEVEL	=============== //
    
    if(TDatabase::ParamDB->MG_DEBUG && rank==0)
    {	 
	printf("==============	end of restriction level ::%d\n		================",i);
    }
    

    
    // calculate u*-representation from u-representation
    if(CoarserLevel->GetType() == 5)
    {
      memcpy(CoarserU, CoarserRhsU, GEO_DIM*CoarserLevel->GetN_UDOF()*SizeOfDouble);
      ((TNSE_MGLevel5 *)CoarserLevel)->GetDstarFromD(CoarserU, CoarserRhsU);
    }

    
    // ============	PREPARE PARAMETERS FOR COARSE GRID COMPUTATION	============= //
    
    CoarserLevel->CorrectDefect(CoarserRhsU);// seet dirichlet dofs to zero
    CoarserLevel->Reset(CoarserU); // set all dofs to zero
    
if(TDatabase::ParamDB->MG_DEBUG && rank==0 )
{	 
  printf("=============~~~~~		start of cycle %d	~~~~~~~===========\n",i);
}
									  
    // ========		CALL NEXT LEVEL IN MULTIGRID CYCLE 	=============== //
    
    for(j=0;j<1;j++)
      Cycle(i-1, res);
    /* F--cycle */

    //     if(rank==0){
    // cout<<"norm1:"<<res<<"\t"<<i<<endl;
    // cout<<"CurrentU:"<<cblas_dnrm2(GEO_DIM*N_UDOF,CurrentU,1)<<endl;
    // cout<<"CurrentP:"<<cblas_dnrm2(N_PDOF,CurrentP,1)<<endl;
    // cout<<"CurrentRhsU:"<<cblas_dnrm2(GEO_DIM*N_UDOF,CurrentRhsU,1)<<endl;
    // cout<<"CurrentRhsP:"<<cblas_dnrm2(N_PDOF,CurrentRhsP,1)<<endl;
    // cout<<"CurrentDefectU:"<<cblas_dnrm2(GEO_DIM*N_UDOF,CurrentDefectU,1)<<endl;
    // cout<<"CurrentDefectP:"<<cblas_dnrm2(N_PDOF,CurrentDefectP,1)<<endl;
    // cout<<"CurrentAux:"<<cblas_dnrm2(GEO_DIM*N_UDOF,CurrentAux,1)<<endl;
    // cout<<"CoarserU:"<<cblas_dnrm2(N_coarse_DOF,CoarserU,1)<<endl;
    // cout<<"CoarserRhsU:"<<cblas_dnrm2(N_coarse_DOF,CoarserRhsU,1)<<endl;
    // cout<<"norm3:"<<res<<"\t"<<i<<endl;
    // }



    if (TDatabase::ParamDB->SC_MG_CYCLE_SADDLE<1) mg_recursions[i] = 1;

    // ==========	END OF COARSE GRID COMPUATION 	===================	//
    
if(TDatabase::ParamDB->MG_DEBUG && rank==0 )
{	 
      printf("=============~~~~~		END of cycle %d	~~~~~~~===========\n",i);
}
     // calculate u from u*
    if(CoarserLevel->GetType() == 5)
    {
      memcpy(CoarserRhsU, CoarserU, GEO_DIM*CoarserLevel->GetN_UDOF()*SizeOfDouble);
      ((TNSE_MGLevel5 *)CoarserLevel)->GetUFromUstar(CoarserRhsU, CoarserU);
    }

 // ===============		POST-SMOOTHING		============== //
 
if(TDatabase::ParamDB->MG_DEBUG && rank==0)
{	 
	  printf("---------	start of prolongation Level :: %d	------------\n",i);
} 

// ===============		PROLONGATE SOLUTION TO FINER LEVEL	=============== //


    #ifdef _MPI
    TDatabase::ParamDB->time_projection_start = MPI_Wtime();
    #else
    TDatabase::ParamDB->time_projection_start = GetTime();
    #endif
    
    Prolongate(USpaces[i-1], USpaces[i],GEO_DIM,
		  CoarserU, CurrentDefectU, CurrentAux);
     
    Prolongate(PSpaces[i-1], PSpaces[i],  
                   CoarserP, CurrentDefectP, CurrentAux);
    
    
    #ifdef _MPI
    ParCommU->CommUpdateReduce(CurrentDefectU);
    ParCommP->CommUpdateReduce(CurrentDefectP);	
    #endif 
    
    #ifdef _MPI      
    ParCommU->CommUpdate(CurrentDefectU);   
    ParCommP->CommUpdate(CurrentDefectP);
    #endif
    
    #ifdef _MPI
    TDatabase::ParamDB->time_projection_end =  MPI_Wtime();
    #else
    TDatabase::ParamDB->time_projection_end =  GetTime();    
    #endif
    TDatabase::ParamDB->time_projection = TDatabase::ParamDB->time_projection_end - TDatabase::ParamDB->time_projection_start;
    
    // calculate u*-representation from u-representation
    if(CurrentLevel->GetType() == 5)
    {
      memcpy(CurrentAux, CurrentDefectU, GEO_DIM*N_UDOF*SizeOfDouble);
      ((TNSE_MGLevel5 *)CurrentLevel)->GetUstarFromU(CurrentAux, CurrentDefectU);
    }
    if(TDatabase::ParamDB->MG_DEBUG && rank==0)
    {	 
      printf("---------	End of prolongation Level :: %d		------------\n",i);
    }
    
    

    // update dofs
    CurrentLevel->CorrectNodes(CurrentDefectU); // Remove updates for Dirichlet RHS
    CurrentLevel->Update(CurrentU, CurrentDefectU); // Add to U 
    CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, res);
 // ===============		PROLONGATE SOLUTION TO FINER LEVEL	=============== //  
    

    // if(rank==0){
    // cout<<"norm1:"<<res<<"\t"<<i<<endl;
    // cout<<"CurrentU:"<<cblas_dnrm2(GEO_DIM*N_UDOF,CurrentU,1)<<endl;
    // cout<<"CurrentP:"<<cblas_dnrm2(N_PDOF,CurrentP,1)<<endl;
    // cout<<"CurrentRhsU:"<<cblas_dnrm2(GEO_DIM*N_UDOF,CurrentRhsU,1)<<endl;
    // cout<<"CurrentRhsP:"<<cblas_dnrm2(N_PDOF,CurrentRhsP,1)<<endl;
    // cout<<"CurrentDefectU:"<<cblas_dnrm2(GEO_DIM*N_UDOF,CurrentDefectU,1)<<endl;
    // cout<<"CurrentDefectP:"<<cblas_dnrm2(N_PDOF,CurrentDefectP,1)<<endl;
    // cout<<"CurrentAux:"<<cblas_dnrm2(GEO_DIM*N_UDOF,CurrentAux,1)<<endl;
    // cout<<"norm3:"<<res<<"\t"<<i<<endl;
    // }

        

    // apply step length control for prolongation
    if(slc)
    {
      alpha = CurrentLevel->StepLengthControl(CurrentU, OldU, CurrentOldDefU, N_Parameters,Parameters);       
      for (j=0;j<N_DOF;j++)
        CurrentU[j] = OldU[j] + alpha *( CurrentU[j]-OldU[j]);
      CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, res);
      memcpy(OldU, CurrentU, N_DOF*SizeOfDouble);
      memcpy(CurrentOldDefU, CurrentDefectU, N_DOF*SizeOfDouble);
    }

    defect_calc = 0;
    CurrentLevel->CorrectNodes(CurrentU);
    
    if (TDatabase::ParamDB->SC_VERBOSE>=2)
    {
      // compute defect
     // CurrentLevel->CorrectNodes(CurrentU);
      oldres = res;
      defect_calc = 1;
      OutPut("level " << i << " ");
      OutPut("res before postsmoothing: " << oldres << endl);       
    }

 //	===========	POST SMOOTHING COMPUTATION BEGINS 	=================== //
if(TDatabase::ParamDB->MG_DEBUG && rank==0)
{	 
  cout << "---------------	start of post-smoothing LEVEL :: "<< i << " residual :: " << res << " -------- "<<endl;
}
 
//  if(i!=N_Levels-1)
//    smoother = 4;

      #ifdef _MPI
          t1 = MPI_Wtime();
        #else
          t1 = GetTime();
        #endif

        
    switch(smoother)
    {
      case 1 :
      case 2 :
        for(j=0;j<TDatabase::ParamDB->SC_POST_SMOOTH_SADDLE;j++)
	{
          CurrentLevel->CellVanka(CurrentU, CurrentRhsU, CurrentAux, 
                N_Parameters, Parameters, smoother, N_Levels);   
	}
	
        break;
    #ifdef _CUDA
          #ifdef _MPI
            #ifdef _HYBRID
    case 10:
    case 20:
         cout<<"CellVanka_GPU"<<endl;
        CurrentLevel->CellVanka_GPU(CurrentU, CurrentRhsU, CurrentAux,
                                  N_Parameters, Parameters,smoother,PostSmooth, stream_smooth, d_ARowPtr1, d_AKCol1,
                                  d_A11Entries1,  d_A12Entries1,  d_A13Entries1,
                                  d_A21Entries1,  d_A22Entries1,  d_A23Entries1,
                                  d_A31Entries1,  d_A32Entries1,  d_A33Entries1,
                                  d_BTRowPtr1,   d_BTKCol1,
                                  d_B1TEntries1,  d_B2TEntries1,  d_B3TEntries1,
                                  d_BRowPtr1,   d_BKCol1,
                                  d_B1Entries1,  d_B2Entries1,  d_B3Entries1);
        break;
        
    case 11:
    case 21:
        
//         cout<<"CellVanka_CPU_GPU"<<endl;
        CurrentLevel->CellVanka_CPU_GPU(CurrentU, CurrentRhsU, CurrentAux,
                                  N_Parameters, Parameters,smoother,PostSmooth, NULL, NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL,
         NULL, NULL, NULL);
        break;
        
    case 12:
    case 22:
//         cout<<"CellVanka_Level_Split"<<endl;
        CurrentLevel->CellVanka_Level_Split(CurrentU, CurrentRhsU, CurrentAux,
                                  N_Parameters, Parameters,smoother,PostSmooth);
        break;
        
    case 13:
    case 23:
//         cout<<"CellVanka_Level_Split"<<endl;
        CurrentLevel->CellVanka_Combo(CurrentU, CurrentRhsU, CurrentAux,
                                  N_Parameters, Parameters,smoother,PostSmooth, NULL, NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL,
         NULL, NULL, NULL,
         NULL, NULL,
         NULL, NULL, NULL);
        break;
    
            #endif
        #endif
    #endif
      case 3 :
      case 4 :  
	cout<<"post smooth"<<endl;

        for(j=0;j<TDatabase::ParamDB->SC_POST_SMOOTH_SADDLE;j++)
	{          
	  CurrentLevel->NodalVanka(CurrentU, CurrentRhsU, CurrentAux, 
                                    N_Parameters, Parameters,smoother,N_Levels);       	  
	}
	break;

#ifdef _CUDA
    #ifdef _MPI
        #ifdef _HYBRID
        
    case 30:
    case 40:
//         cout<<"CellVanka_GPU"<<endl;
        CurrentLevel->NodalVanka_GPU(CurrentU, CurrentRhsU, CurrentAux,
                                  N_Parameters, Parameters,smoother,PostSmooth);
        break;
        
    case 31:
    case 41:
//         cout<<"CellVanka_CPU_GPU"<<endl;
        CurrentLevel->NodalVanka_CPU_GPU(CurrentU, CurrentRhsU, CurrentAux,
                                  N_Parameters, Parameters,smoother,PostSmooth);
        break;
        
    case 32:
    case 42:
//         cout<<"CellVanka_Level_Split"<<endl;
        CurrentLevel->NodalVanka_Level_Split(CurrentU, CurrentRhsU, CurrentAux,
                                  N_Parameters, Parameters,smoother,PostSmooth);
        break;
    
        #endif
    #endif
#endif
        
        break;
      case 5 :  
        // compute defect
        if (!defect_calc) 
          CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, oldres);
        for(j=0;j<TDatabase::ParamDB->SC_POST_SMOOTH_SADDLE;j++)
        {
          CurrentLevel->BraessSarazin(CurrentU, CurrentDefectU, CurrentAux, 
                                      N_Parameters, Parameters, N_Levels);
        }
	#ifdef _MPI        
       break; 
	  case 25:
	  
	   ((TNSE_MGLevel4*)CurrentLevel)->Par_Directsolve(CurrentU,CurrentRhsU);
	   CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, res);
	break;
	    case 15:
	  
	   ((TNSE_MGLevel4*)CurrentLevel)->gmres_solve(CurrentU,CurrentRhsU);
	   CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, res);
	   #endif
        break;
      default :
        OutPut("smoother not found !! Set SC_SMOOTHER properly !!"<< endl);
        Error("smoother not found !! Set SC_SMOOTHER properly !!"<< endl);
        exit(4711);
    }

    #ifdef _MPI
          t2 = MPI_Wtime();
        #else
          t2 = GetTime();
        #endif

        tSmoother2 += t2-t1 ;
 //	===========	END OF POST SMOOTHING COMPUTATION 	=================== //

    CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, res);
 
    if(TDatabase::ParamDB->MG_DEBUG && rank==0)
    {	   
      cout << "---------------	END of post-smoothing LEVEL :: "<< i << " residual :: " << res << " -------- "<<endl;
    }

 // 
    // apply step length control
    if(slc)
    {
      alpha = CurrentLevel->StepLengthControl(CurrentU, OldU, CurrentOldDefU, N_Parameters,Parameters);             
      for (j=0;j<N_DOF;j++)
        CurrentU[j] = OldU[j] + alpha *( CurrentU[j]-OldU[j]);
      CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, res);
    }
    

  }

	  if(TDatabase::ParamDB->MG_DEBUG && rank==0)
	  {	 
	    cout << "// =============================	END MULTIGRID LEVEL:: " << i << "ROUTINE" << res << "====================================="<< endl; 
	    cout << "//===============================================================================================//" << endl;
	    cout << "//============		=============		=====================		===========//" << endl;
	  }
}

void TNSE_MultiGrid::SetDirichletNodes(int i)
{
  TNSE_MGLevel *CurrentLevel;
  double *CurrentU;
  double *CurrentRhsU;
  int HangingNodeBound, N_Dirichlet, N_UDOF, j;

  if(i>=N_Levels) return;

  CurrentLevel = MultiGridLevels[i];
  N_UDOF = CurrentLevel->GetN_UDOF();
  CurrentLevel->GetSolutionVector(CurrentU);
  CurrentLevel->GetRhsVector(CurrentRhsU);

  HangingNodeBound = CurrentLevel->GetHangingNodeBound();
  N_Dirichlet = CurrentLevel->GetN_Dirichlet();

  for (j=0;j<GEO_DIM;j++)
    memcpy(CurrentU+j*N_UDOF+HangingNodeBound, 
           CurrentRhsU+j*N_UDOF+HangingNodeBound,
           SizeOfDouble*N_Dirichlet);
}

/** return residual on grid i */
double TNSE_MultiGrid::GetResidual(int i)
{
  TNSE_MGLevel *CurrentLevel;
  double *CurrentU;
  double *CurrentRhsU;
  double *CurrentDefectU;
  double oldres;
  
  CurrentLevel = MultiGridLevels[i];

  CurrentLevel->GetSolutionVector(CurrentU);
  CurrentLevel->GetRhsVector(CurrentRhsU);
  CurrentDefectU = CurrentLevel->GetAuxVector(0);

  CurrentLevel->Defect(CurrentU, CurrentRhsU, CurrentDefectU, oldres);

  return oldres;
}

/** set recursion for multigrid */ 
void TNSE_MultiGrid::SetRecursion(int levels)
{
  int gam = TDatabase::ParamDB->SC_MG_CYCLE_SADDLE,k;

  // coarsest grid 
  mg_recursions[1] = 1;
  if (gam>0)
    for (k=2;k<=levels;k++)        
      mg_recursions[k] = gam;
  else                /* F -- cycle */
    for (k=2;k<=levels;k++)        
      mg_recursions[k] = 2;
}
