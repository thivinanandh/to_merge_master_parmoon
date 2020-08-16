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
// %W% %G%
//
// Class:       TMultiGrid3D
// Purpose:     store all data for a multi grid method
//
// Author:      Gunar Matthies 26.06.2000
//
// History:     26.06.2000 start of implementation
//
// =======================================================================

#ifndef __MULTIGRID3D__
#define __MULTIGRID3D__

#include <cuda.h>
#include <cuda_runtime.h>
#include <MGLevel3D.h>
#ifdef _MPI   
   #ifdef __3D__
    #include <ParFECommunicator3D.h>
   #else
    #include <ParFECommunicator2D.h>
   #endif
#endif 
// #define MAXN_LEVELS 25

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

class TMultiGrid3D
{
  protected:
    /** number of levels */
    int N_Levels;

    /** number of problems */
    int N_Problems;

    /** number of parameters */
    int N_Parameters;

    /** array of double parameters */
    double *Parameters;

    /** array of multi grid levels */
    TMGLevel3D *MultiGridLevels[MAXN_LEVELS];

    /** array of FE spaces */
    TFESpace3D *FESpaces[MAXN_LEVELS];

    /** array of function vectors on each level */
    double **FunctionVectors[MAXN_LEVELS];

    /** right-hand side vectors */
    double **RhsVectors[MAXN_LEVELS];

    /** auxiliary vectors */
    double **AuxVectors[MAXN_LEVELS];

    /** number of recursions */
    int mg_recursions[MAXN_LEVELS];

    #ifdef _CUDA
    int* d_RowPtr1;
    int* d_KCol1;
    double* d_Entries1;

    int* d_RowPtr2;
    int* d_KCol2;
    double* d_Entries2;

    double* d_sol1;
    int* d_aux1;

    double* d_sol2;
    int* d_aux2;

    int A1, A2;
    
    int nStreams;
    cudaStream_t stream_transfer;
    cudaStream_t *stream_smooth;
    
    #endif

  public:
    /** constructor */
    TMultiGrid3D(int n_problems, int n_parameters, double *parameters);

     ~TMultiGrid3D(){
#ifdef _CUDA
      CUDA_CHECK( cudaStreamDestroy(stream_transfer));
       for (int i = 0; i < nStreams; ++i)
        CUDA_CHECK( cudaStreamDestroy(stream_smooth[i]) );

      if(d_RowPtr1!=NULL){
      CUDA_CHECK(cudaFree(d_RowPtr1));
      CUDA_CHECK(cudaFree(d_KCol1));
      CUDA_CHECK(cudaFree(d_Entries1));
      CUDA_CHECK(cudaFree(d_RowPtr2));
      CUDA_CHECK(cudaFree(d_KCol2));
      CUDA_CHECK(cudaFree(d_Entries2));

      CUDA_CHECK(cudaFree(d_sol1));
      CUDA_CHECK(cudaFree(d_aux1));

      CUDA_CHECK(cudaFree(d_sol2));
      CUDA_CHECK(cudaFree(d_aux2));
      }
      #endif
     }

    /** return number of multi grid levels */
    int GetN_Levels()
    { return N_Levels; }

    /** add new level as finest */
    void AddLevel(TMGLevel3D *MGLevel);

    /** add new level as finest */
    void ReplaceLevel(int i,TMGLevel3D *MGLevel);

    /** return i-th level as TMGLevel object */
    TMGLevel3D *GetLevel(int i)
    { return MultiGridLevels[i]; }

    /** restrict u1, u2 from finest grid to all coarser grids */
    void RestrictToAllGrids();

    /** set correct values for Dirichlet nodes on grid i */
    void SetDirichletNodes(int i);

    /** cycle on level i */
    void Cycle(int i, double &res);

    void CycleIterative(double &res, int cycle_index);

    /** set recursion for multigrid */ 
    void SetRecursion(int levels);
    
    /** Smoother Cycles are called here */
    void Smooth(int smoother_type, TMGLevel3D *Level, 
#ifdef _MPI
		TParFECommunicator3D *ParComm, 
#endif
		double &oldres
    #ifdef _CUDA
    ,cudaStream_t *stream_smooth
    ,int* d_RowPtr
    ,int*  d_KCol
    ,double* d_Entries
    ,double* d_sol
    ,int* d_aux
    #endif
    );


    #ifdef _CUDA

      void InitGPU();
      void DeviceDataTransfer(int next_level, int smooth);
      void Swap_Matrices();

    #endif
};

#endif
