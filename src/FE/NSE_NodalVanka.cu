#include <NSE_MGLevel4.h>
#include <Database.h>
#include <MooNMD_Io.h>
#include <Solver.h>
#include <omp.h>
#ifdef __2D__
  #include <FESpace2D.h>
  #include <FEDatabase2D.h>
#endif
#ifdef __3D__
  #include <FESpace3D.h>
  #include <FEDatabase3D.h>
  #include <Joint.h>
  #include <Edge.h>
  #include <BaseCell.h>  
  #include <Collection.h>
#endif

#include <stdlib.h>
#include <string.h>

#include <LinAlg.h>
#include <Solver.h>
#include <ItMethod.h>
#include <FgmresIte.h>
#include <DirectSolver.h>

#include <LocalProjection.h>

#ifdef _MPI 
#include <ParFECommunicator3D.h>
#include <ParFEMapper3D.h>
#include <ParDirectSolver.h>
#endif


#define THREADS_PER_BLOCK 128

extern double timeVankaAssemble;
extern double timeVankaSolve;

extern double data_transfer_time;
extern double kernel_time;

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


__global__ void NodalVanka_assembleP(int* BRowPtr, int* BKCol, double* B1Entries, double* B2Entries, double*         B3Entries, 
                                double* u1, double* rhs1, int* PDOFReorder, int N_UDOF, double* Rhs,
                                int index1, int index2,
                                char* DofmarkerU, char* DofmarkerP, int* UDOFs, int* NUDOF, int max_N_U, int max_N_LocalDOF
           )
{
    const unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int grid_size = gridDim.x * blockDim.x;
    
    int offset,i,begin,end,l,offset_Rhs, N_UGEO, N_LocalDOF;
    
    double value,value1,value2,value3;
    
    double *rhs2, *rhs3, *rhsp;
    
    double *u2, *u3;
    
    // set pointers
    u2 = u1 + N_UDOF;
    #ifdef __3D__
    u3 = u2 + N_UDOF;
    #endif
    
    rhs2 = rhs1 + N_UDOF;
    #ifdef __3D__
    rhs3 = rhs2 + N_UDOF;
    #endif
    
    rhsp = rhs1 + GEO_DIM*N_UDOF;
    
    for(unsigned int row = thread_id + index1; row < index2; row += grid_size)
    {   
        
            offset = (row-index1) * max_N_U;
            
            offset_Rhs = (row-index1) * max_N_LocalDOF;
        
            i = PDOFReorder[row];

        #ifdef _MPI    
            
//             if(TDatabase::ParamDB->DOF_Average){
            if(DofmarkerP[i] == 'h' || DofmarkerP[i] == 'H')
            continue;
//             }     
//             else{
//             if(DofmarkerP[i] == 'h' || DofmarkerP[i] == 'H'  ||  DofmarkerP[i] == 's')
//             continue;
//             }
            
        #endif
            
            begin = BRowPtr[i];
            end = BRowPtr[i+1];
            value = rhsp[i];      // rhs of this pressure value
            int k1=0;
            
//             Rhs[offset_Rhs+N_LocalDOF-1] = value;  // set rhs
            
            for(int k=begin;k<end;k++)
            { 
                l=BKCol[k]; 
		value1 = B1Entries[k];
		value2 = B2Entries[k];
	  #ifdef __3D__
		value3 = B3Entries[k];
	  #endif
	  #ifdef __2D__
		value = value - (value1*u1[l]+value2*u2[l]); // update rhs
	  #endif      
	  #ifdef __3D__
		value = value - (value1*u1[l]+value2*u2[l]+value3*u3[l]); // update rhs
	  #endif    

	  #ifdef _MPI   
// 	      if(TDatabase::ParamDB->DOF_Average){
		if(DofmarkerU[l] == 'h' || DofmarkerU[l] == 'H')
		  continue;
// 	      }
// 	      
// 	      else{
// 		if(DofmarkerU[l] == 'h' || DofmarkerU[l] == 'H' || DofmarkerU[l] == 's')
// 		  continue;
// 	      }
	  #endif

                UDOFs[offset+k1] = l;
                k1++;
                
            }                    // row done
            
                    NUDOF[i]=k1;
        
//         cout<<NUDOF[i]<<endl;
            N_UGEO = GEO_DIM * k1;
            N_LocalDOF = N_UGEO +1;
        
            Rhs[offset_Rhs+N_LocalDOF-1] = value;  // set rhs
    }
}
        
__global__ void NodalVanka_assembleU(int* ARowPtr, int* AKCol,
                                double* A11Entries, double* A12Entries, double* A13Entries, double* A21Entries, double* A22Entries, double* A23Entries, double* A31Entries, double* A32Entries, double* A33Entries, 
                                int* BTRowPtr, int* BTKCol, double* B1TEntries, double* B2TEntries, double* B3TEntries,
                                double* u1, double* rhs1, int* PDOFReorder, int N_UDOF,
                                int HangingBound, double* System, double* Rhs,
                                int index1, int index2,
                                char* DofmarkerP, int* UDOFs, int* NUDOF, int max_N_U, int max_N_LocalDOF
           )
{
    
    
    double value11,value12,value13,value21,value22;
    double value23,value31,value32,value33;
    
    double value1, value2, value3,value;
    
    int begin,end;
    
    int i, offset_Rhs, UDOF,offset;
    int j1, j2, j3;
    
    double *rhs2, *rhs3, *rhsp;
    
    double *u2, *u3, *p;
    
    
//     HangingBound = USpace->GetHangingBound();

    // set pointers
    u2 = u1 + N_UDOF;
    #ifdef __3D__
    u3 = u2 + N_UDOF;
    #endif
    p  = u1 + GEO_DIM*N_UDOF;
    
    rhs2 = rhs1 + N_UDOF;
    #ifdef __3D__
    rhs3 = rhs2 + N_UDOF;
    #endif
    
    rhsp = rhs1 + GEO_DIM*N_UDOF;
    
    const unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int grid_size = gridDim.x * blockDim.x;
    
    
    
    for(unsigned int row = thread_id + index1; row < index2; row += grid_size)
    {
        i = PDOFReorder[row];
        
        offset_Rhs = (row-index1) * max_N_LocalDOF;
        
        offset = (row-index1) * max_N_U;
        
        #ifdef _MPI    
                
//                 if(TDatabase::ParamDB->DOF_Average){
                if(DofmarkerP[i] == 'h' || DofmarkerP[i] == 'H')
                continue;
//                 }     
//                 else{
//                 if(DofmarkerP[i] == 'h' || DofmarkerP[i] == 'H'  ||  DofmarkerP[i] == 's')
//                 continue;
//                 }
                
            #endif
            
        for(int j=0;j<NUDOF[i];j++){
		
		j1 = j;
		j2 = j+NUDOF[i];
		#ifdef __3D__
		j3 = j2+NUDOF[i];
		#endif
		
		UDOF = UDOFs[offset+j];

		// A block
		begin = ARowPtr[UDOF];
		end = ARowPtr[UDOF+1];

		Rhs[offset_Rhs+j1] = rhs1[UDOF];
		Rhs[offset_Rhs+j2] = rhs2[UDOF];
		#ifdef __3D__
		Rhs[offset_Rhs+j3] = rhs3[UDOF];
		#endif

		for(int k=begin;k<end;k++)
		{
		  int l = AKCol[k];
		  value11 = A11Entries[k];
		  value12 = A12Entries[k];
		  value21 = A21Entries[k];
		  value22 = A22Entries[k];
		  #ifdef __3D__
		  value13 = A13Entries[k];
		  value23 = A23Entries[k];
		  value31 = A31Entries[k];
		  value32 = A32Entries[k];
		  value33 = A33Entries[k];
		  #endif

		  #ifdef __2D__
		  if (UDOF>=HangingBound){ // Dirichlet node
		    value21 = 0;
		    value12 = 0;
          }

		  Rhs[offset_Rhs+j1] = Rhs[offset_Rhs+j1] - (value11*u1[l]+value12*u2[l]);
		  Rhs[offset_Rhs+j2] = Rhs[offset_Rhs+j2] - (value21*u1[l]+value22*u2[l]);
		  #endif
		  
		  #ifdef __3D__
		  if (UDOF>=HangingBound){ // Dirichlet node
		    value12 = 0;
            value13 = 0;
            value21 = 0;
            value23 = 0;
            value31 = 0;
            value32 = 0;
          }
		  
		  Rhs[offset_Rhs+j1] = Rhs[offset_Rhs+j1] - (value11*u1[l]+value12*u2[l]+value13*u3[l]);
		  Rhs[offset_Rhs+j2] = Rhs[offset_Rhs+j2] - (value21*u1[l]+value22*u2[l]+value23*u3[l]);
		  Rhs[offset_Rhs+j3] = Rhs[offset_Rhs+j3] - (value31*u1[l]+value32*u2[l]+value33*u3[l]);

		  #endif

		} // endfor k

		if(UDOF<HangingBound)  // active dof
		{
		  // transpose(B) block for non-Dirichlet nodes
		  begin = BTRowPtr[UDOF];
		  end = BTRowPtr[UDOF+1];

		  for(int k=begin;k<end;k++)
		  {
		    int l = BTKCol[k];
		    value1 = B1TEntries[k];
		    value2 = B2TEntries[k];
		    #ifdef __3D__
		    value3 = B3TEntries[k];
		    #endif
		    value = p[l];
		 
		    
		    Rhs[offset_Rhs+j1] = Rhs[offset_Rhs+j1] - value1*value;
		    Rhs[offset_Rhs+j2] = Rhs[offset_Rhs+j2] - value2*value;
		    #ifdef __3D__
		    Rhs[offset_Rhs+j3] = Rhs[offset_Rhs+j3] - value3*value;
		    #endif
		    

		  } // endfor k
		} // endif UDOF<HangingBound
	      } // endfor j
    
    }
    
}

__global__ void NodalVanka_assembleSysMatB(int* BRowPtr, int* BKCol, double* B1Entries, double* B2Entries, double*         B3Entries, 
                                double* u1, double* rhs1, int* PDOFReorder, int N_UDOF, double* System,
                                int index1, int index2,
                                char* DofmarkerU, char* DofmarkerP, int* UDOFs, int* NUDOF, int max_N_U, int max_N_LocalDOF
           )
{
    const unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int grid_size = gridDim.x * blockDim.x;
    
    int offset,i,begin,end,l,offset_Sys,j1, N_LocalDOF, N_U2, N_UGEO;
    
    double value1,value2,value3;
    
    double *rhs2, *rhs3, *rhsp;
    
    double *u2, *u3;
    
    // set pointers
    u2 = u1 + N_UDOF;
    #ifdef __3D__
    u3 = u2 + N_UDOF;
    #endif
    
    rhs2 = rhs1 + N_UDOF;
    #ifdef __3D__
    rhs3 = rhs2 + N_UDOF;
    #endif
    
    rhsp = rhs1 + GEO_DIM*N_UDOF;
    
    for(unsigned int row = thread_id + index1; row < index2; row += grid_size)
    {   
        
            offset = (row-index1)*max_N_U;
            
            offset_Sys = (row-index1)*(max_N_LocalDOF * max_N_LocalDOF);
        
            i = PDOFReorder[row];

        #ifdef _MPI    
            
//             if(TDatabase::ParamDB->DOF_Average){
            if(DofmarkerP[i] == 'h' || DofmarkerP[i] == 'H')
            continue;
//             }     
//             else{
//             if(DofmarkerP[i] == 'h' || DofmarkerP[i] == 'H'  ||  DofmarkerP[i] == 's')
//             continue;
//             }
            
        #endif
            
        N_U2 = 2 * NUDOF[i];
        N_UGEO = GEO_DIM * NUDOF[i];
        N_LocalDOF = N_UGEO +1;
            
            begin = BRowPtr[i];
            end = BRowPtr[i+1];
            int k1=0;
            
            for(int k=begin;k<end;k++)
            { 
                l=BKCol[k]; 
		value1 = B1Entries[k];
		value2 = B2Entries[k];
	  #ifdef __3D__
		value3 = B3Entries[k];
	  #endif
        
	  #ifdef _MPI   
// 	      if(TDatabase::ParamDB->DOF_Average){
		if(DofmarkerU[l] == 'h' || DofmarkerU[l] == 'H')
		  continue;
// 	      }
// 	      
// 	      else{
// 		if(DofmarkerU[l] == 'h' || DofmarkerU[l] == 'H' || DofmarkerU[l] == 's')
// 		  continue;
// 	      }
	  #endif

            j1 = GEO_DIM*k1;
            System[offset_Sys+k1*N_LocalDOF+N_UGEO] = value1;  // save values for local B
            System[offset_Sys+(k1+NUDOF[i])*N_LocalDOF+N_UGEO] = value2;
        #ifdef __3D__
            System[offset_Sys+(k1+N_U2)*N_LocalDOF+N_UGEO] = value3;
        #endif  
            k1++;  
                
            }                    // row done
            
    }
}

__global__ void NodalVanka_assembleSysMatA(int* ARowPtr, int* AKCol,
                                double* A11Entries, double* A12Entries, double* A13Entries, double* A21Entries, double* A22Entries, double* A23Entries, double* A31Entries, double* A32Entries, double* A33Entries, 
                                int* BTRowPtr, int* BTKCol, double* B1TEntries, double* B2TEntries, double* B3TEntries,
                                double* u1, double* rhs1, int* PDOFReorder, int N_UDOF,
                                int HangingBound, double* System,
                                int index1, int index2,
                                char* DofmarkerP, int* UDOFs, int* NUDOF, int max_N_U, int max_N_LocalDOF
           )
{
    
    
    double value11,value12,value13,value21,value22;
    double value23,value31,value32,value33;
    
    double value1, value2, value3,value;
    
    int begin,end;
    
    int i, m, UDOF,offset,offset_Sys, N_LocalDOF, N_UGEO;
    int j1, j2, j3, j4, k1, k2, k3;
    
    double *rhs2, *rhs3, *rhsp;
    
    double *u2, *u3, *p;
    
    
//     HangingBound = USpace->GetHangingBound();

    // set pointers
    u2 = u1 + N_UDOF;
    #ifdef __3D__
    u3 = u2 + N_UDOF;
    #endif
    p  = u1 + GEO_DIM*N_UDOF;
    
    rhs2 = rhs1 + N_UDOF;
    #ifdef __3D__
    rhs3 = rhs2 + N_UDOF;
    #endif
    
    rhsp = rhs1 + GEO_DIM*N_UDOF;
    
    const unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int grid_size = gridDim.x * blockDim.x;
    
    
    
    for(unsigned int row = thread_id + index1; row < index2; row += grid_size)
    {
        i = PDOFReorder[row];
        
        offset = (row-index1) * max_N_U;
        
        offset_Sys = (row-index1) * (max_N_LocalDOF * max_N_LocalDOF);
        
        #ifdef _MPI    
                
//                 if(TDatabase::ParamDB->DOF_Average){
                if(DofmarkerP[i] == 'h' || DofmarkerP[i] == 'H')
                continue;
//                 }     
//                 else{
//                 if(DofmarkerP[i] == 'h' || DofmarkerP[i] == 'H'  ||  DofmarkerP[i] == 's')
//                 continue;
//                 }
                
            #endif
                
//         cudaMemset(System+offset_Sys+N_UGEO, 0, SizeOfDouble*(N_LocalDOF*N_LocalDOF-N_UGEO));

//         for (int k=0;k<N_U;k++)         // copy local B to the right place
//         {
//         j4 = GEO_DIM*k;
//         System[offset_Sys+k*N_LocalDOF+N_UGEO]=System[offset_Sys+j4];
//         System[offset_Sys+(k+N_U)*N_LocalDOF+N_UGEO]=System[offset_Sys+j4+1];
//         #ifdef __3D__
//         System[offset_Sys+(k+N_U2)*N_LocalDOF+N_UGEO]=System[offset_Sys+j4+2];
//         #endif
//         }
	      
//         cudaMemset(System+offset_Sys, 0, SizeOfDouble*N_UGEO);
            
    
        N_UGEO = GEO_DIM * NUDOF[i];
        N_LocalDOF = N_UGEO +1;
            
        for(int j=0;j<NUDOF[i];j++){
		
		j1 = j;
		j2 = j+NUDOF[i];
		#ifdef __3D__
		j3 = j2+NUDOF[i];
		#endif
		
		UDOF = UDOFs[offset+j];

		// A block
		begin = ARowPtr[UDOF];
		end = ARowPtr[UDOF+1];

		for(int k=begin;k<end;k++)
		{
		  int l = AKCol[k];
		  value11 = A11Entries[k];
		  value12 = A12Entries[k];
		  value21 = A21Entries[k];
		  value22 = A22Entries[k];
		  #ifdef __3D__
		  value13 = A13Entries[k];
		  value23 = A23Entries[k];
		  value31 = A31Entries[k];
		  value32 = A32Entries[k];
		  value33 = A33Entries[k];
		  #endif

		  #ifdef __2D__
		  if (UDOF>=HangingBound){ // Dirichlet node
		    value21 = 0;
		    value12 = 0;
          }

		  #endif
		  
		  #ifdef __3D__
		  if (UDOF>=HangingBound){ // Dirichlet node
		    value12 = 0;
            value13 = 0;
            value21 = 0;
            value23 = 0;
            value31 = 0;
            value32 = 0;
          }

		  #endif
		
		bool flag=true;
        
        for(m=0;m<NUDOF[i] && flag;m++){
            if(UDOFs[offset+m]==l)
            {
            k1 = m*N_LocalDOF;
            k2 = (m+NUDOF[i])*N_LocalDOF;

            System[offset_Sys+k1+j1] = value11;
            System[offset_Sys+k2+j1] = value12;
            System[offset_Sys+k1+j2] = value21;
            System[offset_Sys+k2+j2] = value22;
            #ifdef __3D__
            k3 = (m + 2*NUDOF[i])*N_LocalDOF;
            System[offset_Sys+k3+j1] = value13;
            System[offset_Sys+k3+j2] = value23;
            System[offset_Sys+k1+j3] = value31;
            System[offset_Sys+k2+j3] = value32;
            System[offset_Sys+k3+j3] = value33;
            #endif
            flag=false;
            }
        }

		} // endfor k

		if(UDOF<HangingBound)  // active dof
		{
		  // transpose(B) block for non-Dirichlet nodes
		  begin = BTRowPtr[UDOF];
		  end = BTRowPtr[UDOF+1];

		  for(int k=begin;k<end;k++)
		  {
		    int l = BTKCol[k];
		    value1 = B1TEntries[k];
		    value2 = B2TEntries[k];
		    #ifdef __3D__
		    value3 = B3TEntries[k];
		    #endif
		 
            if(i==l)
            {
            j4 = N_UGEO*N_LocalDOF;
            System[offset_Sys+j4+j1] = value1;
            System[offset_Sys+j4+j2] = value2;
            #ifdef __3D__
            System[offset_Sys+j4+j3] = value3;
            #endif
            }
		    

		  } // endfor k
		} // endif UDOF<HangingBound
	      } // endfor j
    
    }
    
}

// void verify(int* o, int* n,int num, int rank){
//  
//     for(int i=0; i<num; i++){
//         if(o[i] == n[i]){
//             cout<<"neha: matching!!"<<o[i]<<" "<<n[i]<<" rank "<<rank<<endl;
//             continue;
//         }
//         
//         else {cout<<"neha: not matching!! "<<o[i]<<" "<<n[i]<<" rank "<<rank<<endl;
//             return;
//         }
//     }
// //     cout<<"neha: matching!!"<<itr<<" rank "<<rank<<endl;
// }

void verify1(double* o, double* n,int num, int rank){
 
    for(int i=0; i<num; i++){
        if(o[i] == n[i] && rank == 0){
//             cout<<"neha: matching!!"<<o[i]<<" "<<n[i]<<" i "<<i<<endl;
            continue;
        }
        
        else if(o[i] != n[i] && rank == 0) {cout<<"neha: not matching!! "<<o[i]<<" "<<n[i]<<" i "<<i<<endl;
            return;
        }
    }
//     cout<<"neha: matching!!"<<itr<<" rank "<<rank<<endl;
}


#ifdef _MPI
#ifdef _HYBRID
#ifdef _CUDA


void TNSE_MGLevel4::NodalVanka_CPU_GPU(double *u1, double *rhs1, double *aux,
        int N_Parameters, double *Parameters, int smoother, int smooth)
 {
//      cout<<"NV_CPU_GPU"<<endl;
#ifdef _MPI
  TDatabase::ParamDB->time_vanka_start = MPI_Wtime();
#else
  TDatabase::ParamDB->time_vanka_start = GetTime();
#endif
  
#ifdef _MPI
  int rank, *MasterOfDofU,*MasterOfDofP;
  
  char *DofmarkerP = ParCommP->Get_DofMarker();
  char *DofmarkerU = ParCommU->Get_DofMarker();
  
  MPI_Comm_rank(TDatabase::ParamDB->Comm, &rank); 
  MasterOfDofU = ParCommU->GetMaster();
  MasterOfDofP = ParCommP->GetMaster();    
#endif
  
#ifdef __2D__
  const int MaxN_LocalU = 2*MaxN_BaseFunctions2D;
  const int SystemRhs = 3*MaxN_BaseFunctions2D;
  TSquareMatrix2D *sqmatrix[1];
#endif
#ifdef __3D__
  const int MaxN_LocalU = 4*MaxN_BaseFunctions3D;
  const int SystemRhs = 8*MaxN_BaseFunctions3D;
  TSquareMatrix3D *sqmatrix[1];
#endif
  int i,j,k,l,m;
  int order;
  int j1, j2, j3, j4, k1, k2, k3;
  double value, value1, value2, value3;
  double value11,value12,value13,value21,value22;
  double value23,value31,value32,value33;
  double *uold, *pold;
//   double System[SystemRhs*SystemRhs];
//   double Rhs[SystemRhs], 
  
//   double sol[SystemRhs];
  int N_LocalDOF;
  int begin, end, HangingBound, begin1, end1, verbose;
  int UDOF, N_U, N_U2, N_UGEO;
  double *u2, *u3, *p, *rhs2, *rhs3, *rhsp;
  TItMethod *itmethod = NULL;
  double damp = TDatabase::ParamDB->SC_SMOOTH_DAMP_FACTOR_COARSE_SADDLE;
  int LargestDirectSolve = TDatabase::ParamDB->SC_LARGEST_DIRECT_SOLVE;
  MatVecProc *MatVect=MatVectFull;
  DefectProc *Defect=DefectFull;
  TSquareMatrix **matrix= (TSquareMatrix **)sqmatrix;
  
  double gar;

  TDatabase::ParamDB->INTERNAL_LOCAL_DOF = -1;
  
// #ifdef __2D__
//   sqmatrix[0] = (TSquareMatrix2D *)System;
// #endif
// #ifdef __3D__
//   sqmatrix[0] = (TSquareMatrix3D *)System;
// #endif

  HangingBound = USpace->GetHangingBound();

  // set pointers
  u2 = u1 + N_UDOF;
#ifdef __3D__
  u3 = u2 + N_UDOF;
#endif
  p  = u1 + GEO_DIM*N_UDOF;

  rhs2 = rhs1 + N_UDOF;
#ifdef __3D__
  rhs3 = rhs2 + N_UDOF;
#endif
  rhsp = rhs1 + GEO_DIM*N_UDOF;

  // set Dirichlet values
  
  memcpy(u1+HangingNodeBound, rhs1+HangingNodeBound,
         N_Dirichlet*SizeOfDouble);
  memcpy(u2+HangingNodeBound, rhs2+HangingNodeBound,
         N_Dirichlet*SizeOfDouble);
#ifdef __3D__
  memcpy(u3+HangingNodeBound, rhs3+HangingNodeBound,
         N_Dirichlet*SizeOfDouble);
#endif

  // old values
  uold = aux;
  pold = uold+GEO_DIM*N_UDOF;

  // save current solution on 'old' vectors
  memcpy(uold, u1, N_DOF*SizeOfDouble);
  
  
    int nStreams = 2;
    cudaStream_t stream[nStreams];

    for (int i = 0; i < nStreams; ++i)
        CUDA_CHECK( cudaStreamCreate(&stream[i]) );

    int nz_A = A11->GetN_Entries();
    int n_A = A11->GetN_Rows(); 

    int nz_B = B1->GetN_Entries();
    int n_B = B1->GetN_Rows(); 

    int nz_BT = B1T->GetN_Entries();
    int n_BT = B1T->GetN_Rows(); 
    
    int maxDOFPerColor=-1;
  
      for(int i=0;i<N_CPDOF;i++)
        {
            int temp = (ptrPDOFColors[i+1] - ptrPDOFColors[i]);
//                 cout<<"temp:"<<temp<<endl;
                
                if(maxDOFPerColor< temp){
                    
                    maxDOFPerColor = temp;
                    
                }
                
        }
        
//     cout<<"neha:maxDOFPerColor:"<<maxDOFPerColor<<endl;   
//         Coll = USpace->GetCollection();
        
        bool flag=false;
        int max_N_U=-1;
        int max_N_LocalDOF;
        
    for(int ii=0;ii<N_CPDOF;ii++)
        {
            for(int jj=ptrPDOFColors[ii];jj<ptrPDOFColors[ii+1];jj++)
            {
                i = PDOFReorder[jj];
                N_U = 0;

	  #ifdef _MPI
	      if(TDatabase::ParamDB->DOF_Average){
		if(DofmarkerP[i] == 'h' || DofmarkerP[i] == 'H')
		  continue;
	      }     
	      else{
		if(DofmarkerP[i] == 'h' || DofmarkerP[i] == 'H'  ||  DofmarkerP[i] == 's')
		  continue;
	      }
	  #endif
	      // go through row i of B1 and B2
	      begin = BRowPtr[i];
	      end = BRowPtr[i+1];
	      value = rhsp[i];      // rhs of this pressure value
	    for(k=begin;k<end;k++)
	      { 
		l=BKCol[k]; 

	  #ifdef _MPI   
	      if(TDatabase::ParamDB->DOF_Average){
		if(DofmarkerU[l] == 'h' || DofmarkerU[l] == 'H')
		  continue;
	      }
	      
	      else{
		if(DofmarkerU[l] == 'h' || DofmarkerU[l] == 'H' || DofmarkerU[l] == 's')
		  continue;
	      }
	  #endif

	      N_U++;           // count # velo dof connected to the pressure dof
	      }                    // row done
	      
	       if(N_U>max_N_U){
                max_N_U=N_U;
            }
            }
           
        }
        
        N_U=max_N_U;
        N_UGEO = GEO_DIM * max_N_U;
        max_N_LocalDOF = N_UGEO +1;
    
//     cudaEvent_t start, stop;
//     float time;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop); 
//     
//     cudaEventRecord(start, 0);
    
    //     transfer solution and rhs
    double* d_u1 = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_u1, N_DOF * sizeof(double)));
    
    double* d_rhs1 = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_rhs1, N_DOF * sizeof(double)));
    
    int* d_PDOFReorder = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_PDOFReorder, N_PDOF * sizeof(int)));
    
    // Copy to GPU memory
    CUDA_CHECK(cudaMemcpyAsync(d_u1, u1,  N_DOF* sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_rhs1, rhs1, N_DOF * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_PDOFReorder, PDOFReorder, N_PDOF * sizeof(int), cudaMemcpyHostToDevice,stream[0]));

//         cout<<"N_LocalDOF:"<<N_LocalDOF<<endl;
    
    double *System = new double[maxDOFPerColor * (max_N_LocalDOF * max_N_LocalDOF)];
//     double *temp_System = new double[maxDOFPerColor * (max_N_LocalDOF * max_N_LocalDOF)];
    double *Rhs = new double[maxDOFPerColor * max_N_LocalDOF];
    double *sol = new double[maxDOFPerColor * max_N_LocalDOF];
  
    double* d_System = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_System, maxDOFPerColor * (max_N_LocalDOF * max_N_LocalDOF) * sizeof(double)));
    
    double* d_Rhs = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_Rhs, maxDOFPerColor * max_N_LocalDOF * sizeof(double)));
    
    char* d_DofmarkerP = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_DofmarkerP, N_PDOF * sizeof(char)));
    
    CUDA_CHECK(cudaMemcpyAsync(d_DofmarkerP, DofmarkerP, N_PDOF * sizeof(char), cudaMemcpyHostToDevice,stream[0]));
  
    char* d_DofmarkerU = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_DofmarkerU, N_UDOF * sizeof(char)));
    
    CUDA_CHECK(cudaMemcpyAsync(d_DofmarkerU, DofmarkerU, N_UDOF * sizeof(char), cudaMemcpyHostToDevice,stream[0]));
    
    
    //     transfer A matrix
    int* d_ARowPtr = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_ARowPtr, (n_A+1) * sizeof(int)));
    
    int* d_AKCol = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_AKCol, nz_A * sizeof(int)));
    
    double* d_A11Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A11Entries, nz_A * sizeof(double)));
    
    double* d_A12Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A12Entries, nz_A * sizeof(double)));
    
    double* d_A13Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A13Entries, nz_A * sizeof(double)));
    
    double* d_A21Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A21Entries, nz_A * sizeof(double)));
    
    double* d_A22Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A22Entries, nz_A * sizeof(double)));
    
    double* d_A23Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A23Entries, nz_A * sizeof(double)));
    
    double* d_A31Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A31Entries, nz_A * sizeof(double)));
    
    double* d_A32Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A32Entries, nz_A * sizeof(double)));
    
    double* d_A33Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A33Entries, nz_A * sizeof(double)));
    
        // Copy to GPU memory
    CUDA_CHECK(cudaMemcpyAsync(d_ARowPtr, ARowPtr, (n_A+1) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_AKCol, AKCol, nz_A * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_A11Entries, A11Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_A12Entries, A12Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_A13Entries, A13Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_A21Entries, A21Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_A22Entries, A22Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_A23Entries, A23Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_A31Entries, A31Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_A32Entries, A32Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_A33Entries, A33Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    
//     cout<<"neha:color"<<N_CIntCell<<endl;
    
//     transfer BT matrix
    int* d_BTRowPtr = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_BTRowPtr, (n_BT+1) * sizeof(int)));
    
    int* d_BTKCol = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_BTKCol, nz_BT * sizeof(int)));
    
    double* d_B1TEntries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_B1TEntries, nz_BT * sizeof(double)));
    
    double* d_B2TEntries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_B2TEntries, nz_BT * sizeof(double)));
    
    double* d_B3TEntries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_B3TEntries, nz_BT * sizeof(double)));
    
    // Copy to GPU memory
    CUDA_CHECK(cudaMemcpyAsync(d_BTRowPtr, BTRowPtr, (n_BT+1) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_BTKCol, BTKCol, nz_BT * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_B1TEntries, B1TEntries, nz_BT * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_B2TEntries, B2TEntries, nz_BT * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_B3TEntries, B3TEntries, nz_BT * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
//     transfer B matrix
    int* d_BRowPtr = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_BRowPtr, (n_B+1) * sizeof(int)));
    
    int* d_BKCol = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_BKCol, nz_B * sizeof(int)));
    
    double* d_B1Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_B1Entries, nz_B * sizeof(double)));
    
    double* d_B2Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_B2Entries, nz_B * sizeof(double)));
    
    double* d_B3Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_B3Entries, nz_B * sizeof(double)));
    
    // Copy to GPU memory
    CUDA_CHECK(cudaMemcpyAsync(d_BRowPtr, BRowPtr, (n_B+1) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_BKCol, BKCol, nz_B * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_B1Entries, B1Entries, nz_B * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_B2Entries, B2Entries, nz_B * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_B3Entries, B3Entries, nz_B * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
  
//     cout<<"N_U:"<<N_U<<endl;
    int* NUDOF = new int[N_PDOF];
    
    int* UDOFs = new int[max_N_U * maxDOFPerColor];
//     int* temp_UDOFs = new int[N_U * maxDOFPerColor];
    int* d_UDOFs = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_UDOFs, max_N_U * maxDOFPerColor * sizeof(int)));
    
    int* d_NUDOF = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_NUDOF, N_PDOF * sizeof(int)));

    
    CUDA_CHECK(cudaStreamSynchronize(stream[0]));
    
//     cudaEventRecord(stop, 0);
//     cudaEventSynchronize(stop);
    
    float milliseconds;
    
//     cudaEventElapsedTime(&milliseconds, start, stop);
    
//     data_transfer_time += milliseconds;
    
    int end_itr,offset_Rhs,offset_Sys,offset;
    
    if(smooth == -1)
        end_itr = TDatabase::ParamDB->SC_PRE_SMOOTH_SADDLE;
    else if(smooth == 0)
        end_itr = TDatabase::ParamDB->SC_COARSE_MAXIT_SADDLE;
    else
        end_itr = TDatabase::ParamDB->SC_POST_SMOOTH_SADDLE;
    
    int thread_blocks;
    double t1,t2;
    
    int numThreads = TDatabase::ParamDB->OMPNUMTHREADS;
     
    omp_set_num_threads(numThreads);
    
//     double *System = new double[maxDOFPerColor * (N_LocalDOF * N_LocalDOF)];
//     double *Rhs = new double[maxDOFPerColor * N_LocalDOF];
    
    for(int itr=0;itr<end_itr;itr++)
    {
  
        for(int ii=0;ii<N_CPDOF;ii++)
        {
//             cout<<"ii:"<<ii<<endl;
            
            thread_blocks = (ptrPDOFColors[ii+1]-ptrPDOFColors[ii])/THREADS_PER_BLOCK + 1;
            
//             cudaEventRecord(start, 0);
            NodalVanka_assembleP<<<thread_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(d_BRowPtr, d_BKCol, d_B1Entries, d_B2Entries,  d_B3Entries, d_u1, d_rhs1, d_PDOFReorder, N_UDOF, d_Rhs, ptrPDOFColors[ii], ptrPDOFColors[ii+1], d_DofmarkerU, d_DofmarkerP, d_UDOFs, d_NUDOF, max_N_U, max_N_LocalDOF);
            
            CUDA_CHECK(cudaMemcpyAsync(NUDOF, d_NUDOF, N_PDOF * sizeof(int), cudaMemcpyDeviceToHost,stream[0]));
            
            CUDA_CHECK(cudaMemcpyAsync(UDOFs, d_UDOFs, max_N_U * maxDOFPerColor * sizeof(int), cudaMemcpyDeviceToHost,stream[0]));
            
//             cudaEventRecord(stop, 0);
//             cudaEventSynchronize(stop);
//             cudaEventElapsedTime(&milliseconds, start, stop);
//             
//             kernel_time += milliseconds;
// double t1,t2;
#ifdef _MPI
  t1 = MPI_Wtime();
#else
  t1 = GetTime();
#endif
  
            memset(System, 0, maxDOFPerColor*SizeOfDouble*(max_N_LocalDOF*max_N_LocalDOF));
            
            for(int jj=ptrPDOFColors[ii];jj<ptrPDOFColors[ii+1];jj++)
            {
                

//                 if(rank==0){
//                 cout<<"ptrPDOFColors:"<<jj<<endl;
//                 }
                offset_Rhs = (jj-ptrPDOFColors[ii]) * max_N_LocalDOF;
                offset_Sys = (jj-ptrPDOFColors[ii]) * (max_N_LocalDOF * max_N_LocalDOF);
                offset = (jj-ptrPDOFColors[ii]) * max_N_U;
                
                
                
                i = PDOFReorder[jj];
                int k1 = 0;
                
                

            #ifdef _MPI    
                
                if(TDatabase::ParamDB->DOF_Average){
                if(DofmarkerP[i] == 'h' || DofmarkerP[i] == 'H'){
//                     cout<<"neha:continue"<<endl;
                continue;
                
                }
                }     
                else{
                if(DofmarkerP[i] == 'h' || DofmarkerP[i] == 'H'  ||  DofmarkerP[i] == 's'){
//                 cout<<"neha:continue"<<endl;
                    continue;
                
                }
                }
                
            #endif
            

            
//                 cout<<"here"<<endl;
                // go through row i of B1 and B2
                begin = BRowPtr[i];
                end = BRowPtr[i+1];
//                 value = rhsp[i];      // rhs of this pressure value
                for(int k=begin;k<end;k++)
                { 
                    l=BKCol[k]; 
                    value1 = B1Entries[k];
                    value2 = B2Entries[k];
                #ifdef __3D__
                    value3 = B3Entries[k];
                #endif
 

                #ifdef _MPI   
                    if(TDatabase::ParamDB->DOF_Average){
                    if(DofmarkerU[l] == 'h' || DofmarkerU[l] == 'H'){
                        
//                         cout<<"neha:continue"<<endl;
                    continue;
                    
                    }
                    }
                    
                    else{
                    if(DofmarkerU[l] == 'h' || DofmarkerU[l] == 'H' || DofmarkerU[l] == 's'){
//                     cout<<"neha:continue"<<endl;
                        continue;
                    }
                    }
                #endif

//                     UDOFs[offset+k1] = l;
                    
//                 if(rank==1){
//                 cout<<maxDOFPerColor * (N_LocalDOF * N_LocalDOF)<<endl;
//                 cout<<offset_Sys<<" "<<j1<<endl;
//                 cout<<"here"<<endl;
//                 }
                
                    j1 = GEO_DIM*k1;
                    System[offset_Sys+j1] = value1;  // save values for local B
                    System[offset_Sys+j1+1] = value2;
                #ifdef __3D__
                    System[offset_Sys+j1+2] = value3;
                #endif  
                    k1++;           // count # velo dof connected to the pressure dof
                }                    // row done
            }

#ifdef _MPI
  t2 = MPI_Wtime();
#else
  t2 = GetTime();
#endif
timeVankaSolve += t2-t1;
            
            
            CUDA_CHECK(cudaStreamSynchronize(stream[0]));
            
            NodalVanka_assembleU<<<thread_blocks, THREADS_PER_BLOCK, 0, stream[1]>>>(d_ARowPtr, d_AKCol,
                    d_A11Entries, d_A12Entries, d_A13Entries, d_A21Entries, d_A22Entries, d_A23Entries, d_A31Entries,  d_A32Entries, d_A33Entries, 
                    d_BTRowPtr, d_BTKCol, d_B1TEntries, d_B2TEntries, d_B3TEntries,
                    d_u1, d_rhs1, d_PDOFReorder, N_UDOF, HangingBound, d_System, d_Rhs, ptrPDOFColors[ii], ptrPDOFColors[ii+1],d_DofmarkerP, d_UDOFs, d_NUDOF, max_N_U, max_N_LocalDOF);
            
            CUDA_CHECK(cudaMemcpyAsync(Rhs, d_Rhs, maxDOFPerColor * max_N_LocalDOF * sizeof(double), cudaMemcpyDeviceToHost,stream[1]));
            
            
            
            
            
            
            
//             NodalVanka_assembleP<<<thread_blocks, THREADS_PER_BLOCK, 0, stream[1]>>>(d_ARowPtr, d_AKCol,
//                                 d_A11Entries, d_A12Entries, d_A13Entries, d_A21Entries, d_A22Entries, d_A23Entries, d_A31Entries,  d_A32Entries, d_A33Entries, 
//                                 d_BTRowPtr, d_BTKCol, d_B1TEntries, d_B2TEntries, d_B3TEntries,
//                                 d_u1, d_rhs1, d_PDOFReorder, N_U, N_P, N_LocalDOF, N_UDOF,
//                                 d_UGlobalNumbers, d_UBeginIndex, d_PGlobalNumbers, d_PBeginIndex, ActiveBound, d_System, d_Rhs, ptrPDOFColors[i], ptrPDOFColors[i+1], d_DofmarkerP, d_UDOFs);
            
   
// double t1,t2;
#ifdef _MPI
  t1 = MPI_Wtime();
#else
  t1 = GetTime();
#endif
            for(int jj=ptrPDOFColors[ii];jj<ptrPDOFColors[ii+1];jj++)
            {

//                 if(rank==0){
//                 cout<<"ptrPDOFColors:"<<jj<<endl;
//                 }
                offset_Rhs = (jj-ptrPDOFColors[ii]) * max_N_LocalDOF;
                offset_Sys = (jj-ptrPDOFColors[ii]) * (max_N_LocalDOF * max_N_LocalDOF);
                offset = (jj-ptrPDOFColors[ii]) * max_N_U;
                
                i = PDOFReorder[jj];
                int k1 = 0;

            #ifdef _MPI    
                
                if(TDatabase::ParamDB->DOF_Average){
                if(DofmarkerP[i] == 'h' || DofmarkerP[i] == 'H')
                continue;
                }     
                else{
                if(DofmarkerP[i] == 'h' || DofmarkerP[i] == 'H'  ||  DofmarkerP[i] == 's')
                continue;
                }
                
            #endif



                N_U2 = 2 * NUDOF[i];
                N_UGEO = GEO_DIM * NUDOF[i];
                N_LocalDOF = N_UGEO +1;
            
                memset(System+offset_Sys+N_UGEO, 0, SizeOfDouble*(N_LocalDOF*N_LocalDOF-N_UGEO));

                for (k=0;k<NUDOF[i];k++)         // copy local B to the right place
                {
                j4 = GEO_DIM*k;
                System[offset_Sys+k*N_LocalDOF+N_UGEO]=System[offset_Sys+j4];
                System[offset_Sys+(k+NUDOF[i])*N_LocalDOF+N_UGEO]=System[offset_Sys+j4+1];
                #ifdef __3D__
                System[offset_Sys+(k+N_U2)*N_LocalDOF+N_UGEO]=System[offset_Sys+j4+2];
                #endif
                }
	      
                memset(System+offset_Sys, 0, SizeOfDouble*N_UGEO);
	      
                for(j=0;j<NUDOF[i];j++)
                {
                
                j1 = j;
                j2 = j+NUDOF[i];
                #ifdef __3D__
                j3 = j2+NUDOF[i];
                #endif
                
                UDOF = UDOFs[offset+j];

                // A block
                begin = ARowPtr[UDOF];
                end = ARowPtr[UDOF+1];

//                 Rhs[offset_Rhs+j1] = rhs1[UDOF];
//                 Rhs[offset_Rhs+j2] = rhs2[UDOF];
//                 #ifdef __3D__
//                 Rhs[offset_Rhs+j3] = rhs3[UDOF];
//                 #endif

                for(k=begin;k<end;k++)
                {
                l = AKCol[k];
                value11 = A11Entries[k];
                value12 = A12Entries[k];
                value21 = A21Entries[k];
                value22 = A22Entries[k];
                #ifdef __3D__
                value13 = A13Entries[k];
                value23 = A23Entries[k];
                value31 = A31Entries[k];
                value32 = A32Entries[k];
                value33 = A33Entries[k];
                #endif

                #ifdef __2D__
                if (UDOF>=HangingBound) // Dirichlet node
                    value21 = value12 = 0;

//                 Rhs[offset_Rhs+j1] -= value11*u1[l]+value12*u2[l];
//                 Rhs[offset_Rhs+j2] -= value21*u1[l]+value22*u2[l];
                #endif
                
                #ifdef __3D__
                if (UDOF>=HangingBound) // Dirichlet node
                    value12 = value13 = value21 = value23 = value31 = value32 = 0;

        // 		  #ifdef _MPI
        // 		  if(DofmarkerU[UDOF] != 'h' || DofmarkerU[UDOF] != 'H' )
        // 		  #endif
//                 {
//                     Rhs[offset_Rhs+j1] -= value11*u1[l]+value12*u2[l]+value13*u3[l];
//                     Rhs[offset_Rhs+j2] -= value21*u1[l]+value22*u2[l]+value23*u3[l];
//                     Rhs[offset_Rhs+j3] -= value31*u1[l]+value32*u2[l]+value33*u3[l];
//                 }
                #endif

                for(m=0;m<NUDOF[i];m++)
                    if(UDOFs[offset+m]==l)
                    {
                    k1 = m*N_LocalDOF;
                    k2 = (m+NUDOF[i])*N_LocalDOF;

                    System[offset_Sys+k1+j1] = value11;
                    System[offset_Sys+k2+j1] = value12;
                    System[offset_Sys+k1+j2] = value21;
                    System[offset_Sys+k2+j2] = value22;
                    #ifdef __3D__
                    k3 = (m + 2*NUDOF[i])*N_LocalDOF;
                    System[offset_Sys+k3+j1] = value13;
                    System[offset_Sys+k3+j2] = value23;
                    System[offset_Sys+k1+j3] = value31;
                    System[offset_Sys+k2+j3] = value32;
                    System[offset_Sys+k3+j3] = value33;
                    #endif
                    break;
                    }
                } // endfor k

                if(UDOF<HangingBound)  // active dof
                {
                // transpose(B) block for non-Dirichlet nodes
                begin = BTRowPtr[UDOF];
                end = BTRowPtr[UDOF+1];

                for(k=begin;k<end;k++)
                {
                    l = BTKCol[k];
                    value1 = B1TEntries[k];
                    value2 = B2TEntries[k];
                    #ifdef __3D__
                    value3 = B3TEntries[k];
                    #endif
//                     value = p[l];
                
//                     {
//                     Rhs[offset_Rhs+j1] -= value1*value;
//                     Rhs[offset_Rhs+j2] -= value2*value;
//                     #ifdef __3D__
//                     Rhs[offset_Rhs+j3] -= value3*value;
//                     #endif
//                     }

                    if(i==l)
                    {
                    j4 = N_UGEO*N_LocalDOF;
                    System[offset_Sys+j4+j1] = value1;
                    System[offset_Sys+j4+j2] = value2;
                    #ifdef __3D__
                    System[offset_Sys+j4+j3] = value3;
                    #endif
                    }
                } // endfor k
                } // endif UDOF<HangingBound
                } // endfor j
//                 if(rank==0){
//                 cout<<"here2"<<endl;
//                 }
                
//                 CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//                 CUDA_CHECK(cudaStreamSynchronize(stream[2]));

//                 CUDA_CHECK(cudaMemcpyAsync(System, d_System, maxCellsPerColor * (N_LocalDOF * N_LocalDOF) * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
//                 CUDA_CHECK(cudaStreamSynchronize(stream[0]));
            }
            
#ifdef _MPI
  t2 = MPI_Wtime();
#else
  t2 = GetTime();
#endif
timeVankaSolve += t2-t1; 

            
            
            CUDA_CHECK(cudaStreamSynchronize(stream[1]));
            
#ifdef _MPI
  t1 = MPI_Wtime();
#else
  t1 = GetTime();
#endif            
            
//             verify1(temp_Rhs,Rhs, maxDOFPerColor * N_LocalDOF, rank);
            
            #pragma omp parallel for default(shared) schedule(dynamic)
            for(int jj=ptrPDOFColors[ii];jj<ptrPDOFColors[ii+1];jj++)
            {
                int offset_Rhs = (jj-ptrPDOFColors[ii]) * max_N_LocalDOF;
                int offset_Sys = (jj-ptrPDOFColors[ii]) * (max_N_LocalDOF * max_N_LocalDOF);
                int offset = (jj-ptrPDOFColors[ii]) * max_N_U;
                
                int i = PDOFReorder[jj];
                
//                 #ifdef _MPI    
//                 
//                 if(TDatabase::ParamDB->DOF_Average){
//                 if(DofmarkerP[i] == 'h' || DofmarkerP[i] == 'H')
//                 continue;
//                 }     
//                 else{
//                 if(DofmarkerP[i] == 'h' || DofmarkerP[i] == 'H'  ||  DofmarkerP[i] == 's')
//                 continue;
//                 }
//                 
//                 #endif

                int N_U2 = 2 * NUDOF[i];
                int N_UGEO = GEO_DIM * NUDOF[i];
                int N_LocalDOF = N_UGEO +1;
                
                if(C)
                {
                // fill C block if present
                begin = CRowPtr[i];
                end = CRowPtr[i+1];
                for(k=begin;k<end;k++)
                {
                l = CKCol[k];
                value = -CEntries[k]; // minus is right sign
                Rhs[offset_Rhs+N_LocalDOF-1] -= value*p[l];
                if(l==i) // main diagonal
                    System[offset_Sys+N_LocalDOF*N_LocalDOF-1] = value;
                } // endfor k
                } // endif C
	      
//=================================================Start Solving local system ===========================================

//                 #ifdef _MPI
//                 TDatabase::ParamDB->time_vanka_solve_start = MPI_Wtime();
//                 #else
//                 TDatabase::ParamDB->time_vanka_solve_start = GetTime();
//                 #endif
                if (smoother==31 && !C) // no diagonal Vanka for matrix C
                {
            #ifdef __2D__
                // diagonal Vanka
                SolveDiagonalVanka2D(System+offset_Sys,  Rhs+offset_Rhs, NUDOF[i], 1, N_LocalDOF);
            #endif
            #ifdef __3D__
                // diagonal Vanka
                SolveDiagonalVanka3D(System+offset_Sys,  Rhs+offset_Rhs, NUDOF[i], 1, N_LocalDOF);
            #endif
                }
                else
                {
/*                // full Vanka
                if (N_LocalDOF < LargestDirectSolve)
                {
                    if ( TDatabase::ParamDB->INTERNAL_LOCAL_DOF >0)
                {
                    delete itmethod;
                }
                
    //                         cout<<"delete"<<endl;
                itmethod = new TFgmresIte(MatVect, Defect, NULL, 0, N_LocalDOF, 1);
                TDatabase::ParamDB->INTERNAL_LOCAL_DOF = N_LocalDOF;

                #ifdef __2D__
                sqmatrix[0] = (TSquareMatrix2D *)(System+offset_Sys);
                #endif
                #ifdef __3D__
                sqmatrix[0] = (TSquareMatrix3D *)(System+offset_Sys);
                #endif
    
                int iter =0;
                
                double check2 = 0.0;
                for(iter=0;iter<N_LocalDOF;iter++)
                    check2 += Rhs[offset_Rhs+iter]*Rhs[offset_Rhs+iter];
                
                memset(sol+offset_Rhs,0,N_LocalDOF*SizeOfDouble);
                verbose =  TDatabase::ParamDB->SC_VERBOSE;
                TDatabase::ParamDB->SC_VERBOSE = -1;
                
                int tem = TDatabase::ParamDB->SC_FLEXIBLE_KRYLOV_SPACE_SOLVER;
                TDatabase::ParamDB->SC_FLEXIBLE_KRYLOV_SPACE_SOLVER = 0;
                
                if(check2)
                itmethod->Iterate(matrix,NULL,sol+offset_Rhs,Rhs+offset_Rhs);
                
                
                TDatabase::ParamDB->SC_FLEXIBLE_KRYLOV_SPACE_SOLVER=tem;
                TDatabase::ParamDB->SC_VERBOSE = verbose;
                memcpy(Rhs+offset_Rhs, sol+offset_Rhs, N_LocalDOF*SizeOfDouble);
                }
                else
                {   */        
                SolveLinearSystemLapack(System+offset_Sys, Rhs+offset_Rhs, N_LocalDOF, N_LocalDOF);
//                 }
                }

//================================================= End Solving local system ===========================================

  
  // update dof
	      for(int j=0;j<NUDOF[i];j++)
	      {
		int l = UDOFs[offset+j];
        int j1 = j;
		int j2 = j+NUDOF[i];
		#ifdef __3D__
		int j3 = j2+NUDOF[i];
		#endif
        
		u1[l] += damp*Rhs[offset_Rhs+j1];
		u2[l] += damp*Rhs[offset_Rhs+j2];
		#ifdef __3D__
		u3[l] += damp*Rhs[offset_Rhs+j3];
		#endif  		
	      }
	      p[i] += damp*Rhs[offset_Rhs+N_UGEO];
		//	----------------------------===================================-----------------------=====================================-------------------	      
          

	    } // endfor loop over pressure nodes

#ifdef _MPI
  t2 = MPI_Wtime();
#else
  t2 = GetTime();
#endif
timeVankaSolve += t2-t1;    
    CUDA_CHECK(cudaMemcpyAsync(d_u1, u1,  N_DOF* sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    CUDA_CHECK(cudaStreamSynchronize(stream[0]));
        }
    
        

	    
	    // apply damping
	    for(j=0;j<N_DOF;j++)
	      u1[j] = uold[j]+alpha*(u1[j]-uold[j]);

	    // set Dirichlet values
	    memcpy(u1+HangingNodeBound, rhs1+HangingNodeBound,
		  N_Dirichlet*SizeOfDouble);
	    memcpy(u2+HangingNodeBound, rhs2+HangingNodeBound,
		  N_Dirichlet*SizeOfDouble);
	  #ifdef __3D__
	    memcpy(u3+HangingNodeBound, rhs3+HangingNodeBound,
		  N_Dirichlet*SizeOfDouble);
	  #endif
	    

	      // itmethod exists
	    if ( TDatabase::ParamDB->INTERNAL_LOCAL_DOF >0)
	    {
	      TDatabase::ParamDB->INTERNAL_LOCAL_DOF=0;
	      delete itmethod;
	    }
	    
    #ifdef _MPI      
    ParCommU->CommUpdate(u1);   
    ParCommP->CommUpdate(p);
    #endif
    
        // set Dirichlet values
    memcpy(u1+HangingNodeBound, rhs1+HangingNodeBound,
        N_Dirichlet*SizeOfDouble);
    memcpy(u2+HangingNodeBound, rhs2+HangingNodeBound,
        N_Dirichlet*SizeOfDouble);
    #ifdef __3D__
    memcpy(u3+HangingNodeBound, rhs3+HangingNodeBound,
        N_Dirichlet*SizeOfDouble);
    #endif
        
    CUDA_CHECK(cudaMemcpyAsync(d_u1, u1,  N_DOF* sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    CUDA_CHECK(cudaStreamSynchronize(stream[0]));
    
    // save current solution on 'old' vectors
    memcpy(uold, u1, N_DOF*SizeOfDouble);

    }
   
#ifdef _MPI
  TDatabase::ParamDB->time_vanka_end = MPI_Wtime();
#else
  TDatabase::ParamDB->time_vanka_end = GetTime();
#endif

  TDatabase::ParamDB->time_vanka += TDatabase::ParamDB->time_vanka_end - TDatabase::ParamDB->time_vanka_start; 
    
    

    
    delete []System;
    delete []Rhs;
    delete []sol;
    
    delete []NUDOF;
    delete []UDOFs;
    
    for (int i = 0; i < nStreams; ++i)
        CUDA_CHECK( cudaStreamDestroy(stream[i]) );
  
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_ARowPtr));
    CUDA_CHECK(cudaFree(d_AKCol));
    CUDA_CHECK(cudaFree(d_A11Entries));
    CUDA_CHECK(cudaFree(d_A12Entries));
    CUDA_CHECK(cudaFree(d_A13Entries));
    CUDA_CHECK(cudaFree(d_A21Entries));
    CUDA_CHECK(cudaFree(d_A22Entries));
    CUDA_CHECK(cudaFree(d_A23Entries));
    CUDA_CHECK(cudaFree(d_A31Entries));
    CUDA_CHECK(cudaFree(d_A32Entries));
    CUDA_CHECK(cudaFree(d_A33Entries));
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_BTRowPtr));
    CUDA_CHECK(cudaFree(d_BTKCol));
    CUDA_CHECK(cudaFree(d_B1TEntries));
    CUDA_CHECK(cudaFree(d_B2TEntries));
    CUDA_CHECK(cudaFree(d_B3TEntries));
          
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_BRowPtr));
    CUDA_CHECK(cudaFree(d_BKCol));
    CUDA_CHECK(cudaFree(d_B1Entries));
    CUDA_CHECK(cudaFree(d_B2Entries));
    CUDA_CHECK(cudaFree(d_B3Entries));
    
    CUDA_CHECK(cudaFree(d_u1));
    CUDA_CHECK(cudaFree(d_rhs1));
    CUDA_CHECK(cudaFree(d_PDOFReorder));
    CUDA_CHECK(cudaFree(d_UDOFs));
    
    CUDA_CHECK(cudaFree(d_Rhs));
    CUDA_CHECK(cudaFree(d_System));

 }
 
void TNSE_MGLevel4::NodalVanka_GPU(double *u1, double *rhs1, double *aux,
        int N_Parameters, double *Parameters, int smoother, int smooth)
 {
     cout<<"NV_GPU"<<endl;
#ifdef _MPI
  TDatabase::ParamDB->time_vanka_start = MPI_Wtime();
#else
  TDatabase::ParamDB->time_vanka_start = GetTime();
#endif
  
#ifdef _MPI
  int rank, *MasterOfDofU,*MasterOfDofP;
  
  char *DofmarkerP = ParCommP->Get_DofMarker();
  char *DofmarkerU = ParCommU->Get_DofMarker();
  
  MPI_Comm_rank(TDatabase::ParamDB->Comm, &rank); 
  MasterOfDofU = ParCommU->GetMaster();
  MasterOfDofP = ParCommP->GetMaster();    
#endif
  
#ifdef __2D__
  const int MaxN_LocalU = 2*MaxN_BaseFunctions2D;
  const int SystemRhs = 3*MaxN_BaseFunctions2D;
  TSquareMatrix2D *sqmatrix[1];
#endif
#ifdef __3D__
  const int MaxN_LocalU = 4*MaxN_BaseFunctions3D;
  const int SystemRhs = 8*MaxN_BaseFunctions3D;
  TSquareMatrix3D *sqmatrix[1];
#endif
  int i,j,k,l,m;
  int order;
  int j1, j2, j3, j4, k1, k2, k3;
  double value, value1, value2, value3;
  double value11,value12,value13,value21,value22;
  double value23,value31,value32,value33;
  double *uold, *pold;
//   double System[SystemRhs*SystemRhs];
//   double Rhs[SystemRhs], 
  
//   double sol[SystemRhs];
  int N_LocalDOF;
  int begin, end, HangingBound, begin1, end1, verbose;
  int UDOF, N_U, N_U2, N_UGEO;
  double *u2, *u3, *p, *rhs2, *rhs3, *rhsp;
  TItMethod *itmethod = NULL;
  double damp = TDatabase::ParamDB->SC_SMOOTH_DAMP_FACTOR_COARSE_SADDLE;
  int LargestDirectSolve = TDatabase::ParamDB->SC_LARGEST_DIRECT_SOLVE;
  MatVecProc *MatVect=MatVectFull;
  DefectProc *Defect=DefectFull;
  TSquareMatrix **matrix= (TSquareMatrix **)sqmatrix;
  
  double gar;

  TDatabase::ParamDB->INTERNAL_LOCAL_DOF = -1;
  
// #ifdef __2D__
//   sqmatrix[0] = (TSquareMatrix2D *)System;
// #endif
// #ifdef __3D__
//   sqmatrix[0] = (TSquareMatrix3D *)System;
// #endif

  HangingBound = USpace->GetHangingBound();

  // set pointers
  u2 = u1 + N_UDOF;
#ifdef __3D__
  u3 = u2 + N_UDOF;
#endif
  p  = u1 + GEO_DIM*N_UDOF;

  rhs2 = rhs1 + N_UDOF;
#ifdef __3D__
  rhs3 = rhs2 + N_UDOF;
#endif
  rhsp = rhs1 + GEO_DIM*N_UDOF;

  // set Dirichlet values
  
  memcpy(u1+HangingNodeBound, rhs1+HangingNodeBound,
         N_Dirichlet*SizeOfDouble);
  memcpy(u2+HangingNodeBound, rhs2+HangingNodeBound,
         N_Dirichlet*SizeOfDouble);
#ifdef __3D__
  memcpy(u3+HangingNodeBound, rhs3+HangingNodeBound,
         N_Dirichlet*SizeOfDouble);
#endif

  // old values
  uold = aux;
  pold = uold+GEO_DIM*N_UDOF;

  // save current solution on 'old' vectors
  memcpy(uold, u1, N_DOF*SizeOfDouble);
  
  
    int nStreams = 2;
    cudaStream_t stream[nStreams];

    for (int i = 0; i < nStreams; ++i)
        CUDA_CHECK( cudaStreamCreate(&stream[i]) );

    int nz_A = A11->GetN_Entries();
    int n_A = A11->GetN_Rows(); 

    int nz_B = B1->GetN_Entries();
    int n_B = B1->GetN_Rows(); 

    int nz_BT = B1T->GetN_Entries();
    int n_BT = B1T->GetN_Rows(); 
    
    int maxDOFPerColor=-1;
  
      for(int i=0;i<N_CPDOF;i++)
        {
            int temp = (ptrPDOFColors[i+1] - ptrPDOFColors[i]);
//                 cout<<"temp:"<<temp<<endl;
                
                if(maxDOFPerColor< temp){
                    
                    maxDOFPerColor = temp;
                    
                }
                
        }
        
//     cout<<"neha:maxDOFPerColor:"<<maxDOFPerColor<<endl;   
//         Coll = USpace->GetCollection();
        
        bool flag=false;
        int max_N_U=-1;
        int max_N_LocalDOF;
        
    for(int ii=0;ii<N_CPDOF;ii++)
        {
            for(int jj=ptrPDOFColors[ii];jj<ptrPDOFColors[ii+1];jj++)
            {
                i = PDOFReorder[jj];
                N_U = 0;

	  #ifdef _MPI
	      if(TDatabase::ParamDB->DOF_Average){
		if(DofmarkerP[i] == 'h' || DofmarkerP[i] == 'H')
		  continue;
	      }     
	      else{
		if(DofmarkerP[i] == 'h' || DofmarkerP[i] == 'H'  ||  DofmarkerP[i] == 's')
		  continue;
	      }
	  #endif
	      // go through row i of B1 and B2
	      begin = BRowPtr[i];
	      end = BRowPtr[i+1];
	      value = rhsp[i];      // rhs of this pressure value
	    for(k=begin;k<end;k++)
	      { 
		l=BKCol[k]; 

	  #ifdef _MPI   
	      if(TDatabase::ParamDB->DOF_Average){
		if(DofmarkerU[l] == 'h' || DofmarkerU[l] == 'H')
		  continue;
	      }
	      
	      else{
		if(DofmarkerU[l] == 'h' || DofmarkerU[l] == 'H' || DofmarkerU[l] == 's')
		  continue;
	      }
	  #endif

	      N_U++;           // count # velo dof connected to the pressure dof
	      }                    // row done
	      
	       if(N_U>max_N_U){
                max_N_U=N_U;
            }
            }
           
        }
        
        N_U=max_N_U;
        N_UGEO = GEO_DIM * max_N_U;
        max_N_LocalDOF = N_UGEO +1;
    
//     cudaEvent_t start, stop;
//     float time;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop); 
//     
//     cudaEventRecord(start, 0);
    
    //     transfer solution and rhs
    double* d_u1 = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_u1, N_DOF * sizeof(double)));
    
    double* d_rhs1 = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_rhs1, N_DOF * sizeof(double)));
    
    int* d_PDOFReorder = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_PDOFReorder, N_PDOF * sizeof(int)));
    
    // Copy to GPU memory
    CUDA_CHECK(cudaMemcpyAsync(d_u1, u1,  N_DOF* sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_rhs1, rhs1, N_DOF * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_PDOFReorder, PDOFReorder, N_PDOF * sizeof(int), cudaMemcpyHostToDevice,stream[0]));

//         cout<<"N_LocalDOF:"<<N_LocalDOF<<endl;
    
    double *System = new double[maxDOFPerColor * (max_N_LocalDOF * max_N_LocalDOF)];
//     double *temp_System = new double[maxDOFPerColor * (max_N_LocalDOF * max_N_LocalDOF)];
    double *Rhs = new double[maxDOFPerColor * max_N_LocalDOF];
    double *sol = new double[maxDOFPerColor * max_N_LocalDOF];
    

  
    double* d_System = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_System, maxDOFPerColor * (max_N_LocalDOF * max_N_LocalDOF) * sizeof(double)));
    
    double* d_Rhs = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_Rhs, maxDOFPerColor * max_N_LocalDOF * sizeof(double)));
    
    char* d_DofmarkerP = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_DofmarkerP, N_PDOF * sizeof(char)));
    
    CUDA_CHECK(cudaMemcpyAsync(d_DofmarkerP, DofmarkerP, N_PDOF * sizeof(char), cudaMemcpyHostToDevice,stream[0]));
  
    char* d_DofmarkerU = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_DofmarkerU, N_UDOF * sizeof(char)));
    
    CUDA_CHECK(cudaMemcpyAsync(d_DofmarkerU, DofmarkerU, N_UDOF * sizeof(char), cudaMemcpyHostToDevice,stream[0]));
    
    
    //     transfer A matrix
    int* d_ARowPtr = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_ARowPtr, (n_A+1) * sizeof(int)));
    
    int* d_AKCol = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_AKCol, nz_A * sizeof(int)));
    
    double* d_A11Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A11Entries, nz_A * sizeof(double)));
    
    double* d_A12Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A12Entries, nz_A * sizeof(double)));
    
    double* d_A13Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A13Entries, nz_A * sizeof(double)));
    
    double* d_A21Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A21Entries, nz_A * sizeof(double)));
    
    double* d_A22Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A22Entries, nz_A * sizeof(double)));
    
    double* d_A23Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A23Entries, nz_A * sizeof(double)));
    
    double* d_A31Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A31Entries, nz_A * sizeof(double)));
    
    double* d_A32Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A32Entries, nz_A * sizeof(double)));
    
    double* d_A33Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A33Entries, nz_A * sizeof(double)));
    
        // Copy to GPU memory
    CUDA_CHECK(cudaMemcpyAsync(d_ARowPtr, ARowPtr, (n_A+1) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_AKCol, AKCol, nz_A * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_A11Entries, A11Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_A12Entries, A12Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_A13Entries, A13Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_A21Entries, A21Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_A22Entries, A22Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_A23Entries, A23Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_A31Entries, A31Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_A32Entries, A32Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_A33Entries, A33Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    
//     cout<<"neha:color"<<N_CIntCell<<endl;
    
//     transfer BT matrix
    int* d_BTRowPtr = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_BTRowPtr, (n_BT+1) * sizeof(int)));
    
    int* d_BTKCol = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_BTKCol, nz_BT * sizeof(int)));
    
    double* d_B1TEntries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_B1TEntries, nz_BT * sizeof(double)));
    
    double* d_B2TEntries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_B2TEntries, nz_BT * sizeof(double)));
    
    double* d_B3TEntries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_B3TEntries, nz_BT * sizeof(double)));
    
    // Copy to GPU memory
    CUDA_CHECK(cudaMemcpyAsync(d_BTRowPtr, BTRowPtr, (n_BT+1) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_BTKCol, BTKCol, nz_BT * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_B1TEntries, B1TEntries, nz_BT * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_B2TEntries, B2TEntries, nz_BT * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_B3TEntries, B3TEntries, nz_BT * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
//     transfer B matrix
    int* d_BRowPtr = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_BRowPtr, (n_B+1) * sizeof(int)));
    
    int* d_BKCol = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_BKCol, nz_B * sizeof(int)));
    
    double* d_B1Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_B1Entries, nz_B * sizeof(double)));
    
    double* d_B2Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_B2Entries, nz_B * sizeof(double)));
    
    double* d_B3Entries = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_B3Entries, nz_B * sizeof(double)));
    
    // Copy to GPU memory
    CUDA_CHECK(cudaMemcpyAsync(d_BRowPtr, BRowPtr, (n_B+1) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_BKCol, BKCol, nz_B * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_B1Entries, B1Entries, nz_B * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_B2Entries, B2Entries, nz_B * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_B3Entries, B3Entries, nz_B * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
  
//     cout<<"N_U:"<<N_U<<endl;
    int* NUDOF = new int[N_PDOF];
    
    int* UDOFs = new int[max_N_U * maxDOFPerColor];
    
//     int* temp_UDOFs = new int[N_U * maxDOFPerColor];
    int* d_UDOFs = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_UDOFs, max_N_U * maxDOFPerColor * sizeof(int)));
    
    int* d_NUDOF = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_NUDOF, N_PDOF * sizeof(int)));

    
    CUDA_CHECK(cudaStreamSynchronize(stream[0]));
    
//     cudaEventRecord(stop, 0);
//     cudaEventSynchronize(stop);
    
    float milliseconds;
    
//     cudaEventElapsedTime(&milliseconds, start, stop);
    
//     data_transfer_time += milliseconds;
    
    int end_itr,offset_Rhs,offset_Sys,offset;
    
    if(smooth == -1)
        end_itr = TDatabase::ParamDB->SC_PRE_SMOOTH_SADDLE;
    else if(smooth == 0)
        end_itr = TDatabase::ParamDB->SC_COARSE_MAXIT_SADDLE;
    else
        end_itr = TDatabase::ParamDB->SC_POST_SMOOTH_SADDLE;
    
    int thread_blocks;
    double t1,t2;
    
    int numThreads = TDatabase::ParamDB->OMPNUMTHREADS;
     
    omp_set_num_threads(numThreads);
    
//     int* PReorder=ParCommP->GetReorder_M();
//     
//     int PN_OwnDof=ParCommP->GetN_OwnDof();
//     
//     int* UReorder=ParCommU->GetReorder_M();
//     
//     int UN_OwnDof=ParCommU->GetN_OwnDof();
    
//     double *System = new double[maxDOFPerColor * (N_LocalDOF * N_LocalDOF)];
//     double *Rhs = new double[maxDOFPerColor * N_LocalDOF];
    
    for(int itr=0;itr<end_itr;itr++)
    {
  
        for(int ii=0;ii<N_CPDOF;ii++)
        {
//             cout<<"ii:"<<ii<<endl;
            
            thread_blocks = (ptrPDOFColors[ii+1]-ptrPDOFColors[ii])/THREADS_PER_BLOCK + 1;
            
            CUDA_CHECK(cudaMemset(d_System, 0, maxDOFPerColor * (max_N_LocalDOF * max_N_LocalDOF) * sizeof(double) ));
            
//             cudaEventRecord(start, 0);
            NodalVanka_assembleP<<<thread_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(d_BRowPtr, d_BKCol, 
                    d_B1Entries, d_B2Entries,  d_B3Entries, d_u1, d_rhs1, d_PDOFReorder, N_UDOF, d_Rhs, ptrPDOFColors[ii], ptrPDOFColors[ii+1], d_DofmarkerU, d_DofmarkerP, d_UDOFs, d_NUDOF, max_N_U, max_N_LocalDOF);
            
            NodalVanka_assembleU<<<thread_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(d_ARowPtr, d_AKCol,
                    d_A11Entries, d_A12Entries, d_A13Entries, d_A21Entries, d_A22Entries, d_A23Entries, d_A31Entries,  d_A32Entries, d_A33Entries, 
                    d_BTRowPtr, d_BTKCol, d_B1TEntries, d_B2TEntries, d_B3TEntries,
                    d_u1, d_rhs1, d_PDOFReorder, N_UDOF, HangingBound, d_System, d_Rhs, ptrPDOFColors[ii], ptrPDOFColors[ii+1],d_DofmarkerP, d_UDOFs, d_NUDOF, max_N_U, max_N_LocalDOF);
            
            CUDA_CHECK(cudaStreamSynchronize(stream[0]));
            
            CUDA_CHECK(cudaMemcpyAsync(UDOFs, d_UDOFs, N_U * maxDOFPerColor * sizeof(int), cudaMemcpyDeviceToHost,stream[1]));
            
            CUDA_CHECK(cudaMemcpyAsync(Rhs, d_Rhs, maxDOFPerColor * max_N_LocalDOF * sizeof(double), cudaMemcpyDeviceToHost,stream[1]));
            
            NodalVanka_assembleSysMatB<<<thread_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(d_BRowPtr, d_BKCol, 
                    d_B1Entries, d_B2Entries,  d_B3Entries, d_u1, d_rhs1, d_PDOFReorder, N_UDOF, d_System, ptrPDOFColors[ii], ptrPDOFColors[ii+1], d_DofmarkerU, d_DofmarkerP, d_UDOFs, d_NUDOF, max_N_U, max_N_LocalDOF);
            
            NodalVanka_assembleSysMatA<<<thread_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(d_ARowPtr, d_AKCol,
                    d_A11Entries, d_A12Entries, d_A13Entries, d_A21Entries, d_A22Entries, d_A23Entries, d_A31Entries,  d_A32Entries, d_A33Entries, 
                    d_BTRowPtr, d_BTKCol, d_B1TEntries, d_B2TEntries, d_B3TEntries,
                    d_u1, d_rhs1, d_PDOFReorder, N_UDOF, HangingBound, d_System, ptrPDOFColors[ii], ptrPDOFColors[ii+1],d_DofmarkerP, d_UDOFs, d_NUDOF, max_N_U, max_N_LocalDOF);
            
//             NodalVanka_assembleSysMatBT<<<thread_blocks, THREADS_PER_BLOCK, 0, stream[1]>>>(d_ARowPtr, d_AKCol,
//                     d_A11Entries, d_A12Entries, d_A13Entries, d_A21Entries, d_A22Entries, d_A23Entries, d_A31Entries,  d_A32Entries, d_A33Entries, 
//                     d_BTRowPtr, d_BTKCol, d_B1TEntries, d_B2TEntries, d_B3TEntries,
//                     d_u1, d_rhs1, d_PDOFReorder, N_U, N_LocalDOF, N_UDOF, HangingBound, d_System, d_Rhs, ptrPDOFColors[ii], ptrPDOFColors[ii+1],d_DofmarkerP, d_UDOFs);
            
//             cudaEventRecord(stop, 0);
//             cudaEventSynchronize(stop);
//             cudaEventElapsedTime(&milliseconds, start, stop);
//             
//             kernel_time += milliseconds;

            
            
            
            CUDA_CHECK(cudaMemcpyAsync(System, d_System, maxDOFPerColor * max_N_LocalDOF * max_N_LocalDOF * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
            
            CUDA_CHECK(cudaMemcpyAsync(NUDOF, d_NUDOF, N_PDOF * sizeof(int), cudaMemcpyDeviceToHost,stream[0]));
                        
            CUDA_CHECK(cudaStreamSynchronize(stream[0]));
            
            CUDA_CHECK(cudaStreamSynchronize(stream[1]));
            

            
//             NodalVanka_assembleP<<<thread_blocks, THREADS_PER_BLOCK, 0, stream[1]>>>(d_ARowPtr, d_AKCol,
//                                 d_A11Entries, d_A12Entries, d_A13Entries, d_A21Entries, d_A22Entries, d_A23Entries, d_A31Entries,  d_A32Entries, d_A33Entries, 
//                                 d_BTRowPtr, d_BTKCol, d_B1TEntries, d_B2TEntries, d_B3TEntries,
//                                 d_u1, d_rhs1, d_PDOFReorder, N_U, N_P, N_LocalDOF, N_UDOF,
//                                 d_UGlobalNumbers, d_UBeginIndex, d_PGlobalNumbers, d_PBeginIndex, ActiveBound, d_System, d_Rhs, ptrPDOFColors[i], ptrPDOFColors[i+1], d_DofmarkerP, d_UDOFs);
            
            
#ifdef _MPI
  t1 = MPI_Wtime();
#else
  t1 = GetTime();
#endif            
            
//             verify1(temp_Rhs,Rhs, maxDOFPerColor * N_LocalDOF, rank);
            
            #pragma omp parallel for default(shared) schedule(dynamic)
            for(int jj=ptrPDOFColors[ii];jj<ptrPDOFColors[ii+1];jj++)
            {
                int offset_Rhs = (jj-ptrPDOFColors[ii]) * max_N_LocalDOF;
                int offset_Sys = (jj-ptrPDOFColors[ii]) * (max_N_LocalDOF * max_N_LocalDOF);
                int offset = (jj-ptrPDOFColors[ii]) * max_N_U;
                
                int i = PDOFReorder[jj];
                
//                 #ifdef _MPI    
//                 
//                 if(TDatabase::ParamDB->DOF_Average){
//                 if(DofmarkerP[i] == 'h' || DofmarkerP[i] == 'H')
//                 continue;
//                 }     
//                 else{
//                 if(DofmarkerP[i] == 'h' || DofmarkerP[i] == 'H'  ||  DofmarkerP[i] == 's')
//                 continue;
//                 }
//                 
//                 #endif
            
                int N_U2 = 2 * NUDOF[i];
                int N_UGEO = GEO_DIM * NUDOF[i];
                int N_LocalDOF = N_UGEO +1;

                if(C)
                {
                // fill C block if present
                begin = CRowPtr[i];
                end = CRowPtr[i+1];
                for(k=begin;k<end;k++)
                {
                l = CKCol[k];
                value = -CEntries[k]; // minus is right sign
                Rhs[offset_Rhs+N_LocalDOF-1] -= value*p[l];
                if(l==i) // main diagonal
                    System[offset_Sys+N_LocalDOF*N_LocalDOF-1] = value;
                } // endfor k
                } // endif C
	      
//=================================================Start Solving local system ===========================================

//                 #ifdef _MPI
//                 TDatabase::ParamDB->time_vanka_solve_start = MPI_Wtime();
//                 #else
//                 TDatabase::ParamDB->time_vanka_solve_start = GetTime();
//                 #endif
                if ((smoother==30 || smoother==32) && !C) // no diagonal Vanka for matrix C
                {
            #ifdef __2D__
                // diagonal Vanka
                SolveDiagonalVanka2D(System+offset_Sys,  Rhs+offset_Rhs, NUDOF[i], 1, N_LocalDOF);
            #endif
            #ifdef __3D__
                // diagonal Vanka
                SolveDiagonalVanka3D(System+offset_Sys,  Rhs+offset_Rhs, NUDOF[i], 1, N_LocalDOF);
            #endif
                }
                else
                {
                // full Vanka
/*                if (N_LocalDOF < LargestDirectSolve)
                        {
                            if ( TDatabase::ParamDB->INTERNAL_LOCAL_DOF >0)
                            {
                                delete itmethod;
                            }
                            MatVecProc *MatVect=MatVectFull;
                            DefectProc *Defect=DefectFull;
                            
                //                         cout<<"delete"<<endl;
                            TItMethod *itmethod = new TFgmresIte(MatVect, Defect, NULL, 0, N_LocalDOF, 1);
                            TDatabase::ParamDB->INTERNAL_LOCAL_DOF = N_LocalDOF;
                            TSquareMatrix **matrix= (TSquareMatrix **)sqmatrix;
                            #ifdef __2D__
                            sqmatrix[0] = (TSquareMatrix2D *)(System+offset_Sys);
                            #endif
                            #ifdef __3D__
                            sqmatrix[0] = (TSquareMatrix3D *)(System+offset_Sys);
                            #endif
                
                        int iter =0;
                        
                        double check2 = 0.0;
                        for(iter=0;iter<N_LocalDOF;iter++)
                            check2 += Rhs[offset_Rhs+iter]*Rhs[offset_Rhs+iter];
                        
                        memset(sol+offset_Rhs,0,N_LocalDOF*SizeOfDouble);
                        verbose =  TDatabase::ParamDB->SC_VERBOSE;
                        TDatabase::ParamDB->SC_VERBOSE = -1;
                        
                        int tem = TDatabase::ParamDB->SC_FLEXIBLE_KRYLOV_SPACE_SOLVER;
                        TDatabase::ParamDB->SC_FLEXIBLE_KRYLOV_SPACE_SOLVER = 0;
                        
                        if(check2)
                        itmethod->Iterate(matrix,NULL,sol+offset_Rhs,Rhs+offset_Rhs);
                        
                        
                        TDatabase::ParamDB->SC_FLEXIBLE_KRYLOV_SPACE_SOLVER=tem;
                        TDatabase::ParamDB->SC_VERBOSE = verbose;
                        memcpy(Rhs+offset_Rhs, sol+offset_Rhs, N_LocalDOF*SizeOfDouble);
                        
                //                   for(int pp=0; pp<N_LocalDOF; pp++){
                // //                 for(int qq=0; qq<N_LocalDOF; qq++){
                //                         cout<<Rhs[offset_Rhs+pp]<<" ";
                //                         
                // //                 }
                //                 
                //         }
                        }
                else
                {*/           
                SolveLinearSystemLapack(System+offset_Sys, Rhs+offset_Rhs, N_LocalDOF, N_LocalDOF);
//                 }
                }

//================================================= End Solving local system ===========================================

  
  // update dof
	      for(int j=0;j<NUDOF[i];j++)
	      {
		int l = UDOFs[offset+j];
        int j1 = j;
		int j2 = j+NUDOF[i];
		#ifdef __3D__
		int j3 = j2+NUDOF[i];
		#endif
        
		u1[l] += damp*Rhs[offset_Rhs+j1];
		u2[l] += damp*Rhs[offset_Rhs+j2];
		#ifdef __3D__
		u3[l] += damp*Rhs[offset_Rhs+j3];
		#endif  		
	      }
	      p[i] += damp*Rhs[offset_Rhs+N_UGEO];
		//	----------------------------===================================-----------------------=====================================-------------------	      
          

	    } // endfor loop over pressure nodes

#ifdef _MPI
  t2 = MPI_Wtime();
#else
  t2 = GetTime();
#endif
timeVankaSolve += t2-t1;    
    CUDA_CHECK(cudaMemcpyAsync(d_u1, u1,  N_DOF* sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    CUDA_CHECK(cudaStreamSynchronize(stream[0]));
        }
    
        

	    
	    // apply damping
	    for(j=0;j<N_DOF;j++)
	      u1[j] = uold[j]+alpha*(u1[j]-uold[j]);

	    // set Dirichlet values
	    memcpy(u1+HangingNodeBound, rhs1+HangingNodeBound,
		  N_Dirichlet*SizeOfDouble);
	    memcpy(u2+HangingNodeBound, rhs2+HangingNodeBound,
		  N_Dirichlet*SizeOfDouble);
	  #ifdef __3D__
	    memcpy(u3+HangingNodeBound, rhs3+HangingNodeBound,
		  N_Dirichlet*SizeOfDouble);
	  #endif
	    

	      // itmethod exists
	    if ( TDatabase::ParamDB->INTERNAL_LOCAL_DOF >0)
	    {
	      TDatabase::ParamDB->INTERNAL_LOCAL_DOF=0;
	      delete itmethod;
	    }
	    
    #ifdef _MPI      
    ParCommU->CommUpdate(u1);   
    ParCommP->CommUpdate(p);
    #endif
    
        // set Dirichlet values
    memcpy(u1+HangingNodeBound, rhs1+HangingNodeBound,
        N_Dirichlet*SizeOfDouble);
    memcpy(u2+HangingNodeBound, rhs2+HangingNodeBound,
        N_Dirichlet*SizeOfDouble);
    #ifdef __3D__
    memcpy(u3+HangingNodeBound, rhs3+HangingNodeBound,
        N_Dirichlet*SizeOfDouble);
    #endif
        
    CUDA_CHECK(cudaMemcpyAsync(d_u1, u1,  N_DOF* sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    CUDA_CHECK(cudaStreamSynchronize(stream[0]));
    
    // save current solution on 'old' vectors
    memcpy(uold, u1, N_DOF*SizeOfDouble);

    }
   
#ifdef _MPI
  TDatabase::ParamDB->time_vanka_end = MPI_Wtime();
#else
  TDatabase::ParamDB->time_vanka_end = GetTime();
#endif

  TDatabase::ParamDB->time_vanka += TDatabase::ParamDB->time_vanka_end - TDatabase::ParamDB->time_vanka_start; 
    
    

    
    delete []System;
    delete []Rhs;
    delete []sol;
    
    delete []NUDOF;
    delete []UDOFs;
    
    for (int i = 0; i < nStreams; ++i)
        CUDA_CHECK( cudaStreamDestroy(stream[i]) );
  
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_ARowPtr));
    CUDA_CHECK(cudaFree(d_AKCol));
    CUDA_CHECK(cudaFree(d_A11Entries));
    CUDA_CHECK(cudaFree(d_A12Entries));
    CUDA_CHECK(cudaFree(d_A13Entries));
    CUDA_CHECK(cudaFree(d_A21Entries));
    CUDA_CHECK(cudaFree(d_A22Entries));
    CUDA_CHECK(cudaFree(d_A23Entries));
    CUDA_CHECK(cudaFree(d_A31Entries));
    CUDA_CHECK(cudaFree(d_A32Entries));
    CUDA_CHECK(cudaFree(d_A33Entries));
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_BTRowPtr));
    CUDA_CHECK(cudaFree(d_BTKCol));
    CUDA_CHECK(cudaFree(d_B1TEntries));
    CUDA_CHECK(cudaFree(d_B2TEntries));
    CUDA_CHECK(cudaFree(d_B3TEntries));
          
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_BRowPtr));
    CUDA_CHECK(cudaFree(d_BKCol));
    CUDA_CHECK(cudaFree(d_B1Entries));
    CUDA_CHECK(cudaFree(d_B2Entries));
    CUDA_CHECK(cudaFree(d_B3Entries));
    
    CUDA_CHECK(cudaFree(d_u1));
    CUDA_CHECK(cudaFree(d_rhs1));
    CUDA_CHECK(cudaFree(d_PDOFReorder));
    CUDA_CHECK(cudaFree(d_UDOFs));
    
    CUDA_CHECK(cudaFree(d_Rhs));
    CUDA_CHECK(cudaFree(d_System));
 }
 
 void TNSE_MGLevel4::NodalVanka_Level_Split(double *u1, double *rhs1, double *aux,
        int N_Parameters, double *Parameters, int smoother, int smooth)
 {
     int split_level=1;
    
    if(Level > split_level){
        NodalVanka_GPU(u1, rhs1, aux, N_Parameters, Parameters, smoother, smooth);
    }
    else{
        int end;
        
        if(smooth == -1)
            end = TDatabase::ParamDB->SC_PRE_SMOOTH_SADDLE;
        else if(smooth == 0)
            end = TDatabase::ParamDB->SC_COARSE_MAXIT_SADDLE;
        else
            end = TDatabase::ParamDB->SC_POST_SMOOTH_SADDLE;
    
        for(int j=0;j<end;j++)
        {
            NodalVanka(u1, rhs1, aux, 
                N_Parameters, Parameters, smoother, 0);
        }
    }
 }
 
#endif
#endif
#endif
