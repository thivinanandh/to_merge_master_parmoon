#include <NSE_MGLevel4.h>
#include <Database.h>
#include <MooNMD_Io.h>
#include <Solver.h>
#include <omp.h>
#include "nvToolsExt.h"
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


        
#define THREADS_PER_BLOCK 128
        
const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();

        
// extern double timeVankaAssemble;
// extern double timeVankaSolve;

// extern double data_transfer_time;
// extern double kernel_time;

double timeVankaAssemble = 0;
double timeVankaSolve = 0;

double data_transfer_time=0;
double kernel_time=0;


__global__ void CellVanka_assembleSysA(    int* ARowPtr, int* AKCol,
                                double* A11Entries, double* A12Entries, double* A13Entries, double* A21Entries, double* A22Entries, double* A23Entries, double* A31Entries, double* A32Entries, double* A33Entries, 
                                int* BTRowPtr, int* BTKCol, double* B1TEntries, double* B2TEntries, double* B3TEntries, int* BRowPtr, int* BKCol, double* B1Entries, double* B2Entries, double* B3Entries, 
                                double* u1, double* rhs1, int* CellReorder, int N_U, int N_P, int N_LocalDOF, int N_UDOF,
                                int* UGlobalNumbers, int* UBeginIndex, int* PGlobalNumbers, int* PBeginIndex, int ActiveBound, double* System, double* Rhs,
                                int index1, int index2
                        
           )
{
    
    const unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int grid_size = gridDim.x * blockDim.x;
    
    int UDOF, PDOF;
    int *UDOFs,*PDOFs;
    int j,j1,j2,j3, ii, k,l,m, k1, k2, k3;
    int begin, end, begin1, end1;
    
    double *u2, *u3, *p, *rhs2, *rhs3, *rhsp;
    
    double value, value1, value2, value3;
    double value11,value12,value13,value21,value22;
    double value23,value31,value32,value33;
    
    int offset_Rhs,offset_Sys;
    
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
    

    for(unsigned int row = thread_id + index1; row < index2; row += grid_size)
    {
        ii = CellReorder[row];
        
        offset_Rhs = (row-index1) * N_LocalDOF;
        offset_Sys = (row-index1) * (N_LocalDOF * N_LocalDOF);
    
//                 Cell = Coll->GetCell(ii);
                
//             #ifdef _MPI
//                 if(haloCell[ii]){
//             //       cout << "this should" << endl;
//                 continue;
//             //       cout << "this shouldnt" << endl;
//                 }   
//             #endif
//             //    OutPut(i << downwind[i] << endl);
//             #ifdef __2D__
//                 UEle = TFEDatabase2D::GetFE2D(USpace->GetFE2D(ii, Cell));
//                 PEle = TFEDatabase2D::GetFE2D(PSpace->GetFE2D(ii, Cell));
//             #endif
//             #ifdef __3D__
//                 UEle = TFEDatabase3D::GetFE3D(USpace->GetFE3D(ii, Cell));
//                 PEle = TFEDatabase3D::GetFE3D(PSpace->GetFE3D(ii, Cell));
//             #endif

                // get local number of dof
                
                UDOFs = UGlobalNumbers+UBeginIndex[ii];
                PDOFs = PGlobalNumbers+PBeginIndex[ii];

                // fill local matrix
                for(int j=0;j<N_U;j++)
                {
                j1 = j;
                j2 = j+N_U;
            #ifdef __3D__
                j3 = j2+N_U;
            #endif
                UDOF = UDOFs[j];

                // A block
                begin = ARowPtr[UDOF];
                end = ARowPtr[UDOF+1];

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
                    if (UDOF>=ActiveBound){ // Dirichlet node
                    value12 = 0;
                    value21 = 0;
                    
                    }


            #endif
            #ifdef __3D__
                    if (UDOF>=ActiveBound){ // Dirichlet node
                    value12 = 0;
                    value13 = 0;
                    value21 = 0;
                    value23 = 0;
                    value31 = 0;
                    value32 = 0;
                    }

            #endif

                    for(m=0;m<N_U;m++)
                    if(UDOFs[m]==l)
                    {
                        // column belongs to local system
                        k1 = m*N_LocalDOF;
                        k2 = (m+N_U)*N_LocalDOF;
                        System[offset_Sys + k1+j1] = value11;
                        System[offset_Sys + k2+j1] = value12;
                        System[offset_Sys + k1+j2] = value21;
                        System[offset_Sys + k2+j2] = value22;
            #ifdef __3D__
                        k3 = (m+2*N_U)*N_LocalDOF;
                        System[offset_Sys + k3+j1] = value13;
                        System[offset_Sys + k3+j2] = value23;
                        System[offset_Sys + k1+j3] = value31;
                        System[offset_Sys + k2+j3] = value32;
                        System[offset_Sys + k3+j3] = value33;
            #endif
                        break;
                    }
                } // endfor k

                } // endfor j
                

                
    }   

}

__global__ void CellVanka_assembleSysBT(    int* ARowPtr, int* AKCol,
                                double* A11Entries, double* A12Entries, double* A13Entries, double* A21Entries, double* A22Entries, double* A23Entries, double* A31Entries, double* A32Entries, double* A33Entries, 
                                int* BTRowPtr, int* BTKCol, double* B1TEntries, double* B2TEntries, double* B3TEntries, int* BRowPtr, int* BKCol, double* B1Entries, double* B2Entries, double* B3Entries, 
                                double* u1, double* rhs1, int* CellReorder, int N_U, int N_P, int N_LocalDOF, int N_UDOF,
                                int* UGlobalNumbers, int* UBeginIndex, int* PGlobalNumbers, int* PBeginIndex, int ActiveBound, double* System, double* Rhs,
                                int index1, int index2
           )
{
    
    const unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int grid_size = gridDim.x * blockDim.x;
    
    int UDOF, PDOF;
    int *UDOFs,*PDOFs;
    int j,j1,j2,j3, ii, k,l,m, k1, k2, k3;
    int begin, end, begin1, end1;
    
    double *u2, *u3, *p, *rhs2, *rhs3, *rhsp;
    
    double value, value1, value2, value3;
    double value11,value12,value13,value21,value22;
    double value23,value31,value32,value33;
    
    int offset_Rhs,offset_Sys;
    
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
    

    for(unsigned int row = thread_id + index1; row < index2; row += grid_size)
    {
        ii = CellReorder[row];
        
        offset_Rhs = (row-index1) * N_LocalDOF;
        offset_Sys = (row-index1) * (N_LocalDOF * N_LocalDOF);
    
//                 Cell = Coll->GetCell(ii);
                
//             #ifdef _MPI
//                 if(haloCell[ii]){
//             //       cout << "this should" << endl;
//                 continue;
//             //       cout << "this shouldnt" << endl;
//                 }   
//             #endif
//             //    OutPut(i << downwind[i] << endl);
//             #ifdef __2D__
//                 UEle = TFEDatabase2D::GetFE2D(USpace->GetFE2D(ii, Cell));
//                 PEle = TFEDatabase2D::GetFE2D(PSpace->GetFE2D(ii, Cell));
//             #endif
//             #ifdef __3D__
//                 UEle = TFEDatabase3D::GetFE3D(USpace->GetFE3D(ii, Cell));
//                 PEle = TFEDatabase3D::GetFE3D(PSpace->GetFE3D(ii, Cell));
//             #endif

                // get local number of dof
                
                UDOFs = UGlobalNumbers+UBeginIndex[ii];
                PDOFs = PGlobalNumbers+PBeginIndex[ii];

                // fill local matrix
                for(int j=0;j<N_U;j++)
                {
                j1 = j;
                j2 = j+N_U;
            #ifdef __3D__
                j3 = j2+N_U;
            #endif
                UDOF = UDOFs[j];

                if(UDOF<ActiveBound)  // active dof
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
                    value = p[l];

                    for(m=0;m<N_P;m++)
                        if(PDOFs[m]==l)
                        {
                        // column belongs to local system
                        k1 = (m+GEO_DIM*N_U)*N_LocalDOF;
                        System[offset_Sys + k1+j1] = value1;
                        System[offset_Sys + k1+j2] = value2;
            #ifdef __3D__
                        System[offset_Sys + k1+j3] = value3;
            #endif
                        break;
                        }

                    } // endfor k
                } // endif UDOF<ActiveBound
                } // endfor j
                

                
    }   

}


__global__ void CellVanka_assembleSysB(    int* ARowPtr, int* AKCol,
                                double* A11Entries, double* A12Entries, double* A13Entries, double* A21Entries, double* A22Entries, double* A23Entries, double* A31Entries, double* A32Entries, double* A33Entries, 
                                int* BTRowPtr, int* BTKCol, double* B1TEntries, double* B2TEntries, double* B3TEntries, int* BRowPtr, int* BKCol, double* B1Entries, double* B2Entries, double* B3Entries, 
                                double* u1, double* rhs1, int* CellReorder, int N_U, int N_P, int N_LocalDOF, int N_UDOF,
                                int* UGlobalNumbers, int* UBeginIndex, int* PGlobalNumbers, int* PBeginIndex, int ActiveBound, double* System, double* Rhs,
                                int index1, int index2
           )
{
    
    const unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int grid_size = gridDim.x * blockDim.x;
    
    int UDOF, PDOF;
    int *UDOFs,*PDOFs;
    int j,j1,j2,j3, ii, k,l,m, k1, k2, k3;
    int begin, end, begin1, end1;
    
    double *u2, *u3, *p, *rhs2, *rhs3, *rhsp;
    
    double value, value1, value2, value3;
    double value11,value12,value13,value21,value22;
    double value23,value31,value32,value33;
    
    int offset_Rhs,offset_Sys;
    
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
    

    for(unsigned int row = thread_id + index1; row < index2; row += grid_size)
    {
        ii = CellReorder[row];
        
        offset_Rhs = (row-index1) * N_LocalDOF;
        offset_Sys = (row-index1) * (N_LocalDOF * N_LocalDOF);
    
//                 Cell = Coll->GetCell(ii);
                
//             #ifdef _MPI
//                 if(haloCell[ii]){
//             //       cout << "this should" << endl;
//                 continue;
//             //       cout << "this shouldnt" << endl;
//                 }   
//             #endif
//             //    OutPut(i << downwind[i] << endl);
//             #ifdef __2D__
//                 UEle = TFEDatabase2D::GetFE2D(USpace->GetFE2D(ii, Cell));
//                 PEle = TFEDatabase2D::GetFE2D(PSpace->GetFE2D(ii, Cell));
//             #endif
//             #ifdef __3D__
//                 UEle = TFEDatabase3D::GetFE3D(USpace->GetFE3D(ii, Cell));
//                 PEle = TFEDatabase3D::GetFE3D(PSpace->GetFE3D(ii, Cell));
//             #endif

                // get local number of dof
                
                UDOFs = UGlobalNumbers+UBeginIndex[ii];
                PDOFs = PGlobalNumbers+PBeginIndex[ii];

                
                for(j=0;j<N_P;j++)
                {
                j1 = j+GEO_DIM*N_U;
                PDOF = PDOFs[j];
                begin = BRowPtr[PDOF];
                end = BRowPtr[PDOF+1];
//                 Rhs[offset_Rhs + j1] = rhsp[PDOF];

                for(k=begin;k<end;k++)
                {
                    l=BKCol[k];
                    value1 = B1Entries[k];
                    value2 = B2Entries[k];
            #ifdef __3D__
                    value3 = B3Entries[k];
            #endif
                    for(m=0;m<N_U;m++)
                    if(UDOFs[m]==l)
                    {
                        // column belongs to local system
                        k1 = m;
                        k2 = m+N_U;
            #ifdef __3D__
                        k3 = k2 + N_U;
            #endif
                        System[offset_Sys + k1*N_LocalDOF+j1] = value1;
                        System[offset_Sys + k2*N_LocalDOF+j1] = value2;
            #ifdef __3D__
                        System[offset_Sys + k3*N_LocalDOF+j1] = value3;
            #endif
                        break;
                    }
                } // endfor k
                } // endfor j
                

                
    }   

}



__global__ void CellVanka_assembleU(    int* ARowPtr, int* AKCol,
                                double* A11Entries, double* A12Entries, double* A13Entries, double* A21Entries, double* A22Entries, double* A23Entries, double* A31Entries, double* A32Entries, double* A33Entries, 
                                int* BTRowPtr, int* BTKCol, double* B1TEntries, double* B2TEntries, double* B3TEntries, int* BRowPtr, int* BKCol, double* B1Entries, double* B2Entries, double* B3Entries, 
                                double* u1, double* rhs1, int* CellReorder, int N_U, int N_P, int N_LocalDOF, int N_UDOF,
                                int* UGlobalNumbers, int* UBeginIndex, int* PGlobalNumbers, int* PBeginIndex, int ActiveBound, double* System, double* Rhs,
                                int index1, int index2
           )
{
    
    const unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int grid_size = gridDim.x * blockDim.x;
    
    int UDOF, PDOF;
    int *UDOFs,*PDOFs;
    int j,j1,j2,j3, ii, k,l,m, k1, k2, k3;
    int begin, end, begin1, end1;
    
    double *u2, *u3, *p, *rhs2, *rhs3, *rhsp;
    
    double value, value1, value2, value3;
    double value11,value12,value13,value21,value22;
    double value23,value31,value32,value33;
    const unsigned int THREADS_PER_VECTOR = 16;
    const unsigned int warp_id = thread_id   /  THREADS_PER_VECTOR;
    const unsigned int VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
    const unsigned int num_vectors = VECTORS_PER_BLOCK * gridDim.x; 
    const unsigned int lane = threadIdx.x & (THREADS_PER_VECTOR - 1);
    
    int offset_Rhs,offset_Sys;
    
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
    

    for(unsigned int row = warp_id + index1; row < index2; row += num_vectors)
    {
        int ii = CellReorder[row];
        
        int offset_Rhs = (row-index1) * N_LocalDOF;
        int offset_Sys = (row-index1) * (N_LocalDOF * N_LocalDOF);
    
//                 Cell = Coll->GetCell(ii);
                
            // #ifdef _MPI
            //     if(haloCell[ii]){
            // //       cout << "this should" << endl;
            //     continue;
            // //       cout << "this shouldnt" << endl;
            //     }   
            // #endif
//             //    OutPut(i << downwind[i] << endl);
//             #ifdef __2D__
//                 UEle = TFEDatabase2D::GetFE2D(USpace->GetFE2D(ii, Cell));
//                 PEle = TFEDatabase2D::GetFE2D(PSpace->GetFE2D(ii, Cell));
//             #endif
//             #ifdef __3D__
//                 UEle = TFEDatabase3D::GetFE3D(USpace->GetFE3D(ii, Cell));
//                 PEle = TFEDatabase3D::GetFE3D(PSpace->GetFE3D(ii, Cell));
//             #endif

                // get local number of dof
                
                UDOFs = UGlobalNumbers+UBeginIndex[ii];
                PDOFs = PGlobalNumbers+PBeginIndex[ii];

                // fill local matrix
                for(int j=lane;j<N_U;j += THREADS_PER_VECTOR)
                {
                j1 = j;
                j2 = j+N_U;
            #ifdef __3D__
                j3 = j2+N_U;
            #endif
                UDOF = UDOFs[j];

                // A block
                begin = ARowPtr[UDOF];
                end = ARowPtr[UDOF+1];

                Rhs[offset_Rhs + j1] = rhs1[UDOF];
                Rhs[offset_Rhs + j2] = rhs2[UDOF];
            #ifdef __3D__
                Rhs[offset_Rhs + j3] = rhs3[UDOF];
            #endif

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
                    if (UDOF>=ActiveBound){ // Dirichlet node
                    value12 = 0;
                    value21 = 0;
                    
                    }

                    Rhs[offset_Rhs + j1] = Rhs[offset_Rhs + j1] - (value11*u1[l]+value12*u2[l]);
                    Rhs[offset_Rhs + j2] = Rhs[offset_Rhs + j2] - (value21*u1[l]+value22*u2[l]);
            #endif
            #ifdef __3D__
                    if (UDOF>=ActiveBound){ // Dirichlet node
                    value12 = 0;
                    value13 = 0;
                    value21 = 0;
                    value23 = 0;
                    value31 = 0;
                    value32 = 0;
                    }

                    Rhs[offset_Rhs + j1] = Rhs[offset_Rhs + j1] - (value11*u1[l]+value12*u2[l]+value13*u3[l]);
                    Rhs[offset_Rhs + j2] = Rhs[offset_Rhs + j2] - (value21*u1[l]+value22*u2[l]+value23*u3[l]);
                    Rhs[offset_Rhs + j3] = Rhs[offset_Rhs + j3] - (value31*u1[l]+value32*u2[l]+value33*u3[l]);
            #endif

                } // endfor k

                if(UDOF<ActiveBound)  // active dof
                {
                    // transpose(B) block for non-Dirichlet nodes
                    begin = BTRowPtr[UDOF];
                    end = BTRowPtr[UDOF+1];

                    for(k=begin;k<end;k++)
                    {
                    l = BTKCol[k];
                    value = p[l];
                    
                    value1 = B1TEntries[k];
                    value2 = B2TEntries[k];
                #ifdef __3D__
                    value3 = B3TEntries[k];
                #endif
                    
                    
//                     Rhs[offset_Rhs + j1]= __dsub_rn(Rhs[offset_Rhs + j1],(value1*value));
                    Rhs[offset_Rhs + j1] = Rhs[offset_Rhs + j1] - (value1*value);
                    Rhs[offset_Rhs + j2] = Rhs[offset_Rhs + j2] - (value2*value);
//                     Rhs[offset_Rhs + j2]= __dsub_rn(Rhs[offset_Rhs + j2],(value2*value));
            #ifdef __3D__
                    Rhs[offset_Rhs + j3] = Rhs[offset_Rhs + j3] - (value3*value);
//                     Rhs[offset_Rhs + j3]= __dsub_rn(Rhs[offset_Rhs + j3],(value3*value));
            #endif

                    } // endfor k
                } // endif UDOF<ActiveBound
                } // endfor j
                
    }   

}
__global__ void CellVanka_assembleP(    int* ARowPtr, int* AKCol,
                                double* A11Entries, double* A12Entries, double* A13Entries, double* A21Entries, double* A22Entries, double* A23Entries, double* A31Entries, double* A32Entries, double* A33Entries, 
                                int* BTRowPtr, int* BTKCol, double* B1TEntries, double* B2TEntries, double* B3TEntries, int* BRowPtr, int* BKCol, double* B1Entries, double* B2Entries, double* B3Entries, 
                                double* u1, double* rhs1, int* CellReorder, int N_U, int N_P, int N_LocalDOF, int N_UDOF,
                                int* UGlobalNumbers, int* UBeginIndex, int* PGlobalNumbers, int* PBeginIndex, int ActiveBound, double* System, double* Rhs,
                                int index1, int index2
           )
{
    
    const unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int grid_size = gridDim.x * blockDim.x;
    
    int UDOF, PDOF;
    int *UDOFs,*PDOFs;
    int j,j1,j2,j3, ii, k,l,m, k1, k2, k3;
    int begin, end, begin1, end1;
    
    double *u2, *u3, *p, *rhs2, *rhs3, *rhsp;
    
    double value, value1, value2, value3;
    double value11,value12,value13,value21,value22;
    double value23,value31,value32,value33;
    
    int offset_Rhs,offset_Sys;
    
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

    const unsigned int THREADS_PER_VECTOR = 16;
    const unsigned int warp_id = thread_id   /  THREADS_PER_VECTOR;
    const unsigned int VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
    const unsigned int num_vectors = VECTORS_PER_BLOCK * gridDim.x; 
    const unsigned int lane = threadIdx.x & (THREADS_PER_VECTOR - 1);
    

    for(unsigned int row = warp_id + index1; row < index2; row += num_vectors)
    {
        int ii = CellReorder[row];
        
        int offset_Rhs = (row-index1) * N_LocalDOF;
        int offset_Sys = (row-index1) * (N_LocalDOF * N_LocalDOF);
    
//                 Cell = Coll->GetCell(ii);
                
//             #ifdef _MPI
//                 if(haloCell[ii]){
//             //       cout << "this should" << endl;
//                 continue;
//             //       cout << "this shouldnt" << endl;
//                 }   
//             #endif
//             //    OutPut(i << downwind[i] << endl);
//             #ifdef __2D__
//                 UEle = TFEDatabase2D::GetFE2D(USpace->GetFE2D(ii, Cell));
//                 PEle = TFEDatabase2D::GetFE2D(PSpace->GetFE2D(ii, Cell));
//             #endif
//             #ifdef __3D__
//                 UEle = TFEDatabase3D::GetFE3D(USpace->GetFE3D(ii, Cell));
//                 PEle = TFEDatabase3D::GetFE3D(PSpace->GetFE3D(ii, Cell));
//             #endif

                // get local number of dof
                
                UDOFs = UGlobalNumbers+UBeginIndex[ii];
                PDOFs = PGlobalNumbers+PBeginIndex[ii];

                
                for(j=lane;j<N_P;j += THREADS_PER_VECTOR)
                {
                j1 = j+GEO_DIM*N_U;
                PDOF = PDOFs[j];
                begin = BRowPtr[PDOF];
                end = BRowPtr[PDOF+1];
                Rhs[offset_Rhs + j1] = rhsp[PDOF];

                for(k=begin;k<end;k++)
                {
                    l=BKCol[k];
                    value1 = B1Entries[k];
                    value2 = B2Entries[k];
            #ifdef __3D__
                    value3 = B3Entries[k];
            #endif
                    Rhs[offset_Rhs + j1] = Rhs[offset_Rhs + j1] - value1*u1[l];
//                     Rhs[offset_Rhs + j1]= __dsub_rn(Rhs[offset_Rhs + j1],(value1*u1[l]));
                    Rhs[offset_Rhs + j1] = Rhs[offset_Rhs + j1] - value2*u2[l];
//                     Rhs[offset_Rhs + j1]= __dsub_rn(Rhs[offset_Rhs + j1],(value1*u2[l]));
            #ifdef __3D__
                    Rhs[offset_Rhs + j1] = Rhs[offset_Rhs + j1] - value3*u3[l];
//                     Rhs[offset_Rhs + j1]= __dsub_rn(Rhs[offset_Rhs + j1],(value1*u3[l]));
            #endif


                } // endfor k
                } // endfor j
                

                
    }   

}

__global__ void updateSolution(double* u1, double* rhs1, int* CellReorder, int N_U, int N_P, int N_LocalDOF, int N_UDOF,
    int* UGlobalNumbers, int* UBeginIndex, int* PGlobalNumbers, int* PBeginIndex, int ActiveBound, double* System, double* Rhs,
    int index1, int index2, double damp)
                            
{

    const unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int grid_size = gridDim.x * blockDim.x;
    
    int UDOF, PDOF;
    int *UDOFs,*PDOFs;
    int j,j1,j2,j3, ii, k,l,m, k1, k2, k3;
    int begin, end, begin1, end1;
    
    double *u2, *u3, *p, *rhs2, *rhs3, *rhsp;
    
    int offset_Rhs,offset_Sys;

    const unsigned int THREADS_PER_VECTOR = 32;
    const unsigned int warp_id = thread_id   /  THREADS_PER_VECTOR;
    const unsigned int VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
    const unsigned int num_vectors = VECTORS_PER_BLOCK * gridDim.x; 
    const unsigned int lane = threadIdx.x & (THREADS_PER_VECTOR - 1);
    
    for(unsigned int row = warp_id + index1; row < index2; row += num_vectors)
    {
        int offset_Rhs = (row-index1) * N_LocalDOF;
        int offset_Sys = (row-index1) * (N_LocalDOF * N_LocalDOF);
        int ii = CellReorder[row];
        
    //                 TBaseCell *Cell = Coll->GetCell(ii);
        
    // //             #ifdef _MPI
    // //                 if(Cell->IsHaloCell()){
    // //             //       cout << "this should" << endl;
    // //                 continue;
    // //             //       cout << "this shouldnt" << endl;
    // //                 }   
    // //             #endif
    //             //    OutPut(i << downwind[i] << endl);
    //             #ifdef __2D__
    //                 TFE2D *UEle = TFEDatabase2D::GetFE2D(USpace->GetFE2D(ii, Cell));
    //                 TFE2D *PEle = TFEDatabase2D::GetFE2D(PSpace->GetFE2D(ii, Cell));
    //             #endif
    //             #ifdef __3D__
    //                 TFE3D *UEle = TFEDatabase3D::GetFE3D(USpace->GetFE3D(ii, Cell));
    //                 TFE3D *PEle = TFEDatabase3D::GetFE3D(PSpace->GetFE3D(ii, Cell));
    //             #endif

    //                 // // get local number of dof
    //                 // int N_U = UEle->GetN_DOF();
    //                 // int N_P = PEle->GetN_DOF();
    //                 // int N_LocalDOF = GEO_DIM*N_U+N_P;

        int *UDOFs = UGlobalNumbers+UBeginIndex[ii];
        int *PDOFs = PGlobalNumbers+PBeginIndex[ii];
        
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

        #ifdef __3D__
        int j1 = 2*N_U;
    #endif
        for(int j=lane;j<N_U;j = j + THREADS_PER_VECTOR)
        {
        int l = UDOFs[j];
        u1[l] += damp*Rhs[offset_Rhs+j];
        u2[l] += damp*Rhs[offset_Rhs+j+N_U];
    #ifdef __3D__
        u3[l] += damp*Rhs[offset_Rhs+j+j1];
    #endif  
        }

        j1 = GEO_DIM*N_U;
        for(int j=lane;j<N_P;j = j + THREADS_PER_VECTOR)
        {
        int l = PDOFs[j];
        p[l] += damp*Rhs[offset_Rhs+j+j1];
        }
    }

}

void verify(double* o, double* n, int num,int rank){
 
    for(int i=0; i<num; i++){
        if(o[i] == n[i] && rank ==1) {
            cout<<"neha: matching!!"<<o[i]<<" "<<n[i]<<" i "<<i<<" rank "<<rank<<endl;
            continue;
        }
        
        if(o[i] != n[i] && rank ==1) {
            cout<<"neha: not matching!! "<<o[i]<<" "<<n[i]<<" i "<<i<<" rank "<<rank<<endl;
            
        }
    }
    
}

#ifdef _MPI

#ifdef _HYBRID

#ifdef _CUDA


void TNSE_MGLevel4::CellVanka_CPU_GPU(double *u1, double *rhs1, double *aux,
        int N_Parameters, double *Parameters, int smoother, int smooth, cudaStream_t *stream, int* d_ARowPtr, int*  d_AKCol,
        double* d_A11Entries, double* d_A12Entries, double* d_A13Entries,
        double* d_A21Entries, double* d_A22Entries, double* d_A23Entries,
        double* d_A31Entries, double* d_A32Entries, double* d_A33Entries,
        int* d_BTRowPtr, int*  d_BTKCol,
        double* d_B1TEntries, double* d_B2TEntries, double* d_B3TEntries,
        int* d_BRowPtr, int*  d_BKCol,
        double* d_B1Entries, double* d_B2Entries, double* d_B3Entries)
        {
                // cout<<"cellvanka_CG"<<endl;
                double *u2, *u3, *p, *rhs2, *rhs3, *rhsp;
                int N_Cells;
                double *uold, *pold;
                TCollection *Coll;
                int *UGlobalNumbers, *UBeginIndex, *UDOFs;
                int *PGlobalNumbers, *PBeginIndex, *PDOFs;
                // int N_LocalDOF;
                int ActiveBound;
                double damp = TDatabase::ParamDB->SC_SMOOTH_DAMP_FACTOR_COARSE_SADDLE;
                
                TBaseCell *Cell;
                #ifdef __2D__
              const int RhsDim =  3*MaxN_BaseFunctions2D;
              TFE2D *UEle, *PEle;
              TSquareMatrix2D *sqmatrix[1];
            #endif
            #ifdef __3D__
              const int RhsDim =  4*MaxN_BaseFunctions3D;
              TFE3D *UEle, *PEle;
              TSquareMatrix3D *sqmatrix[1];
            #endif
                TItMethod *itmethod = NULL;
                int LargestDirectSolve = TDatabase::ParamDB->SC_LARGEST_DIRECT_SOLVE;
              MatVecProc *MatVect=MatVectFull;
              DefectProc *Defect=DefectFull;
              TSquareMatrix **matrix= (TSquareMatrix **)sqmatrix;
                
                int j1,l,j;
                
                Coll = USpace->GetCollection();
                    
                bool flag=false;
                
                // int nStreams = 2;
                // cudaStream_t stream[nStreams];
                
                // for (int i = 0; i < nStreams; ++i)
                //     CUDA_CHECK( cudaStreamCreate(&stream[i]) );

                #ifdef _MPI
      int rank;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        if(rank==0){
            cout<<"CellVanka_CPU_GPU"<<endl;
        }
#endif
                
                int nz_A = A11->GetN_Entries();
                int n_A = A11->GetN_Rows(); 
                
                int nz_B = B1->GetN_Entries();
                int n_B = B1->GetN_Rows(); 
                
                int nz_BT = B1T->GetN_Entries();
                int n_BT = B1T->GetN_Rows(); 
            
                
                Coll = USpace->GetCollection();
                N_Cells = Coll->GetN_Cells();
                
                
                Coll = PSpace->GetCollection();
                int N_PCells = Coll->GetN_Cells();
               
                
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
                    N_Dirichlet*sizeof(double));
                memcpy(u2+HangingNodeBound, rhs2+HangingNodeBound,
                    N_Dirichlet*sizeof(double));
                #ifdef __3D__
                memcpy(u3+HangingNodeBound, rhs3+HangingNodeBound,
                    N_Dirichlet*sizeof(double));
                #endif
            
                SetHangingNodes(u1);
            
                // old values
                uold = aux;
                pold  = uold+GEO_DIM*N_UDOF;
            
                // save current solution on 'old' vectors
                memcpy(uold, u1, N_DOF*SizeOfDouble);
                
                // bool *haloCell = new bool[N_Cells];
                
                Coll = USpace->GetCollection();
                
                // for(int i=0;i<N_CIntCell;i++)
                // {
                //     int temp = (ptrCellColors[i+1] - ptrCellColors[i]);
                    
                //     if(maxCellsPerColor< temp){
                            
                //             maxCellsPerColor = temp;
                            
                //     }
                    
                //     for(int jj=ptrCellColors[i];jj<ptrCellColors[i+1];jj++)
                //     {
                //         int ii = CellReorder[jj];
                //         Cell = Coll->GetCell(ii);
                        
                //         #ifdef _MPI
                //             if(Cell->IsHaloCell()){
                //                 haloCell[jj]=true;
                //             }
                //             else{
                //                 haloCell[jj]=false;
                                
                //                 if(!flag){
                //             #ifdef __2D__
                //                 UEle = TFEDatabase2D::GetFE2D(USpace->GetFE2D(ii, Cell));
                //                 PEle = TFEDatabase2D::GetFE2D(PSpace->GetFE2D(ii, Cell));
                //             #endif
                //             #ifdef __3D__
                //                 UEle = TFEDatabase3D::GetFE3D(USpace->GetFE3D(ii, Cell));
                //                 PEle = TFEDatabase3D::GetFE3D(PSpace->GetFE3D(ii, Cell));
                //             #endif
            
                //                 // get local number of dof
                //                 N_U = UEle->GetN_DOF();
                //                 N_P = PEle->GetN_DOF();
                //                 N_LocalDOF = GEO_DIM*N_U+N_P;
                //                 flag=true;
                //                 }
                //             }
                //         #endif
                //     }
                    
                // }
                
                cudaEvent_t start, stop;
                float time;
                cudaEventCreate(&start);
                cudaEventCreate(&stop); 
                
                cudaEventRecord(start, 0);
              
                 //     transfer solution and rhs
                double* d_u1 = NULL;
                CUDA_CHECK(cudaMalloc((void**)&d_u1, N_DOF * sizeof(double)));
                
                double* d_rhs1 = NULL;
                CUDA_CHECK(cudaMalloc((void**)&d_rhs1, N_DOF * sizeof(double)));
                
                int* d_CellReorder = NULL;
                CUDA_CHECK(cudaMalloc((void**)&d_CellReorder, N_Cells * sizeof(int)));
                
                // Copy to GPU memory
                CUDA_CHECK(cudaMemcpyAsync(d_u1, u1,  N_DOF* sizeof(double), cudaMemcpyHostToDevice,stream[0]));
                
                CUDA_CHECK(cudaMemcpyAsync(d_rhs1, rhs1, N_DOF * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
                
                CUDA_CHECK(cudaMemcpyAsync(d_CellReorder, CellReorder, N_Cells * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
                
                
            
                
            //     for(int i=0; i<N_Cells; i++){
            //         haloCell[i]=FALSE;
            //     }
                
                // bool* d_haloCell = NULL;
                // CUDA_CHECK(cudaMalloc((void**)&d_haloCell, N_Cells * sizeof(bool)));
                
                // CUDA_CHECK(cudaMemcpyAsync(d_haloCell, haloCell, N_Cells * sizeof(bool), cudaMemcpyHostToDevice,stream[0]));
                    
                // cout<<"neha:maxCellsPerColor:"<<maxCellsPerColor<<endl;    
                
                double *System = new double[maxCellsPerColor * (N_LocalDOF * N_LocalDOF)];
                double *Rhs = new double[maxCellsPerColor * N_LocalDOF];
                // double *temp_Rhs = new double[maxCellsPerColor * N_LocalDOF];
                
                
            //     double CSystem[maxCellsPerColor * (N_LocalDOF * N_LocalDOF)];
            //     double CRhs[maxCellsPerColor * N_LocalDOF];
                
                double* d_System = NULL;
                // CUDA_CHECK(cudaMalloc((void**)&d_System, maxCellsPerColor * (N_LocalDOF * N_LocalDOF) * sizeof(double)));
                
                double* d_Rhs = NULL;
                CUDA_CHECK(cudaMalloc((void**)&d_Rhs, maxCellsPerColor * N_LocalDOF * sizeof(double)));
                
                // CUDA_CHECK(cudaMemset(d_System, 0, maxCellsPerColor * (N_LocalDOF * N_LocalDOF) * sizeof(double) ));
                
                
            //     cout<<"neha:N_LocalDOF:"<<N_LocalDOF<<endl;
                
            //     transfer A matrix
            //     int* d_ARowPtr = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_ARowPtr, (n_A+1) * sizeof(int)));
                
            //     int* d_AKCol = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_AKCol, nz_A * sizeof(int)));
                
            //     double* d_A11Entries = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_A11Entries, nz_A * sizeof(double)));
                
            //     double* d_A12Entries = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_A12Entries, nz_A * sizeof(double)));
                
            //     double* d_A13Entries = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_A13Entries, nz_A * sizeof(double)));
                
            //     double* d_A21Entries = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_A21Entries, nz_A * sizeof(double)));
                
            //     double* d_A22Entries = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_A22Entries, nz_A * sizeof(double)));
                
            //     double* d_A23Entries = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_A23Entries, nz_A * sizeof(double)));
                
            //     double* d_A31Entries = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_A31Entries, nz_A * sizeof(double)));
                
            //     double* d_A32Entries = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_A32Entries, nz_A * sizeof(double)));
                
            //     double* d_A33Entries = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_A33Entries, nz_A * sizeof(double)));
                
            //         // Copy to GPU memory
            //     CUDA_CHECK(cudaMemcpyAsync(d_ARowPtr, ARowPtr, (n_A+1) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
                
            //     CUDA_CHECK(cudaMemcpyAsync(d_AKCol, AKCol, nz_A * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
                
            //     CUDA_CHECK(cudaMemcpyAsync(d_A11Entries, A11Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
                
            //     CUDA_CHECK(cudaMemcpyAsync(d_A12Entries, A12Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
                
            //     CUDA_CHECK(cudaMemcpyAsync(d_A13Entries, A13Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
                
            //     CUDA_CHECK(cudaMemcpyAsync(d_A21Entries, A21Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
                
            //     CUDA_CHECK(cudaMemcpyAsync(d_A22Entries, A22Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
                
            //     CUDA_CHECK(cudaMemcpyAsync(d_A23Entries, A23Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
                
            //     CUDA_CHECK(cudaMemcpyAsync(d_A31Entries, A31Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
                
            //     CUDA_CHECK(cudaMemcpyAsync(d_A32Entries, A32Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
                
            //     CUDA_CHECK(cudaMemcpyAsync(d_A33Entries, A33Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
                
                
            // //     cout<<"neha:color"<<N_CIntCell<<endl;
                
            // //     transfer BT matrix
            //     int* d_BTRowPtr = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_BTRowPtr, (n_BT+1) * sizeof(int)));
                
            //     int* d_BTKCol = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_BTKCol, nz_BT * sizeof(int)));
                
            //     double* d_B1TEntries = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_B1TEntries, nz_BT * sizeof(double)));
                
            //     double* d_B2TEntries = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_B2TEntries, nz_BT * sizeof(double)));
                
            //     double* d_B3TEntries = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_B3TEntries, nz_BT * sizeof(double)));
                
            //     // Copy to GPU memory
            //     CUDA_CHECK(cudaMemcpyAsync(d_BTRowPtr, BTRowPtr, (n_BT+1) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
                
            //     CUDA_CHECK(cudaMemcpyAsync(d_BTKCol, BTKCol, nz_BT * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
                
            //     CUDA_CHECK(cudaMemcpyAsync(d_B1TEntries, B1TEntries, nz_BT * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
                
            //     CUDA_CHECK(cudaMemcpyAsync(d_B2TEntries, B2TEntries, nz_BT * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
                
            //     CUDA_CHECK(cudaMemcpyAsync(d_B3TEntries, B3TEntries, nz_BT * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
                
            // //     transfer B matrix
            //     int* d_BRowPtr = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_BRowPtr, (n_B+1) * sizeof(int)));
                
            //     int* d_BKCol = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_BKCol, nz_B * sizeof(int)));
                
            //     double* d_B1Entries = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_B1Entries, nz_B * sizeof(double)));
                
            //     double* d_B2Entries = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_B2Entries, nz_B * sizeof(double)));
                
            //     double* d_B3Entries = NULL;
            //     CUDA_CHECK(cudaMalloc((void**)&d_B3Entries, nz_B * sizeof(double)));
                
            //     // Copy to GPU memory
            //     CUDA_CHECK(cudaMemcpyAsync(d_BRowPtr, BRowPtr, (n_B+1) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
                
            //     CUDA_CHECK(cudaMemcpyAsync(d_BKCol, BKCol, nz_B * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
                
            //     CUDA_CHECK(cudaMemcpyAsync(d_B1Entries, B1Entries, nz_B * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
                
            //     CUDA_CHECK(cudaMemcpyAsync(d_B2Entries, B2Entries, nz_B * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
                
            //     CUDA_CHECK(cudaMemcpyAsync(d_B3Entries, B3Entries, nz_B * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
                
            
            
                UGlobalNumbers = USpace->GetGlobalNumbers();
                UBeginIndex = USpace->GetBeginIndex();
                ActiveBound = USpace->GetActiveBound();
            
                PGlobalNumbers = PSpace->GetGlobalNumbers();
                PBeginIndex = PSpace->GetBeginIndex();
                
                int* d_UGlobalNumbers = NULL;
                CUDA_CHECK(cudaMalloc((void**)&d_UGlobalNumbers, N_U*N_Cells * sizeof(int)));
                
                int* d_UBeginIndex = NULL;
                CUDA_CHECK(cudaMalloc((void**)&d_UBeginIndex, N_Cells * sizeof(int)));
                
                int* d_PGlobalNumbers = NULL;
                CUDA_CHECK(cudaMalloc((void**)&d_PGlobalNumbers, (N_P*N_Cells) * sizeof(int)));
                
                int* d_PBeginIndex = NULL;
                CUDA_CHECK(cudaMalloc((void**)&d_PBeginIndex, N_PCells * sizeof(int)));
                
                
                // Copy to GPU memory
                CUDA_CHECK(cudaMemcpyAsync(d_UGlobalNumbers, UGlobalNumbers, N_U*N_Cells * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
                
                CUDA_CHECK(cudaMemcpyAsync(d_UBeginIndex, UBeginIndex, N_Cells * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
                
                CUDA_CHECK(cudaMemcpyAsync(d_PGlobalNumbers, PGlobalNumbers, (N_P*N_Cells) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
                
                CUDA_CHECK(cudaMemcpyAsync(d_PBeginIndex, PBeginIndex, N_Cells * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
                
                CUDA_CHECK(cudaStreamSynchronize(stream[0]));
                
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                
                float milliseconds;
                
                cudaEventElapsedTime(&milliseconds, start, stop);
                
                data_transfer_time += milliseconds;
                
                int offset_Rhs, offset_Sys;
                
                int thread_blocks;
                int jj, itr;
                int count=0;
                
            
                        double value, value1, value2, value3;
                        double value11,value12,value13,value21,value22;
                        double value23,value31,value32,value33;
                        int k1, k2, k3, j2, j3, k,m, UDOF, PDOF, begin, end,end_itr, ii;
                        
                        if(smooth == -1)
                        end_itr = TDatabase::ParamDB->SC_PRE_SMOOTH_SADDLE;
                    else if(smooth == 0)
                        end_itr = TDatabase::ParamDB->SC_COARSE_MAXIT_SADDLE;
                    else
                        end_itr = TDatabase::ParamDB->SC_POST_SMOOTH_SADDLE;
            //             double GSystem[maxCellsPerColor * (N_LocalDOF * N_LocalDOF)];
            //             double GRhs[maxCellsPerColor * N_LocalDOF];
                    
                if (N_LocalDOF > LargestDirectSolve)
                {
                  // size of local system has changed
                  if (N_LocalDOF != TDatabase::ParamDB->INTERNAL_LOCAL_DOF)
                  {
                    // itmethod exists already
                    if ( TDatabase::ParamDB->INTERNAL_LOCAL_DOF >0)
                      delete itmethod;
                    // allocate new itmethod
                    itmethod = new TFgmresIte(MatVect, Defect, NULL, 0, N_LocalDOF, 1);
                    TDatabase::ParamDB->INTERNAL_LOCAL_DOF = N_LocalDOF;
                  }
                }
            
                    
                int numThreads = TDatabase::ParamDB->OMPNUMTHREADS;
                 
                 omp_set_num_threads(numThreads);
                
                for(itr=0;itr<end_itr;itr++)
                {
                    for(int i=0;i<N_CIntCell;i++)
                    {
            //             if(rank==1)
            //             cout<<"color:"<<i<<endl;
                        thread_blocks = (ptrCellColors[i+1]-ptrCellColors[i])*32/THREADS_PER_BLOCK + 1;
                        
                        // CUDA_CHECK(cudaMemset(d_System, 0, maxCellsPerColor * (N_LocalDOF * N_LocalDOF) * sizeof(double) ));
                        memset(System, 0, maxCellsPerColor*SizeOfDouble*N_LocalDOF*N_LocalDOF);
            //             cudaEventRecord(start, 0);
                        CellVanka_assembleU<<<thread_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(d_ARowPtr, d_AKCol,
                                            d_A11Entries, d_A12Entries, d_A13Entries, d_A21Entries, d_A22Entries, d_A23Entries, d_A31Entries,  d_A32Entries, d_A33Entries, 
                                            d_BTRowPtr, d_BTKCol, d_B1TEntries, d_B2TEntries, d_B3TEntries, d_BRowPtr, d_BKCol, d_B1Entries, d_B2Entries,  d_B3Entries, 
                                            d_u1, d_rhs1, d_CellReorder, N_U, N_P, N_LocalDOF, N_UDOF,
                                            d_UGlobalNumbers, d_UBeginIndex, d_PGlobalNumbers, d_PBeginIndex, ActiveBound, d_System, d_Rhs, ptrCellColors[i], ptrCellColors[i+1]);
                        
                        CellVanka_assembleP<<<thread_blocks, THREADS_PER_BLOCK, 0, stream[1]>>>(d_ARowPtr, d_AKCol,
                                            d_A11Entries, d_A12Entries, d_A13Entries, d_A21Entries, d_A22Entries, d_A23Entries, d_A31Entries,  d_A32Entries, d_A33Entries, 
                                            d_BTRowPtr, d_BTKCol, d_B1TEntries, d_B2TEntries, d_B3TEntries, d_BRowPtr, d_BKCol, d_B1Entries, d_B2Entries,  d_B3Entries, 
                                            d_u1, d_rhs1, d_CellReorder, N_U, N_P, N_LocalDOF, N_UDOF,
                                            d_UGlobalNumbers, d_UBeginIndex, d_PGlobalNumbers, d_PBeginIndex, ActiveBound, d_System, d_Rhs, ptrCellColors[i], ptrCellColors[i+1]);
                        
            //             cudaEventRecord(stop, 0);
            //             cudaEventSynchronize(stop);
            //             cudaEventElapsedTime(&milliseconds, start, stop);
            //             
            //             kernel_time += milliseconds;
                        
                        
                         
            //              if(rank==1)
            //              verify(temp_Rhs,Rhs,maxCellsPerColor * N_LocalDOF,1);
            //           cout<<"i:"<<ptrCellColors[i]<<endl;
            //           cout<<"i+1:"<<ptrCellColors[i+1]<<endl;
                        
                        
                        PUSH_RANGE("assembly",1)
                        memset(System, 0, maxCellsPerColor*SizeOfDouble*N_LocalDOF*N_LocalDOF);
                        #pragma omp parallel for default(shared) schedule(dynamic)
                        for(int jj=ptrCellColors[i];jj<ptrCellColors[i+1];jj++)
                        {
            
                            double value, value1, value2, value3;
                            double value11,value12,value13,value21,value22;
                            double value23,value31,value32,value33;
            
                            int k1, k2, k3, j1, j2, j3, k,m, UDOF, PDOF, begin, end, l;
                            
                            int offset_Rhs = (jj-ptrCellColors[i]) * N_LocalDOF;
                            int offset_Sys = (jj-ptrCellColors[i]) * (N_LocalDOF * N_LocalDOF);
                            
                            int ii = CellReorder[jj];
                            
            //                 cout<<"cell:"<<ii<<endl;
            //     for(ii=0;ii<N_Cells;ii++)
            //   {
                //ii = downwind[i];
                
                            TBaseCell *Cell = Coll->GetCell(ii);
                            
                        #ifdef _MPI
                            if(Cell->IsHaloCell()){
                        //       cout << "this should" << endl;
                            continue;
                        //       cout << "this shouldnt" << endl;
                            }   
                        #endif
                        //    OutPut(i << downwind[i] << endl);
                        // #ifdef __2D__
                        //     UEle = TFEDatabase2D::GetFE2D(USpace->GetFE2D(ii, Cell));
                        //     PEle = TFEDatabase2D::GetFE2D(PSpace->GetFE2D(ii, Cell));
                        // #endif
                        // #ifdef __3D__
                        //     UEle = TFEDatabase3D::GetFE3D(USpace->GetFE3D(ii, Cell));
                        //     PEle = TFEDatabase3D::GetFE3D(PSpace->GetFE3D(ii, Cell));
                        // #endif
            
                        //     // get local number of dof
                        //     N_U = UEle->GetN_DOF();
                        //     N_P = PEle->GetN_DOF();
                        //     N_LocalDOF = GEO_DIM*N_U+N_P;
                            
            //                 if(N_LocalDOF != 85){
            //                     cout<<"neha:not same"<<endl;
            //                     }
                            
                        //     if(rank==0){
                        //         cout<<"N_U:"<<N_U<<endl;
                        //         cout<<"N_P:"<<N_P<<endl;
                        //         cout<<"N_LocalDOF:"<<N_LocalDOF<<endl;
                        //     }
            
                            // reset local systems
                            /*if (N_LocalDOF > RhsDim)
                            {
                            OutPut(
                            "TNSE_MGLevel4::CellVanka - Not enough memory in array Rhs!!!"
                            << endl << "available " << RhsDim << " needed " <<
                            N_LocalDOF << endl);
                            exit(4711);
                            }*/
                            // reset local systems
                            
                            if (N_LocalDOF > LargestDirectSolve)
                            {
                            // size of local system has changed
                            if (N_LocalDOF != TDatabase::ParamDB->INTERNAL_LOCAL_DOF)
                            {
                                // itmethod exists already
                                if ( TDatabase::ParamDB->INTERNAL_LOCAL_DOF >0)
                                delete itmethod;
                                // allocate new itmethod
                                itmethod = new TFgmresIte(MatVect, Defect, NULL, 0, N_LocalDOF, 1);
                                TDatabase::ParamDB->INTERNAL_LOCAL_DOF = N_LocalDOF;
                            }
                            }
            
                            int *UDOFs = UGlobalNumbers+UBeginIndex[ii];
                            int *PDOFs = PGlobalNumbers+PBeginIndex[ii];
            
                            // fill local matrix
                            for(int j=0;j<N_U;j++)
                            {
                            j1 = j;
                            j2 = j+N_U;
                        #ifdef __3D__
                            j3 = j2+N_U;
                        #endif
                            UDOF = UDOFs[j];
            
                            // A block
                            begin = ARowPtr[UDOF];
                            end = ARowPtr[UDOF+1];
                            
            //                 cout<<"A:"<<(end-begin)<<endl;
            
            
            
                            for(int k=begin;k<end;k++)
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
                    if (UDOF>=ActiveBound) // Dirichlet node
                      value12 = value21 = 0;
            
            #endif
            #ifdef __3D__
                    if (UDOF>=ActiveBound) // Dirichlet node
                      value12 = value13 = value21 = value23 = value31 = value32 = 0;
            
            #endif
            
                                for(int m=0;m<N_U;m++)
                                if(UDOFs[m]==l)
                                {
                                    // column belongs to local system
                                    k1 = m*N_LocalDOF;
                                    k2 = (m+N_U)*N_LocalDOF;
                                    System[offset_Sys+k1+j1] = value11;
                                    System[offset_Sys+k2+j1] = value12;
                                    System[offset_Sys+k1+j2] = value21;
                                    System[offset_Sys+k2+j2] = value22;
                        #ifdef __3D__
                                    k3 = (m+2*N_U)*N_LocalDOF;
                                    System[offset_Sys+k3+j1] = value13;
                                    System[offset_Sys+k3+j2] = value23;
                                    System[offset_Sys+k1+j3] = value31;
                                    System[offset_Sys+k2+j3] = value32;
                                    System[offset_Sys+k3+j3] = value33;
                        #endif
                                    break;
                                }
                            } // endfor k
                            
                            if(UDOF<ActiveBound)  // active dof
                            {
                                // transpose(B) block for non-Dirichlet nodes
                                begin = BTRowPtr[UDOF];
                                end = BTRowPtr[UDOF+1];
            
            //                     cout<<"BT:"<<(end-begin)<<endl;
                                for(int k=begin;k<end;k++)
                                {
                                l = BTKCol[k];
                                value1 = B1TEntries[k];
                                value2 = B2TEntries[k];
                        #ifdef __3D__
                                value3 = B3TEntries[k];
                        #endif
                                value = p[l];
            
            
                                for(int m=0;m<N_P;m++)
                                    if(PDOFs[m]==l)
                                    {
                                    // column belongs to local system
                                    k1 = (m+GEO_DIM*N_U)*N_LocalDOF;
                                    System[offset_Sys+k1+j1] = value1;
                                    System[offset_Sys+k1+j2] = value2;
                        #ifdef __3D__
                                    System[offset_Sys+k1+j3] = value3;
                        #endif
                                    break;
                                    }
            
                                } // endfor k
                            } // endif UDOF<ActiveBound
                            
                            } // endfor j
                            
                                            // fill local matrix
                            for(int j=0;j<N_P;j++)
                            {
                            j1 = j+GEO_DIM*N_U;
                            PDOF = PDOFs[j];
                            begin = BRowPtr[PDOF];
                            end = BRowPtr[PDOF+1];
            
            
            //                 cout<<"B:"<<(end-begin)<<endl;
                            for(int k=begin;k<end;k++)
                            {
                                l=BKCol[k];
                                value1 = B1Entries[k];
                                value2 = B2Entries[k];
                        #ifdef __3D__
                                value3 = B3Entries[k];
                        #endif
            
                                for(m=0;m<N_U;m++)
                                if(UDOFs[m]==l)
                                {
                                    // column belongs to local system
                                    k1 = m;
                                    k2 = m+N_U;
                        #ifdef __3D__
                                    k3 = k2 + N_U;
                        #endif
                                    System[offset_Sys+k1*N_LocalDOF+j1] = value1;
                                    System[offset_Sys+k2*N_LocalDOF+j1] = value2;
                        #ifdef __3D__
                                    System[offset_Sys+k3*N_LocalDOF+j1] = value3;
                        #endif
                                    break;
                                }
                            } // endfor k
                            } // endfor j
            
            
                        }
                        
                        POP_RANGE
                        
                        CUDA_CHECK(cudaStreamSynchronize(stream[0]));
                        CUDA_CHECK(cudaStreamSynchronize(stream[1]));
            //             CUDA_CHECK(cudaStreamSynchronize(stream[2]));
                        CUDA_CHECK(cudaMemcpyAsync(Rhs, d_Rhs, maxCellsPerColor * N_LocalDOF * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
            //             CUDA_CHECK(cudaMemcpyAsync(System, d_System, maxCellsPerColor * (N_LocalDOF * N_LocalDOF) * sizeof(double), cudaMemcpyDeviceToHost,stream[1]));
                        CUDA_CHECK(cudaStreamSynchronize(stream[0]));
            //             CUDA_CHECK(cudaStreamSynchronize(stream[1]));
                    
            double t1,t2;
            #ifdef _MPI
              t1 = MPI_Wtime();
            #else
              t1 = GetTime();
            #endif
                PUSH_RANGE("solve", 2)
                #pragma omp parallel for default(shared) schedule(dynamic)
                for(int jj=ptrCellColors[i];jj<ptrCellColors[i+1];jj++)
                        {
            //                 cout<<"thrds:"<<omp_get_num_threads()<<endl;
            
                            int offset_Rhs = (jj-ptrCellColors[i]) * N_LocalDOF;
                            int offset_Sys = (jj-ptrCellColors[i]) * (N_LocalDOF * N_LocalDOF);
                            
            
                            int ii = CellReorder[jj];
                            
                            TBaseCell *Cell = Coll->GetCell(ii);
                            
            //             #ifdef _MPI
            //                 if(Cell->IsHaloCell()){
            //             //       cout << "this should" << endl;
            //                 continue;
            //             //       cout << "this shouldnt" << endl;
            //                 }   
            //             #endif
                        //    OutPut(i << downwind[i] << endl);
                        #ifdef __2D__
                            TFE2D *UEle = TFEDatabase2D::GetFE2D(USpace->GetFE2D(ii, Cell));
                            TFE2D *PEle = TFEDatabase2D::GetFE2D(PSpace->GetFE2D(ii, Cell));
                        #endif
                        #ifdef __3D__
                            TFE3D *UEle = TFEDatabase3D::GetFE3D(USpace->GetFE3D(ii, Cell));
                            TFE3D *PEle = TFEDatabase3D::GetFE3D(PSpace->GetFE3D(ii, Cell));
                        #endif
            
                            // get local number of dof
                            int N_U = UEle->GetN_DOF();
                            int N_P = PEle->GetN_DOF();
                            int N_LocalDOF = GEO_DIM*N_U+N_P;
            
                            int *UDOFs = UGlobalNumbers+UBeginIndex[ii];
                            int *PDOFs = PGlobalNumbers+PBeginIndex[ii];
              
                            // solve local system
                            if (smoother==11  && !C) // no diagonal Vanka for matrix C
                            {
                            // diagonal Vanka
                        #ifdef __2D__
                            SolveDiagonalVanka2D(System+offset_Sys, Rhs+offset_Rhs, N_U, N_P, N_LocalDOF);
                        #endif
                        #ifdef __3D__
                            SolveDiagonalVanka3D(System+offset_Sys, Rhs+offset_Rhs, N_U, N_P, N_LocalDOF);
                        #endif
                            }
                            else
                            {
                            // full Vanka
                        //       cout << "full vanka dof :: " << N_LocalDOF << endl;
                            
            //                 if (N_LocalDOF > LargestDirectSolve)
            //                 {
            //                     memset(sol,0,N_LocalDOF*SizeOfDouble);
            //                     verbose =  TDatabase::ParamDB->SC_VERBOSE;
            //                     TDatabase::ParamDB->SC_VERBOSE = -1;
            //                     itmethod->Iterate(matrix,NULL,sol,Rhs);
            //                     TDatabase::ParamDB->SC_VERBOSE = verbose;
            //                     memcpy(Rhs, sol, N_LocalDOF*SizeOfDouble);
            //                 }
            //                 else
            //                 {
                                SolveLinearSystemLapack(System+offset_Sys, Rhs+offset_Rhs, N_LocalDOF, N_LocalDOF);
                                
            //                     CUDA_CHECK(cudaMemcpyAsync(d_System, System, maxCellsPerColor * N_LocalDOF * N_LocalDOF *  sizeof(double), cudaMemcpyHostToDevice, stream[0]));
                                
            //                     CUDA_CHECK(cudaStreamSynchronize(stream[0]));
                                
            //                     SolveLinearSystemCuSolver(d_System+offset_Sys, d_Rhs+offset_Rhs, Rhs+offset_Rhs, N_LocalDOF, N_LocalDOF);
            //                 }
                            }
                            
            
                        #ifdef __3D__
                            int j1 = 2*N_U;
                        #endif
                            for(int j=0;j<N_U;j++)
                            {
                            int l = UDOFs[j];
                            u1[l] += damp*Rhs[offset_Rhs+j];
                            u2[l] += damp*Rhs[offset_Rhs+j+N_U];
                        #ifdef __3D__
                            u3[l] += damp*Rhs[offset_Rhs+j+j1];
                        #endif  
                            }
            
                            j1 = GEO_DIM*N_U;
                            for(int j=0;j<N_P;j++)
                            {
                            int l = PDOFs[j];
                            p[l] += damp*Rhs[offset_Rhs+j+j1];
                            }
            //   } // endfor loop over cells
            
                    
                }
                POP_RANGE
                
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
              if (fabs(1-alpha)>1e-3)
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
            
              SetHangingNodes(u1);
              
              #ifdef _MPI      
               ParCommU->CommUpdate(u1);   
               ParCommP->CommUpdate(p);
            #endif
               
                CUDA_CHECK(cudaMemcpyAsync(d_u1, u1,  N_DOF* sizeof(double), cudaMemcpyHostToDevice,stream[0]));
                CUDA_CHECK(cudaStreamSynchronize(stream[0]));
                
                // save current solution on 'old' vectors
                memcpy(uold, u1, N_DOF*SizeOfDouble);
                }
              
                delete []System;
                delete []Rhs;
                
                // for (int i = 0; i < nStreams; ++i)
                //     CUDA_CHECK( cudaStreamDestroy(stream[i]) );
              
                // // Free GPU memory
                // CUDA_CHECK(cudaFree(d_ARowPtr));
                // CUDA_CHECK(cudaFree(d_AKCol));
                // CUDA_CHECK(cudaFree(d_A11Entries));
                // CUDA_CHECK(cudaFree(d_A12Entries));
                // CUDA_CHECK(cudaFree(d_A13Entries));
                // CUDA_CHECK(cudaFree(d_A21Entries));
                // CUDA_CHECK(cudaFree(d_A22Entries));
                // CUDA_CHECK(cudaFree(d_A23Entries));
                // CUDA_CHECK(cudaFree(d_A31Entries));
                // CUDA_CHECK(cudaFree(d_A32Entries));
                // CUDA_CHECK(cudaFree(d_A33Entries));
                
                // // Free GPU memory
                // CUDA_CHECK(cudaFree(d_BTRowPtr));
                // CUDA_CHECK(cudaFree(d_BTKCol));
                // CUDA_CHECK(cudaFree(d_B1TEntries));
                // CUDA_CHECK(cudaFree(d_B2TEntries));
                // CUDA_CHECK(cudaFree(d_B3TEntries));
                      
                // // Free GPU memory
                // CUDA_CHECK(cudaFree(d_BRowPtr));
                // CUDA_CHECK(cudaFree(d_BKCol));
                // CUDA_CHECK(cudaFree(d_B1Entries));
                // CUDA_CHECK(cudaFree(d_B2Entries));
                // CUDA_CHECK(cudaFree(d_B3Entries));
                
                CUDA_CHECK(cudaFree(d_u1));
                CUDA_CHECK(cudaFree(d_rhs1));
                CUDA_CHECK(cudaFree(d_CellReorder));
                
                CUDA_CHECK(cudaFree(d_Rhs));
                // CUDA_CHECK(cudaFree(d_System));

                CUDA_CHECK(cudaFree(d_UGlobalNumbers));
                CUDA_CHECK(cudaFree(d_UBeginIndex));
                CUDA_CHECK(cudaFree(d_PGlobalNumbers));
                CUDA_CHECK(cudaFree(d_PBeginIndex));
            }

void TNSE_MGLevel4::CellVanka_GPU(double *u1, double *rhs1, double *aux,
        int N_Parameters, double *Parameters, int smoother, int smooth, cudaStream_t *stream, int* d_ARowPtr, int*  d_AKCol,
        double* d_A11Entries, double* d_A12Entries, double* d_A13Entries,
        double* d_A21Entries, double* d_A22Entries, double* d_A23Entries,
        double* d_A31Entries, double* d_A32Entries, double* d_A33Entries,
        int* d_BTRowPtr, int*  d_BTKCol,
        double* d_B1TEntries, double* d_B2TEntries, double* d_B3TEntries,
        int* d_BRowPtr, int*  d_BKCol,
        double* d_B1Entries, double* d_B2Entries, double* d_B3Entries)
 {
     
    double *u2, *u3, *p, *rhs2, *rhs3, *rhsp;
    int N_Cells;
    double *uold, *pold;
    TCollection *Coll;
    int *UGlobalNumbers, *UBeginIndex, *UDOFs;
    int *PGlobalNumbers, *PBeginIndex, *PDOFs;
    // int N_LocalDOF;
    int ActiveBound;
    double damp = TDatabase::ParamDB->SC_SMOOTH_DAMP_FACTOR_COARSE_SADDLE;
    
    TBaseCell *Cell;
    #ifdef __2D__
  const int RhsDim =  3*MaxN_BaseFunctions2D;
  TFE2D *UEle, *PEle;
  TSquareMatrix2D *sqmatrix[1];
#endif
#ifdef __3D__
  const int RhsDim =  4*MaxN_BaseFunctions3D;
  TFE3D *UEle, *PEle;
  TSquareMatrix3D *sqmatrix[1];
#endif
    TItMethod *itmethod = NULL;
    int LargestDirectSolve = TDatabase::ParamDB->SC_LARGEST_DIRECT_SOLVE;
  MatVecProc *MatVect=MatVectFull;
  DefectProc *Defect=DefectFull;
  TSquareMatrix **matrix= (TSquareMatrix **)sqmatrix;
    
    int j1,l,j;
    
    Coll = USpace->GetCollection();
        
    bool flag=false;
    
    int nStreams = 2;

    #ifdef _MPI
      int rank;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        if(rank==0){
            cout<<"CellVanka_GPU"<<endl;
        }
#endif


    // cudaStream_t stream[nStreams];
    
    // for (int i = 0; i < nStreams; ++i)
    //     CUDA_CHECK( cudaStreamCreate(&stream[i]) );
    
    int nz_A = A11->GetN_Entries();
    int n_A = A11->GetN_Rows(); 
    
    int nz_B = B1->GetN_Entries();
    int n_B = B1->GetN_Rows(); 
    
    int nz_BT = B1T->GetN_Entries();
    int n_BT = B1T->GetN_Rows(); 

    
    Coll = USpace->GetCollection();
    N_Cells = Coll->GetN_Cells();
    
    
    Coll = PSpace->GetCollection();
    int N_PCells = Coll->GetN_Cells();
   
    
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
        N_Dirichlet*sizeof(double));
    memcpy(u2+HangingNodeBound, rhs2+HangingNodeBound,
        N_Dirichlet*sizeof(double));
    #ifdef __3D__
    memcpy(u3+HangingNodeBound, rhs3+HangingNodeBound,
        N_Dirichlet*sizeof(double));
    #endif

    SetHangingNodes(u1);

    // old values
    uold = aux;
    pold  = uold+GEO_DIM*N_UDOF;

    // save current solution on 'old' vectors
    memcpy(uold, u1, N_DOF*SizeOfDouble);
    
    // bool *haloCell = new bool[N_Cells];
    
    // Coll = USpace->GetCollection();

    // Cell = Coll->GetCell(0);

    // #ifdef __2D__
    //     UEle = TFEDatabase2D::GetFE2D(USpace->GetFE2D(0, Cell));
    //     PEle = TFEDatabase2D::GetFE2D(PSpace->GetFE2D(0, Cell));
    // #endif
    // #ifdef __3D__
    //     UEle = TFEDatabase3D::GetFE3D(USpace->GetFE3D(0, Cell));
    //     PEle = TFEDatabase3D::GetFE3D(PSpace->GetFE3D(0, Cell));
    // #endif

    //     // get local number of dof
    //     N_U = UEle->GetN_DOF();
    //     N_P = PEle->GetN_DOF();
    //     N_LocalDOF = GEO_DIM*N_U+N_P;

    
    // for(int i=0;i<N_CIntCell;i++)
    // {
    //     // int temp = (ptrCellColors[i+1] - ptrCellColors[i]);
        
    //     // if(maxCellsPerColor< temp){
                
    //     //         maxCellsPerColor = temp;
                
    //     // }
        
    //     for(int jj=ptrCellColors[i];jj<ptrCellColors[i+1];jj++)
    //     {
    //         int ii = CellReorder[jj];
    //         Cell = Coll->GetCell(ii);
            
    //         #ifdef _MPI
    //             if(Cell->IsHaloCell()){
    //                 haloCell[jj]=true;
    //             }
    //             else{
    //                 haloCell[jj]=false;
                    
    //                 if(!flag){
    //             #ifdef __2D__
    //                 UEle = TFEDatabase2D::GetFE2D(USpace->GetFE2D(ii, Cell));
    //                 PEle = TFEDatabase2D::GetFE2D(PSpace->GetFE2D(ii, Cell));
    //             #endif
    //             #ifdef __3D__
    //                 UEle = TFEDatabase3D::GetFE3D(USpace->GetFE3D(ii, Cell));
    //                 PEle = TFEDatabase3D::GetFE3D(PSpace->GetFE3D(ii, Cell));
    //             #endif

    //                 // get local number of dof
    //                 N_U = UEle->GetN_DOF();
    //                 N_P = PEle->GetN_DOF();
    //                 N_LocalDOF = GEO_DIM*N_U+N_P;
    //                 flag=true;
    //                 }
    //             }
    //         #endif
    //     }
        
    // }
    
    // cout<<"maxCellsPerColor"<<maxCellsPerColor<<endl;
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop); 
    
    cudaEventRecord(start, 0);
  
     //     transfer solution and rhs
    double* d_u1 = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_u1, N_DOF * sizeof(double)));
    
    double* d_rhs1 = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_rhs1, N_DOF * sizeof(double)));
    
    int* d_CellReorder = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_CellReorder, N_Cells * sizeof(int)));
    
    // Copy to GPU memory
    CUDA_CHECK(cudaMemcpyAsync(d_u1, u1,  N_DOF* sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_rhs1, rhs1, N_DOF * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_CellReorder, CellReorder, N_Cells * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    

    
//     for(int i=0; i<N_Cells; i++){
//         haloCell[i]=FALSE;
//     }
    
    // bool* d_haloCell = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_haloCell, N_Cells * sizeof(bool)));
    


    
    // CUDA_CHECK(cudaMemcpyAsync(d_haloCell, haloCell, N_Cells * sizeof(bool), cudaMemcpyHostToDevice,stream[0]));
        
//     cout<<"neha:N_LocalDOF:"<<N_LocalDOF<<endl;    
    
    double *System = new double[maxCellsPerColor * (N_LocalDOF * N_LocalDOF)];
    double *Rhs = new double[maxCellsPerColor * N_LocalDOF];
    // double *temp_Rhs = new double[maxCellsPerColor * N_LocalDOF];
    
    
//     double CSystem[maxCellsPerColor * (N_LocalDOF * N_LocalDOF)];
//     double CRhs[maxCellsPerColor * N_LocalDOF];
    

    double* d_System = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_System, maxCellsPerColor * (N_LocalDOF * N_LocalDOF) * sizeof(double)));
    
    double* d_Rhs = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_Rhs, maxCellsPerColor * N_LocalDOF * sizeof(double)));
    
    CUDA_CHECK(cudaMemsetAsync(d_System, 0, maxCellsPerColor * (N_LocalDOF * N_LocalDOF) * sizeof(double), stream[0] ));
    
    
    // cout<<"neha:N_LocalDOF:"<<N_LocalDOF<<endl;
    
//     transfer A matrix
    // int* d_ARowPtr = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_ARowPtr, (n_A+1) * sizeof(int)));
    
    // int* d_AKCol = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_AKCol, nz_A * sizeof(int)));
    
    // double* d_A11Entries = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_A11Entries, nz_A * sizeof(double)));
    
    // double* d_A12Entries = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_A12Entries, nz_A * sizeof(double)));
    
    // double* d_A13Entries = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_A13Entries, nz_A * sizeof(double)));
    
    // double* d_A21Entries = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_A21Entries, nz_A * sizeof(double)));
    
    // double* d_A22Entries = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_A22Entries, nz_A * sizeof(double)));
    
    // double* d_A23Entries = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_A23Entries, nz_A * sizeof(double)));
    
    // double* d_A31Entries = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_A31Entries, nz_A * sizeof(double)));
    
    // double* d_A32Entries = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_A32Entries, nz_A * sizeof(double)));
    
    // double* d_A33Entries = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_A33Entries, nz_A * sizeof(double)));
    
    //     // Copy to GPU memory
    // CUDA_CHECK(cudaMemcpyAsync(d_ARowPtr, ARowPtr, (n_A+1) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_AKCol, AKCol, nz_A * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_A11Entries, A11Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_A12Entries, A12Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_A13Entries, A13Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_A21Entries, A21Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_A22Entries, A22Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_A23Entries, A23Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_A31Entries, A31Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_A32Entries, A32Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_A33Entries, A33Entries, nz_A * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    
//     cout<<"neha:color"<<N_CIntCell<<endl;
    
//     transfer BT matrix
//     int* d_BTRowPtr = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_BTRowPtr, (n_BT+1) * sizeof(int)));
    
//     int* d_BTKCol = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_BTKCol, nz_BT * sizeof(int)));
    
//     double* d_B1TEntries = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_B1TEntries, nz_BT * sizeof(double)));
    
//     double* d_B2TEntries = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_B2TEntries, nz_BT * sizeof(double)));
    
//     double* d_B3TEntries = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_B3TEntries, nz_BT * sizeof(double)));
    
    // // Copy to GPU memory
    // CUDA_CHECK(cudaMemcpyAsync(d_BTRowPtr, BTRowPtr, (n_BT+1) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_BTKCol, BTKCol, nz_BT * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_B1TEntries, B1TEntries, nz_BT * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_B2TEntries, B2TEntries, nz_BT * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_B3TEntries, B3TEntries, nz_BT * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
// //     transfer B matrix
//     int* d_BRowPtr = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_BRowPtr, (n_B+1) * sizeof(int)));
    
//     int* d_BKCol = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_BKCol, nz_B * sizeof(int)));
    
//     double* d_B1Entries = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_B1Entries, nz_B * sizeof(double)));
    
//     double* d_B2Entries = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_B2Entries, nz_B * sizeof(double)));
    
//     double* d_B3Entries = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_B3Entries, nz_B * sizeof(double)));
    
    // // Copy to GPU memory
    // CUDA_CHECK(cudaMemcpyAsync(d_BRowPtr, BRowPtr, (n_B+1) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_BKCol, BKCol, nz_B * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_B1Entries, B1Entries, nz_B * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_B2Entries, B2Entries, nz_B * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_B3Entries, B3Entries, nz_B * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    


    UGlobalNumbers = USpace->GetGlobalNumbers();
    UBeginIndex = USpace->GetBeginIndex();
    ActiveBound = USpace->GetActiveBound();

    PGlobalNumbers = PSpace->GetGlobalNumbers();
    PBeginIndex = PSpace->GetBeginIndex();
    
    int* d_UGlobalNumbers = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_UGlobalNumbers, N_U*N_Cells * sizeof(int)));
    
    int* d_UBeginIndex = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_UBeginIndex, N_Cells * sizeof(int)));
    
    int* d_PGlobalNumbers = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_PGlobalNumbers, (N_P*N_Cells) * sizeof(int)));
    
    int* d_PBeginIndex = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_PBeginIndex, N_PCells * sizeof(int)));
    
    
    // Copy to GPU memory
    CUDA_CHECK(cudaMemcpyAsync(d_UGlobalNumbers, UGlobalNumbers, N_U*N_Cells * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_UBeginIndex, UBeginIndex, N_Cells * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_PGlobalNumbers, PGlobalNumbers, (N_P*N_Cells) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_PBeginIndex, PBeginIndex, N_Cells * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaStreamSynchronize(stream[0]));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float milliseconds;
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    data_transfer_time += milliseconds;
    
    int offset_Rhs, offset_Sys;
    
    int thread_blocks;
    int jj, itr;
    int count=0;
    

            double value, value1, value2, value3;
            double value11,value12,value13,value21,value22;
            double value23,value31,value32,value33;
            int k1, k2, k3, j2, j3, k,m, UDOF, PDOF, begin, end,end_itr, ii;
            
            if(smooth == -1)
            end_itr = TDatabase::ParamDB->SC_PRE_SMOOTH_SADDLE;
        else if(smooth == 0)
            end_itr = TDatabase::ParamDB->SC_COARSE_MAXIT_SADDLE;
        else
            end_itr = TDatabase::ParamDB->SC_POST_SMOOTH_SADDLE;
//             double GSystem[maxCellsPerColor * (N_LocalDOF * N_LocalDOF)];
//             double GRhs[maxCellsPerColor * N_LocalDOF];
        
    if (N_LocalDOF > LargestDirectSolve)
    {
      // size of local system has changed
      if (N_LocalDOF != TDatabase::ParamDB->INTERNAL_LOCAL_DOF)
      {
        // itmethod exists already
        if ( TDatabase::ParamDB->INTERNAL_LOCAL_DOF >0)
          delete itmethod;
        // allocate new itmethod
        itmethod = new TFgmresIte(MatVect, Defect, NULL, 0, N_LocalDOF, 1);
        TDatabase::ParamDB->INTERNAL_LOCAL_DOF = N_LocalDOF;
      }
    }
    
     
        
    int numThreads = TDatabase::ParamDB->OMPNUMTHREADS;
     
     omp_set_num_threads(numThreads);
    
    for(itr=0;itr<end_itr;itr++)
    {
        for(int i=0;i<N_CIntCell;i++)
        {
//             if(rank==1)
//             cout<<"color:"<<i<<endl;
            thread_blocks = (ptrCellColors[i+1]-ptrCellColors[i])*32/THREADS_PER_BLOCK + 1;
            
            CUDA_CHECK(cudaMemsetAsync(d_System, 0, maxCellsPerColor * (N_LocalDOF * N_LocalDOF) * sizeof(double), stream[0] ));
            memset(System, 0, maxCellsPerColor*SizeOfDouble*N_LocalDOF*N_LocalDOF);
//             cudaEventRecord(start, 0);
            
            CellVanka_assembleSysA<<<thread_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(d_ARowPtr, d_AKCol,
                                d_A11Entries, d_A12Entries, d_A13Entries, d_A21Entries, d_A22Entries, d_A23Entries, d_A31Entries,  d_A32Entries, d_A33Entries, 
                                d_BTRowPtr, d_BTKCol, d_B1TEntries, d_B2TEntries, d_B3TEntries, d_BRowPtr, d_BKCol, d_B1Entries, d_B2Entries,  d_B3Entries, 
                                d_u1, d_rhs1, d_CellReorder, N_U, N_P, N_LocalDOF, N_UDOF,
                                d_UGlobalNumbers, d_UBeginIndex, d_PGlobalNumbers, d_PBeginIndex, ActiveBound, d_System, d_Rhs, ptrCellColors[i], ptrCellColors[i+1]);
    
            CellVanka_assembleU<<<thread_blocks, THREADS_PER_BLOCK, 0, stream[1]>>>(d_ARowPtr, d_AKCol,
                                d_A11Entries, d_A12Entries, d_A13Entries, d_A21Entries, d_A22Entries, d_A23Entries, d_A31Entries,  d_A32Entries, d_A33Entries, 
                                d_BTRowPtr, d_BTKCol, d_B1TEntries, d_B2TEntries, d_B3TEntries, d_BRowPtr, d_BKCol, d_B1Entries, d_B2Entries,  d_B3Entries, 
                                d_u1, d_rhs1, d_CellReorder, N_U, N_P, N_LocalDOF, N_UDOF,
                                d_UGlobalNumbers, d_UBeginIndex, d_PGlobalNumbers, d_PBeginIndex, ActiveBound, d_System, d_Rhs, ptrCellColors[i], ptrCellColors[i+1]);
            
            CellVanka_assembleP<<<thread_blocks, THREADS_PER_BLOCK, 0, stream[1]>>>(d_ARowPtr, d_AKCol,
                                d_A11Entries, d_A12Entries, d_A13Entries, d_A21Entries, d_A22Entries, d_A23Entries, d_A31Entries,  d_A32Entries, d_A33Entries, 
                                d_BTRowPtr, d_BTKCol, d_B1TEntries, d_B2TEntries, d_B3TEntries, d_BRowPtr, d_BKCol, d_B1Entries, d_B2Entries,  d_B3Entries, 
                                d_u1, d_rhs1, d_CellReorder, N_U, N_P, N_LocalDOF, N_UDOF,
                                d_UGlobalNumbers, d_UBeginIndex, d_PGlobalNumbers, d_PBeginIndex, ActiveBound, d_System, d_Rhs, ptrCellColors[i], ptrCellColors[i+1]);
            
            
            
            CellVanka_assembleSysB<<<thread_blocks, THREADS_PER_BLOCK, 0, stream[1]>>>(d_ARowPtr, d_AKCol,
                                d_A11Entries, d_A12Entries, d_A13Entries, d_A21Entries, d_A22Entries, d_A23Entries, d_A31Entries,  d_A32Entries, d_A33Entries, 
                                d_BTRowPtr, d_BTKCol, d_B1TEntries, d_B2TEntries, d_B3TEntries, d_BRowPtr, d_BKCol, d_B1Entries, d_B2Entries,  d_B3Entries, 
                                d_u1, d_rhs1, d_CellReorder, N_U, N_P, N_LocalDOF, N_UDOF,
                                d_UGlobalNumbers, d_UBeginIndex, d_PGlobalNumbers, d_PBeginIndex, ActiveBound, d_System, d_Rhs, ptrCellColors[i], ptrCellColors[i+1]);
            
            CellVanka_assembleSysBT<<<thread_blocks, THREADS_PER_BLOCK, 0, stream[1]>>>(d_ARowPtr, d_AKCol,
                                d_A11Entries, d_A12Entries, d_A13Entries, d_A21Entries, d_A22Entries, d_A23Entries, d_A31Entries,  d_A32Entries, d_A33Entries, 
                                d_BTRowPtr, d_BTKCol, d_B1TEntries, d_B2TEntries, d_B3TEntries, d_BRowPtr, d_BKCol, d_B1Entries, d_B2Entries,  d_B3Entries, 
                                d_u1, d_rhs1, d_CellReorder, N_U, N_P, N_LocalDOF, N_UDOF,
                                d_UGlobalNumbers, d_UBeginIndex, d_PGlobalNumbers, d_PBeginIndex, ActiveBound, d_System, d_Rhs, ptrCellColors[i], ptrCellColors[i+1]);
//             
//             cudaEventRecord(stop, 0);
//             cudaEventSynchronize(stop);
//             cudaEventElapsedTime(&milliseconds, start, stop);
//             
//             kernel_time += milliseconds;
            
            
             
//              if(rank==1)
//              verify(temp_Rhs,Rhs,maxCellsPerColor * N_LocalDOF,1);
//           cout<<"i:"<<ptrCellColors[i]<<endl;
//           cout<<"i+1:"<<ptrCellColors[i+1]<<endl;
            

            

            // CUDA_CHECK(cudaStreamSynchronize(stream[0]));
            // CUDA_CHECK(cudaStreamSynchronize(stream[1]));
//             CUDA_CHECK(cudaStreamSynchronize(stream[2]));
            // CUDA_CHECK(cudaMemcpyAsync(Rhs, d_Rhs, maxCellsPerColor * N_LocalDOF * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
            // CUDA_CHECK(cudaMemcpyAsync(System, d_System, maxCellsPerColor * (N_LocalDOF * N_LocalDOF) * sizeof(double), cudaMemcpyDeviceToHost,stream[1]));
            CUDA_CHECK(cudaStreamSynchronize(stream[0]));
            CUDA_CHECK(cudaStreamSynchronize(stream[1]));
        
double t1,t2;
#ifdef _MPI
  t1 = MPI_Wtime();
#else
  t1 = GetTime();
#endif
  
        
//             int batch=2;
//             int* offset_Sysarr=new int[batch];
//             int* offset_Rhsarr=new int[batch];
// //   #pragma parallel for default(private) shared(System,j1,begin,end,N_U,N_P,N_LocalDOF,CellReorder)
//             for(int jj=ptrCellColors[i];jj<ptrCellColors[i+1];)
//             {
//                 int count=0;
                
//                 int kk=jj;
// //                 cout<<jj<<endl;
                
//                 for(count=0; count<batch && (kk<ptrCellColors[i+1]); kk++){
                    
//                     ii = CellReorder[kk];
                
//                     Cell = Coll->GetCell(ii);
                    
// //                     kk++;
                    
//                 #ifdef _MPI
//                     if(Cell->IsHaloCell()){
//                 //       cout << "this should" << endl;
//                     continue;
//                 //       cout << "this shouldnt" << endl;
//                     }   
//                 #endif
            
//                     offset_Sysarr[count]= ((kk)-ptrCellColors[i]) * (N_LocalDOF * N_LocalDOF);
//                     offset_Rhsarr[count]= ((kk)-ptrCellColors[i]) * N_LocalDOF;
                    
//                     count++;
// //                     kk++;
                
//                 }
                
//                 SolveLinearSystemCuSolver(d_System, d_Rhs, Rhs, offset_Sysarr,offset_Rhsarr, count,  N_LocalDOF, N_LocalDOF);
// //                 cout<<"solved"<<endl;
                
//                 count=0;
//                 kk=jj;
                
//                 for(count=0; count<batch && (kk<ptrCellColors[i+1]); kk++,jj++){
                    
//                     offset_Rhs = (kk-ptrCellColors[i]) * N_LocalDOF;
                    
//                     ii = CellReorder[kk];
                
//                     Cell = Coll->GetCell(ii);
                    
                    
//                 #ifdef _MPI
//                     if(Cell->IsHaloCell()){
//                 //       cout << "this should" << endl;
//                     continue;
//                 //       cout << "this shouldnt" << endl;
//                     }   
//                 #endif
                    
//                     #ifdef __2D__
//                     UEle = TFEDatabase2D::GetFE2D(USpace->GetFE2D(ii, Cell));
//                     PEle = TFEDatabase2D::GetFE2D(PSpace->GetFE2D(ii, Cell));
//                 #endif
//                 #ifdef __3D__
//                     UEle = TFEDatabase3D::GetFE3D(USpace->GetFE3D(ii, Cell));
//                     PEle = TFEDatabase3D::GetFE3D(PSpace->GetFE3D(ii, Cell));
//                 #endif

//                     // get local number of dof
//                     N_U = UEle->GetN_DOF();
//                     N_P = PEle->GetN_DOF();
//                     N_LocalDOF = GEO_DIM*N_U+N_P;

//                     UDOFs = UGlobalNumbers+UBeginIndex[ii];
//                     PDOFs = PGlobalNumbers+PBeginIndex[ii];
                    
//                 #ifdef __3D__
//                     j1 = 2*N_U;
//                 #endif
//                     for(j=0;j<N_U;j++)
//                     {
//                     l = UDOFs[j];
//                     u1[l] += damp*Rhs[offset_Rhs+j];
//                     u2[l] += damp*Rhs[offset_Rhs+j+N_U];
//                 #ifdef __3D__
//                     u3[l] += damp*Rhs[offset_Rhs+j+j1];
//                 #endif  
//                     }

//                     j1 = GEO_DIM*N_U;
//                     for(j=0;j<N_P;j++)
//                     {
//                     l = PDOFs[j];
//                     p[l] += damp*Rhs[offset_Rhs+j+j1];
//                     }
                    
//                     count++;
                    
                
//                 }

//             }
            

    PUSH_RANGE("solve",4)
    mg_solver->Magma_Batch_Solver(d_System, d_Rhs, Rhs, ptrCellColors[i+1] - ptrCellColors[i]);
    // cout<<"here"<<endl;
    
    updateSolution<<<thread_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(d_u1, d_rhs1, d_CellReorder, N_U, N_P, N_LocalDOF, N_UDOF,
                                    d_UGlobalNumbers, d_UBeginIndex, d_PGlobalNumbers, d_PBeginIndex, ActiveBound, d_System, d_Rhs, ptrCellColors[i], ptrCellColors[i+1], damp);

//     #pragma omp parallel for default(shared) schedule(dynamic)
//     for(int jj=ptrCellColors[i];jj<ptrCellColors[i+1];jj++)
//             {
// //                 cout<<"thrds:"<<omp_get_num_threads()<<endl;
//                 int offset_Rhs = (jj-ptrCellColors[i]) * N_LocalDOF;
//                 int offset_Sys = (jj-ptrCellColors[i]) * (N_LocalDOF * N_LocalDOF);
                

//                 int ii = CellReorder[jj];
                
// //                 TBaseCell *Cell = Coll->GetCell(ii);
                
// // //             #ifdef _MPI
// // //                 if(Cell->IsHaloCell()){
// // //             //       cout << "this should" << endl;
// // //                 continue;
// // //             //       cout << "this shouldnt" << endl;
// // //                 }   
// // //             #endif
// //             //    OutPut(i << downwind[i] << endl);
// //             #ifdef __2D__
// //                 TFE2D *UEle = TFEDatabase2D::GetFE2D(USpace->GetFE2D(ii, Cell));
// //                 TFE2D *PEle = TFEDatabase2D::GetFE2D(PSpace->GetFE2D(ii, Cell));
// //             #endif
// //             #ifdef __3D__
// //                 TFE3D *UEle = TFEDatabase3D::GetFE3D(USpace->GetFE3D(ii, Cell));
// //                 TFE3D *PEle = TFEDatabase3D::GetFE3D(PSpace->GetFE3D(ii, Cell));
// //             #endif

// //                 // // get local number of dof
// //                 // int N_U = UEle->GetN_DOF();
// //                 // int N_P = PEle->GetN_DOF();
// //                 // int N_LocalDOF = GEO_DIM*N_U+N_P;

//                 int *UDOFs = UGlobalNumbers+UBeginIndex[ii];
//                 int *PDOFs = PGlobalNumbers+PBeginIndex[ii];
  
// //                 // solve local system
// //                 if ((smoother==10 || smoother==12) && !C) // no diagonal Vanka for matrix C
// //                 {
// //                 // diagonal Vanka
// //             #ifdef __2D__
// //                 SolveDiagonalVanka2D(System+offset_Sys, Rhs+offset_Rhs, N_U, N_P, N_LocalDOF);
// //             #endif
// //             #ifdef __3D__
// //                 SolveDiagonalVanka3D(System+offset_Sys, Rhs+offset_Rhs, N_U, N_P, N_LocalDOF);
// //             #endif
// //                 }
// //                 else
// //                 {
// //                 // full Vanka
// //             //       cout << "full vanka dof :: " << N_LocalDOF << endl;
                
// // //                 if (N_LocalDOF > LargestDirectSolve)
// // //                 {
// // //                     memset(sol,0,N_LocalDOF*SizeOfDouble);
// // //                     verbose =  TDatabase::ParamDB->SC_VERBOSE;
// // //                     TDatabase::ParamDB->SC_VERBOSE = -1;
// // //                     itmethod->Iterate(matrix,NULL,sol,Rhs);
// // //                     TDatabase::ParamDB->SC_VERBOSE = verbose;
// // //                     memcpy(Rhs, sol, N_LocalDOF*SizeOfDouble);
// // //                 }
// // //                 else
// // //                 {
// //                     SolveLinearSystemLapack(System+offset_Sys, Rhs+offset_Rhs, N_LocalDOF, N_LocalDOF);
                    
// // //                     CUDA_CHECK(cudaMemcpyAsync(d_System, System, maxCellsPerColor * N_LocalDOF * N_LocalDOF *  sizeof(double), cudaMemcpyHostToDevice, stream[0]));
                    
// // //                     CUDA_CHECK(cudaStreamSynchronize(stream[0]));
                    
// //                     // SolveLinearSystemCuSolver(d_System+offset_Sys, d_Rhs+offset_Rhs, Rhs+offset_Rhs, N_LocalDOF, N_LocalDOF);
// // //                 }
// //                 }
                

//             #ifdef __3D__
//                 int j1 = 2*N_U;
//             #endif
//                 for(int j=0;j<N_U;j++)
//                 {
//                 int l = UDOFs[j];
//                 u1[l] += damp*Rhs[offset_Rhs+j];
//                 u2[l] += damp*Rhs[offset_Rhs+j+N_U];
//             #ifdef __3D__
//                 u3[l] += damp*Rhs[offset_Rhs+j+j1];
//             #endif  
//                 }

//                 j1 = GEO_DIM*N_U;
//                 for(int j=0;j<N_P;j++)
//                 {
//                 int l = PDOFs[j];
//                 p[l] += damp*Rhs[offset_Rhs+j+j1];
//                 }


        
//     }
    POP_RANGE
    
#ifdef _MPI
  t2 = MPI_Wtime();
#else
  t2 = GetTime();
#endif
timeVankaSolve += t2-t1;

    // CUDA_CHECK(cudaMemcpyAsync(d_u1, u1,  N_DOF* sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    CUDA_CHECK(cudaStreamSynchronize(stream[0]));


}
CUDA_CHECK(cudaMemcpyAsync(u1, d_u1,  N_DOF* sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
CUDA_CHECK(cudaStreamSynchronize(stream[0]));
  // apply damping
  if (fabs(1-alpha)>1e-3)
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

  SetHangingNodes(u1);
  
  #ifdef _MPI      
   ParCommU->CommUpdate(u1);   
   ParCommP->CommUpdate(p);
#endif
   
    CUDA_CHECK(cudaMemcpyAsync(d_u1, u1,  N_DOF* sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    CUDA_CHECK(cudaStreamSynchronize(stream[0]));
    
    // save current solution on 'old' vectors
    memcpy(uold, u1, N_DOF*SizeOfDouble);
    }
  
    delete []System;
    delete []Rhs;
    
    // for (int i = 0; i < nStreams; ++i)
    //     CUDA_CHECK( cudaStreamDestroy(stream[i]) );
  
    // Free GPU memory
    // CUDA_CHECK(cudaFree(d_ARowPtr));
    // CUDA_CHECK(cudaFree(d_AKCol));
    // CUDA_CHECK(cudaFree(d_A11Entries));
    // CUDA_CHECK(cudaFree(d_A12Entries));
    // CUDA_CHECK(cudaFree(d_A13Entries));
    // CUDA_CHECK(cudaFree(d_A21Entries));
    // CUDA_CHECK(cudaFree(d_A22Entries));
    // CUDA_CHECK(cudaFree(d_A23Entries));
    // CUDA_CHECK(cudaFree(d_A31Entries));
    // CUDA_CHECK(cudaFree(d_A32Entries));
    // CUDA_CHECK(cudaFree(d_A33Entries));
    
    // // Free GPU memory
    // CUDA_CHECK(cudaFree(d_BTRowPtr));
    // CUDA_CHECK(cudaFree(d_BTKCol));
    // CUDA_CHECK(cudaFree(d_B1TEntries));
    // CUDA_CHECK(cudaFree(d_B2TEntries));
    // CUDA_CHECK(cudaFree(d_B3TEntries));
          
    // // Free GPU memory
    // CUDA_CHECK(cudaFree(d_BRowPtr));
    // CUDA_CHECK(cudaFree(d_BKCol));
    // CUDA_CHECK(cudaFree(d_B1Entries));
    // CUDA_CHECK(cudaFree(d_B2Entries));
    // CUDA_CHECK(cudaFree(d_B3Entries));
    
    CUDA_CHECK(cudaFree(d_u1));
    CUDA_CHECK(cudaFree(d_rhs1));
    CUDA_CHECK(cudaFree(d_CellReorder));
    
    CUDA_CHECK(cudaFree(d_Rhs));
    CUDA_CHECK(cudaFree(d_System));

    CUDA_CHECK(cudaFree(d_UGlobalNumbers));
    CUDA_CHECK(cudaFree(d_UBeginIndex));
    CUDA_CHECK(cudaFree(d_PGlobalNumbers));
    CUDA_CHECK(cudaFree(d_PBeginIndex));

}

// void TNSE_MGLevel4::CellVanka_Level_Split(double *u1, double *rhs1, double *aux,
//         int N_Parameters, double *Parameters, int smoother, int smooth)
//  {
     
//     int split_level=1;
    
//     if(Level > split_level){
//         CellVanka_GPU(u1, rhs1, aux, N_Parameters, Parameters, smoother, smooth);
//     }
//     else{
//         int end;
        
//         if(smooth == -1)
//             end = TDatabase::ParamDB->SC_PRE_SMOOTH_SADDLE;
//         else if(smooth == 0)
//             end = TDatabase::ParamDB->SC_COARSE_MAXIT_SADDLE;
//         else
//             end = TDatabase::ParamDB->SC_POST_SMOOTH_SADDLE;
    
//         for(int j=0;j<end;j++)
//         {
//             CellVanka(u1, rhs1, aux, 
//                 N_Parameters, Parameters, smoother, 0);
//         }
//     }
        
// }

void TNSE_MGLevel4::CellVanka_Combo(double *u1, double *rhs1, double *aux,
        int N_Parameters, double *Parameters, int smoother, int smooth, cudaStream_t *stream, int* d_ARowPtr, int*  d_AKCol,
        double* d_A11Entries, double* d_A12Entries, double* d_A13Entries,
        double* d_A21Entries, double* d_A22Entries, double* d_A23Entries,
        double* d_A31Entries, double* d_A32Entries, double* d_A33Entries,
        int* d_BTRowPtr, int*  d_BTKCol,
        double* d_B1TEntries, double* d_B2TEntries, double* d_B3TEntries,
        int* d_BRowPtr, int*  d_BKCol,
        double* d_B1Entries, double* d_B2Entries, double* d_B3Entries)
 {
    
    TCollection *Coll = USpace->GetCollection();
    int N_Cells = Coll->GetN_Cells();
    
    int split_level=1;
    
    if(N_Cells > 10000 ){
        // cout<<"CellVanka_GPU"<<endl;
        CellVanka_GPU(u1, rhs1, aux, N_Parameters, Parameters, smoother, smooth, stream, d_ARowPtr, d_AKCol,
            d_A11Entries, d_A12Entries, d_A13Entries,
            d_A21Entries, d_A22Entries, d_A23Entries,
            d_A31Entries, d_A32Entries, d_A33Entries,
            d_BTRowPtr, d_BTKCol,
            d_B1TEntries, d_B2TEntries, d_B3TEntries,
            d_BRowPtr, d_BKCol,
            d_B1Entries, d_B2Entries, d_B3Entries);
    }
    else if(N_Cells > 1000 && N_Cells<=10000 ){
        // cout<<"CellVanka_CPU_GPU"<<endl;
        CellVanka_CPU_GPU(u1, rhs1, aux, N_Parameters, Parameters, smoother, smooth, stream, d_ARowPtr, d_AKCol,
            d_A11Entries, d_A12Entries, d_A13Entries,
            d_A21Entries, d_A22Entries, d_A23Entries,
            d_A31Entries, d_A32Entries, d_A33Entries,
            d_BTRowPtr, d_BTKCol,
            d_B1TEntries, d_B2TEntries, d_B3TEntries,
            d_BRowPtr, d_BKCol,
            d_B1Entries, d_B2Entries, d_B3Entries);
    }
    else{
        

    #ifdef _MPI
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(rank==0)
    {        
    #endif
            cout<<"CellVanka_CPU"<<endl;
    #ifdef _MPI
        }
    #endif
        int end;
        
        if(smooth == -1)
            end = TDatabase::ParamDB->SC_PRE_SMOOTH_SADDLE;
        else if(smooth == 0)
            end = TDatabase::ParamDB->SC_COARSE_MAXIT_SADDLE;
        else
            end = TDatabase::ParamDB->SC_POST_SMOOTH_SADDLE;
    
        for(int j=0;j<end;j++)
        {
            CellVanka(u1, rhs1, aux, 
                N_Parameters, Parameters, smoother, 0);
        }
    }
        
}

#endif
#endif
#endif
