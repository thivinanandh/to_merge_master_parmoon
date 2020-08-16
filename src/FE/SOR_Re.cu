#include <ParFECommunicator3D.h>
#include <ParFEMapper3D.h>
#include <Database.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <cmath> 
#include <LinAlg.h>
#include <MGLevel3D.h>
#include <omp.h>
#include "nvToolsExt.h"
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

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




//     if((call) != cudaSuccess) { \
//         cudaError_t err = cudaGetLastError(); \
//         fprintf(stderr,"CUDA error calling \""#call"\", code is %d\n",err); \
//         my_abort(err); }

#define THREADS_PER_BLOCK 1024
        
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

using namespace std;

__global__ void mergeSolution ( double*  d_sol,
                                double* sol,
                                int* Reorder,
                                int Bound,
                                int N_Active
                            )
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x ; // global thread index
    
    int i;
    
    if(thread_id < Bound){
        
        if(Reorder != nullptr)
            i = Reorder[thread_id];
        else
            i=thread_id;
    
        if(i < N_Active ){
            
            d_sol[i] = sol[i];
        }
    }
    
//     int hanging_id=thread_id+N_Active;
//     
//     if(hanging_id < HangingNodeBound){
//         d_sol[hanging_id] = sol[hanging_id];
//         
//     }
    
}

__global__ void jacobi_update(double* x,double* h,double* d,double omega,int nrow)
{
    int id = blockDim.x*blockIdx.x + threadIdx.x;
    if(id < nrow)
    {
        x[id] -= omega*h[id]/d[id];
    }
}

void mergeSolutionCPU(double*  sol,
                                double* sol_temp,
                                int* Reorder1,
                                int Bound1,
                                int N_Active){
    
    #pragma omp parallel for
    for(int ii=0; ii< Bound1 ; ii++){
        int i = Reorder1[ii];
    
        if(i < N_Active ){
            
            sol[i] = sol_temp[i];
        }
        
    }
    
    
}

// void expandedSol(double* sol, double* ex_sol, int N_colours, int* ptrC, int N_Active, int* Reorder, int* RowPtr, int* KCol){
    
//     int i;
//     for(int ii=0;ii<N_colours;ii++)
//         {
//             for(int jj=ptrC[ii];jj<ptrC[ii+1];jj++)
//             {
//                 i = Reorder[jj];
//                 if(i >= N_Active)     continue;

//                 for(int j=RowPtr[i];j<RowPtr[i+1];j++)
//                 {
//                     ex_sol[j] = sol[KCol[j]];
                    
//                 } // endfor j
//                 // cout << "sol[i]: " << sol[i] << endl;
//             } // endfor jj
//         } //end for ii 
    
// }

// void compressedSol(double* sol, double* ex_sol, int N_colours, int* ptrC, int N_Active, int* Reorder, int* RowPtr, int* KCol){
    
//     int i;
//     for(int ii=0;ii<N_colours;ii++)
//         {
//             for(int jj=ptrC[ii];jj<ptrC[ii+1];jj++)
//             {
//                 i = Reorder[jj];
//                 if(i >= N_Active)     continue;

//                 for(int j=RowPtr[i];j<RowPtr[i+1];j++)
//                 {
//                     sol[KCol[j]] = ex_sol[j];
                    
//                 } // endfor j
//                 // cout << "sol[i]: " << sol[i] << endl;
//             } // endfor jj
//         } //end for ii  
// }

// void vectorCoalesce(double* sol, double* ex_sol, TParFEMapper3D *ParMapper, int N_CMaster, int N_CDept2, int N_CDept1, int N_CInt, int* ptrCMaster, int* ptrCDept2, int* ptrCDept1, int* ptrCInt, int N_Active, int* RowPtr, int* KCol, int option){
    
//     int* Reorder;
    
//     if(option == 0){
        
//         Reorder = ParMapper->GetReorder_M();
//         expandedSol(sol, ex_sol, N_CMaster, ptrCMaster, N_Active, Reorder, RowPtr, KCol);
        
//         Reorder = ParMapper->GetReorder_D2();
//         expandedSol(sol, ex_sol, N_CDept2, ptrCDept2, N_Active, Reorder, RowPtr, KCol);
        
//         Reorder = ParMapper->GetReorder_D1();
//         expandedSol(sol, ex_sol, N_CDept1, ptrCDept1, N_Active, Reorder, RowPtr, KCol);
        
//         Reorder = ParMapper->GetReorder_I();
//         expandedSol(sol, ex_sol, N_CInt, ptrCInt, N_Active, Reorder, RowPtr, KCol);
    
//     }
    
//     if(option == 1){
        
//         Reorder = ParMapper->GetReorder_M();
//         compressedSol(sol, ex_sol, N_CMaster, ptrCMaster, N_Active, Reorder, RowPtr, KCol);
        
//         Reorder = ParMapper->GetReorder_D2();
//         compressedSol(sol, ex_sol, N_CDept2, ptrCDept2, N_Active, Reorder, RowPtr, KCol);
        
//         Reorder = ParMapper->GetReorder_D1();
//         compressedSol(sol, ex_sol, N_CDept1, ptrCDept1, N_Active, Reorder, RowPtr, KCol);
        
//         Reorder = ParMapper->GetReorder_I();
//         compressedSol(sol, ex_sol, N_CInt, ptrCInt, N_Active, Reorder, RowPtr, KCol);
    
//     }
    
 
// }



__global__ void SOR_Re_Kernel ( int start , const int end,
                                        const int* RowPtr ,
                                        const int* Reorder,
                                        const int* KCol ,
                                        const double* Entries ,
                                        const double* f ,
                                        double* sol,
                                        int N_Active,
                                        double omega,
                                        int sub_itr
                                    )
{
//     int size= blockDim.x*THREADS_PER_BLOCK;
    
    const unsigned int THREADS_PER_VECTOR = 32;
    const unsigned int VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
    
    const unsigned int num_vectors = VECTORS_PER_BLOCK * gridDim.x; 
    
    
//     cout<<"neha: val size: "<<(VECTORS_PER_BLOCK + 1) * THREADS_PER_VECTOR<<endl;
    
    __shared__ volatile double vals [VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];
//     __shared__ volatile double vals [THREADS_PER_BLOCK];
    
    const unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
//     const unsigned int vector_id   = thread_id   /  THREADS_PER_VECTOR; 
    
    const unsigned int warp_id = thread_id   /  THREADS_PER_VECTOR;  // global warp index
    
    unsigned int warp_id_loc = threadIdx.x   /  THREADS_PER_VECTOR;  // global warp index
    
    const unsigned int lane = threadIdx.x & (THREADS_PER_VECTOR - 1); // thread index within the warp
    
//     int SHARED_SIZE=VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2;
    
//     for (int i = threadIdx.x; i < SHARED_SIZE; i += blockDim.x) 
//     vals[i] = 0;
//     __syncthreads();
    
//     // one warp per row
//     int row = warp_id ;
    
    int index;
    double t, sum;
    __shared__ double diag[VECTORS_PER_BLOCK];
    double temp;
//    int sub_itr=2;
//         if ( row < ndofs ){

        for( int ii=0; ii < sub_itr; ii++){
            for(unsigned int row = warp_id + start; row < end; row += num_vectors){
                
                int i = Reorder[row];
                if(i < N_Active){
                    
                    int row_start = RowPtr[i];
                    int row_end = RowPtr[i+1];
                    
                    sum = 0;

                        // compute running sum per thread
                        for ( int jj = row_start + lane ; jj < row_end ; jj += THREADS_PER_VECTOR){
                            
                            
                            
                            index = KCol[jj];
                            
                            if(index == i){
//                                 warp_id_loc = threadIdx.x/THREADS_PER_VECTOR;
                                
        //                         master = jj;
                                diag[warp_id_loc] = Entries[jj];
                            }
                            else{
                                sum = sum + ( Entries [ jj ] * sol [index]);
//                                 sum = __dadd_rn(sum, Entries [ jj ] * sol [index]);
                            }
                            
                        }

                        
                    vals [ threadIdx.x] = sum;
//                     __syncthreads();
                    
    //                 parallel reduction in shared memory
                    if (THREADS_PER_VECTOR > 16) vals[threadIdx.x] = sum = sum + vals[threadIdx.x + 16];
                    if (THREADS_PER_VECTOR >  8) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  8];
                    if (THREADS_PER_VECTOR >  4) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  4];
                    if (THREADS_PER_VECTOR >  2) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  2];
                    if (THREADS_PER_VECTOR >  1) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  1];
                    
    //                 if (lane < 16) vals[threadIdx.x] = sum = sum + vals[threadIdx.x + 16];
    //                 if (lane <  8) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  8];
    //                 if (lane <  4) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  4];
    //                 if (lane <  2) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  2];
    //                 if (lane <  1) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  1];
                    
//                     __syncthreads();
                    // first thread writes the result
                    if ( lane == 0){
        //                 f[i] -= vals [ threadIdx.x ];
//                         double diag1;
//                         
//                         int flag=1;
//                         
//                         for(int kk=row_start; (kk<row_end) && (flag==1); kk++){
//                             
//                             if(KCol[kk] == i){
//                                 diag1 = Entries[kk];
//                                 flag=0;
//                             }
//                         }
//                         warp_id_loc = threadIdx.x/THREADS_PER_VECTOR;
                        
                        t=sol[i];
                        
    //                     t= omega*((f[i]-vals[threadIdx.x])/diag-sol[i]);
                        
                        temp= omega*((f[i]-vals[threadIdx.x])/diag[warp_id_loc]-t) + t; 
                        sol[i]=temp;
//                         sol[i]=(vals[threadIdx.x]);
    //                     atomicAdd(&sol[i],t);
    //                     t = old_sol[i];
    //                     sol[i] = omega*((f[i]-vals[threadIdx.x])/diag -t) + t;
    //                     sol_temp[i]=0.0123;
                    }
                
            
            }
        }
        }
    
}


__global__ void IND_kernel ( int start , const int end,
    const int* RowPtr ,
    const int* Reorder,
    const int* KCol ,
    const double* Entries ,
    const double* f ,
    double* sol,
    int N_Active,
    double omega,
    int sub_itr
)
{
//     int size= blockDim.x*THREADS_PER_BLOCK;

const unsigned int THREADS_PER_VECTOR = 32;
const unsigned int VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;

const unsigned int num_vectors = VECTORS_PER_BLOCK * gridDim.x; 


//     cout<<"neha: val size: "<<(VECTORS_PER_BLOCK + 1) * THREADS_PER_VECTOR<<endl;

__shared__ volatile double vals [VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];
//     __shared__ volatile double vals [THREADS_PER_BLOCK];

const unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
//     const unsigned int vector_id   = thread_id   /  THREADS_PER_VECTOR; 

const unsigned int warp_id = thread_id   /  THREADS_PER_VECTOR;  // global warp index

unsigned int warp_id_loc = threadIdx.x   /  THREADS_PER_VECTOR;  // global warp index

const unsigned int lane = threadIdx.x & (THREADS_PER_VECTOR - 1); // thread index within the warp

//     int SHARED_SIZE=VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2;

//     for (int i = threadIdx.x; i < SHARED_SIZE; i += blockDim.x) 
//     vals[i] = 0;
//     __syncthreads();

//     // one warp per row
//     int row = warp_id ;

int index;
double t, sum;
__shared__ double diag[VECTORS_PER_BLOCK];
double temp;
//    int sub_itr=2;
//         if ( row < ndofs ){

for( int ii=0; ii < sub_itr; ii++){
for(unsigned int row = warp_id + start; row < end; row += num_vectors){

int i = Reorder[row];
if(i < N_Active){

int row_start = RowPtr[i];
int row_end = RowPtr[i+1];

sum = 0;

// compute running sum per thread
for ( int jj = row_start + lane ; jj < row_end ; jj += THREADS_PER_VECTOR){



index = KCol[jj];

if(index == i){
//                                 warp_id_loc = threadIdx.x/THREADS_PER_VECTOR;

//                         master = jj;
diag[warp_id_loc] = Entries[jj];
}
else{
sum = sum + ( Entries [ jj ] * sol [index]);
//                                 sum = __dadd_rn(sum, Entries [ jj ] * sol [index]);
}

}


vals [ threadIdx.x] = sum;
//                     __syncthreads();

//                 parallel reduction in shared memory
if (THREADS_PER_VECTOR > 16) vals[threadIdx.x] = sum = sum + vals[threadIdx.x + 16];
if (THREADS_PER_VECTOR >  8) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  8];
if (THREADS_PER_VECTOR >  4) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  4];
if (THREADS_PER_VECTOR >  2) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  2];
if (THREADS_PER_VECTOR >  1) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  1];

//                 if (lane < 16) vals[threadIdx.x] = sum = sum + vals[threadIdx.x + 16];
//                 if (lane <  8) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  8];
//                 if (lane <  4) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  4];
//                 if (lane <  2) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  2];
//                 if (lane <  1) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  1];

//                     __syncthreads();
// first thread writes the result
if ( lane == 0){
//                 f[i] -= vals [ threadIdx.x ];
//                         double diag1;
//                         
//                         int flag=1;
//                         
//                         for(int kk=row_start; (kk<row_end) && (flag==1); kk++){
//                             
//                             if(KCol[kk] == i){
//                                 diag1 = Entries[kk];
//                                 flag=0;
//                             }
//                         }
//                         warp_id_loc = threadIdx.x/THREADS_PER_VECTOR;

t=sol[i];

//                     t= omega*((f[i]-vals[threadIdx.x])/diag-sol[i]);

temp= omega*((f[i]-vals[threadIdx.x])/diag[warp_id_loc]-t) + t; 
sol[i]=temp;
//                         sol[i]=(vals[threadIdx.x]);
//                     atomicAdd(&sol[i],t);
//                     t = old_sol[i];
//                     sol[i] = omega*((f[i]-vals[threadIdx.x])/diag -t) + t;
//                     sol_temp[i]=0.0123;
}


}
}
}

}

__global__ void Jacobi_Scalar_Kernel(   const int start , const int end,
                                        const int* RowPtr ,
                                        #ifdef _MPI 
                                        const int* master,
                                        #endif
                                        const int* KCol ,
                                        const double* Entries ,
                                        const double* f ,
                                        double*  old_sol,
                                        double* sol,
                                        double omega,
                                        int rank
                                    )
{
    const unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int grid_size = gridDim.x * blockDim.x;
    
    int index;
    double t, sum;
    double diag;

    for(unsigned int i = thread_id + start; i < end; i += grid_size)
    {
        #ifdef _MPI 
        if(master[i] == rank){
        #endif
            
//         int i = Reorder[row];
//             if(i < N_Active){
                
                int row_start = RowPtr[i];
                int row_end = RowPtr[i+1];
                
                sum = f[i];
                
                for ( int jj = row_start; jj < row_end ; jj ++){
                    
                    
                    
                    index = KCol[jj];
                    
                    if(index == i){
                        
//                         master = jj;
                        diag = Entries[jj];
                    }
                    else{
                        sum = sum - ( Entries [ jj ] * old_sol [index]);
                    }
                    
                }
                
                t = old_sol[i];
                sol[i] = (1-omega)*t + omega*sum/diag;
//                 sol[i]=diag;
                
//         }
    #ifdef _MPI 
        }
    #endif
    }
}


__global__ void Jacobi_Kernel_Hyb (   const int start , const int end,
                                        const int* RowPtr ,
                                        #ifdef _MPI
                                        const int* master,
                                        #endif
                                        const int* KCol ,
                                        const double* Entries ,
                                        const double* f ,
                                        double*  old_sol,
                                        double* sol,
                                        double omega
                                    #ifdef _MPI
                                        ,int rank
                                    #endif
                                    )
{
const unsigned int THREADS_PER_VECTOR = 32;
    const unsigned int VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
    
    const unsigned int num_vectors = VECTORS_PER_BLOCK * gridDim.x; 
    
    
//     cout<<"neha: val size: "<<(VECTORS_PER_BLOCK + 1) * THREADS_PER_VECTOR<<endl;
    
    __shared__ volatile double vals [VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];
//     __shared__ volatile double vals [THREADS_PER_BLOCK];
    
    const unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
//     const unsigned int vector_id   = thread_id   /  THREADS_PER_VECTOR; 
    
    const unsigned int warp_id = thread_id   /  THREADS_PER_VECTOR;  // global warp index
    
    unsigned int warp_id_loc = threadIdx.x   /  THREADS_PER_VECTOR;  // global warp index
    
    const unsigned int lane = threadIdx.x & (THREADS_PER_VECTOR - 1); // thread index within the warp
    
//     int SHARED_SIZE=VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2;
    
//     for (int i = threadIdx.x; i < SHARED_SIZE; i += blockDim.x) 
//     vals[i] = 0;
//     __syncthreads();
    
//     // one warp per row
//     int row = warp_id ;
    
    int index;
    double t, sum;
    __shared__ double diag[VECTORS_PER_BLOCK];
    double temp;
   
//         if ( row < ndofs ){
        for(unsigned int row = warp_id + start; row < end; row += num_vectors){
            
//             int i = Reorder[row];
//             if(i < N_Active){
            int i=row;
            
            #ifdef _MPI
                if(master[i] == rank){
            #endif
                
                
                int row_start = RowPtr[i];
                int row_end = RowPtr[i+1];
                
                sum = 0;

                // compute running sum per thread
                for ( int jj = row_start + lane ; jj < row_end ; jj += 32){
                    
                    
                    
                    index = KCol[jj];
                    
                    if(index == i){
                        
//                         master = jj;
//                         diag = Entries[jj];
                        diag[warp_id_loc] = Entries[jj];
                    }
                    else{
//                         sum += ( Entries [ jj ] * old_sol [index]);
                        sum = __dadd_rn(sum, Entries [ jj ] * old_sol [index]);
                    }
                    
                }
                    
                vals [ threadIdx.x] = sum;
//                 __syncthreads();
                
//                 parallel reduction in shared memory
                if (THREADS_PER_VECTOR > 16) vals[threadIdx.x] = sum = sum + vals[threadIdx.x + 16];
                if (THREADS_PER_VECTOR >  8) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  8];
                if (THREADS_PER_VECTOR >  4) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  4];
                if (THREADS_PER_VECTOR >  2) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  2];
                if (THREADS_PER_VECTOR >  1) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  1];
                
//                 if (lane < 16) vals[threadIdx.x] = sum = sum + vals[threadIdx.x + 16];
// //                 __syncthreads();
//                 if (lane <  8) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  8];
// //                 __syncthreads();
//                 if (lane <  4) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  4];
// //                 __syncthreads();
//                 if (lane <  2) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  2];
// //                 __syncthreads();
//                 if (lane <  1) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  1];
//                 __syncthreads();
                
//                 __syncthreads();
                // first thread writes the result
                if ( lane == 0){
    //                 f[i] -= vals [ threadIdx.x ];
//                     t= omega*((f[i]-vals[threadIdx.x])/diag-sol[i]);
//                     atomicAdd(&sol[i],t);
                    t = old_sol[i];
                    sol[i] = omega*((f[i]-vals[threadIdx.x])/diag[warp_id_loc]-t) + t;
//                     sol_temp[i]=0.0123;
                }
            
        #ifdef _MPI
        }
        #endif
    }
}

__global__ void SOR_Scalar_Kernel(   const int start , const int end,
                                        const int* RowPtr ,
                                        const int* Reorder,
                                        const int* KCol ,
                                        const double* Entries ,
                                        const double* f ,
                                        double* sol,
                                        int N_Active,
                                        double omega
                                    )
{
    const unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int grid_size = gridDim.x * blockDim.x;
    
    int index;
    double t, sum;
    double diag;

    for(unsigned int row = thread_id + start; row < end; row += grid_size)
    {
        
//         if(master[i] == rank){
            
        int i = Reorder[row];
            if(i < N_Active){
                
                int row_start = RowPtr[i];
                int row_end = RowPtr[i+1];
                
                sum = f[i];
                
                for ( int jj = row_start; jj < row_end ; jj ++){
                    
                    
                    
                    index = KCol[jj];
                    
                    if(index == i){
                        
//                         master = jj;
                        diag = Entries[jj];
                    }
                    else{
                        sum = sum - ( Entries [ jj ] * sol [index]);
                    }
                    
                }
                
                t = sol[i];
                sol[i] = omega*(sum/diag-t) + t;
//                 sol[i]=diag;
                
//         }
        }
    }
}
__global__ void SOR_Re_Kernel ( int start , const int end,
                                        const int* RowPtr ,
                                        const int* Reorder,
                                        const int* KCol ,
                                        const double* Entries ,
                                        const double* f ,
                                        double* sol,
                                        int N_Active,
                                        double omega
                                    )
{
//     int size= blockDim.x*THREADS_PER_BLOCK;
    
    const unsigned int THREADS_PER_VECTOR = 32;
    const unsigned int VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
    
    const unsigned int num_vectors = VECTORS_PER_BLOCK * gridDim.x; 
    
    
//     cout<<"neha: val size: "<<(VECTORS_PER_BLOCK + 1) * THREADS_PER_VECTOR<<endl;
    
    __shared__ volatile double vals [VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];
//     __shared__ volatile double vals [THREADS_PER_BLOCK];
    
    const unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
//     const unsigned int vector_id   = thread_id   /  THREADS_PER_VECTOR; 
    
    const unsigned int warp_id = thread_id   /  THREADS_PER_VECTOR;  // global warp index
    
    const unsigned int lane = threadIdx.x & (THREADS_PER_VECTOR - 1); // thread index within the warp
    
//     // one warp per row
//     int row = warp_id ;
    
    int index;
    double t, sum;
    __shared__ double diag;
    double temp;
   int sub_itr=2;
//         if ( row < ndofs ){

//         for( int ii=0; ii < sub_itr; ii++){
            for(unsigned int row = warp_id + start; row < end; row += num_vectors){
                
                int i = Reorder[row];
                if(i < N_Active){
                    
                    int row_start = RowPtr[i];
                    int row_end = RowPtr[i+1];
                    
                    sum = 0;

                    // compute running sum per thread
                    for ( int jj = row_start + lane ; jj < row_end ; jj += 32){
                        
                        
                        
                        index = KCol[jj];
                        
                        if(index == i){
                            
    //                         master = jj;
                            diag = Entries[jj];
                        }
                        else{
                            sum += ( Entries [ jj ] * sol [index]);
                        }
                        
                    }
                        
                    vals [ threadIdx.x] = sum;
                    
    //                 parallel reduction in shared memory
                    if (THREADS_PER_VECTOR > 16) vals[threadIdx.x] = sum = sum + vals[threadIdx.x + 16];
                    if (THREADS_PER_VECTOR >  8) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  8];
                    if (THREADS_PER_VECTOR >  4) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  4];
                    if (THREADS_PER_VECTOR >  2) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  2];
                    if (THREADS_PER_VECTOR >  1) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  1];
                    
    //                 if (lane < 16) vals[threadIdx.x] = sum = sum + vals[threadIdx.x + 16];
    //                 if (lane <  8) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  8];
    //                 if (lane <  4) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  4];
    //                 if (lane <  2) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  2];
    //                 if (lane <  1) vals[threadIdx.x] = sum = sum + vals[threadIdx.x +  1];
                    
                    
                    // first thread writes the result
                    if ( lane == 0){
        //                 f[i] -= vals [ threadIdx.x ];
                        
                        t=sol[i];
                        
    //                     t= omega*((f[i]-vals[threadIdx.x])/diag-sol[i]);
                        
                        temp= omega*((f[i]-vals[threadIdx.x])/diag-t) + t; 
                        sol[i]=temp;
    //                     atomicAdd(&sol[i],t);
    //                     t = old_sol[i];
    //                     sol[i] = omega*((f[i]-vals[threadIdx.x])/diag -t) + t;
    //                     sol_temp[i]=0.0123;
                    }
                
            
            }
//         }
        }
    
}

void verify(int* o, int* n,int num, int itr, int rank){
 
    for(int i=0; i<num; i++){
        if(o[i] == n[i]){
            cout<<"neha: matching!!"<<o[i]<<" "<<n[i]<<" rank "<<rank<<endl;
            continue;
        }
        
        else {cout<<"neha: not matching!! "<<o[i]<<" "<<n[i]<<" rank "<<rank<<endl;
            return;
        }
    }
//     cout<<"neha: matching!!"<<itr<<" rank "<<rank<<endl;
}

void verify(double* o, double* n, int num, int itr,int rank){
 
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
#ifdef _CUDA
void TMGLevel3D::Jacobi_GPU(double *sol, double *f, double *aux, int N_Parameters, double *Parameters, int smooth, cudaStream_t *stream, int* d_RowPtr, int*  d_KCol, double* d_Entries, double* d_sol, int* d_master)
{
    // cout<<"Jacobi_GPU"<<endl;
    double omega = Parameters[0];
  
    #ifdef _MPI
    int rank;
    MPI_Comm_rank(TDatabase::ParamDB->Comm, &rank);
    int *master =ParComm->GetMaster();
    #endif
    
//     omega = Parameters[0];
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
    int ii, i,j,k,l,index,loop;
    int itr,jj,tid,nrows=0,numThreads,end;
    double s, t, diag;
    // int nStreams = 1;
    // cudaStream_t stream[nStreams];

    int nz= A->GetN_Entries();
    int n = A->GetN_Rows();
    
    if(smooth == -1)
        end = TDatabase::ParamDB->SC_PRE_SMOOTH_SCALAR;
    else if(smooth == 0)
        end = TDatabase::ParamDB->SC_COARSE_MAXIT_SCALAR;
    else{
        end = TDatabase::ParamDB->SC_POST_SMOOTH_SCALAR;
        // CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    }

//     const int streamSize = N_DOF / nStreams;
//     const int streamBytes = streamSize * sizeof(int);
    
    // for (i = 0; i < nStreams; ++i)
    //     CUDA_CHECK( cudaStreamCreate(&stream[i]) );
  
   // Allocate data on GPU memory
    // int* d_RowPtr = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_RowPtr, (n+1) * sizeof(int)));
    
    // int* d_Reorder = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_Reorder, (N_InterfaceM + N_Int + N_Dept1 +  N_Dept2) * sizeof(int)));
    // #ifdef _MPI 
    //     int* d_master = NULL;
    //     CUDA_CHECK(cudaMalloc((void**)&d_master, (N_DOF) * sizeof(int)));
    // #endif
    // int* d_KCol = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_KCol, nz * sizeof(int)));
    
    // double* d_Entries = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_Entries, nz * sizeof(double)));
    
    double* d_f = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_f, n * sizeof(double)));
    
    // double* d_sol = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_sol, n * sizeof(double)));

    // double* d_diag_mat = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_diag_mat, n * sizeof(double)));

    // double* d_helper = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_helper, n * sizeof(double)));

    double* d_sol_temp = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_sol_temp, N_Active * sizeof(double)));
    
    
    memcpy(sol+HangingNodeBound, f+HangingNodeBound, N_Dirichlet*sizeof(double));
    
    // Copy to GPU memory
    // CUDA_CHECK(cudaMemcpyAsync(d_RowPtr, RowPtr, (n+1) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_Entries, Entries, nz * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_KCol, KCol, nz * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_f, f, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));

    // CUDA_CHECK(cudaMemcpyAsync(d_diag_mat, A->GetMklDiagonal(), n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));

    // cusparseStatus_t stat;
    // cusparseHandle_t hndl;

    // cublasHandle_t hndl1;
    // cublasStatus_t stat1;

    // cusparseMatDescr_t descrA;
    // cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;

    // stat  = cusparseCreate(&hndl);
    // stat1 = cublasCreate(&hndl1);
    // stat = cusparseCreateMatDescr(&descrA);
    // stat = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    // transA = CUSPARSE_OPERATION_NON_TRANSPOSE;

    // stat  = cusparseSetStream(hndl,stream[0]);
    // stat1 = cublasSetStream(hndl1,stream[0]);

    // double s1 = 1.0;
    // double s2 = -1.0;
    // double s3 = 0.0;

    #ifdef _MPI
        CUDA_CHECK(cudaMemcpyAsync(d_master, master, N_DOF * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    #endif

    int num_blocks;
    
//     num_blocks= ceil(N_Active/THREADS_PER_BLOCK) +1 ;
        
    num_blocks= ceil(N_Active * 32 /THREADS_PER_BLOCK) +1 ;
              
    if ( num_blocks == 0) num_blocks=1;
    
//     CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//     CUDA_CHECK(cudaStreamSynchronize(stream[1]));
    
//     cout<<"N_Active:"<<N_Active<<endl;
    

        
         for(itr=0;itr<end;itr++)
        {
        //    stat = cusparseDcsrmv(hndl,transA,N_Active,N_DOF,nz,&s1,descrA,d_Entries,d_RowPtr,d_KCol,d_sol,&s3,d_helper);
        //    stat1 = cublasDaxpy(hndl1,N_Active,&s2,d_f,1,d_helper,1);
        //    jacobi_update<<<num_blocks,THREADS_PER_BLOCK, 0, stream[0]>>>(d_sol,d_helper,d_diag_mat,omega,N_Active);

           Jacobi_Kernel_Hyb<<<num_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(0, N_Active,d_RowPtr
            #ifdef _MPI 
            ,d_master 
            #endif
            ,d_KCol ,d_Entries ,d_f ,d_sol , d_sol_temp , omega
                                                                        #ifdef _MPI
                                                                           ,rank);
                                                                        #else
                                                                        );
                                                                        #endif

//         Jacobi_Scalar_Kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(0, N_Active,d_RowPtr ,d_master ,d_KCol ,d_Entries ,d_f ,d_sol , d_sol_temp, N_Active , omega, rank);
        
        
//         CUDA_CHECK(cudaStreamSynchronize(stream[0]));
        
        
        #ifdef _MPI
            CUDA_CHECK(cudaMemcpyAsync(sol, d_sol_temp, N_Active * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
            CUDA_CHECK(cudaStreamSynchronize(stream[0]));

            ParComm->CommUpdate(sol);
//                 ParComm->CommUpdate(sol);

            CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
        
        #else
            
            CUDA_CHECK(cudaMemcpyAsync(d_sol, d_sol_temp, N_Active * sizeof(double), cudaMemcpyDeviceToDevice,stream[0]));
            
        #endif    

        
        CUDA_CHECK(cudaStreamSynchronize(stream[0]));
        }

        #ifndef _MPI
        CUDA_CHECK(cudaMemcpyAsync(sol, d_sol_temp, N_Active * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
        CUDA_CHECK(cudaStreamSynchronize(stream[0]));
        #endif
            
//     CUDA_CHECK(cudaDeviceSynchronize());
    
    // for (i = 0; i < nStreams; ++i)
    //     CUDA_CHECK( cudaStreamDestroy(stream[i]) );

  // Free GPU memory
    // CUDA_CHECK(cudaFree(d_RowPtr));
    // CUDA_CHECK(cudaFree(d_KCol));
    // CUDA_CHECK(cudaFree(d_Entries));
    CUDA_CHECK(cudaFree(d_f));
    // CUDA_CHECK(cudaFree(d_sol));
    CUDA_CHECK(cudaFree(d_sol_temp));
    // #ifdef _MPI
    // CUDA_CHECK(cudaFree(d_master));
    // #endif
}

#ifdef _MPI
void TMGLevel3D::Jacobi_CPU_GPU(double *sol, double *f, double *aux, int N_Parameters, double *Parameters, int smooth, cudaStream_t *stream, int* d_RowPtr, int*  d_KCol, double* d_Entries, double* d_sol, int* d_master)
{
    double omega = Parameters[0];
  
    #ifdef _MPI
        int rank;
        MPI_Comm_rank(TDatabase::ParamDB->Comm, &rank);
        int *master =ParComm->GetMaster();
    #endif
  
    int ii, i,j,k,l,index,loop;
    int itr,jj,tid,nrows=0,numThreads,end;
    double s, t, diag;
    // int nStreams = 1;
    // cudaStream_t stream[nStreams];
    
    if(smooth == -1)
        end = TDatabase::ParamDB->SC_PRE_SMOOTH_SCALAR;
    else if(smooth == 0)
        end = TDatabase::ParamDB->SC_COARSE_MAXIT_SCALAR;
    else
        end = TDatabase::ParamDB->SC_POST_SMOOTH_SCALAR;

    int nz= A->GetN_Entries();
    int n = A->GetN_Rows();

    
    // for (i = 0; i < nStreams; ++i)
    //     CUDA_CHECK( cudaStreamCreate(&stream[i]) );
    
    int N_Active_gpu = ceil(0.8*N_Active);
    
    int N_Active_cpu = N_Active-N_Active_gpu;

  
   // Allocate data on GPU memory
    // int* d_RowPtr = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_RowPtr, (n+1) * sizeof(int)));
    
    // int* d_Reorder = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_Reorder, (N_InterfaceM + N_Int + N_Dept1 +  N_Dept2) * sizeof(int)));
    
    // #ifdef _MPI
    //     int* d_master = NULL;
    //     CUDA_CHECK(cudaMalloc((void**)&d_master, (N_DOF) * sizeof(int)));
    // #endif

    // int* d_KCol = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_KCol, nz * sizeof(int)));
    
    // double* d_Entries = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_Entries, nz * sizeof(double)));
    
    double* d_f = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_f, n * sizeof(double)));
    
    // double* d_sol = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_sol, n * sizeof(double)));
    
    double* d_sol_temp = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_sol_temp, N_Active * sizeof(double)));

    memcpy(sol+HangingNodeBound, f+HangingNodeBound, N_Dirichlet*sizeof(double));
                
    memcpy(aux, sol, N_DOF*SizeOfDouble);
    
    
    // Copy to GPU memory
    // CUDA_CHECK(cudaMemcpyAsync(d_RowPtr, RowPtr, (n+1) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_Entries, Entries, nz * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_KCol, KCol, nz * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_f, f, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    #ifdef _MPI
        CUDA_CHECK(cudaMemcpyAsync(d_master, master, N_DOF * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    #endif

    int num_blocks, merge_blocks;
    
//     num_blocks= ceil(N_Active/THREADS_PER_BLOCK) +1 ;
        
    num_blocks= ceil(N_Active_gpu * 32 /THREADS_PER_BLOCK) +1 ;

    omp_set_num_threads(TDatabase::ParamDB->OMPNUMTHREADS);
    
              
    if ( num_blocks == 0) num_blocks=1;
    
    
        
         for(itr=0;itr<end;itr++)
        {
                
    
                memcpy(sol+HangingNodeBound, f+HangingNodeBound, N_Dirichlet*sizeof(double));
                
                memcpy(aux, sol, N_DOF*SizeOfDouble);

                Jacobi_Kernel_Hyb<<<num_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(0, N_Active_gpu,d_RowPtr
                    #ifdef _MPI 
                    ,d_master 
                    #endif
                    ,d_KCol ,d_Entries ,d_f ,d_sol , d_sol_temp , omega
                                                                                #ifdef _MPI
                                                                                   ,rank);
                                                                                #else
                                                                                );
                                                                                #endif
    
            // Jacobi_Kernel_Hyb<<<num_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(0, N_Active_gpu,d_RowPtr ,d_master ,d_KCol ,d_Entries ,d_f ,d_sol , d_sol_temp , omega
            //                                                                 #ifdef _MPI
            //                                                                    ,rank);
            //                                                                 #endif
//         Jacobi_Scalar_Kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(0, N_Active,d_RowPtr ,d_master ,d_KCol ,d_Entries ,d_f ,d_sol , d_sol_temp, N_Active , omega, rank);
            
            

            j = RowPtr[0];

            #pragma omp parallel for private(s,k,diag,j,index,t)
            for(i=N_Active_gpu; i<N_Active;i++)
            {
            #ifdef _MPI
                if(master[i] != rank)
                continue;
            #endif
                s = f[i];
                k = RowPtr[i+1];
                for(j = RowPtr[i];j<k;j++)
                {
                index = KCol[j];
                if(index == i)
                    diag = Entries[j];
                else
                    s -= Entries[j] * aux[index];
                } // endfor j
                t = aux[i];
                sol[i] = (1-omega)*t + omega*s/diag;
            } // endfor i
            
            // set active nodes
            for(i=N_Active;i<HangingNodeBound;i++)
            {

                #ifdef _MPI
                    if(master[i] != rank)
                    continue;
                #endif

                s = f[i];
                k = RowPtr[i+1];
                for(j=RowPtr[i];j<k;j++)
                {
                index = KCol[j];
                if(index != i)
                s -= Entries[j] * sol[index];
                else
                diag = Entries[j];
                } // endfor j
                sol[i] = s/diag;
            } // endfor i
            
            


            #ifdef _MPI

                CUDA_CHECK(cudaMemcpyAsync(sol, d_sol_temp, N_Active_gpu * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
                CUDA_CHECK(cudaStreamSynchronize(stream[0]));
                
                ParComm->CommUpdate(sol);
        
                CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
                #else
            
                CUDA_CHECK(cudaMemcpyAsync(d_sol, d_sol_temp, N_Active * sizeof(double), cudaMemcpyDeviceToDevice,stream[0]));
                
            #endif    
    
            
            CUDA_CHECK(cudaStreamSynchronize(stream[0]));
            }
    
            #ifndef _MPI
            CUDA_CHECK(cudaMemcpyAsync(sol, d_sol_temp, N_Active * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
            CUDA_CHECK(cudaStreamSynchronize(stream[0]));
            #endif

            
            
    
//     CUDA_CHECK(cudaDeviceSynchronize());
    
    // for (i = 0; i < nStreams; ++i)
    //     CUDA_CHECK( cudaStreamDestroy(stream[i]) );

  // Free GPU memory
    // CUDA_CHECK(cudaFree(d_RowPtr));
    // CUDA_CHECK(cudaFree(d_KCol));
    // CUDA_CHECK(cudaFree(d_Entries));
    CUDA_CHECK(cudaFree(d_f));
    // CUDA_CHECK(cudaFree(d_sol));
    CUDA_CHECK(cudaFree(d_sol_temp));
    // #ifdef _MPI
    //     CUDA_CHECK(cudaFree(d_master));
    // #endif
}
#endif

#endif
// void SOR_Re_cpu_gpu(double *sol, double *f, int *RowPtr, int *KCol, double *Entries, TParFEMapper3D *ParMapper, TParFECommunicator3D *ParComm, int N_CMaster, int N_CDept1, int N_CDept2, int N_CInt, int *ptrCMaster, int *ptrCDept1, int *ptrCDept2, int *ptrCInt, int repeat, int end, int HangingNodeBound, int N_Dirichlet,int N_Int,int N_InterfaceM,int N_Dept1, int N_Dept2, int N_Active, double omega, int N_DOF ,int nz, int n, int rank)
// {
//     int ii, i,j,k,l,index,loop;
//     int itr,jj,tid,nrows=0,numThreads;
//     double s, t, diag;
//     int* Reorder;
//     int nStreams = 3;
//     cudaStream_t stream[nStreams];
// 
//     const int streamSize = N_DOF / nStreams;
// //     const int streamBytes = streamSize * sizeof(int);
//     
//     for (i = 0; i < nStreams; ++i)
//         CUDA_CHECK( cudaStreamCreate(&stream[i]) );
//   
//    // Allocate data on GPU memory
//     int* d_RowPtr = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_RowPtr, (n+1) * sizeof(int)));
//     
//     int* d_Reorder = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_Reorder, (N_InterfaceM + N_Int + N_Dept1 +  N_Dept2) * sizeof(int)));
//     
//     int* d_KCol = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_KCol, nz * sizeof(int)));
//     
//     double* d_Entries = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_Entries, nz * sizeof(double)));
//     
//     double* d_f = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_f, n * sizeof(double)));
//     
//     double* d_sol = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_sol, n * sizeof(double)));
//     
//     double* d_sol_temp = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_sol_temp, n * sizeof(double)));
//     
// //     double* d_sol_temp1 = NULL;
// //     CUDA_CHECK(cudaMalloc((void**)&d_sol_temp1, n * sizeof(double)));
// //     
// //     double* d_current = NULL;
//     
//     memcpy(sol+HangingNodeBound, f+HangingNodeBound, N_Dirichlet*sizeof(double));
//     
//     double *sol_t= (double *) malloc(sizeof(double) * n);
//     
//     
// // Copy to GPU memory
//     CUDA_CHECK(cudaMemcpyAsync(d_RowPtr, RowPtr, (n+1) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
//     
//     CUDA_CHECK(cudaMemcpyAsync(d_Entries, Entries, nz * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//     
//     CUDA_CHECK(cudaMemcpyAsync(d_KCol, KCol, nz * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
//     
//     CUDA_CHECK(cudaMemcpyAsync(d_f, f, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//     
//     CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//     
// //     CUDA_CHECK(cudaMemcpyAsync(d_sol_temp, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//     
//     Reorder = ParMapper->GetReorder_M();
// 
//     CUDA_CHECK( cudaMemcpyAsync(d_Reorder, Reorder, (N_InterfaceM + N_Int + N_Dept1 +  N_Dept2) * sizeof(int), cudaMemcpyHostToDevice, stream[0]) );
//     
// //     PUSH_RANGE("test",10)
// //     int c;
// //     for(int p=0; p<10000; p++){
// //         c=c*1.89;
// //     }
// //     POP_RANGE
//     
//     int merge_blocks, indp_blocks;
//     
//     for(itr=0;itr<end;itr++)
//     {
//       for(loop=0;loop<repeat;loop++)
//       {
//         
//       //########################################## MASTERS DOFS ########################################################//
// // 	    if(itr == 0)
// // 	    {
// //             PUSH_RANGE("master",1)
// // //             Reorder = ParMapper->GetReorder_M();
// //             
// //             for(ii=0;ii<N_CMaster;ii++)
// //             {
// //     //             #pragma omp for schedule(guided) 
// //                 for(jj=ptrCMaster[ii];jj<ptrCMaster[ii+1];jj++)
// //                 {
// //                 i = Reorder[jj];
// //                 if(i >= N_Active)     continue;
// //             
// //                 s = f[i];
// //                 k = RowPtr[i+1];
// //         //             if(RowPtr[i+1]- RowPtr[i] > 32)
// //         //                 cout<<"neha: no of nz master: "<<RowPtr[i+1]- RowPtr[i]<<endl;
// //                 for(j=RowPtr[i];j<k;j++)
// //                 {
// //                     index = KCol[j];
// //                     if(index == i)
// //                     {
// //                     diag = Entries[j];
// //                     }
// //                     else
// //                     {
// //                     s -= Entries[j] * sol[index];
// //                     }
// //                 } // endfor j
// //                 
// //                 t = sol[i];
// //                 sol[i] = omega*(s/diag-t) + t;
// //                 // cout << "sol[i]: " << sol[i] << endl;
// //                 } // endfor jj
// //             } //end for ii
// //                 
// //             if(loop == (repeat-1))
// //             {
// // 
// //                 ParComm->CommUpdateMS(sol);
// // 
// //             }
// //             
// //             POP_RANGE
// //             
// //             CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
// //     
// //         } //end firstTime
//         
//         
//         
// //         CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//         
//         for(ii=0;ii<N_CInt;ii++)
//         {
//         
//             indp_blocks= (((ptrCInt[ii+1] - ptrCInt[ii])*32)/THREADS_PER_BLOCK) + 1;
//             
//         
//             if ( indp_blocks == 0) indp_blocks=1;
//             
//             
//             SOR_Re_Kernel<<<indp_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCInt[ii], ptrCInt[ii+1], d_RowPtr, (d_Reorder + N_InterfaceM) ,d_KCol ,d_Entries ,d_f ,d_sol, N_Active, omega);
//         
//         }
//             
// // 	    if(itr != 0){  
//             
//             PUSH_RANGE("master",1)
//             Reorder = ParMapper->GetReorder_M();
//             for(ii=0;ii<N_CMaster;ii++)
//                 {
//         //             #pragma omp for schedule(guided) 
//                     for(jj=ptrCMaster[ii];jj<ptrCMaster[ii+1];jj++)
//                     {
//                     i = Reorder[jj];
//                     if(i >= N_Active)     continue;
//                 
//                     s = f[i];
//                     k = RowPtr[i+1];
//             //             if(RowPtr[i+1]- RowPtr[i] > 32)
//             //                 cout<<"neha: no of nz master: "<<RowPtr[i+1]- RowPtr[i]<<endl;
//                     for(j=RowPtr[i];j<k;j++)
//                     {
//                         index = KCol[j];
//                         if(index == i)
//                         {
//                         diag = Entries[j];
//                         }
//                         else
//                         {
//                         s -= Entries[j] * sol[index];
//                         }
//                     } // endfor j
//                     
//                     t = sol[i];
//                     sol[i] = omega*(s/diag-t) + t;
//                     // cout << "sol[i]: " << sol[i] << endl;
//                     } // endfor jj
//                 } //end for ii
//                     
//             if(loop == (repeat-1))
//             {
// 
//                 ParComm->CommUpdateMS(sol);
// 
//             }
//             
//             POP_RANGE
// //         }
//         //########################################## DEPENDENT2 DOFS #####################################################//
// 
//         if(N_Dept2 != 0){
//             PUSH_RANGE("D2",3)
// 
//             Reorder = ParMapper->GetReorder_D2();
//             for(ii=0;ii<N_CDept2;ii++)
//             {
// //             #pragma omp for schedule(guided) 
//             for(jj=ptrCDept2[ii];jj<ptrCDept2[ii+1];jj++)
//             {
//             i = Reorder[jj];
//             if(i >= N_Active)     continue;
//         
//             s = f[i];
//             k = RowPtr[i+1];
// //             if(RowPtr[i+1]- RowPtr[i] > 32)
// //                 cout<<"neha: no of nz d2: "<<RowPtr[i+1]- RowPtr[i]<<endl;
//             for(j=RowPtr[i];j<k;j++)
//             {
//             index = KCol[j];
//             if(index == i)
//             {
//             diag = Entries[j];
//             }
//             else
//             {
//             s -= Entries[j] * sol[index];    
//             }
//             } // endfor j
//             t = sol[i];
//             sol[i] = omega*(s/diag-t) + t;
//             // cout << "sol[i]: " << sol[i] << endl;
//             } // endfor jj
//         } //end for ii
//         
//             POP_RANGE
//         
//         merge_blocks = (N_Dept2/THREADS_PER_BLOCK ) + 1;
//         
//         if ( merge_blocks == 0) merge_blocks=1;
//         
//         CUDA_CHECK(cudaMemcpyAsync(d_sol_temp, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[1]));
// 	    mergeSolution<<<merge_blocks, THREADS_PER_BLOCK, 1, stream[1]>>>(d_sol ,d_sol_temp ,(d_Reorder + N_InterfaceM + N_Int + N_Dept1) , N_Dept2, N_Active);
// 
//         }        
//         //########################################## DEPENDENT1 DOFS #####################################################//
//         
//         if(N_Dept1 != 0){
//         PUSH_RANGE("D1",4)
// 	    Reorder = ParMapper->GetReorder_D1();
// 	    
//         for(ii=0;ii<N_CDept1;ii++)
//             {
// //             #pragma omp for schedule(guided) 
//             for(jj=ptrCDept1[ii];jj<ptrCDept1[ii+1];jj++)
//             {
//             i = Reorder[jj];
//             if(i >= N_Active)     continue;
//             
//             s = f[i];
//             k = RowPtr[i+1];
// //             if(RowPtr[i+1]- RowPtr[i] > 32)
// //                 cout<<"neha: no of nz d1: "<<RowPtr[i+1]- RowPtr[i]<<endl;
//             for(j=RowPtr[i];j<k;j++)
//             {
//             index = KCol[j];
//             if(index == i)
//             {
//                 diag = Entries[j];
//             }
//             else
//             {
//                 s -= Entries[j] * sol[index];
//             }
//             } // endfor j
//             t = sol[i];
//             sol[i] = omega*(s/diag-t) + t;
//             // cout << "sol[i]: " << sol[i] << endl;
//             } // endfor jj
//             } //end for ii  
// 	    
// 	    POP_RANGE
// 	    
// 	    merge_blocks = (N_Dept1/THREADS_PER_BLOCK) + 1;
//         if ( merge_blocks == 0) merge_blocks=1;
//         
//         CUDA_CHECK(cudaMemcpyAsync(d_sol_temp, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[2]));
// 	    mergeSolution<<<merge_blocks, THREADS_PER_BLOCK, 0, stream[2]>>>(d_sol ,d_sol_temp ,(d_Reorder + N_InterfaceM + N_Int) , N_Dept1, N_Active);
//         }
//         
// //         CUDA_CHECK(cudaMemcpyAsync(sol_t, d_sol, n * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
// 	    
//       //################################################################################################################//
// 	    
//         PUSH_RANGE("H1",5)
// 	    if(loop == (repeat-1)){
//             
//             ParComm->CommUpdateH1(sol);
// 	        
//         }
//         
//         POP_RANGE
//         
// //         CUDA_CHECK(cudaStreamSynchronize(stream[0]));
// //         CUDA_CHECK(cudaStreamSynchronize(stream[1]));
//         
// //         PUSH_RANGE("mergeCPU",6)
// //         
// //         Reorder = ParMapper->GetReorder_I();
// //         
// //         mergeSolutionCPU(sol, sol_t, Reorder, N_Int, N_Active );
// //         
// //         POP_RANGE
//         
//         
// //         CUDA_CHECK(cudaStreamSynchronize(stream[2]));
// //         
//         merge_blocks = (N_Int/THREADS_PER_BLOCK) + 1;
// 
//         if ( merge_blocks == 0) merge_blocks=1;
//         CUDA_CHECK(cudaMemcpyAsync(d_sol_temp, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
// 	    mergeSolution<<<merge_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(d_sol_temp ,d_sol ,(d_Reorder + N_InterfaceM) , N_Int, N_Active);
//         CUDA_CHECK(cudaMemcpyAsync(sol, d_sol_temp, n * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
// //         
//         
// 
//       //############################################# Hanging NODES ####################################################//  
// 	    // set hanging nodes
// 	    int *master = ParComm->GetMaster();
//         
//         
// 	    for(i=N_Active;i<HangingNodeBound;i++)
// 	    {
// 
//             if(master[i] != rank)
//             continue;
// 
//             s = f[i];
//             k = RowPtr[i+1];
//             for(j=RowPtr[i];j<k;j++)
//             {
//             index = KCol[j];
//             if(index != i)
//             s -= Entries[j] * sol[index];
//             else
//             diag = Entries[j];
//             } // endfor j
//             sol[i] = s/diag;
//         } // endfor i
// 	  
// // 	  CUDA_CHECK(cudaStreamSynchronize(stream[1]));
// //       CUDA_CHECK(cudaStreamSynchronize(stream[2]));
//       
//         
//       
//       
// // 	  CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
// // 	    CUDA_CHECK(cudaStreamSynchronize(stream[0]));
// 	  
// //     CUDA_CHECK(cudaStreamSynchronize(stream[1]));
// 
//     
// 
//       }//loop
//       
//     }
//     
//     CUDA_CHECK(cudaDeviceSynchronize());
//     
//     for (i = 0; i < nStreams; ++i)
//         CUDA_CHECK( cudaStreamDestroy(stream[i]) );
//   
//   // Free GPU memory
//     CUDA_CHECK(cudaFree(d_RowPtr));
//     CUDA_CHECK(cudaFree(d_Reorder));
//     CUDA_CHECK(cudaFree(d_KCol));
//     CUDA_CHECK(cudaFree(d_Entries));
//     CUDA_CHECK(cudaFree(d_f));
//     CUDA_CHECK(cudaFree(d_sol));
//     CUDA_CHECK(cudaFree(d_sol_temp));
// //     CUDA_CHECK(cudaFree(d_sol_temp1));
// }

#ifdef _HYBRID
#ifdef _MPI
void TMGLevel3D::SOR_Re_CPU_GPU(double *sol, double *f, double *aux, int N_Parameters, double *Parameters,int smooth, cudaStream_t *stream, int* d_RowPtr, int*  d_KCol, double* d_Entries, double* d_sol, int* d_Reorder)
{
//     cout<<"neha:cpu_gpu"<<endl;
    int ii, i,j,k,l,index,loop;
    int itr,jj,tid,nrows=0,numThreads, end;
    double s, t, diag;
    int* Reorder;
    int nStreams = 2;
    // cudaStream_t stream[nStreams];

    const int streamSize = N_DOF / nStreams;
//     const int streamBytes = streamSize * sizeof(int);
    
    // for (i = 0; i < nStreams; ++i)
    //     CUDA_CHECK( cudaStreamCreate(&stream[i]) );
    
    if(smooth == -1)
        end = TDatabase::ParamDB->SC_PRE_SMOOTH_SCALAR;
    else if(smooth == 0)
        end = TDatabase::ParamDB->SC_COARSE_MAXIT_SCALAR;
    else
        end = TDatabase::ParamDB->SC_POST_SMOOTH_SCALAR;

    int nz= A->GetN_Entries();
    int n = A->GetN_Rows();
    
    double omega = Parameters[0];
    int repeat = TDatabase::ParamDB->Par_P6;

    if(repeat <= 0)
        repeat = 1;
  
    #ifdef _MPI
    int rank;
    MPI_Comm_rank(TDatabase::ParamDB->Comm, &rank);
    #endif
  
   // Allocate data on GPU memory
    // int* d_RowPtr = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_RowPtr, (n+1) * sizeof(int)));
    
    // int* d_Reorder = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_Reorder, (N_InterfaceM + N_Int + N_Dept1 +  N_Dept2) * sizeof(int)));
    
    // int* d_KCol = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_KCol, nz * sizeof(int)));
    
    // double* d_Entries = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_Entries, nz * sizeof(double)));
    
    double* d_f = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_f, n * sizeof(double)));
    
    // double* d_sol = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_sol, n * sizeof(double)));
    
    double* d_sol_temp = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_sol_temp, n * sizeof(double)));
    
//     double* d_sol_temp1 = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_sol_temp1, n * sizeof(double)));
//     
//     double* d_current = NULL;
    
    memcpy(sol+HangingNodeBound, f+HangingNodeBound, N_Dirichlet*sizeof(double));
    
//     double *sol_t= (double *) malloc(sizeof(double) * n);
    
    
    
// Copy to GPU memory
    // CUDA_CHECK(cudaMemcpyAsync(d_RowPtr, RowPtr, (n+1) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_Entries, Entries, nz * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_KCol, KCol, nz * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_f, f, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
//     CUDA_CHECK(cudaMemcpyAsync(d_sol_temp, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    Reorder = ParMapper->GetReorder_M();

    CUDA_CHECK( cudaMemcpyAsync(d_Reorder, Reorder, (N_InterfaceM + N_Int + N_Dept1 +  N_Dept2) * sizeof(int), cudaMemcpyHostToDevice, stream[0]) );
    
//     PUSH_RANGE("test",10)
//     int c;
//     for(int p=0; p<10000; p++){
//         c=c*1.89;
//     }
//     POP_RANGE
    
    int merge_blocks, indp_blocks;
    
//     cout<<"N_Int:"<<(N_Int == ptrCInt[N_CInt])<<endl;
    float ind_ratio,master_ratio,dept1_ratio, dept2_ratio;
    int sub_itr=1;
    
    omp_set_num_threads(TDatabase::ParamDB->OMPNUMTHREADS);
    
    for(itr=0;itr<end;itr++)
    {
      for(loop=0;loop<repeat;loop++)
      {
        
          
        if( N_Int != 0){
        ind_ratio = N_Int*1.0/N_CInt;
        
        if(ind_ratio < 0){
                //         cout<<"ind_ratio:"<<ind_ratio<<endl;
            Reorder = ParMapper->GetReorder_I();
            for(ii=0;ii<N_Int;ii++)
            {
                i = Reorder[ii];
                if(i >= N_Active)     continue;
                
                s = f[i];
                k = RowPtr[i+1];
                for(j=RowPtr[i];j<k;j++)
                {
                index = KCol[j];
                if(index == i)
                {
                    diag = Entries[j];
                }
                else
                {
                    s -= Entries[j] * sol[index];
                }
                } // endfor j
                t = sol[i];
                sol[i] = omega*(s/diag-t) + t;
                // cout << "sol[i]: " << sol[i] << endl;
            } // endfor i
                
        }
        
        else{
            
        
        if ( ind_ratio > 100.0){
//             cout<<"ind_ratio:"<<ind_ratio<<endl;
            for(ii=0;ii<N_CInt;ii++)
            {
            
                indp_blocks= (((ptrCInt[ii+1] - ptrCInt[ii])*32)/THREADS_PER_BLOCK) + 1;
                
            
                if ( indp_blocks == 0) indp_blocks=1;
                
                
                SOR_Re_Kernel<<<indp_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCInt[ii], ptrCInt[ii+1], d_RowPtr, (d_Reorder + N_InterfaceM) ,d_KCol ,d_Entries ,d_f ,d_sol, N_Active, omega, sub_itr);
            
            }
        }
        
        else if(ind_ratio <= 100.0 ){
//             else{
            
//             cout<<"ind_ratio:"<<ind_ratio<<endl;
            if(( N_CInt % 2 ) == 0){
//                 cout<<"neha: even "<<endl;
                
                for(ii=0;ii<N_CInt;ii += N_CInt/2)
                {
                
                    indp_blocks = (((ptrCInt[ii + N_CInt/2] - ptrCInt[ii])*32)/THREADS_PER_BLOCK) + 1;
                    
                
//                     if ( indp_blocks == 0) indp_blocks=1;
                    
                    
                    SOR_Re_Kernel<<<indp_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCInt[ii], ptrCInt[ii + N_CInt/2], d_RowPtr, (d_Reorder + N_InterfaceM) ,d_KCol ,d_Entries ,d_f ,d_sol, N_Active, omega, sub_itr);
                
                }
            }
            
            else{
//                 cout<<"neha: odd "<<endl;
                for(ii=0;ii < N_CInt - 1 ;ii += N_CInt/2)
                {
                
                    indp_blocks= (((ptrCInt[ii + N_CInt/2] - ptrCInt[ii])*32)/THREADS_PER_BLOCK) + 1;
                    
                
                    if ( indp_blocks == 0) indp_blocks=1;
                    
                    
                    SOR_Re_Kernel<<<indp_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCInt[ii], ptrCInt[ii + N_CInt/2], d_RowPtr, (d_Reorder + N_InterfaceM) ,d_KCol ,d_Entries ,d_f ,d_sol, N_Active, omega, sub_itr);
                
                }
                
                indp_blocks= (((ptrCInt[ii + 1] - ptrCInt[ii])*32)/THREADS_PER_BLOCK) + 1;
                    
                    
                SOR_Re_Kernel<<<indp_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCInt[ii], ptrCInt[ii+1], d_RowPtr, (d_Reorder + N_InterfaceM) ,d_KCol ,d_Entries ,d_f ,d_sol, N_Active, omega, sub_itr);
                
            }
                

//             indp_blocks = (((ptrCInt[N_CInt] - ptrCInt[0])*32)/THREADS_PER_BLOCK) + 1;
//                     
//                     
//             SOR_Re_Kernel<<<indp_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCInt[0], ptrCInt[N_CInt], d_RowPtr, (d_Reorder + N_InterfaceM) ,d_KCol ,d_Entries ,d_f ,d_sol, N_Active, omega);
            
        }
        }
        

        }
            
                 
            if(N_InterfaceM != 0){
            Reorder = ParMapper->GetReorder_M();
            
            
            // master_ratio = N_InterfaceM*1.0/N_CMaster;
            
                
            PUSH_RANGE("master",1)
            
            #pragma omp parallel default(shared) private(ii,jj)
            {
                for(ii=0;ii<N_CMaster;ii++)
                    {

                        // cout<<"got threads:"<<omp_get_num_threads()<<endl;  
                        #pragma omp for schedule(dynamic)
                        for(jj=ptrCMaster[ii];jj<ptrCMaster[ii+1];jj++)
                        {
                        int i = Reorder[jj];
                        if(i >= N_Active)     continue;
                    
                        double s = f[i];
                        int k = RowPtr[i+1];
                        double diag;
                //             if(RowPtr[i+1]- RowPtr[i] > 32)
                //                 cout<<"neha: no of nz master: "<<RowPtr[i+1]- RowPtr[i]<<endl;
                        for(int j=RowPtr[i];j<k;j++)
                        {
                            int index = KCol[j];
                            if(index == i)
                            {
                            diag = Entries[j];
                            }
                            else
                            {
                            s -= Entries[j] * sol[index];
                            }
                        } // endfor j
                        
                        double t = sol[i];
                        sol[i] = omega*(s/diag-t) + t;
                        // cout << "sol[i]: " << sol[i] << endl;
                        } // endfor jj
                        }
                    } //end for ii
                POP_RANGE
                
                merge_blocks = (N_InterfaceM/THREADS_PER_BLOCK ) + 1;
                
                CUDA_CHECK(cudaMemcpyAsync(d_sol_temp, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[1]));
                mergeSolution<<<merge_blocks, THREADS_PER_BLOCK, 0, stream[1]>>>(d_sol ,d_sol_temp ,(d_Reorder) , N_InterfaceM, N_Active);
       
            }
            
        //########################################## DEPENDENT2 DOFS #####################################################//

        if(N_Dept2 != 0){
            
                for(ii=0;ii<N_CDept2;ii++)
                {
            
                int  d2_blocks= (((ptrCDept2[ii+1]-ptrCDept2[ii])*32)/THREADS_PER_BLOCK)+ 1;
        //             d2_blocks= (((ptrCDept2[ii+1]-ptrCDept2[ii])*1)/THREADS_PER_BLOCK) + 1;

                    
                    if ( d2_blocks == 0) d2_blocks=1;

                    
                    SOR_Re_Kernel<<<d2_blocks, THREADS_PER_BLOCK, 0, stream[1]>>>(ptrCDept2[ii], ptrCDept2[ii+1],d_RowPtr ,(d_Reorder + N_InterfaceM + N_Int + N_Dept1) ,d_KCol ,d_Entries ,d_f ,d_sol, N_Active, omega, sub_itr);
        //             SOR_Scalar_Kernel<<<d2_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCDept2[ii], ptrCDept2[ii+1],d_RowPtr ,(d_Reorder + N_InterfaceM + N_Int + N_Dept1) ,d_KCol ,d_Entries ,d_f ,d_sol, N_Active, omega);
                    

            
                }

        }
        
        PUSH_RANGE("MS",2)
            if(loop == (repeat-1))
            {

                ParComm->CommUpdateMS(sol);

            }
        POP_RANGE
        //########################################## DEPENDENT1 DOFS #####################################################//
        
        if(N_Dept1 != 0){
        PUSH_RANGE("D1",4)
	    
        Reorder = ParMapper->GetReorder_D1();
        #pragma omp parallel default(shared) private(ii,jj)
        {
            for(ii=0;ii<N_CDept1;ii++)
                {
                #pragma omp for schedule(dynamic)
                for(jj=ptrCDept1[ii];jj<ptrCDept1[ii+1];jj++)
                {
                int i = Reorder[jj];
                if(i >= N_Active)     continue;
                
                double s = f[i];
                int k = RowPtr[i+1];
                double diag;
    //             if(RowPtr[i+1]- RowPtr[i] > 32)
    //                 cout<<"neha: no of nz d1: "<<RowPtr[i+1]- RowPtr[i]<<endl;
                for(int j=RowPtr[i];j<k;j++)
                {
                int index = KCol[j];
                if(index == i)
                {
                    diag = Entries[j];
                }
                else
                {
                    s -= Entries[j] * sol[index];
                }
                } // endfor j
                double t = sol[i];
                sol[i] = omega*(s/diag-t) + t;
                // cout << "sol[i]: " << sol[i] << endl;
                } // endfor jj
                } //end for ii  
                }
	    
	    POP_RANGE
	    
                merge_blocks = (N_Dept1/THREADS_PER_BLOCK) + 1;
                
                CUDA_CHECK(cudaMemcpyAsync(d_sol_temp, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[1]));
                mergeSolution<<<merge_blocks, THREADS_PER_BLOCK, 0, stream[1]>>>(d_sol ,d_sol_temp ,(d_Reorder + N_InterfaceM + N_Int) , N_Dept1, N_Active);

        }
        

        PUSH_RANGE("H1",5)
	    if(loop == (repeat-1)){
            
            ParComm->CommUpdateH1(sol);
	        
        }
        
        POP_RANGE
        
        if( ind_ratio > 0.0){
            merge_blocks = (N_Int/THREADS_PER_BLOCK) + 1;

//         if ( merge_blocks == 0) merge_blocks=1;
            CUDA_CHECK(cudaMemcpyAsync(d_sol_temp, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
            mergeSolution<<<merge_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(d_sol_temp ,d_sol ,(d_Reorder + N_InterfaceM) , N_Int, N_Active);
//             CUDA_CHECK(cudaMemcpyAsync(sol, d_sol_temp, n * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
            
            
            merge_blocks = (N_Dept2/THREADS_PER_BLOCK) + 1;

//         if ( merge_blocks == 0) merge_blocks=1;
//             CUDA_CHECK(cudaMemcpyAsync(d_sol_temp, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[1]));
            mergeSolution<<<merge_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(d_sol_temp ,d_sol ,(d_Reorder + N_InterfaceM + N_Int + N_Dept1) , N_Dept2, N_Active);
            CUDA_CHECK(cudaMemcpyAsync(sol, d_sol_temp, n * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
            

        }
        

//         
        

      //############################################# Hanging NODES ####################################################//  
	    // set hanging nodes
	    int *master = ParComm->GetMaster();
        
        
	    for(i=N_Active;i<HangingNodeBound;i++)
	    {

            if(master[i] != rank)
            continue;
            s = f[i];
            k = RowPtr[i+1];
            for(j=RowPtr[i];j<k;j++)
            {
            index = KCol[j];
            if(index != i)
            s -= Entries[j] * sol[index];
            else
            diag = Entries[j];
            } // endfor j
            sol[i] = s/diag;
        } // endfor i
	  
	  CUDA_CHECK(cudaStreamSynchronize(stream[0]));
      CUDA_CHECK(cudaStreamSynchronize(stream[1]));
      
        

    

      }//loop
      
    }
    
//     CUDA_CHECK(cudaDeviceSynchronize());
    
    // for (i = 0; i < nStreams; ++i)
    //     CUDA_CHECK( cudaStreamDestroy(stream[i]) );
  
  // Free GPU memory
    // CUDA_CHECK(cudaFree(d_RowPtr));
    // CUDA_CHECK(cudaFree(d_Reorder));
    // CUDA_CHECK(cudaFree(d_KCol));
    // CUDA_CHECK(cudaFree(d_Entries));
    CUDA_CHECK(cudaFree(d_f));
    // CUDA_CHECK(cudaFree(d_sol));
    CUDA_CHECK(cudaFree(d_sol_temp));
//     CUDA_CHECK(cudaFree(d_sol_temp1));
}

void TMGLevel3D::SOR_Re_GPU(double *sol, double *f, double *aux, int N_Parameters, double *Parameters,int smooth, cudaStream_t *stream, int* d_RowPtr, int*  d_KCol, double* d_Entries, double* d_sol, int* d_Reorder)
{
//     cout<<"SOR_Re_Hyb"<<endl;
    int ii, i,j,k,l,index,loop;
    int itr,jj,tid,nrows=0,numThreads,end;
    double s, t, diag;
    int* Reorder;
    int nStreams = 1;
    // cudaStream_t stream[nStreams];
    
    // for (i = 0; i < nStreams; ++i)
    //     CUDA_CHECK( cudaStreamCreate(&stream[i]) );
    
    if(smooth == -1)
        end = TDatabase::ParamDB->SC_PRE_SMOOTH_SCALAR;
    else if(smooth == 0)
        end = TDatabase::ParamDB->SC_COARSE_MAXIT_SCALAR;
    else
        end = TDatabase::ParamDB->SC_POST_SMOOTH_SCALAR;

    int nz= A->GetN_Entries();
    int n = A->GetN_Rows();
    
    double omega = Parameters[0];
    int repeat = TDatabase::ParamDB->Par_P6;

    if(repeat <= 0)
        repeat = 1;
  
    #ifdef _MPI
    int rank;
    MPI_Comm_rank(TDatabase::ParamDB->Comm, &rank);
    #endif
  
   // Allocate data on GPU memory
    // int* d_RowPtr = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_RowPtr, (n+1) * sizeof(int)));
    
    // int* d_Reorder = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_Reorder, (N_InterfaceM + N_Int + N_Dept1 +  N_Dept2) * sizeof(int)));
    
    // int* d_KCol = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_KCol, nz * sizeof(int)));
    
    // double* d_Entries = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_Entries, nz * sizeof(double)));
    
    double* d_f = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_f, n * sizeof(double)));
    
    // double* d_sol = NULL;
    // CUDA_CHECK(cudaMalloc((void**)&d_sol, n * sizeof(double)));
    
//     double* d_sol_temp = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_sol_temp, n * sizeof(double)));
//     
//     double* d_current = NULL;
    
    memcpy(sol+HangingNodeBound, f+HangingNodeBound, N_Dirichlet*sizeof(double));
    
    // // Copy to GPU memory
    // CUDA_CHECK(cudaMemcpyAsync(d_RowPtr, RowPtr, (n+1) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_Entries, Entries, nz * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_KCol, KCol, nz * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_f, f, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
//     CUDA_CHECK(cudaMemcpyAsync(d_sol_temp, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
    
    Reorder = ParMapper->GetReorder_M();

    CUDA_CHECK( cudaMemcpyAsync(d_Reorder, Reorder, (N_InterfaceM + N_Int + N_Dept1 +  N_Dept2) * sizeof(int), cudaMemcpyHostToDevice, stream[0]) );
    
    double *sol_t= (double *) malloc(sizeof(double) * n);
    
    memcpy(sol_t, sol, n*sizeof(double));
    
    int master_blocks, indp_blocks, d1_blocks, d2_blocks;
    
    int sub_itr=1;
    
    
    CUDA_CHECK(cudaStreamSynchronize(stream[0]));
    
//     CUDA_CHECK(cudaMemcpyAsync(sol_t, d_sol, n * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
//     CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//     verify(sol_t,sol,n,0,rank);
    
//     CUDA_CHECK(cudaStreamSynchronize(stream[1]));
    
    for(itr=0;itr<end;itr++)
    {
      for(loop=0;loop<repeat;loop++)
      {
        //         CUDA_CHECK(cudaMemcpyAsync(d_sol+HangingNodeBound, f+HangingNodeBound, N_Dirichlet*sizeof(double), cudaMemcpyHostToDevice,stream[0]));
            
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             CUDA_CHECK(cudaMemcpyAsync(sol, d_sol, n * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
      //########################################## MASTERS DOFS ########################################################//
	    if(itr == 0)
	    {
            

// if(rank ==0)
	      for(ii=0;ii<N_CMaster;ii++)
	      {
              
              master_blocks= (((ptrCMaster[ii+1]-ptrCMaster[ii])*32)/THREADS_PER_BLOCK) + 1;
//               master_blocks= (((ptrCMaster[ii+1]-ptrCMaster[ii])*1.0)/THREADS_PER_BLOCK) + 1;
//               cout<<"neha master_blocks:"<<ptrCMaster[ii+1]-ptrCMaster[ii]<<endl;
              if ( master_blocks == 0) master_blocks=1;
              
              
              SOR_Re_Kernel<<<master_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCMaster[ii], ptrCMaster[ii+1],d_RowPtr ,d_Reorder ,d_KCol ,d_Entries ,d_f ,d_sol , N_Active , omega, sub_itr);
//               SOR_Scalar_Kernel<<<master_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCMaster[ii], ptrCMaster[ii+1],d_RowPtr ,d_Reorder ,d_KCol ,d_Entries ,d_f ,d_sol , N_Active , omega);
        
		
	      } //end for ii
	      
//         Reorder = ParMapper->GetReorder_M();
// //         if(rank ==0)
// 	      for(ii=0;ii<N_CMaster;ii++)
//       {
// // 	#pragma omp for schedule(guided) 
// 	for(jj=ptrCMaster[ii];jj<ptrCMaster[ii+1];jj++)
// 	{
// 	  i = Reorder[jj];
// //       if(i==34){
// //           cout<<"jj:"<<jj<<endl;
// //       }
//           if(i >= N_Active)     continue;
//             
//           
//           s = f[i];
//           k = RowPtr[i+1];
//                       if(RowPtr[i+1]- RowPtr[i] > 32)
//                 cout<<"neha: no of nz master: "<<RowPtr[i+1]- RowPtr[i]<<endl;
//           for(j=RowPtr[i];j<k;j++)
//           {
//               
//             index = KCol[j];
//             if(i==104){
//                   cout<<index<<endl;
//               }
//             if(index == i)
//             {
//               diag = Entries[j];
//             }
//             else
//             {
//               s -= (Entries[j] * sol[index]);
// //                 s -= sol[index];
//             }
//            } // endfor j
//  
//            t = sol[i];
//            sol[i] = omega*(s/diag-t) + t;
// //            sol[i] = s;
//            if(rank ==0)
//            cout <<std::setprecision(16)<< "sol[i]: " << sol[i] <<" "<<ii<<endl;
//         } // endfor jj
//       } //end for ii
//       
//       CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             CUDA_CHECK(cudaMemcpyAsync(sol_t, d_sol, n * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             if(rank ==0)
//             verify(sol_t,sol,n,0,rank);
            
          if(loop == (repeat-1)){
              
            // CUDA_CHECK(cudaStreamSynchronize(stream[0]));
            CUDA_CHECK(cudaMemcpyAsync(sol, d_sol, n * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
            CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             
//             verify(sol_t,sol,n,0,rank);
            PUSH_RANGE("master",1)
            ParComm->CommUpdateMS(sol);
            POP_RANGE
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
            CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
            CUDA_CHECK(cudaStreamSynchronize(stream[0]));
              
          }
	      
	    } //end firstTime
	    
        
        for(ii=0;ii<N_CDept1;ii++)
	    {
            
            d1_blocks= (((ptrCDept1[ii+1]-ptrCDept1[ii])*32)/THREADS_PER_BLOCK)+ 1;
//             d1_blocks= (((ptrCDept1[ii+1]-ptrCDept1[ii])*1)/THREADS_PER_BLOCK) + 1;
            
            if ( d1_blocks == 0) d1_blocks=1;
        
            SOR_Re_Kernel<<<d1_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCDept1[ii], ptrCDept1[ii+1],d_RowPtr ,( d_Reorder + N_InterfaceM + N_Int) ,d_KCol ,d_Entries ,d_f, d_sol, N_Active, omega, sub_itr);
//             SOR_Scalar_Kernel<<<d1_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCDept1[ii], ptrCDept1[ii+1],d_RowPtr ,( d_Reorder + N_InterfaceM + N_Int) ,d_KCol ,d_Entries ,d_f, d_sol, N_Active, omega);


        } //end for ii  
        
        if(loop == (repeat-1)){
            
            // CUDA_CHECK(cudaStreamSynchronize(stream[0]));
            CUDA_CHECK(cudaMemcpyAsync(sol, d_sol, n * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
            CUDA_CHECK(cudaStreamSynchronize(stream[0]));
            PUSH_RANGE("H1",2)
            ParComm->CommUpdateH1(sol);
            POP_RANGE
            
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
            CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
            CUDA_CHECK(cudaStreamSynchronize(stream[0]));
	        
        }
        
        //########################################## DEPENDENT2 DOFS #####################################################//


        for(ii=0;ii<N_CDept2;ii++)
	    {
    
            d2_blocks= (((ptrCDept2[ii+1]-ptrCDept2[ii])*32)/THREADS_PER_BLOCK)+ 1;
//             d2_blocks= (((ptrCDept2[ii+1]-ptrCDept2[ii])*1)/THREADS_PER_BLOCK) + 1;

            
            if ( d2_blocks == 0) d2_blocks=1;

            
            SOR_Re_Kernel<<<d2_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCDept2[ii], ptrCDept2[ii+1],d_RowPtr ,(d_Reorder + N_InterfaceM + N_Int + N_Dept1) ,d_KCol ,d_Entries ,d_f ,d_sol, N_Active, omega, sub_itr);
//             SOR_Scalar_Kernel<<<d2_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCDept2[ii], ptrCDept2[ii+1],d_RowPtr ,(d_Reorder + N_InterfaceM + N_Int + N_Dept1) ,d_KCol ,d_Entries ,d_f ,d_sol, N_Active, omega);
            

    
        }
//         Reorder = ParMapper->GetReorder_D2();
//                     for(ii=0;ii<N_CDept2;ii++)
//             {
// //             #pragma omp for schedule(dynamic) 
//             for(jj=ptrCDept2[ii];jj<ptrCDept2[ii+1];jj++)
//             {
//             i = Reorder[jj];
//       #ifdef _MPI      if(i >= N_Active)     continue;
//         
//             s = f[i];
//             k = RowPtr[i+1];
// //             if(RowPtr[i+1]- RowPtr[i] > 32)
// //                 cout<<"neha: no of nz d2: "<<RowPtr[i+1]- RowPtr[i]<<endl;
//             for(j=RowPtr[i];j<k;j++)
//             {
//             index = KCol[j];
//             if(index == i)
//             {
//             diag = Entries[j];
//             }
//             else
//             {
//             s -= Entries[j] * sol[index];    
//             }
//             } // endfor j
//             t = sol[i];
//             sol[i] = omega*(s/diag-t) + t;
//             // cout << "sol[i]: " << sol[i] << endl;
//             } // endfor jj
//         } //end for ii
        

        
        //########################################## DEPENDENT1 DOFS #####################################################//
// 	    Reorder = ParMapper->GetReorder_D1();
	    
        
// 	    Reorder = ParMapper->GetReorder_D1();
// 	                for(ii=0;ii<N_CDept1;ii++)
//             {
// //             #pragma omp for schedule(dynamic) 
//             for(jj=ptrCDept1[ii];jj<ptrCDept1[ii+1];jj++)
//             {
//             i = Reorder[jj];
//             if(i >= N_Active)     continue;
//             
//             s = f[i];
//             k = RowPtr[i+1];
// //             if(RowPtr[i+1]- RowPtr[i] > 32)
// //                 cout<<"neha: no of nz d1: "<<RowPtr[i+1]- RowPtr[i]<<endl;
//             for(j=RowPtr[i];j<k;j++)
//             {
//             index = KCol[j];
//             if(index == i)
//             {
//                 diag = Entries[j];
//             }
//             else
//             {
//                 s -= Entries[j] * sol[index];
//             }
//             } // endfor j
//             t = sol[i];
//             sol[i] = omega*(s/diag-t) + t;
//             // cout << "sol[i]: " << sol[i] << endl;
//             } // endfor jj
//             } //end for ii  
	    
	    
	    
      //################################################################################################################//
	    
	    
      

        
//         //########################################## MASTERS DOFS ########################################################//
	    if(itr!=(end-1))
	    {

            for(ii=0;ii<N_CMaster;ii++)
	      {
              
              master_blocks= (((ptrCMaster[ii+1]-ptrCMaster[ii])*32)/THREADS_PER_BLOCK) + 1;
//               master_blocks= (((ptrCMaster[ii+1]-ptrCMaster[ii])*1.0)/THREADS_PER_BLOCK) + 1;
//               cout<<"neha master_blocks:"<<ptrCMaster[ii+1]-ptrCMaster[ii]<<endl;
              if ( master_blocks == 0) master_blocks=1;
              
              
              SOR_Re_Kernel<<<master_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCMaster[ii], ptrCMaster[ii+1],d_RowPtr ,d_Reorder ,d_KCol ,d_Entries ,d_f ,d_sol , N_Active , omega, sub_itr);
//               SOR_Scalar_Kernel<<<master_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCMaster[ii], ptrCMaster[ii+1],d_RowPtr ,d_Reorder ,d_KCol ,d_Entries ,d_f ,d_sol , N_Active , omega);
        
		
	      } //end for ii
//             
//             Reorder = ParMapper->GetReorder_M();
//             for(ii=0;ii<N_CMaster;ii++)
//             {
//                 
// //                 master_blocks= (((ptrCMaster[ii+1]-ptrCMaster[ii])*32)/THREADS_PER_BLOCK)+ 1;
//                 master_blocks= (((ptrCMaster[ii+1]-ptrCMaster[ii])*1)/THREADS_PER_BLOCK) + 1;
//                 
//                 if ( master_blocks == 0) master_blocks=1;
//                 
// //                 SOR_Re_Kernel<<<master_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCMaster[ii], ptrCMaster[ii+1],d_RowPtr ,d_Reorder ,d_KCol ,d_Entries ,d_f ,d_sol  ,N_Active, omega);
//                 SOR_Scalar_Kernel<<<master_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCMaster[ii], ptrCMaster[ii+1],d_RowPtr ,d_Reorder ,d_KCol ,d_Entries ,d_f ,d_sol  ,N_Active, omega);
//                 
// 
//             
//             
//             } //end for ii
//             
// //             Reorder = ParMapper->GetReorder_M();
// //             for(ii=0;ii<N_CMaster;ii++)
// //             {
// // //             #pragma omp for schedule(dynamic) 
// //             for(jj=ptrCMaster[ii];jj<ptrCMaster[ii+1];jj++)
// //             {
// //             i = Reorder[jj];
// //             if(i >= N_Active)     continue;
// //         
// //             s = f[i];
// //             k = RowPtr[i+1];
// // //             if(RowPtr[i+1]- RowPtr[i] > 32)
// // //                 cout<<"neha: no of nz master: "<<RowPtr[i+1]- RowPtr[i]<<endl;
// //             for(j=RowPtr[i];j<k;j++)
// //             {
// //             index = KCol[j];
// //             if(index == i)
// //             {
// //                 diag = Entries[j];
// //             }
// //             else
// //             {
// //                 s -= Entries[j] * sol[index];
// //             }
// //             } // endfor j
// //         
// //             t = sol[i];
// //             sol[i] = omega*(s/diag-t) + t;
// //             // cout << "sol[i]: " << sol[i] << endl;
// //             } // endfor jj
// //             } //end for ii
//             
// 
//             
//             if(loop == (repeat-1)){
//                 
//                 CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//                 CUDA_CHECK(cudaMemcpyAsync(sol, d_sol, n * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
//                 CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//                 
//                 ParComm->CommUpdateMS(sol);
//                 
// //                 CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//                 CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//                 CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//                 
//             }
//             
        } //end !lastTime

        if(itr!=(end-1))
	    {

            if(loop == (repeat-1)){
            // CUDA_CHECK(cudaStreamSynchronize(stream[0]));
            CUDA_CHECK(cudaMemcpyAsync(sol, d_sol, n * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
            CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             
//             verify(sol_t,sol,n,0,rank);
            PUSH_RANGE("master",1)
            ParComm->CommUpdateMS(sol);
            POP_RANGE
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
            CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
            CUDA_CHECK(cudaStreamSynchronize(stream[0]));
            }

        }


        for(ii=0;ii<N_CInt;ii++)
	    {
        
            indp_blocks= (((ptrCInt[ii+1] - ptrCInt[ii])*32)/THREADS_PER_BLOCK)+ 1;
            
//             cout<<"ind blocks:"<<indp_blocks<<endl;
//             indp_blocks= (((ptrCInt[ii+1] - ptrCInt[ii])*1)/THREADS_PER_BLOCK) + 1;
            
        
            if ( indp_blocks == 0) indp_blocks=1;
            
            
            IND_kernel<<<indp_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCInt[ii], ptrCInt[ii+1], d_RowPtr, (d_Reorder + N_InterfaceM) ,d_KCol ,d_Entries ,d_f ,d_sol, N_Active, omega, sub_itr);
//             SOR_Scalar_Kernel<<<indp_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCInt[ii], ptrCInt[ii+1], d_RowPtr, (d_Reorder + N_InterfaceM) ,d_KCol ,d_Entries ,d_f ,d_sol, N_Active, omega);
        
        }
      //################################################################################################################//

//         CUDA_CHECK(cudaStreamSynchronize(stream[1]));

//         Reorder = ParMapper->GetReorder_I();
//                     for(ii=0;ii<N_CInt;ii++)
//             {
// //             #pragma omp for schedule(dynamic) 
//             for(jj=ptrCInt[ii];jj<ptrCInt[ii+1];jj++)
//             {
//             i = Reorder[jj];
//             if(i >= N_Active)     continue;
//             
//             s = f[i];
//             k = RowPtr[i+1];
// //             if(RowPtr[i+1]- RowPtr[i] > 32)
// //                 cout<<"neha: no of nz indp: "<<RowPtr[i+1]- RowPtr[i]<<endl;
//             for(j=RowPtr[i];j<k;j++)
//             {
//             index = KCol[j];
//             if(index == i)
//             {
//             diag = Entries[j];
//             }
//             else
//             {
//             s -= Entries[j] * sol[index];
//             }
//             } // endfor jj
//             t = sol[i];
//             sol[i] = omega*(s/diag-t) + t;
//             // cout << "sol[i]: " << sol[i] << endl;
//     //           cout << "sol[i]: indp" << sol[i] <<" iter "<<itr<<" i "<<i<< endl;
//             } // endfor jj
//         } //endfor ii
//         
//               CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             CUDA_CHECK(cudaMemcpyAsync(sol_t, d_sol, n * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             if(rank ==0)
//             verify(sol_t,sol,n,0,rank);
      //############################################# Hanging NODES ####################################################//  
	    // set hanging nodes
	    int *master = ParComm->GetMaster();
        
        
	    for(i=N_Active;i<HangingNodeBound;i++)
	    {

            if(master[i] != rank)
            continue;

            s = f[i];
            k = RowPtr[i+1];
            for(j=RowPtr[i];j<k;j++)
            {
            index = KCol[j];
            if(index != i)
            s -= Entries[j] * sol[index];
            else
            diag = Entries[j];
            } // endfor j
            sol[i] = s/diag;
        } // endfor i
	  
	  
// 	  CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
// 	    CUDA_CHECK(cudaStreamSynchronize(stream[0]));
	  
    CUDA_CHECK(cudaStreamSynchronize(stream[0]));

      }//loop
      
    }
    
//     CUDA_CHECK(cudaDeviceSynchronize());
    
    // for (i = 0; i < nStreams; ++i)
    //     CUDA_CHECK( cudaStreamDestroy(stream[i]) );
  
  // Free GPU memory
    // CUDA_CHECK(cudaFree(d_RowPtr));
    // CUDA_CHECK(cudaFree(d_Reorder));
    // CUDA_CHECK(cudaFree(d_KCol));
    // CUDA_CHECK(cudaFree(d_Entries));
    CUDA_CHECK(cudaFree(d_f));
    // CUDA_CHECK(cudaFree(d_sol));
//     CUDA_CHECK(cudaFree(d_sol_temp));
}

// void TMGLevel3D::SOR_Re_Level_Split(double *sol, double *f, double *aux, int N_Parameters, double *Parameters, int smooth)
// {
//     int split_level=1;
    
//     if(Level > split_level){
//         SOR_Re_GPU(sol, f, aux, N_Parameters, Parameters, smooth);
//     }
//     else{
//         int end;
        
//         if(smooth == -1)
//             end = TDatabase::ParamDB->SC_PRE_SMOOTH_SCALAR;
//         else if(smooth == 0)
//             end = TDatabase::ParamDB->SC_COARSE_MAXIT_SCALAR;
//         else
//             end = TDatabase::ParamDB->SC_POST_SMOOTH_SCALAR;
    
//         for(int j=0;j<end;j++)
//         {
//             SOR_Re(sol, f, aux, N_Parameters, Parameters);
//         }
//     }
    
// }

// void TMGLevel3D::Jacobi_Level_Split(double *sol, double *f, double *aux, int N_Parameters, double *Parameters, int smooth)
// {
//     int split_level=1;
        
//     if(Level > split_level){
//         Jacobi_GPU(sol, f, aux, N_Parameters, Parameters, smooth);
//     }
//     else{
//         int end;
        
//         if(smooth == -1)
//             end = TDatabase::ParamDB->SC_PRE_SMOOTH_SCALAR;
//         else if(smooth == 0)
//             end = TDatabase::ParamDB->SC_COARSE_MAXIT_SCALAR;
//         else
//             end = TDatabase::ParamDB->SC_POST_SMOOTH_SCALAR;
        
//         for(int j=0;j<end;j++)
//         {
//             Jacobi(sol, f, aux, N_Parameters, Parameters);
//             #ifdef _MPI  
//                     ParComm->CommUpdate(sol);
//             #endif
//         }
//     }
    
// }

void TMGLevel3D::SOR_Re_Combo(double *sol, double *f, double *aux, int N_Parameters, double *Parameters, int smooth, cudaStream_t *stream, int* d_RowPtr, int*  d_KCol, double* d_Entries, double* d_sol, int* d_master)
{
    int ind_ratio;
    if(N_Int != 0){
       ind_ratio = N_Int/N_CInt; 
    }
    if(ind_ratio > 1000){
        cout<<"level:"<<Level<<endl;
        cout<<"SOR_Re_GPU"<<endl;
        SOR_Re_GPU(sol, f, aux, N_Parameters, Parameters, smooth, stream, d_RowPtr, d_KCol, d_Entries, d_sol, d_master);
    }
    else if(ind_ratio > 500 && ind_ratio <=1000){
        cout<<"level:"<<Level<<endl;
        cout<<"SOR_Re_CPU_GPU"<<endl;
        SOR_Re_CPU_GPU(sol, f, aux, N_Parameters, Parameters, smooth, stream, d_RowPtr, d_KCol, d_Entries, d_sol, d_master);
    }
    else{
        cout<<"level:"<<Level<<endl;
        cout<<"SOR_Re_CPU"<<endl;
        int end;
        
        if(smooth == -1)
            end = TDatabase::ParamDB->SC_PRE_SMOOTH_SCALAR;
        else if(smooth == 0)
            end = TDatabase::ParamDB->SC_COARSE_MAXIT_SCALAR;
        else
            end = TDatabase::ParamDB->SC_POST_SMOOTH_SCALAR;
    
        for(int j=0;j<end;j++)
        {
            SOR_Re(sol, f, aux, N_Parameters, Parameters);
        }
    }
    
}


void TMGLevel3D::Jacobi_Combo(double *sol, double *f, double *aux, int N_Parameters, double *Parameters, int smooth, cudaStream_t *stream, int* d_RowPtr, int*  d_KCol, double* d_Entries, double* d_sol, int* d_master)
{

    #ifdef _MPI
    int rank;
    MPI_Comm_rank(TDatabase::ParamDB->Comm, &rank);
    
    #endif
    if(N_Active > 100000){
        
        #ifdef _MPI
        if(rank==0){
            cout<<"Jacobi_GPU"<<endl;
        }
        #endif
        
        Jacobi_GPU(sol, f, aux, N_Parameters, Parameters, smooth, stream, d_RowPtr, d_KCol, d_Entries, d_sol, d_master);
    }
    else if(N_Active > 10000 && N_Active <=100000){
        
        #ifdef _MPI
        if(rank==0){
            cout<<"Jacobi_CPU_GPU"<<endl;
        }
        #endif
        
        Jacobi_CPU_GPU(sol, f, aux, N_Parameters, Parameters, smooth, stream, d_RowPtr, d_KCol, d_Entries, d_sol, d_master);
    }
    else{
        
        #ifdef _MPI
        if(rank==0){
            cout<<"Jacobi"<<endl;
        }
        #endif
        int end;
        
        if(smooth == -1)
            end = TDatabase::ParamDB->SC_PRE_SMOOTH_SCALAR;
        else if(smooth == 0)
            end = TDatabase::ParamDB->SC_COARSE_MAXIT_SCALAR;
        else
            end = TDatabase::ParamDB->SC_POST_SMOOTH_SCALAR;
    
        for(int j=0;j<end;j++)
        {
            Jacobi(sol, f, aux, N_Parameters, Parameters);
            #ifdef _MPI  
                ParComm->CommUpdate(sol);
            #endif
        }
    }
    
}
#endif
#endif

// void SOR_Re_Hyb(double *sol, double *f, int *RowPtr, int *KCol, double *Entries, TParFEMapper3D *ParMapper, TParFECommunicator3D *ParComm, int N_CMaster, int N_CDept1, int N_CDept2, int N_CInt, int *ptrCMaster, int *ptrCDept1, int *ptrCDept2, int *ptrCInt, int repeat, int end, int HangingNodeBound, int N_Dirichlet,int N_Int,int N_InterfaceM,int N_Dept1, int N_Dept2, int N_Active, double omega, int N_DOF ,int nz, int n, int rank)
// {
//     int ii, i,j,k,l,index,loop;
//     int itr,jj,tid,nrows=0,numThreads;
//     double s, t, diag;
//     int* Reorder;
//     int nStreams = 1;
//     cudaStream_t stream[nStreams];
// 
//     const int streamSize = N_DOF / nStreams;
// //     const int streamBytes = streamSize * sizeof(int);
//     
//     for (i = 0; i < nStreams; ++i)
//         CUDA_CHECK( cudaStreamCreate(&stream[i]) );
//   
//    // Allocate data on GPU memory
//     int* d_RowPtr = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_RowPtr, (n+1) * sizeof(int)));
//     
//     int* d_Reorder = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_Reorder, (N_InterfaceM + N_Int + N_Dept1 +  N_Dept2) * sizeof(int)));
//     
//     int* d_KCol = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_KCol, nz * sizeof(int)));
//     
//     double* d_Entries = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_Entries, nz * sizeof(double)));
//     
//     double* d_f = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_f, n * sizeof(double)));
//     
//     double* d_sol = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_sol, n * sizeof(double)));
//     
// //     double* d_sol_temp = NULL;
// //     CUDA_CHECK(cudaMalloc((void**)&d_sol_temp, n * sizeof(double)));
// //     
// //     double* d_current = NULL;
//     
//     memcpy(sol+HangingNodeBound, f+HangingNodeBound, N_Dirichlet*sizeof(double));
//     
//     // Copy to GPU memory
//     CUDA_CHECK(cudaMemcpyAsync(d_RowPtr, RowPtr, (n+1) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
//     
//     CUDA_CHECK(cudaMemcpyAsync(d_Entries, Entries, nz * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//     
//     CUDA_CHECK(cudaMemcpyAsync(d_KCol, KCol, nz * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
//     
//     CUDA_CHECK(cudaMemcpyAsync(d_f, f, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//     
//     CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//     
// //     CUDA_CHECK(cudaMemcpyAsync(d_sol_temp, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//     
//     Reorder = ParMapper->GetReorder_M();
// 
//     CUDA_CHECK( cudaMemcpyAsync(d_Reorder, Reorder, (N_InterfaceM + N_Int + N_Dept1 +  N_Dept2) * sizeof(int), cudaMemcpyHostToDevice, stream[0]) );
//     
// //     double *sol_t= (double *) malloc(sizeof(double) * n);
//     
//     int master_blocks, indp_blocks, d1_blocks, d2_blocks;
//     
//     int sub_itr=1;
//     
//     
//     CUDA_CHECK(cudaStreamSynchronize(stream[0]));
// //     CUDA_CHECK(cudaStreamSynchronize(stream[1]));
//     
//     for(itr=0;itr<end;itr++)
//     {
//       for(loop=0;loop<repeat;loop++)
//       {
//         
//       //########################################## MASTERS DOFS ########################################################//
// 	    if(itr == 0)
// 	    {
//             
// 
// 
// 	      for(ii=0;ii<N_CMaster;ii++)
// 	      {
//               
//               master_blocks= ceil(((ptrCMaster[ii+1]-ptrCMaster[ii])*32)/THREADS_PER_BLOCK) + 1;
// //               master_blocks= ceil(((ptrCMaster[ii+1]-ptrCMaster[ii])*1)/THREADS_PER_BLOCK);
//               
//               if ( master_blocks == 0) master_blocks=1;
//               
//               
//               SOR_Re_Kernel<<<master_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCMaster[ii], ptrCMaster[ii+1],d_RowPtr ,d_Reorder ,d_KCol ,d_Entries ,d_f ,d_sol , N_Active , omega, sub_itr);
// //               SOR_Scalar_Kernel<<<master_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCMaster[ii], ptrCMaster[ii+1],d_RowPtr ,d_Reorder ,d_KCol ,d_Entries ,d_f ,d_sol , N_Active , omega);
//         
// 		
// 	      } //end for ii
//           
//           if(loop == (repeat-1)){
//               
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             CUDA_CHECK(cudaMemcpyAsync(sol, d_sol, n * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             
//             ParComm->CommUpdateMS(sol);
//             
// //             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//               
//           }
// 	      
// 	    } //end firstTime
// 	    
//         //########################################## DEPENDENT2 DOFS #####################################################//
// 
// 
//         for(ii=0;ii<N_CDept2;ii++)
// 	    {
//     
//             d2_blocks= ceil(((ptrCDept2[ii+1]-ptrCDept2[ii])*32)/THREADS_PER_BLOCK);
// //             d2_blocks= ceil(((ptrCDept2[ii+1]-ptrCDept2[ii])*1)/THREADS_PER_BLOCK);
// 
//             
//             if ( d2_blocks == 0) d2_blocks=1;
// 
//             
//             SOR_Re_Kernel<<<d2_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCDept2[ii], ptrCDept2[ii+1],d_RowPtr ,(d_Reorder + N_InterfaceM + N_Int + N_Dept1) ,d_KCol ,d_Entries ,d_f ,d_sol, N_Active, omega, sub_itr);
// //             SOR_Scalar_Kernel<<<d2_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCDept2[ii], ptrCDept2[ii+1],d_RowPtr ,(d_Reorder + N_InterfaceM + N_Int + N_Dept1) ,d_KCol ,d_Entries ,d_f ,d_sol, N_Active, omega);
//             
// 
//     
//         }
//         
//         //########################################## DEPENDENT1 DOFS #####################################################//
// // 	    Reorder = ParMapper->GetReorder_D1();
// 	    
//         for(ii=0;ii<N_CDept1;ii++)
// 	    {
//             
//             d1_blocks= ceil(((ptrCDept1[ii+1]-ptrCDept1[ii])*32)/THREADS_PER_BLOCK);
// //             d1_blocks= ceil(((ptrCDept1[ii+1]-ptrCDept1[ii])*1)/THREADS_PER_BLOCK);
//             
//             if ( d1_blocks == 0) d1_blocks=1;
//         
//             SOR_Re_Kernel<<<d1_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCDept1[ii], ptrCDept1[ii+1],d_RowPtr ,( d_Reorder + N_InterfaceM + N_Int) ,d_KCol ,d_Entries ,d_f, d_sol, N_Active, omega, sub_itr);
// //             SOR_Scalar_Kernel<<<d1_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCDept1[ii], ptrCDept1[ii+1],d_RowPtr ,( d_Reorder + N_InterfaceM + N_Int) ,d_KCol ,d_Entries ,d_f, d_sol, N_Active, omega);
// 
// 
// 	    } //end for ii  
// 	    
// 	    
// 	    
//       //################################################################################################################//
// 	    
// 	    if(loop == (repeat-1)){
//             
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             CUDA_CHECK(cudaMemcpyAsync(sol, d_sol, n * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             
//             ParComm->CommUpdateH1(sol);
//             
// //             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
// 	        
//         }
//       
// 
//         
//         //########################################## MASTERS DOFS ########################################################//
// 	    if(itr!=(end-1) && itr != 0)
// 	    {
//             
// //             Reorder = ParMapper->GetReorder_M();
//             for(ii=0;ii<N_CMaster;ii++)
//             {
//                 
//                 master_blocks= ceil(((ptrCMaster[ii+1]-ptrCMaster[ii])*32)/THREADS_PER_BLOCK);
// //                 master_blocks= ceil(((ptrCMaster[ii+1]-ptrCMaster[ii])*1)/THREADS_PER_BLOCK);
//                 
//                 if ( master_blocks == 0) master_blocks=1;
//                 
//                 SOR_Re_Kernel<<<master_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCMaster[ii], ptrCMaster[ii+1],d_RowPtr ,d_Reorder ,d_KCol ,d_Entries ,d_f ,d_sol  ,N_Active, omega, sub_itr);
// //                 SOR_Scalar_Kernel<<<master_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCMaster[ii], ptrCMaster[ii+1],d_RowPtr ,d_Reorder ,d_KCol ,d_Entries ,d_f ,d_sol  ,N_Active, omega);
//                 
// 
//             
//             
//             } //end for ii
//             
// 
//             
//             if(loop == (repeat-1)){
//                 
//                 CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//                 CUDA_CHECK(cudaMemcpyAsync(sol, d_sol, n * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
//                 CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//                 
//                 ParComm->CommUpdateMS(sol);
//                 
// //                 CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//                 CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//                 CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//                 
//             }
//             
//         } //end !lastTime
//       //################################################################################################################//
// 
// //         CUDA_CHECK(cudaStreamSynchronize(stream[1]));
//         for(ii=0;ii<N_CInt;ii++)
// 	    {
//         
//             indp_blocks= ceil(((ptrCInt[ii+1] - ptrCInt[ii])*32)/THREADS_PER_BLOCK);
//             
// //             cout<<"ind blocks:"<<indp_blocks<<endl;
// //             indp_blocks= ceil(((ptrCInt[ii+1] - ptrCInt[ii])*1)/THREADS_PER_BLOCK);
//             
//         
//             if ( indp_blocks == 0) indp_blocks=1;
//             
//             
//             SOR_Re_Kernel<<<indp_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCInt[ii], ptrCInt[ii+1], d_RowPtr, (d_Reorder + N_InterfaceM) ,d_KCol ,d_Entries ,d_f ,d_sol, N_Active, omega, sub_itr);
// //             SOR_Scalar_Kernel<<<indp_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCInt[ii], ptrCInt[ii+1], d_RowPtr, (d_Reorder + N_InterfaceM) ,d_KCol ,d_Entries ,d_f ,d_sol, N_Active, omega);
//         
//         }
//         
// 
//       //############################################# Hanging NODES ####################################################//  
// 	    // set hanging nodes
// 	    int *master = ParComm->GetMaster();
//         
//         
// 	    for(i=N_Active;i<HangingNodeBound;i++)
// 	    {
// 
//             if(master[i] != rank)
//             continue;
// 
//             s = f[i];
//             k = RowPtr[i+1];
//             for(j=RowPtr[i];j<k;j++)
//             {
//             index = KCol[j];
//             if(index != i)
//             s -= Entries[j] * sol[index];
//             else
//             diag = Entries[j];
//             } // endfor j
//             sol[i] = s/diag;
//         } // endfor i
// 	  
// 	  
// // 	  CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
// // 	    CUDA_CHECK(cudaStreamSynchronize(stream[0]));
// 	  
// //     CUDA_CHECK(cudaStreamSynchronize(stream[1]));
// 
//       }//loop
//       
//     }
//     
// //     CUDA_CHECK(cudaDeviceSynchronize());
//     
//     for (i = 0; i < nStreams; ++i)
//         CUDA_CHECK( cudaStreamDestroy(stream[i]) );
//   
//   // Free GPU memory
//     CUDA_CHECK(cudaFree(d_RowPtr));
//     CUDA_CHECK(cudaFree(d_Reorder));
//     CUDA_CHECK(cudaFree(d_KCol));
//     CUDA_CHECK(cudaFree(d_Entries));
//     CUDA_CHECK(cudaFree(d_f));
//     CUDA_CHECK(cudaFree(d_sol));
// //     CUDA_CHECK(cudaFree(d_sol_temp));
// }

// void SOR_Re_GPU(double *sol, double *f, int *RowPtr, int *KCol, double *Entries, TParFEMapper3D *ParMapper, TParFECommunicator3D *ParComm, int N_CMaster, int N_CDept1, int N_CDept2, int N_CInt, int *ptrCMaster, int *ptrCDept1, int *ptrCDept2, int *ptrCInt, int repeat, int end, int HangingNodeBound, int N_Dirichlet,int N_Int,int N_InterfaceM,int N_Dept1, int N_Dept2, int N_Active, double omega, int N_DOF ,int nz, int n, int rank)
// {
//     int ii, i,j,k,l,index,loop;
//     int itr,jj,tid,nrows=0,numThreads;
//     double s, t, diag;
//     int* Reorder;
//     int nStreams = 2;
//     cudaStream_t stream[nStreams];
// 
//     const int streamSize = N_DOF / nStreams;
// //     const int streamBytes = streamSize * sizeof(int);
//     
//     for (i = 0; i < nStreams; ++i)
//         CUDA_CHECK( cudaStreamCreate(&stream[i]) );
//     
// //     int* 
// //     
// //     double* ex_sol = (double *) malloc(sizeof(double) * nz);
// //     
// //     vectorCoalesce(sol, ex_sol, ParMapper, N_CMaster, N_CDept2, N_CDept1, N_CInt, ptrCMaster, ptrCDept2, ptrCDept1, ptrCInt, N_Active, RowPtr, KCol, 0);
//   
//    // Allocate data on GPU memory
//     int* d_RowPtr = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_RowPtr, (n+1) * sizeof(int)));
//     
//     int* d_Reorder = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_Reorder, (N_InterfaceM + N_Int + N_Dept1 +  N_Dept2) * sizeof(int)));
//     
//     int* d_KCol = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_KCol, nz * sizeof(int)));
//     
//     double* d_Entries = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_Entries, nz * sizeof(double)));
//     
//     double* d_f = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_f, n * sizeof(double)));
//     
//     double* d_sol = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_sol, nz * sizeof(double)));
//     
// //     double* d_sol_temp = NULL;
// //     CUDA_CHECK(cudaMalloc((void**)&d_sol_temp, n * sizeof(double)));
// //     
// //     double* d_current = NULL;
//     
//     memcpy(sol+HangingNodeBound, f+HangingNodeBound, N_Dirichlet*sizeof(double));
//     
//     // Copy to GPU memory
//     CUDA_CHECK(cudaMemcpyAsync(d_RowPtr, RowPtr, (n+1) * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
//     
//     CUDA_CHECK(cudaMemcpyAsync(d_Entries, Entries, nz * sizeof(double), cudaMemcpyHostToDevice,stream[1]));
//     
//     CUDA_CHECK(cudaMemcpyAsync(d_KCol, KCol, nz * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
//     
//     CUDA_CHECK(cudaMemcpyAsync(d_f, f, n * sizeof(double), cudaMemcpyHostToDevice,stream[1]));
//     
//     CUDA_CHECK(cudaMemcpyAsync(d_sol, ex_sol, nz * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//     
// //     CUDA_CHECK(cudaMemcpyAsync(d_sol_temp, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//     
//     Reorder = ParMapper->GetReorder_M();
// 
//     CUDA_CHECK( cudaMemcpyAsync(d_Reorder, Reorder, (N_InterfaceM + N_Int + N_Dept1 +  N_Dept2) * sizeof(int), cudaMemcpyHostToDevice, stream[1]) );
//     
//     double *sol_t= (double *) malloc(sizeof(double) * n);
//     
//     int master_blocks, indp_blocks, d1_blocks, d2_blocks;
//     
//     
//     CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//     CUDA_CHECK(cudaStreamSynchronize(stream[1]));
//     
//     for(itr=0;itr<end;itr++)
//     {
//       for(loop=0;loop<repeat;loop++)
//       {
//         
//       //########################################## MASTERS DOFS ########################################################//
// 	    if(itr == 0)
// 	    {
//             
// 
// 
// 	      for(ii=0;ii<N_CMaster;ii++)
// 	      {
//               
//               master_blocks= ceil(((ptrCMaster[ii+1]-ptrCMaster[ii])*32)/THREADS_PER_BLOCK);
// //               master_blocks= ceil(((ptrCMaster[ii+1]-ptrCMaster[ii])*1)/THREADS_PER_BLOCK);
//               
//               if ( master_blocks == 0) master_blocks=1;
//               
//               
//               SOR_Re_Kernel<<<master_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCMaster[ii], ptrCMaster[ii+1],d_RowPtr ,d_Reorder ,d_KCol ,d_Entries ,d_f ,d_sol , N_Active , omega);
// //               SOR_Scalar_Kernel<<<master_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCMaster[ii], ptrCMaster[ii+1],d_RowPtr ,d_Reorder ,d_KCol ,d_Entries ,d_f ,d_sol , N_Active , omega);
//         
// 		
// 	      } //end for ii
//           
//           if(loop == (repeat-1)){
//               
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             CUDA_CHECK(cudaMemcpyAsync(ex_sol, d_sol, nz * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
//             vectorCoalesce(sol, ex_sol, ParMapper, N_CMaster, N_CDept2, N_CDept1, N_CInt, ptrCMaster, ptrCDept2, ptrCDept1, ptrCInt, N_Active, RowPtr, KCol, 1);
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             
//             ParComm->CommUpdateMS(sol);
//             
// //             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             vectorCoalesce(sol, ex_sol, ParMapper, N_CMaster, N_CDept2, N_CDept1, N_CInt, ptrCMaster, ptrCDept2, ptrCDept1, ptrCInt, N_Active, RowPtr, KCol, 0);
//             CUDA_CHECK(cudaMemcpyAsync(d_sol, ex_sol, nz * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//               
//           }
// 	      
// 	    } //end firstTime
// 	    
//         //########################################## DEPENDENT2 DOFS #####################################################//
// 
// 
//         for(ii=0;ii<N_CDept2;ii++)
// 	    {
//     
//             d2_blocks= ceil(((ptrCDept2[ii+1]-ptrCDept2[ii])*32)/THREADS_PER_BLOCK);
// //             d2_blocks= ceil(((ptrCDept2[ii+1]-ptrCDept2[ii])*1)/THREADS_PER_BLOCK);
// 
//             
//             if ( d2_blocks == 0) d2_blocks=1;
// 
//             
//             SOR_Re_Kernel<<<d2_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCDept2[ii], ptrCDept2[ii+1],d_RowPtr ,(d_Reorder + N_InterfaceM + N_Int + N_Dept1) ,d_KCol ,d_Entries ,d_f ,d_sol, N_Active, omega);
// //             SOR_Scalar_Kernel<<<d2_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCDept2[ii], ptrCDept2[ii+1],d_RowPtr ,(d_Reorder + N_InterfaceM + N_Int + N_Dept1) ,d_KCol ,d_Entries ,d_f ,d_sol, N_Active, omega);
//             
// 
//     
//         }
//         
//         //########################################## DEPENDENT1 DOFS #####################################################//
// // 	    Reorder = ParMapper->GetReorder_D1();
// 	    
//         for(ii=0;ii<N_CDept1;ii++)
// 	    {
//             
//             d1_blocks= ceil(((ptrCDept1[ii+1]-ptrCDept1[ii])*32)/THREADS_PER_BLOCK);
// //             d1_blocks= ceil(((ptrCDept1[ii+1]-ptrCDept1[ii])*1)/THREADS_PER_BLOCK);
//             
//             if ( d1_blocks == 0) d1_blocks=1;
//         
//             SOR_Re_Kernel<<<d1_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCDept1[ii], ptrCDept1[ii+1],d_RowPtr ,( d_Reorder + N_InterfaceM + N_Int) ,d_KCol ,d_Entries ,d_f, d_sol, N_Active, omega);
// //             SOR_Scalar_Kernel<<<d1_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCDept1[ii], ptrCDept1[ii+1],d_RowPtr ,( d_Reorder + N_InterfaceM + N_Int) ,d_KCol ,d_Entries ,d_f, d_sol, N_Active, omega);
// 
// 
// 	    } //end for ii  
// 	    
// 	    
// 	    
//       //################################################################################################################//
// 	    
// 	    if(loop == (repeat-1)){
//             
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             CUDA_CHECK(cudaMemcpyAsync(ex_sol, d_sol, nz * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
//             vectorCoalesce(sol, ex_sol, ParMapper, N_CMaster, N_CDept2, N_CDept1, N_CInt, ptrCMaster, ptrCDept2, ptrCDept1, ptrCInt, N_Active, RowPtr, KCol, 1);
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             
//             ParComm->CommUpdateH1(sol);
//             
// //             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             vectorCoalesce(sol, ex_sol, ParMapper, N_CMaster, N_CDept2, N_CDept1, N_CInt, ptrCMaster, ptrCDept2, ptrCDept1, ptrCInt, N_Active, RowPtr, KCol, 0);
//             CUDA_CHECK(cudaMemcpyAsync(d_sol, ex_sol, nz * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
// 	        
//         }
//       
// 
//         
//         //########################################## MASTERS DOFS ########################################################//
// 	    if(itr!=(end-1) && itr != 0)
// 	    {
//             
// //             Reorder = ParMapper->GetReorder_M();
//             for(ii=0;ii<N_CMaster;ii++)
//             {
//                 
//                 master_blocks= ceil(((ptrCMaster[ii+1]-ptrCMaster[ii])*32)/THREADS_PER_BLOCK);
// //                 master_blocks= ceil(((ptrCMaster[ii+1]-ptrCMaster[ii])*1)/THREADS_PER_BLOCK);
//                 
//                 if ( master_blocks == 0) master_blocks=1;
//                 
//                 SOR_Re_Kernel<<<master_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCMaster[ii], ptrCMaster[ii+1],d_RowPtr ,d_Reorder ,d_KCol ,d_Entries ,d_f ,d_sol  ,N_Active, omega);
// //                 SOR_Scalar_Kernel<<<master_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCMaster[ii], ptrCMaster[ii+1],d_RowPtr ,d_Reorder ,d_KCol ,d_Entries ,d_f ,d_sol  ,N_Active, omega);
//                 
// 
//             
//             
//             } //end for ii
//             
// 
//             
//             if(loop == (repeat-1)){
//                 
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             CUDA_CHECK(cudaMemcpyAsync(ex_sol, d_sol, nz * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
//             vectorCoalesce(sol, ex_sol, ParMapper, N_CMaster, N_CDept2, N_CDept1, N_CInt, ptrCMaster, ptrCDept2, ptrCDept1, ptrCInt, N_Active, RowPtr, KCol, 1);
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//                 
//                 ParComm->CommUpdateMS(sol);
//                 
// //                 CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//             vectorCoalesce(sol, ex_sol, ParMapper, N_CMaster, N_CDept2, N_CDept1, N_CInt, ptrCMaster, ptrCDept2, ptrCDept1, ptrCInt, N_Active, RowPtr, KCol, 0);
//             CUDA_CHECK(cudaMemcpyAsync(d_sol, ex_sol, nz * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//             CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//                 
//             }
//             
//         } //end !lastTime
//       //################################################################################################################//
// 
// //         CUDA_CHECK(cudaStreamSynchronize(stream[1]));
//         for(ii=0;ii<N_CInt;ii++)
// 	    {
//         
//             indp_blocks= ceil(((ptrCInt[ii+1] - ptrCInt[ii])*32)/THREADS_PER_BLOCK);
// //             indp_blocks= ceil(((ptrCInt[ii+1] - ptrCInt[ii])*1)/THREADS_PER_BLOCK);
//             
//         
//             if ( indp_blocks == 0) indp_blocks=1;
//             
//             
//             SOR_Re_Kernel<<<indp_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCInt[ii], ptrCInt[ii+1], d_RowPtr, (d_Reorder + N_InterfaceM) ,d_KCol ,d_Entries ,d_f ,d_sol, N_Active, omega);
// //             SOR_Scalar_Kernel<<<indp_blocks, THREADS_PER_BLOCK, 0, stream[0]>>>(ptrCInt[ii], ptrCInt[ii+1], d_RowPtr, (d_Reorder + N_InterfaceM) ,d_KCol ,d_Entries ,d_f ,d_sol, N_Active, omega);
//         
//         }
//         
// 
//       //############################################# Hanging NODES ####################################################//  
// 	    // set hanging nodes
// 	    int *master = ParComm->GetMaster();
//         
//         
// 	    for(i=N_Active;i<HangingNodeBound;i++)
// 	    {
// 
//             if(master[i] != rank)
//             continue;
// 
//             s = f[i];
//             k = RowPtr[i+1];
//             for(j=RowPtr[i];j<k;j++)
//             {
//             index = KCol[j];
//             if(index != i)
//             s -= Entries[j] * sol[index];
//             else
//             diag = Entries[j];
//             } // endfor j
//             sol[i] = s/diag;
//         } // endfor i
// 	  
// 	  
// // 	  CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
// // 	    CUDA_CHECK(cudaStreamSynchronize(stream[0]));
// 	  
// //     CUDA_CHECK(cudaStreamSynchronize(stream[1]));
// 
//       }//loop
//       
//     }
//     
//     CUDA_CHECK(cudaDeviceSynchronize());
//     
//     for (i = 0; i < nStreams; ++i)
//         CUDA_CHECK( cudaStreamDestroy(stream[i]) );
//   
//   // Free GPU memory
//     CUDA_CHECK(cudaFree(d_RowPtr));
//     CUDA_CHECK(cudaFree(d_Reorder));
//     CUDA_CHECK(cudaFree(d_KCol));
//     CUDA_CHECK(cudaFree(d_Entries));
//     CUDA_CHECK(cudaFree(d_f));
//     CUDA_CHECK(cudaFree(d_sol));
// //     CUDA_CHECK(cudaFree(d_sol_temp));
// }

// void SOR_Re_New(double *sol, double *f, double *aux,
//         int N_Parameters, double *Parameters, int *RowPtr, int *KCol, double *Entries, TParFEMapper3D *ParMapper, TParFECommunicator3D *ParComm, int repeat, int HangingNodeBound,int N_Dirichlet,int N_Int,int N_InterfaceM,int N_Dept1,int N_Dept2,int N_Active, int N_DOF ,int nz, int n)
// {
//   int ii, i,j,k,l,index,rank,loop;
//   double s, t, diag;
//   double omega;
//   
//   int* Reorder;
// //   int repeat = TDatabase::ParamDB->Par_P6;
//   if(repeat <= 0)
//     repeat = 1;
//   
// //   cout<<"neha: New SOR_Re "<<endl;
//   
// //   cout<<"neha: repeat "<<repeat<<endl;
//   
//   omega = Parameters[0];
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//   
//   
// //     int n = sizeof(f)/ sizeof(double);
// //     int nz = sizeof(Entries)/ sizeof(double);
//   
// 
// //     int RowPtr_size = sizeof(RowPtr)/ sizeof(int);
// //     int KCol_size = sizeof(KCol)/ sizeof(int);
//     
// //     double* sol_temp= (double*)malloc (n * sizeof(double));
//     
// //     cout<<"neha: size of entries nz "<<nz<<" "<<sizeof(Entries)<<" "<<sizeof(double)<<Entries[0]<<Entries[1]<<Entries[2]<<endl;
//     
//     cout<<"neha: sol"<<n<<" "<<sol[0]<<" "<<sol[1]<<endl;
//     
// //     int THREADS_PER_BLOCK=1024;
// //     int gridSize=10;
//     
//     int nStreams = 2;
//     cudaStream_t stream[nStreams];
//     
//     const int streamSize = N_DOF / nStreams;
// //     const int streamBytes = streamSize * sizeof(int);
//     
//     for (int i = 0; i < nStreams; ++i)
//         CUDA_CHECK( cudaStreamCreate(&stream[i]) );
//   
//    // Allocate data on GPU memory
//     int* d_RowPtr = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_RowPtr, n * sizeof(int)));
//     
//     int* d_Reorder = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_Reorder, (N_Int + N_Dept1 +  N_Dept2) * sizeof(int)));
//     
//     int* d_KCol = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_KCol, nz * sizeof(int)));
//     
//     double* d_Entries = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_Entries, nz * sizeof(double)));
//     
//     double* d_f = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_f, n * sizeof(double)));
//     
//     double* d_sol = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_sol, n * sizeof(double)));
//     
//     double* d_sol_temp = NULL;
//     CUDA_CHECK(cudaMalloc((void**)&d_sol_temp, n * sizeof(double)));
//     
//   
// //     memcpy(sol+HangingNodeBound, f+HangingNodeBound, N_Dirichlet*sizeof(double));
//     
//     // Copy to GPU memory
//     CUDA_CHECK(cudaMemcpyAsync(d_RowPtr, RowPtr, n * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
//     
//     CUDA_CHECK(cudaMemcpyAsync(d_Entries, Entries, nz * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//     
//     CUDA_CHECK(cudaMemcpyAsync(d_KCol, KCol, nz * sizeof(int), cudaMemcpyHostToDevice,stream[0]));
//     
//     CUDA_CHECK(cudaMemcpyAsync(d_f, f, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//     
//     CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//     
//     Reorder = ParMapper->GetReorder_I();
// 
//     CUDA_CHECK( cudaMemcpyAsync(d_Reorder, Reorder, (N_Int + N_Dept1 +  N_Dept2) * sizeof(int), cudaMemcpyHostToDevice, stream[0]) );
//     
//     
//   for(loop=0;loop<repeat;loop++)
//   {
//       // set Dirichlet nodes
//       if(loop!=0){
//         memcpy(sol+HangingNodeBound, f+HangingNodeBound, N_Dirichlet*sizeof(double));
//       }
// 
//       
// 
//     //########################################## MASTERS DOFS ########################################################//
//       Reorder = ParMapper->GetReorder_M();
//       for(ii=0;ii<N_InterfaceM;ii++)
//       {
//             i = Reorder[ii];
//             if(i >= N_Active)     continue;
//             
//             s = f[i];
//             k = RowPtr[i+1];
// //             if(rank==0)
// //                 cout<<"neha: no. of non-zero in each row(master) "<<(k-RowPtr[i])<<endl;
//             for(j=RowPtr[i];j<k;j++)
//             {
//                 index = KCol[j];
//                 if(index == i)
//                 {
//                     diag = Entries[j];
//                 }
//                 else
//                 {
//                     s -= Entries[j] * sol[index];
//                 }
//             } // endfor j
//             
//             t = sol[i];
//             sol[i] = omega*(s/diag-t) + t;
//             cout << "sol[i]: master" << sol[i] << endl;
//       } // endfor i
//       
//       if(loop == (repeat-1))
//             ParComm->CommUpdateMS(sol);
//     //################################################################################################################//
// 
// //     CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[1]));
//     CUDA_CHECK(cudaMemcpyAsync(d_sol, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//     
//     CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//     
// //     if(loop==0){
// //         CUDA_CHECK(cudaStreamSynchronize(stream[0]));
// //     }
//     
// //     CUDA_CHECK(cudaStreamSynchronize(stream[1]));
//     
//     //########################################## INDEPENDENT DOFS ####################################################//  
// 
//     
// //     Reorder = ParMapper->GetReorder_I();
// //     
// //     CUDA_CHECK(cudaMalloc((void**)&d_Reorder, (N_Int + N_Dept1 +  N_Dept2) * sizeof(int)));
// //         
// //     CUDA_CHECK( cudaMemcpyAsync(d_Reorder, Reorder, (N_Int + N_Dept1 +  N_Dept2) * sizeof(int), cudaMemcpyHostToDevice, stream[0]) );
//     
//     
//     int numblocks1= ceil((N_Int*32)/THREADS_PER_BLOCK);
//     
//     if ( numblocks1 == 0) numblocks1=1;
//     
// //     cout<<"neha: numblocks1 "<<numblocks1<<endl;
//     
//     // Run kernel
//     SOR_Re_Kernel<<<numblocks1, THREADS_PER_BLOCK, 0, stream[0]>>>(N_Int ,d_RowPtr ,d_Reorder ,d_KCol ,d_Entries ,d_f ,d_sol ,N_Active, omega);
//     
//     // Copy data back to CPU memory
// //     CUDA_CHECK( cudaMemcpyAsync(&sol, &d_sol, N_Int * sizeof(double), cudaMemcpyDeviceToHost, stream[0]) );
//     
//     CUDA_CHECK(cudaMemcpyAsync(sol, d_sol, n * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
//     CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//     
// 
//     //########################################## DEPENDENT1 DOFS #####################################################//
//       Reorder = ParMapper->GetReorder_D1();
//       for(ii=0;ii<N_Dept1;ii++)
//       {
//             i = Reorder[ii];
//             if(i >= N_Active)     continue;
//             
//             s = f[i];
//             k = RowPtr[i+1];
//             if(rank==0)
// //                 cout<<"neha: no. of non-zero in each row(dept1) "<<(k-RowPtr[i])<<endl;
//             for(j=RowPtr[i];j<k;j++)
//             {
//                 index = KCol[j];
//                 if(index == i)
//                 {
//                     diag = Entries[j];
//                 }
//                 else
//                 {
//                     s -= Entries[j] * sol[index];
//                 }
//             } // endfor j
//             t = sol[i];
//             sol[i] = omega*(s/diag-t) + t;
//             cout << "sol[i]: d1" << sol[i] << endl;
//       } // endfor i
//     //################################################################################################################//
// 
// 
//     
//     
//    if(loop == (repeat-1))
//       ParComm->CommUpdateH1(sol);
//    
//    
//    //########################################## DEPENDENT2 DOFS #####################################################//
// 
//     
//     Reorder = ParMapper->GetReorder_D2();
//     
//     int numblocks2= ceil((N_Dept2*32)/THREADS_PER_BLOCK);
//     
//     if ( numblocks2 == 0) numblocks2=1;
//     
// //     cout<<"neha: numblocks2 "<<numblocks2<<endl;
//         
// //     CUDA_CHECK( cudaMemcpyAsync(&d_Reorder, &Reorder, N_Dept2 * sizeof(int), cudaMemcpyHostToDevice, stream[1]) );
//     
//     // Run kernel
// //     SOR_Re_Kernel<<<numblocks2, THREADS_PER_BLOCK, 0, stream[1]>>>(N_Dept2 ,d_RowPtr ,(d_Reorder + N_Int + N_Dept1) ,d_KCol ,d_Entries ,d_f ,d_sol ,N_Active, omega);
//     SOR_Re_Kernel<<<numblocks2, THREADS_PER_BLOCK, 0, stream[0]>>>(N_Dept2 ,d_RowPtr ,(d_Reorder + N_Int + N_Dept1) ,d_KCol ,d_Entries ,d_f ,d_sol ,N_Active, omega);
//     
//     // Copy data back to CPU memory
// //     CUDA_CHECK( cudaMemcpyAsync(&sol, &d_sol, N_Dept2 * sizeof(double), cudaMemcpyDeviceToHost, stream[1]) );
// 
//     //############################################# Hanging NODES ####################################################//  
//       // set hanging nodes
//       int *master = ParComm->GetMaster();
//       for(i=N_Active;i<HangingNodeBound;i++)
//       {
// 	if(master[i] != rank)
// 	  continue;
// 	s = f[i];
// 	k = RowPtr[i+1];
// 	for(j=RowPtr[i];j<k;j++)
// 	{
// 	  index = KCol[j];
// 	  if(index != i)
// 	    s -= Entries[j] * sol[index];
// 	  else
// 	    diag = Entries[j];
// 	} // endfor j
// 	sol[i] = s/diag;
//       } // endfor i
//     
//     //############################################# Hanging NODES ####################################################// 
//     
//     CUDA_CHECK(cudaStreamSynchronize(stream[0]));
// //     CUDA_CHECK(cudaStreamSynchronize(stream[1]));
//     
//     
//     int numblocks3= ceil(N_Dept1/THREADS_PER_BLOCK);
//     
// //     cout<<"neha: N_Dept1 "<<N_Dept1<<endl;
//     
//     if ( numblocks3 == 0) numblocks3=1;
//     
// //     cout<<"neha: numblocks3 "<<numblocks3<<endl;
//     
// //     cout<<"neha: N_Active "<<N_Active<<endl;
//     
// //     cout<<"neha: HangingNodeBound "<<HangingNodeBound<<endl;
//     //merge sol_temp and sol
//     
//     // transfer updated sol to GPU memory
//     CUDA_CHECK(cudaMemcpyAsync(d_sol_temp, sol, n * sizeof(double), cudaMemcpyHostToDevice,stream[0]));
//     
// //     Reorder = ParMapper->GetReorder_D1();
//     if(N_Dept1 != 0)
// //     cout<<"neha: Reorder "<<Reorder[0]<<" "<<Reorder[1]<<endl;
//     
// //     mergeSolution<<<numblocks3, THREADS_PER_BLOCK, 0, stream[0]>>>(d_sol ,d_sol_temp ,(d_Reorder + N_Int) ,N_Dept1 ,HangingNodeBound, N_Active);
//     
//     CUDA_CHECK(cudaMemcpyAsync(sol, d_sol, n * sizeof(double), cudaMemcpyDeviceToHost,stream[0]));
//     
//     CUDA_CHECK(cudaStreamSynchronize(stream[0]));
//     
//     Reorder = ParMapper->GetReorder_I();
//       for(ii=0;ii<N_Int;ii++)
//       {
// 	i = Reorder[ii];
// 	if(i >= N_Active)     continue;
// 	
// 	cout << "sol[i]: indp" << sol[i] << endl;
//       } // endfor i
//       
//       
//   }//loop
//   
// //   cout<<"neha: out of repeat loop"<<endl;
//   
//   CUDA_CHECK(cudaDeviceSynchronize());
//   
//   // Free GPU memory
//     CUDA_CHECK(cudaFree(d_RowPtr));
//     CUDA_CHECK(cudaFree(d_Reorder));
//     CUDA_CHECK(cudaFree(d_KCol));
//     CUDA_CHECK(cudaFree(d_Entries));
//     CUDA_CHECK(cudaFree(d_f));
//     CUDA_CHECK(cudaFree(d_sol));
//     CUDA_CHECK(cudaFree(d_sol_temp));
// }

// void my_abort(int err) {
//     printf("Test FAILED\n");
//     MPI_Abort(MPI_COMM_WORLD, err);
// }

