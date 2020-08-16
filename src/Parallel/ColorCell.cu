
#ifdef _MPI

#include "mpi.h"
#include <ParFEMapper3D.h>
#include <FEDatabase3D.h>
#include <Database.h>
#include <SubDomainJoint.h>
#include <Edge.h>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <omp.h>


#define GLOBAL_NO 0
#define DOF_NO 1

#define HALOCELL 0
#define NONHALO  1

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
        
using namespace std;

extern double timeC;

__global__ void MatchKernel(int* UDOFs,int* adjlist, int N_OwnCells, int N_U){
    
    const unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int grid_size = gridDim.x * blockDim.x;
    const unsigned int totalCells=N_OwnCells;
    
    
//     unsigned int pp= thread_id;
    
//     unsigned int qq= thread_id - (pp*N_OwnCells);
    
    for(unsigned int row = thread_id; row < totalCells; row += grid_size)
    {
        int k=0;
        
        for(int qq=0; qq<N_OwnCells; qq++){
            
            bool flag=false;
            
            for(int ii=0; ii<N_U && !flag; ii++){
                
                for(int jj=0; jj<N_U; jj++){
                    
        //             cout<<"list1"<<list1[ii]<<endl;
        //             cout<<"list2"<<list2[jj]<<endl;
                    int ind1=row*N_U+ii;
                    int ind2=qq*N_U+jj;
                    
                    if(UDOFs[ind1]!= -1 && UDOFs[ind2] != -1){
                    if(UDOFs[ind1]==UDOFs[ind2]){
                    
        //                 if(rank==0){
        //             cout<<"list1"<<list1[ii]<<endl;
        //             cout<<"list2"<<list2[jj]<<endl;
        //                 }
                        adjlist[row*N_U+k]=qq;
                        k++;
                        flag=true;
                        break;
                        
                    }
                    }
                    
                }
                
            }
            
            
            
        }
    
    
    }
    
}

void TParFEMapper3D::colorCellGPU(int N_U, int N_Cells, int N_OwnCells, TCollection *Coll){

// #ifdef __3D__
  const int MaxN_LocalU =  4*125;//4*MaxN_BaseFunctions3D;
  TFE3D *UEle, *PEle;
// #endif
    
//     TCollection *Coll;
    int *PGlobalNumbers, *PBeginIndex, *PDOFs, PDOF, N_P;
    TBaseCell *Cell;
    
//     int N_U=0;
    cout<<"color GPU"<<endl;
    double t1,t2;
    
    int rank, size;
    MPI_Comm_rank(Comm, &rank);
    MPI_Comm_size(Comm, &size);
    

//     Coll = USpace->GetCollection();
//     int N_Cells = Coll->GetN_Cells();
    
    int *UDOFs= new int[N_OwnCells*N_U];
    
    for(int i=0; i<N_U*N_OwnCells; i++){
        
        UDOFs[i]=-1;;
    }
    
    vector<int> nonHaloCell(N_OwnCells);
    int r=0;
       
    for(int ii=0;ii<N_Cells;ii++)
  {
    //ii = downwind[i];
    
    Cell = Coll->GetCell(ii);
    
#ifdef _MPI
    if(!Cell->IsHaloCell()){
//       cout << "this should" << endl;
      nonHaloCell[r]=ii;
      r++;
//       cout << "this shouldnt" << endl;
    }
        
#endif

  }
    
    int k=0;
    
//     cout<<"neha:N_Cells"<<N_Cells<<endl;
//     cout<<"neha:N_Dof"<<N_Dof<<endl;
//     cout<<"neha:MaxN_LocalU"<<MaxN_LocalU<<endl;

   
    for(int ii=0;ii<N_OwnCells;ii++)
  {
    //ii = downwind[i];
    
    Cell = Coll->GetCell(nonHaloCell[ii]);
    
// #ifdef _MPI
//     if(Cell->IsHaloCell()){
// //       cout << "this should" << endl;
//       continue;
// //       cout << "this shouldnt" << endl;
//     }   
// #endif

    PGlobalNumbers = FESpace->GetGlobalNumbers();
    PBeginIndex = FESpace->GetBeginIndex();
    
#ifdef __2D__
//     UEle = TFEDatabase2D::GetFE2D(USpace->GetFE2D(ii, Cell));
    PEle = TFEDatabase2D::GetFE2D(FESpace->GetFE2D(ii, Cell));
#endif
#ifdef __3D__
//     UEle = TFEDatabase3D::GetFE3D(USpace->GetFE3D(ii, Cell));
    PEle = TFEDatabase3D::GetFE3D(FESpace->GetFE3D(ii, Cell));
#endif

    // get local number of dof
//     N_U = UEle->GetN_DOF();
    N_P = PEle->GetN_DOF();
//     UDOFs[ii]=new int[27];
    
//     cout<<"N_P"<<N_P<<endl;
    
    PDOFs = PGlobalNumbers+PBeginIndex[nonHaloCell[ii]];
    
//     for(int j=0;j<1;j++)
//     {
        PDOF = PDOFs[0];
        k=0;
        
        for(int jj=RowPtr[PDOF];jj<RowPtr[PDOF+1];jj++)
        {
//             if(rank==0){
//             cout<<"cell:"<<ii<<endl;
// //             cout<<"PDOF:"<<PDOF<<endl;
// //             cout<<"UDOF:"<<KCol[jj]<<endl;
//             }
            UDOFs[ii*N_U+k]=KCol[jj];
            
//             if((ii*N_OwnCells+k)> (N_OwnCells*N_U)){
//                 cout<<"t"<<N_OwnCells*N_U<<endl;
//                 cout<<"s"<<(ii*N_OwnCells+k)<<endl;
//             }
//             if(rank==0)
//             cout<<"UDOFs:"<<UDOFs[ii][N_U]<<endl;
            k++;
            
        }
//         cout<<"k:"<<k<<endl;
//         if(rank==0){
//             cout<<"N_U:"<<N_U<<endl;
//         }

    }
    
    
    
    unsigned int totalCells=N_OwnCells*N_U;
//     cout<<"totalCells"<<totalCells<<endl;
//     int* abc = new int[1000];
    int* adjlist = new int[totalCells];
//     cout<<"here"<<endl;
    int* d_UDOFs = NULL;
    
    CUDA_CHECK(cudaMalloc((void**)&d_UDOFs, N_U*N_OwnCells * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_UDOFs, UDOFs, N_U*N_OwnCells * sizeof(int), cudaMemcpyHostToDevice));
    
    int* d_adjlist = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_adjlist, N_OwnCells*N_U * sizeof(int)));
    
    
    
    CUDA_CHECK(cudaMemset(d_adjlist, -1, N_OwnCells * N_U * sizeof(int) ));
    
    int thread_blocks = (N_OwnCells)/THREADS_PER_BLOCK + 1;
    
    
    
    MatchKernel<<<thread_blocks, THREADS_PER_BLOCK, 0>>>(d_UDOFs,d_adjlist,N_OwnCells,N_U);
    
    CUDA_CHECK(cudaMemcpy(adjlist, d_adjlist, N_OwnCells * N_U * sizeof(int),cudaMemcpyDeviceToHost));
    
    

//   }
  
//   vector<vector<int>> adjlist(N_OwnCells);

  unsigned long int pp,qq;
  
// #pragma omp parallel for default(shared) num_threads(2) private(pp) collapse(2)
    
//     double start = omp_get_wtime();  
//     omp_set_num_threads(8);
//     #pragma omp parallel for default(shared) collapse(2)
//   for(pp=0; pp<N_OwnCells; pp++){
//       
//       for(int qq=0; qq<N_OwnCells; qq++){
//     
//     if(match(UDOFs[pp], UDOFs[qq], N_U)){
//         
//         adjlist[pp].emplace_back(qq);
// //         if(rank==0){
// //         cout<<"ii"<<ii<<endl;
// //         cout<<"jj"<<jj<<endl;
// //         }
//     }
//     
// //     if(rank==0)
// //         cout<<"neib:"<<k<<endl;
//   }
//   }
//   cout<<"time red:"<<omp_get_wtime()-start<<endl;
  
  
    int *allocatedColor = new int[N_Cells];
    //initialize all colors with default
    for(int i=0;i<N_Cells;i++)
        allocatedColor[i] = -1;
    
//     allocatedColor[0]=0;
  
  bool avail[N_OwnCells];
  for(int i=0;i<N_OwnCells;i++)
    avail[i] = false;
  
  int numColors=0;
  
  t1=MPI_Wtime();
  
    for(int ii=0;ii<N_OwnCells;ii++)
    {
        
//         Cell = Coll->GetCell(ii);
//             #ifdef _MPI
//         if(Cell->IsHaloCell()){
//           cout << "this should" << endl;
//         continue;
//     //       cout << "this shouldnt" << endl;
//         }   
//     #endif

        for(int jj=0;jj<N_U;jj++)
        {
//             if(rank==0 && adjlist[ii*N_U+jj]!=-1 )
//             cout<<"adj:"<<ii<<":"<<jj<<":"<<adjlist[ii*N_U+jj]<<endl;
            
            if((adjlist[ii*N_U+jj] != -1) && allocatedColor[adjlist[ii*N_U+jj]] != -1){
                avail[allocatedColor[adjlist[ii*N_U+jj]]]= true;
            }
            
        }
        
        int cr; 
        for (cr = 0; cr < N_OwnCells; cr++) {
            if (avail[cr] == false) 
                break;
        }
        
        allocatedColor[ii] = cr;
        
        for(int jj=0;jj<N_U;jj++)
        {
            
            if((adjlist[ii*N_U+jj] != -1) && allocatedColor[adjlist[ii*N_U+jj]] != -1){
                avail[allocatedColor[adjlist[ii*N_U+jj]]]= false;
            }
            
        }
        
        if(numColors < allocatedColor[ii])
        numColors = allocatedColor[ii];
    }
//     }
    //colors were numbered from 0, hece total will be 1 more
    
    t2=MPI_Wtime();
    printf("total time taken for coloring cells = %lf\n",t2-t1);  
    //colors were numbered from 0, hece total will be 1 more
    numColors++;
    
    N_CIntCell=numColors;
    
    ptrCellColors = new int[numColors+1];
    CellReorder = new int[N_OwnCells];
    
    k = 0;
    //arrange the dofs, such that same color dofs are together
    for(int i=0;i<numColors;i++)
    {
        ptrCellColors[i] = k;
        for(int j=0;j<N_Cells;j++)
        {

            if(allocatedColor[j] == i)
            {
                CellReorder[k] = j;
//                 cout<<"CellReorder:"<<CellReorder[k]<<endl;
                k++;
                allocatedColor[j] = -1;
            }
        }
    }
    
    ptrCellColors[numColors] = k;
    
//     for(int i=0;i<numColors+1;i++){
//         cout<<"ptrcell:"<<ptrCellColors[i]<<endl;
//     }
    
    
//     delete []allocatedColor;
    
//     for(int i=0; i<N_Cells; i++){
//         delete UDOFs[i];
//     }
    
//     delete []UDOFs;
  
        maxCellsPerColor = -1;

        for(int i=0;i<N_CIntCell;i++)
        {
            int temp = (ptrCellColors[i+1] - ptrCellColors[i]);
            
            if(maxCellsPerColor< temp){
                    
                    maxCellsPerColor = temp;
                    
            }

        }
        cout<<"maxCellsPerColor"<<maxCellsPerColor<<endl;
  printf("numcolors:: %d\t total cells = %d\n",N_CIntCell,N_OwnCells);
  
  // Free GPU memory
    CUDA_CHECK(cudaFree(d_adjlist));
    CUDA_CHECK(cudaFree(d_UDOFs));
    
}


void TParFEMapper3D::colorPDOFGPU(int N_U, int N_Cells, TCollection *Coll, char *DofmarkerU){
    
    cout<< "color GPU"<<endl;
    // #ifdef __3D__
  const int MaxN_LocalU =  4*125;//4*MaxN_BaseFunctions3D;
  TFE3D *UEle, *PEle;
// #endif
    
//     TCollection *Coll;
    int *PGlobalNumbers, *PBeginIndex, *PDOFs, PDOF, N_P;
    TBaseCell *Cell;
    
//     int N_U=0;
    
    double t1,t2;
    
    int rank, size;
    MPI_Comm_rank(Comm, &rank);
    MPI_Comm_size(Comm, &size);
    

//     Coll = USpace->GetCollection();
//     int N_Cells = Coll->GetN_Cells();
    
    
    
    int N_PDOF = FESpace->GetN_DegreesOfFreedom();
    
//     vector<vector<int>> UDOFs(N_OwnDof);
    
//     for(int i=0; i<N_PDOF; i++){
//         UDOFs[i]=new int[N_U];
//     }
    
    int k=0;
    
    
    cout<<"neha:N_Cells"<<N_Cells<<endl;
//     cout<<"neha:N_Dof"<<N_Dof<<endl;
//     cout<<"neha:MaxN_LocalU"<<MaxN_LocalU<<endl;
    
    PGlobalNumbers = FESpace->GetGlobalNumbers();
    PBeginIndex = FESpace->GetBeginIndex();
    
    int N_DOF=N_OwnDof+N_InterfaceS;
    
    vector<int> nonHaloDOF(N_DOF); 
    int r=0;
    
    int max_N_U=-1;
    
    for(int j=0;j<N_PDOF;j++)
    {
            
        if(TDatabase::ParamDB->DOF_Average){
		if(DofMarker[j] == 'h' || DofMarker[j] == 'H')
		  continue;
	      }     
	      else{
		if(DofMarker[j] == 'h' || DofMarker[j] == 'H'  ||  DofMarker[j] == 's')
		  continue;
	      }
        
          N_U = 0;
          int begin = RowPtr[j];
	      int end = RowPtr[j+1];
	    for(int k=begin;k<end;k++)
	      { 
		int l=KCol[k]; 

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
        
        nonHaloDOF[r++]=j;
        
    }
    int l=0;
    
    int *UDOFs= new int[N_DOF*max_N_U];
    
    for(int i=0; i<max_N_U*N_DOF; i++){
        
        UDOFs[i]=-1;;
    }
    
    for(int j=0;j<N_DOF;j++)
    {
        k=0;
        
        for(int jj=RowPtr[nonHaloDOF[j]];jj<RowPtr[nonHaloDOF[j]+1];jj++)
        {
                l=KCol[jj]; 
        
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

            UDOFs[j*max_N_U+k]=KCol[jj];
            
            k++;
            
        }

    }
    
    
        unsigned int totalDOF=N_DOF*max_N_U;
    cout<<"totalCells"<<max_N_U<<endl;
//     int* abc = new int[1000];
    int* adjlist = new int[totalDOF];
//     cout<<"here"<<endl;
    int* d_UDOFs = NULL;
    
    CUDA_CHECK(cudaMalloc((void**)&d_UDOFs, totalDOF * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_UDOFs, UDOFs, totalDOF * sizeof(int), cudaMemcpyHostToDevice));
    
    int* d_adjlist = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_adjlist, totalDOF * sizeof(int)));
    
    
    
    CUDA_CHECK(cudaMemset(d_adjlist, -1, totalDOF * sizeof(int) ));
    
    int thread_blocks = (N_DOF)/THREADS_PER_BLOCK + 1;
    
    
    
    MatchKernel<<<thread_blocks, THREADS_PER_BLOCK, 0>>>(d_UDOFs,d_adjlist,N_DOF,max_N_U);
    
    CUDA_CHECK(cudaMemcpy(adjlist, d_adjlist, totalDOF * sizeof(int),cudaMemcpyDeviceToHost));
    
  
//     vector<vector<int>> adjlist(N_PDOF);
//     vector<bool> isHalo(N_Cells,false);
    int ii,jj;
//     
//     for( ii=0; ii<N_Cells; ii++){
//       
//       Cell = Coll->GetCell(ii);
//     
//     #ifdef _MPI
//         if(Cell->IsHaloCell()){
//             isHalo[ii]=true;
//         }
//     #endif
//         
//     }
    
//     double start=omp_get_wtime();
// #pragma omp parallel for default(shared) num_threads(8) private(ii) collapse(2)
//   for( ii=0; ii<N_DOF; ii++){
//       
// //       if(DofMarker[ii] == 'h' || DofMarker[ii] == 'H'  ||  DofMarker[ii] == 's')
// //             continue;
//       
//       for(jj=0; jj<N_DOF; jj++){
//                 
// //             if(DofMarker[jj] == 'h' || DofMarker[jj] == 'H'  ||  DofMarker[jj] == 's')
// //             continue;
//             
//             if(match(UDOFs[nonHaloDOF[ii]], UDOFs[nonHaloDOF[jj]])){
//                 
//                 adjlist[nonHaloDOF[ii]].emplace_back(nonHaloDOF[jj]);
//         //         if(rank==0){
//         //         cout<<"ii"<<ii<<endl;
//         //         cout<<"jj"<<jj<<endl;
//         //         }
//             }
//     
//     }
//     
//   }
//   
//     cout<<"time red:"<<omp_get_wtime()-start<<endl;
//     int *allocatedColor = new int[N_Cells];
    int *allocatedPColor = new int[N_Dof];
    
    //initialize all colors with default
    for(int i=0;i<N_PDOF;i++)
        allocatedPColor[i] = -1;
    
//         Cell = Coll->GetCell(0);

    l=0;
    
      bool avail[N_DOF];
      for(int i=0;i<N_DOF;i++)
        avail[i] = false;

  int numColors=0;
  
  t1=MPI_Wtime();
  
    for(int ii=0;ii<N_DOF;ii++)
    {
        
        for(int jj=0;jj<max_N_U;jj++)
        {

            if((adjlist[ii*max_N_U+jj] != -1) && allocatedPColor[nonHaloDOF[adjlist[ii*max_N_U+jj]]] != -1){
                avail[allocatedPColor[nonHaloDOF[adjlist[ii*max_N_U+jj]]]]= true;
            }
            
        }
        
        int cr; 
        for (cr = 0; cr < N_DOF; cr++) {
            if (avail[cr] == false) 
                break;
        }
        
        allocatedPColor[nonHaloDOF[ii]] = cr;
        
        for(int jj=0;jj<max_N_U;jj++)
        {
            
            if((adjlist[ii*max_N_U+jj] != -1) && allocatedPColor[nonHaloDOF[adjlist[ii*max_N_U+jj]]] != -1){
                avail[allocatedPColor[nonHaloDOF[adjlist[ii*max_N_U+jj]]]]= false;
            }
            
        }
        
        if(numColors < allocatedPColor[nonHaloDOF[ii]])
        numColors = allocatedPColor[nonHaloDOF[ii]];
    }
//     }
    //colors were numbered from 0, hece total will be 1 more
    
    t2=MPI_Wtime();
    printf("total time taken for coloring PDOFs = %lf\n",t2-t1);  
    //colors were numbered from 0, hece total will be 1 more
    numColors++;
    
    N_CPDOF=numColors;
    
    ptrPDOFColors = new int[N_CPDOF+1];
    PDOFReorder = new int[N_DOF];
    
    k = 0;
    
    
    cout<<"N_PDOF:"<<N_PDOF<<endl;
    //arrange the dofs, such that same color dofs are together
    for(int i=0;i<N_CPDOF;i++)
    {
        ptrPDOFColors[i] = k;
        
//         if(rank==0)
//         cout<<"PDOFReorder:"<<ptrPDOFColors[i]<<"k:"<<k<<endl;
                
        for(int j=0;j<N_PDOF;j++)
        {

            if(allocatedPColor[j] == i)
            {
                PDOFReorder[k] = j;

                k++;
                allocatedPColor[j] = -1;
            }
        }
    }
    
    ptrPDOFColors[N_CPDOF] = k;
    
//         for(int i=0;i<N_CPDOF+1;i++)
//         {
//                                     if(rank==0)
//         cout<<"PDOFReorder:"<<ptrPDOFColors[i]<<"i:"<<i<<endl;
//         }
        
//             for(int j=0;j<N_DOF;j++)
//         {
//             if(rank==0)
//         cout<<"PDOFReorder:"<<PDOFReorder[j]<<"j:"<<j<<endl;
//         }
    
//     for(int i=0;i<numColors+1;i++){
//         cout<<"ptrcell:"<<ptrCellColors[i]<<endl;
//     }
    
    
//     delete []allocatedColor;
    delete []allocatedPColor;
    
//     for(int i=0; i<N_DOF; i++){
//         delete UDOFs[i];
//     }
    
    delete []UDOFs;
    
    delete []adjlist;
  
  printf("numcolors:: %d\t total PDOFS = %d\n",N_CPDOF,N_DOF);
  
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_adjlist));
    CUDA_CHECK(cudaFree(d_UDOFs));
    
}

#endif
