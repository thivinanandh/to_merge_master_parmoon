// =======================================================================
// %W% %G%
//
// Class:       TMultiGrid3D
// Purpose:     store all data for a multi grid method in 3d
//
// Author:      Gunar Matthies 26.06.2000
//
// History:     26.06.2000 start of implementation
//
// =======================================================================

#include <MultiGrid3D.h>
#include <FEDatabase3D.h>
#include <Database.h>
#include <MooNMD_Io.h>
#include <LinAlg.h>

#include <stdlib.h>
#include <string.h>
#ifdef _MPI  
#include <ParFECommunicator3D.h>
#include <FEFunction3D.h>
#include <ParFEMapper3D.h>
#endif

#include "nvToolsExt.h"
#define PreSmooth -1
#define CoarseSmooth 0
#define PostSmooth 1

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

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


double tSmoother = 0.0;

/** constructor */
TMultiGrid3D::TMultiGrid3D(int n_problems, int n_parameters, 
                       double *parameters)
{
  N_Levels = 0;
  
  N_Problems = n_problems;

  N_Parameters = n_parameters;

  Parameters = parameters;

  
}

#ifdef _CUDA
void TMultiGrid3D::InitGPU(){

  TMGLevel3D *finest_level = MultiGridLevels[N_Levels-1];
  TSquareMatrix3D *A_finest = finest_level->GetMatrix();

  nStreams = 2;

  stream_smooth = new cudaStream_t[nStreams];

  for (int i = 0; i < nStreams; ++i)
        CUDA_CHECK( cudaStreamCreate(&stream_smooth[i]) );

  CUDA_CHECK( cudaStreamCreate(&stream_transfer));
  int nz= A_finest->GetN_Entries();
  int n = A_finest->GetN_Rows();
  
  // cout<<nz<<endl;
  // cout<<n<<endl;

  
      CUDA_CHECK(cudaMalloc((void**)&d_RowPtr1, (n+1) * sizeof(int)));

      CUDA_CHECK(cudaMalloc((void**)&d_KCol1, nz * sizeof(int)));

      CUDA_CHECK(cudaMalloc((void**)&d_Entries1, nz * sizeof(double)));

      CUDA_CHECK(cudaMalloc((void**)&d_RowPtr2, (n+1) * sizeof(int)));

      CUDA_CHECK(cudaMalloc((void**)&d_KCol2, nz * sizeof(int)));

      CUDA_CHECK(cudaMalloc((void**)&d_Entries2, nz * sizeof(double)));

      CUDA_CHECK(cudaMalloc((void**)&d_sol1, n * sizeof(double)));

      CUDA_CHECK(cudaMalloc((void**)&d_aux1, n * sizeof(int)));

      CUDA_CHECK(cudaMalloc((void**)&d_sol2, n * sizeof(double)));

      CUDA_CHECK(cudaMalloc((void**)&d_aux2, n * sizeof(int)));

  

}



      void TMultiGrid3D::DeviceDataTransfer(int next_level, int smooth){

        TMGLevel3D *next_level1 = MultiGridLevels[next_level];
        TSquareMatrix3D *A_next_level = next_level1->GetMatrix();

        int nz= A_next_level->GetN_Entries();
        int n = A_next_level->GetN_Rows();

        CUDA_CHECK(cudaMemcpyAsync(d_RowPtr2, A_next_level->GetRowPtr(), (n+1) * sizeof(int), cudaMemcpyHostToDevice,stream_transfer));
    
        CUDA_CHECK(cudaMemcpyAsync(d_Entries2, A_next_level->GetEntries(), nz * sizeof(double), cudaMemcpyHostToDevice,stream_transfer));
    
        CUDA_CHECK(cudaMemcpyAsync(d_KCol2, A_next_level->GetKCol(), nz * sizeof(int), cudaMemcpyHostToDevice,stream_transfer));

        A2 = next_level;

        int smoother = TDatabase::ParamDB->SC_SMOOTHER_SCALAR;

        // if(smooth == -1 || smooth == 0 )
        // CUDA_CHECK(cudaMemcpyAsync(d_sol2, next_level1->GetSolution(), n * sizeof(double), cudaMemcpyHostToDevice,stream_transfer));
        
        // TParFECommunicator3D *ParComm = next_level1->GetParComm();
        // TParFEMapper3D *ParMapper = next_level1->GetParMapper();

        // if(smoother == 10 || smoother == 11 || smoother == 12 || smoother == 13){
        //   CUDA_CHECK(cudaMemcpyAsync(d_aux2, ParComm->GetMaster() , n * sizeof(int), cudaMemcpyHostToDevice,stream_transfer));
        // }

        // if(smoother == 20 || smoother == 21 || smoother == 22 || smoother == 23){
        //   CUDA_CHECK(cudaMemcpyAsync(d_aux2, ParMapper->GetReorder_M() , ParMapper->GetN_OwnDof() * sizeof(int), cudaMemcpyHostToDevice,stream_transfer));
        // }

      }

      void TMultiGrid3D::Swap_Matrices(){
        int* t;
        double* t1;

        int t2;

        t2 = A1;
        A1 = A2;
        A2 = t2;

        t = d_RowPtr1;
        d_RowPtr1 = d_RowPtr2;
        d_RowPtr2 = t;

        t = d_KCol1;
        d_KCol1 = d_KCol2;
        d_KCol2 = t;

        t1 = d_Entries1;
        d_Entries1 = d_Entries2;
        d_Entries2 = t1;

        t1 = d_sol1;
        d_sol1 = d_sol2;
        d_sol2 = t1;

        t = d_aux1;
        d_aux1 = d_aux2;
        d_aux2 = t;

      }

#endif


/** add new level as finest */
void TMultiGrid3D::AddLevel(TMGLevel3D *MGLevel)
{ 
  MultiGridLevels[N_Levels] = MGLevel;

  FESpaces[N_Levels] = MGLevel->GetFESpace();

  N_Levels++;
}
/** add new level as finest */
void TMultiGrid3D::ReplaceLevel(int i,TMGLevel3D *MGLevel)
{ 
  TMGLevel3D *ret;

  ret = MultiGridLevels[i];
  MultiGridLevels[i] = MGLevel;

//  MultiGridLevels[N_Levels] = MGLevel;

  FESpaces[i] = MGLevel->GetFESpace();

  if (i>=N_Levels)
    N_Levels = i+1;
}

/** restrict solution from finest grid to all coarser grids */
void TMultiGrid3D::RestrictToAllGrids()
{
  int lev, j;
  TMGLevel3D *CurrentLevel, *CoarserLevel;
  double *X1, *R1;
  double *X2, *R2, *Aux;

  for(lev=N_Levels-1;lev>0;lev--)
  {
    CurrentLevel = MultiGridLevels[lev];
    X1 = CurrentLevel->GetSolution();

    CoarserLevel = MultiGridLevels[lev-1];
    X2 = CoarserLevel->GetSolution();
    Aux = CoarserLevel->GetAuxVector(0);

    RestrictFunction(FESpaces[lev-1], FESpaces[lev], X2, 
                     X1, Aux);
  } // endfor lev
} // RestrictToAllGrids

void TMultiGrid3D::Smooth(int smoother_type, TMGLevel3D *Level, 
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
        )
{
  int i,j,k;
  double *CurrentSol, *CurrentRhs, *CurrentDefect, *CurrentAux;
#ifdef _HYBRID
  bool firstTime, LastTime;
#endif
  
  CurrentSol    = Level->GetSolution();
  CurrentRhs    = Level->GetRhs();
  CurrentDefect = Level->GetAuxVector(0);
  CurrentAux    = Level->GetAuxVector(1);
  
  if(smoother_type == PreSmooth)			//presmoothing
  {
    switch(TDatabase::ParamDB->SC_SMOOTHER_SCALAR)
    {
      case 1: // Jacobi
        for(j=0;j<TDatabase::ParamDB->SC_PRE_SMOOTH_SCALAR;j++)
	{
          Level->Jacobi(CurrentSol, CurrentRhs, CurrentAux,
                N_Parameters, Parameters);
#ifdef _MPI  
         ParComm->CommUpdate(CurrentSol);
#endif
	}
        break;
      case 2: // SOR
        for(j=0;j<TDatabase::ParamDB->SC_PRE_SMOOTH_SCALAR;j++)
	{
          Level->SOR(CurrentSol, CurrentRhs, CurrentAux,
                N_Parameters, Parameters);
#ifdef _MPI  
         ParComm->CommUpdate(CurrentSol);
#endif
	}
        break;
      case 3: // SSOR
        for(j=0;j<TDatabase::ParamDB->SC_PRE_SMOOTH_SCALAR;j++)
	{
          Level->SSOR(CurrentSol, CurrentRhs, CurrentAux,
                N_Parameters, Parameters);
#ifdef _MPI  
         ParComm->CommUpdate(CurrentSol);
#endif
	}
        break;
      case 4: // ILU
        for(j=0;j<TDatabase::ParamDB->SC_PRE_SMOOTH_SCALAR;j++)
        {
          Level->Defect(CurrentSol, CurrentRhs, CurrentDefect, 
                oldres);
          Level->ILU(CurrentSol, CurrentRhs, CurrentDefect,
                N_Parameters, Parameters);
        }
        break;
#ifdef _MPI
  #ifdef _HYBRID
	case 5: //SOR_Reorder
	printf("Not Working\n");
	MPI_Finalize();
	exit(0);
        break;
  #else	
	case 5: //SOR_Reorder
	for(j=0;j<TDatabase::ParamDB->SC_PRE_SMOOTH_SCALAR;j++)
	{
          Level->SOR(CurrentSol, CurrentRhs, CurrentAux,
                N_Parameters, Parameters);
	}
        break;
  #endif
#endif

#ifdef _MPI
  #ifdef _HYBRID
	case 6: //SOR Reorder and color
	  Level->SOR_Re_Color(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, PreSmooth);
        break;
  #else	
	case 6: //SOR_Reorder
	for(j=0;j<TDatabase::ParamDB->SC_PRE_SMOOTH_SCALAR;j++)
	{
          Level->SOR_Re(CurrentSol, CurrentRhs, CurrentAux,
                N_Parameters, Parameters);
	}
        break;
  #endif
#ifdef _MPI
	case 7: //SOR_Reorder
	for(j=0;j<TDatabase::ParamDB->SC_PRE_SMOOTH_SCALAR;j++)
	{
          Level->SOR_Re(CurrentSol, CurrentRhs, CurrentAux,
                N_Parameters, Parameters);
	}
        break;
#endif
#endif	

#ifdef _CUDA 
    case 10: //Jacobi GPU
    
    Level->Jacobi_GPU(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, PreSmooth, stream_smooth, d_RowPtr, d_KCol, d_Entries, d_sol, d_aux);

    break;
    
    #ifdef _MPI
        #ifdef _HYBRID

                // case 11: //SOR_Re GPU
                
                // Level->Jacobi_Level_Split(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, PreSmooth);

                // break;

                case 12: //SOR_Re GPU

                Level->Jacobi_CPU_GPU(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, PreSmooth, stream_smooth, d_RowPtr, d_KCol, d_Entries, d_sol, d_aux);

                break;

                case 13: //SOR_Re GPU

                Level->Jacobi_Combo(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, PreSmooth, stream_smooth, d_RowPtr, d_KCol, d_Entries, d_sol, d_aux);

                break;
                
                case 20: //SOR_Re GPU
                
                Level->SOR_Re_GPU(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, PreSmooth, stream_smooth, d_RowPtr, d_KCol, d_Entries, d_sol, d_aux);
                
                break;
                
                case 21: //SOR_Re GPU
                
                Level->SOR_Re_CPU_GPU(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, PreSmooth, stream_smooth, d_RowPtr, d_KCol, d_Entries, d_sol, d_aux);
                
                break;
                
                // case 22: //SOR_Re GPU
                
                // Level->SOR_Re_Level_Split(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, PreSmooth);
                
                // break;
                
                case 23: //SOR_Re GPU
                
                Level->SOR_Re_Combo(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, PreSmooth, stream_smooth, d_RowPtr, d_KCol, d_Entries, d_sol, d_aux);
                
                break;
        #endif
    #endif

#endif
    
      default:
        for(j=0;j<TDatabase::ParamDB->SC_PRE_SMOOTH_SCALAR;j++)
          Level->SOR(CurrentSol, CurrentRhs, CurrentAux,
            N_Parameters, Parameters);
    } // endswitch SC_SMOOTHER
  }
  else if(smoother_type == CoarseSmooth)
  {
#ifdef _MPI  
  int rank;
  MPI_Comm_rank(TDatabase::ParamDB->Comm, &rank); 
#endif  
    double res;
    int it = 0;
    int maxit =  TDatabase::ParamDB->SC_COARSE_MAXIT_SCALAR;
    
    Level->Defect(CurrentSol, CurrentRhs, CurrentDefect, res);
     if(TDatabase::ParamDB->SC_VERBOSE>=2 
#ifdef _MPI  
        && rank==TDatabase::ParamDB->Par_P0
#endif  
     )
     {
      OutPut("residual before on coarse "<<res << endl);
     }
     
    double reduction = TDatabase::ParamDB->SC_COARSE_RED_FACTOR_SCALAR*res;
  
    while ((res>reduction)&&(it<maxit))
    {  
      switch(TDatabase::ParamDB->SC_COARSE_SMOOTHER_SCALAR)
            {
                 case 1: // Jacobi
                         Level->Jacobi(CurrentSol, CurrentRhs, CurrentAux,
                                                       N_Parameters, Parameters);
#ifdef _MPI  
                         ParComm->CommUpdate(CurrentSol);
#endif
                         break;
		       
                 case 2: // SOR
                         Level->SOR(CurrentSol, CurrentRhs, CurrentAux,
                                                     N_Parameters, Parameters);
#ifdef _MPI  
                         ParComm->CommUpdate(CurrentSol);
#endif
                         break;
		       
                 case 3: // SSOR
                         Level->SSOR(CurrentSol, CurrentRhs, CurrentAux,
                                                    N_Parameters, Parameters);
#ifdef _MPI  
                         ParComm->CommUpdate(CurrentSol);
#endif
                         break;
        
	         case 4: // ILU
                         Level->ILU(CurrentSol, CurrentRhs, CurrentDefect,
                                                        N_Parameters, Parameters);
                         break;
		       
                 case 17: // solution with Gaussian elimination
                          Level->SolveExact(CurrentSol, CurrentRhs);
                          break;
#ifdef _MPI
  #ifdef _HYBRID
	         case 5: //SOR_Reorder
	                 printf("Not Working\n");
	                 MPI_Finalize();
	                 exit(0);
                         break;
  #else
                 case 5: //SOR_Reorder
	                 for(j=0;j<TDatabase::ParamDB->SC_PRE_SMOOTH_SCALAR;j++)
	                 {
                               Level->SOR(CurrentSol, CurrentRhs, CurrentAux,
                                                            N_Parameters, Parameters);
	                 }
                         break;
  #endif
#endif

#ifdef _MPI
  #ifdef _HYBRID
	  case 6: //SOR_Reorder
	    Level->SOR_Re_Color(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, CoarseSmooth);
          break;
  #else	
	  case 6: //SOR_Reorder
            Level->SOR_Re(CurrentSol, CurrentRhs, CurrentAux,
                  N_Parameters, Parameters);
          break;
  #endif
		
	   case 7: //SOR_Reorder
            Level->SOR_Re(CurrentSol, CurrentRhs, CurrentAux,
                  N_Parameters, Parameters);
          break;
#endif	  

#ifdef _CUDA          
        case 10: //Jacobi gpu
        
        Level->Jacobi_GPU(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, CoarseSmooth, stream_smooth, d_RowPtr, d_KCol, d_Entries, d_sol, d_aux);
        
        it=maxit; //to break while loop
        
        break;
                
        #ifdef _MPI
            #ifdef _HYBRID
                // case 11: //Jacobi GPU
                
                // Level->Jacobi_Level_Split(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, CoarseSmooth);
                // it=maxit; //to break while loop
                
                // break;
                
                case 12: //Jacobi GPU
                
                Level->Jacobi_CPU_GPU(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, CoarseSmooth, stream_smooth, d_RowPtr, d_KCol, d_Entries, d_sol, d_aux);
                it=maxit; //to break while loop
                
                break;
                
                case 13: //Jacobi GPU
                
                Level->Jacobi_Combo(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, CoarseSmooth, stream_smooth, d_RowPtr, d_KCol, d_Entries, d_sol, d_aux);
                it=maxit;
                
                break;
                
                case 20: //SOR_Re GPU
                
                Level->SOR_Re_GPU(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, CoarseSmooth, stream_smooth, d_RowPtr, d_KCol, d_Entries, d_sol, d_aux);
                it=maxit; //to break while loop
                
                break;
                
                case 21: //SOR_Re GPU
                
                Level->SOR_Re_CPU_GPU(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, CoarseSmooth, stream_smooth, d_RowPtr, d_KCol, d_Entries, d_sol, d_aux);
                it=maxit; //to break while loop
                
                break;
                
                // case 22: //SOR_Re GPU
                
                // Level->SOR_Re_Level_Split(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, CoarseSmooth);
                // it=maxit; //to break while loop
                
                // break;
                
                case 23: //SOR_Re GPU
                
                Level->SOR_Re_Combo(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, CoarseSmooth, stream_smooth, d_RowPtr, d_KCol, d_Entries, d_sol, d_aux);
                it=maxit; //to break while loop
                
                break;
                
                
            #endif
        #endif

#endif
        
        default:
                       OutPut("Coarse smoother not implemented !! Use coarse smoother 3" << endl);
                       Level->SSOR(CurrentSol, CurrentRhs, CurrentAux,
                                                    N_Parameters, Parameters);
#ifdef _MPI  
                       ParComm->CommUpdate(CurrentSol);
#endif
          } // endswitch SC_COARSE_SMOOTHER_SCALAR
          
       Level->Defect(CurrentSol, CurrentRhs, CurrentDefect, res);
       it++;
       if(TDatabase::ParamDB->SC_VERBOSE>=2
#ifdef _MPI  
        && rank==TDatabase::ParamDB->Par_P0
#endif
        )
         OutPut("itr no. :: "<<it-1<<"        res on coarse: " << res << endl);   
    }//endwhile
    oldres = res;
  }
  else if(smoother_type == PostSmooth)
  {
    switch(TDatabase::ParamDB->SC_SMOOTHER_SCALAR)
    {
      case 1: // Jacobi
        for(j=0;j<TDatabase::ParamDB->SC_POST_SMOOTH_SCALAR;j++)
	{
          Level->Jacobi(CurrentSol, CurrentRhs, CurrentAux,
                N_Parameters, Parameters);
#ifdef _MPI  
          // communicate the values (sol & rhs) to the slave DOFs from master DOF
          ParComm->CommUpdate(CurrentSol);
#endif 
	}
        break;
      case 2: // SOR
        for(j=0;j<TDatabase::ParamDB->SC_POST_SMOOTH_SCALAR;j++)
	{
          Level->SOR(CurrentSol, CurrentRhs, CurrentAux,
                N_Parameters, Parameters);
#ifdef _MPI  
          // communicate the values (sol & rhs) to the slave DOFs from master DOF
          ParComm->CommUpdate(CurrentSol);
#endif 
	}
        break;
      case 3: // SSOR
        for(j=0;j<TDatabase::ParamDB->SC_POST_SMOOTH_SCALAR;j++)
	{
          Level->SSOR(CurrentSol, CurrentRhs, CurrentAux,
                N_Parameters, Parameters);
#ifdef _MPI  
          // communicate the values (sol & rhs) to the slave DOFs from master DOF
          ParComm->CommUpdate(CurrentSol);
#endif 
	}
        break;
      case 4: // ILU
        for(j=0;j<TDatabase::ParamDB->SC_POST_SMOOTH_SCALAR;j++)
        {
          Level->Defect(CurrentSol, CurrentRhs, CurrentDefect, 
                oldres);
          Level->ILU(CurrentSol, CurrentRhs, CurrentDefect,
                N_Parameters, Parameters);
        }
        break;
#ifdef _MPI
  #ifdef _HYBRID
	case 5: //SOR_Reorder
	printf("Not Working\n");
	MPI_Finalize();
	exit(0);
        break;
  #else	
	case 5: //SOR_Reorder
	for(j=0;j<TDatabase::ParamDB->SC_PRE_SMOOTH_SCALAR;j++)
	{
          Level->SOR(CurrentSol, CurrentRhs, CurrentAux,
                N_Parameters, Parameters);
	}
        break;
  #endif
#endif

#ifdef _MPI
  #ifdef _HYBRID
	case 6: //SOR Reorder and color

	  Level->SOR_Re_Color(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, PostSmooth);

        break;
  #else	
	case 6: //SOR_Reorder
	for(j=0;j<TDatabase::ParamDB->SC_POST_SMOOTH_SCALAR;j++)
	{
          Level->SOR_Re(CurrentSol, CurrentRhs, CurrentAux,
                N_Parameters, Parameters);
	}
        break;
  #endif	

	case 7: //SOR_Reorder
	for(j=0;j<TDatabase::ParamDB->SC_POST_SMOOTH_SCALAR;j++)
	{
          Level->SOR_Re(CurrentSol, CurrentRhs, CurrentAux,
                N_Parameters, Parameters);
	}
        break;
#endif	

#ifdef _CUDA        
    case 10: //Jacobi gpu
    
    Level->Jacobi_GPU(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, PostSmooth, stream_smooth, d_RowPtr, d_KCol, d_Entries, d_sol, d_aux);
    
    break;
    
    #ifdef _MPI
        #ifdef _HYBRID

                // case 11: //SOR_Re GPU
                
                // Level->Jacobi_Level_Split(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, PostSmooth);
                
                // break;
                
                case 12: //SOR_Re GPU
                
                Level->Jacobi_CPU_GPU(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, PostSmooth, stream_smooth, d_RowPtr, d_KCol, d_Entries, d_sol, d_aux);
                
                break;
                
                case 13: //SOR_Re GPU
                
                Level->Jacobi_Combo(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, PostSmooth, stream_smooth, d_RowPtr, d_KCol, d_Entries, d_sol, d_aux);
                
                break;
                
                case 20: //SOR_Re GPU
                
                Level->SOR_Re_GPU(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, PostSmooth, stream_smooth, d_RowPtr, d_KCol, d_Entries, d_sol, d_aux);
                
                break;
                
                case 21: //SOR_Re GPU
                
                Level->SOR_Re_CPU_GPU(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, PostSmooth, stream_smooth, d_RowPtr, d_KCol, d_Entries, d_sol, d_aux);
                
                break;
                
                // case 22: //SOR_Re GPU
                
                // Level->SOR_Re_Level_Split(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, PostSmooth);
                
                // break;
                
                case 23: //SOR_Re GPU
                
                Level->SOR_Re_Combo(CurrentSol, CurrentRhs, CurrentAux, N_Parameters, Parameters, PostSmooth, stream_smooth, d_RowPtr, d_KCol, d_Entries, d_sol, d_aux);
                
                break;
        #endif
    #endif

#endif
    
      default:
        for(j=0;j<TDatabase::ParamDB->SC_POST_SMOOTH_SCALAR;j++)
	{
          Level->SOR(CurrentSol, CurrentRhs, CurrentAux,
            N_Parameters, Parameters);
#ifdef _MPI  
          // communicate the values (sol & rhs) to the slave DOFs from master DOF
          ParComm->CommUpdate(CurrentSol);
#endif 
	}
    } // endswitch SC_SMOOTHER_SCALAR
  }
  else{
    printf("Wrong smoother_type \n");
#ifdef _MPI
    MPI_Finalize();
#endif
    exit(0);
    }
}


/** one cycle on level i */
void TMultiGrid3D::Cycle(int i, double &res)
{
  double s,t1,t2;
  
  TMGLevel3D *CurrentLevel, *CoarserLevel;
  double *CurrentSol, *CoarserSol, *CoarserRhs;
  double *CurrentRhs, *CurrentDefect, *CurrentAux;
  double *CurrentAux2, *OldSol, *OldDefect;
  double oldres,reduction, alpha;
  int j, N_DOF, maxit, it, slc,gam;
  double initres, normsol, firstres;

  CurrentLevel = MultiGridLevels[i];
  CurrentDefect = CurrentLevel->GetAuxVector(0);
  CurrentAux = CurrentLevel->GetAuxVector(1);
  slc =0;
  if ((TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_ALL_SCALAR)||
      (TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_FINE_SCALAR))
    slc = 1;
  if (slc)
  {
    OldSol = CurrentLevel->GetAuxVector(2);
    OldDefect = CurrentLevel->GetAuxVector(3);
  }
  CurrentAux2 = CurrentDefect;
  CurrentSol = CurrentLevel->GetSolution();
  CurrentRhs = CurrentLevel->GetRhs();
  N_DOF = CurrentLevel->GetN_DOF();
  
#ifdef _MPI  
  TParFECommunicator3D *ParComm, *CoarseParComm; 
  int rank;
 
   ParComm = CurrentLevel->GetParComm();  

   MPI_Comm_rank(TDatabase::ParamDB->Comm, &rank); 
#endif  
 // OutPut("Norm of B rhs in cycle " <<  sqrt(Ddot(N_DOF,CurrentRhs,CurrentRhs)) <<"i is:"<<i <<endl); 
  
  
  if(i==0)
  {
    PUSH_RANGE("coarse_solve",1)
    // coarse grid
//     cout << "coarse grid" << endl;
//     res = 1;
//     maxit =  TDatabase::ParamDB->SC_COARSE_MAXIT_SCALAR;
//     it = 0;
//     CurrentLevel->Defect(CurrentSol, CurrentRhs, CurrentDefect, res);
    
//    if(TDatabase::ParamDB->SC_VERBOSE>=2 
// #ifdef _MPI  
//         && rank==TDatabase::ParamDB->Par_P0
// #endif  
//      )
//      {
//       OutPut("residual before on coarse "<<res << endl);
//      }
     
//     reduction = TDatabase::ParamDB->SC_COARSE_RED_FACTOR_SCALAR*res;
//     while ((res>reduction)&&(it<maxit))
//     {
#ifdef _MPI
     t1 = MPI_Wtime();
     Smooth(CoarseSmooth, CurrentLevel, ParComm, res
     #ifdef _CUDA
            ,stream_smooth
          ,NULL
          ,NULL
          ,NULL
          ,NULL
          ,NULL
     #endif
     ); 
     t2 = MPI_Wtime();
#else
     t1 = GetTime();
     Smooth(CoarseSmooth, CurrentLevel, res, NULL, NULL, NULL, NULL, NULL, NULL );
     t2 = GetTime();
#endif
     tSmoother += t2-t1 ;
     
//       CurrentLevel->Defect(CurrentSol, CurrentRhs, CurrentDefect, res);
//       it++;
//       if(TDatabase::ParamDB->SC_VERBOSE>=2
// #ifdef _MPI  
//         && rank==TDatabase::ParamDB->Par_P0
// #endif
//         )
//          OutPut("itr no. :: "<<it-1<<"        res on coarse: " << res << endl);
//     }//end while
POP_RANGE
  }
  
  else
  {
    slc =0;

    if (TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_ALL_SCALAR)
      slc = 1;
    else if ((TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_FINE_SCALAR)
             &&(i==N_Levels-1))
      slc = 1;
    
    CoarserLevel = MultiGridLevels[i-1];
    CoarserSol = CoarserLevel->GetSolution();
    CoarserRhs = CoarserLevel->GetRhs();
  
#ifdef _MPI  
  CoarseParComm = CoarserLevel->GetParComm();   
#endif  
    // smoothing
    CurrentLevel->Defect(CurrentSol, CurrentRhs, CurrentDefect, oldres);  
    firstres = initres = oldres;  
    normsol = sqrt(Ddot(N_DOF, CurrentSol, CurrentSol));
  
    if(TDatabase::ParamDB->SC_VERBOSE>=2
#ifdef _MPI  
        && rank==TDatabase::ParamDB->Par_P0
#endif   
      )
      {
       OutPut("level " << i << " ");
       OutPut("res before presmoothing: " << oldres << endl);
      }

    if (slc)
    {
      memcpy(OldSol, CurrentSol, N_DOF*SizeOfDouble);
      memcpy(OldDefect, CurrentDefect, N_DOF*SizeOfDouble);
    }

#ifdef _MPI
     t1 = MPI_Wtime();
     Smooth(PreSmooth, CurrentLevel, ParComm, oldres
    #ifdef _CUDA
      ,stream_smooth
          ,NULL
          ,NULL
          ,NULL
          ,NULL
          ,NULL
     #endif
     ); 
     t2 = MPI_Wtime();
#else
     t1 = GetTime();
     Smooth(PreSmooth, CurrentLevel, oldres, NULL, NULL, NULL, NULL, NULL, NULL );
     t2 = GetTime();
#endif
     tSmoother += t2-t1 ;
    
    PUSH_RANGE("defect", 6)
    // calculate defect
    CurrentLevel->Defect(CurrentSol, CurrentRhs, CurrentDefect, oldres);
    
    if (TDatabase::ParamDB->SC_VERBOSE>=2
#ifdef _MPI  
        && rank==TDatabase::ParamDB->Par_P0
#endif 
       )
      {
         OutPut("normsol: " << normsol << " oldres: " << oldres << endl);
        OutPut("level " << i << " ");
        OutPut("res after presmoothing: " << oldres << endl);
         OutPut("Smoothing (" << i << "): " << oldres/normsol << endl);
      }

      POP_RANGE
    // restrict defect
//     exit(0);

PUSH_RANGE("restriction", 2)
#ifdef _MPI  
        ParComm->CommUpdate(CurrentDefect);
  
	memcpy(CurrentLevel->GetTemp_arr(),CurrentDefect,CurrentLevel->GetN_DOF());
    
	DefectRestriction(CoarserLevel->GetFESpace(), CurrentLevel->GetFESpace(),
                           CoarserRhs, CurrentDefect,
                           CurrentAux);
// 	OutPut("restriction "<<endl);
// 	exit(0);

	  CoarseParComm->CommUpdateReduce(CoarserRhs);	 

// 	OutPut("2.restriction "<<endl);
	//exit(0);
#else
        DefectRestriction(FESpaces[i-1], FESpaces[i],CoarserRhs, CurrentDefect, CurrentAux);
#endif
POP_RANGE

    CoarserLevel->CorrectDefect(CoarserRhs);  //non-active part set to 0
    CoarserLevel->Reset(CoarserSol);          //all set to 0
     
    // coarse grid correction
    // coarse grid correction, apply mg recursively*/
    for(j=0;j<mg_recursions[i];j++)
       Cycle(i-1, res);
    if (TDatabase::ParamDB->SC_MG_CYCLE_SCALAR<1) mg_recursions[i] = 1;              // F--cycle 

PUSH_RANGE("prolongate",3)
    // prolongate correction
#ifdef _MPI  
      Prolongate(CoarserLevel->GetFESpace(), CurrentLevel->GetFESpace(),
                       CoarserSol, CurrentLevel->GetTemp_arr(),
                       CurrentLevel->GetAuxVector(1));


	ParComm->CommUpdateReduce(CurrentLevel->GetTemp_arr());

      
      CurrentLevel->Update(CurrentSol, CurrentLevel->GetTemp_arr());
      
      ParComm->CommUpdate(CurrentSol);
#else 
    Prolongate(FESpaces[i-1], FESpaces[i], 
                   CoarserSol, CurrentAux2, CurrentAux);

    CurrentLevel->CorrectNodes(CurrentAux2);

    CurrentLevel->Update(CurrentSol, CurrentAux2);
#endif  
POP_RANGE
    CurrentLevel->Defect(CurrentSol, CurrentRhs, CurrentDefect, oldres);
  
    initres = oldres;
    normsol = sqrt(Ddot(N_DOF, CurrentSol, CurrentSol));
    
    if (TDatabase::ParamDB->SC_VERBOSE>=2
#ifdef _MPI  
        && rank==TDatabase::ParamDB->Par_P0
#endif  
       )
      {
        OutPut("level " << i << " ");
        OutPut("res before postsmoothing: " << oldres << endl);
      }
//        exit(0);
    // smoothing
#ifdef _MPI
     t1 = MPI_Wtime();
     Smooth(PostSmooth, CurrentLevel, ParComm, oldres
    #ifdef _CUDA
            ,stream_smooth
          ,NULL
          ,NULL
          ,NULL
          ,NULL
          ,NULL
    #endif
     ); 
     t2 = MPI_Wtime();
#else
     t1 = GetTime();
     Smooth(PostSmooth, CurrentLevel, oldres, NULL, NULL, NULL, NULL, NULL, NULL );
     t2 = GetTime();
#endif
     tSmoother += t2-t1 ;
     
    if (slc)
    {
      alpha = CurrentLevel->StepLengthControl(CurrentSol, OldSol,
                          OldDefect,                  
                          N_Parameters,Parameters);       
      
      for (j=0;j<N_DOF;j++)
        CurrentSol[j] = OldSol[j] + alpha *( CurrentSol[j]-OldSol[j]);
    }

    CurrentLevel->Defect(CurrentSol, CurrentRhs, CurrentDefect, res);
    
    if (TDatabase::ParamDB->SC_VERBOSE>=2
#ifdef _MPI  
        && rank==TDatabase::ParamDB->Par_P0
#endif  
    )
      {
        OutPut("level " << i << " ");
        OutPut("res after postsmoothing: " << res);
        OutPut(" rate: " << res/firstres << endl);
        // OutPut("Smoothing2 (" << i << "): " << initres/normsol << endl);
      }
  }
}


// void TMultiGrid3D::CycleIterative(double &res)
// {
//   double s,t1,t2;
//   cout<<"new cycle"<<endl;
//   TMGLevel3D *CurrentLevel, *CoarserLevel;
//   double *CurrentSol, *CoarserSol, *CoarserRhs;
//   double *CurrentRhs, *CurrentDefect, *CurrentAux;
//   double *CurrentAux2, *OldSol, *OldDefect;
//   double oldres,reduction, alpha;
//   int j, N_DOF, maxit, it, slc,gam;
//   double initres, normsol, firstres;

//   int i;
  
//   for(i=N_Levels-1; i>0; i--){

//       CurrentLevel = MultiGridLevels[i];
//       CurrentDefect = CurrentLevel->GetAuxVector(0);
//       CurrentAux = CurrentLevel->GetAuxVector(1);
//       slc =0;
//       if ((TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_ALL_SCALAR)||
//           (TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_FINE_SCALAR))
//         slc = 1;
//       if (slc)
//       {
//         OldSol = CurrentLevel->GetAuxVector(2);
//         OldDefect = CurrentLevel->GetAuxVector(3);
//       }
//       CurrentAux2 = CurrentDefect;
//       CurrentSol = CurrentLevel->GetSolution();
//       CurrentRhs = CurrentLevel->GetRhs();
//       N_DOF = CurrentLevel->GetN_DOF();
      
//     #ifdef _MPI  
//       TParFECommunicator3D *ParComm, *CoarseParComm; 
//       int rank;
    
//       ParComm = CurrentLevel->GetParComm();  

//       MPI_Comm_rank(TDatabase::ParamDB->Comm, &rank); 
//     #endif  
//     // OutPut("Norm of B rhs in cycle " <<  sqrt(Ddot(N_DOF,CurrentRhs,CurrentRhs)) <<"i is:"<<i <<endl); 
      
//       slc =0;

//         if (TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_ALL_SCALAR)
//           slc = 1;
//         else if ((TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_FINE_SCALAR)
//                 &&(i==N_Levels-1))
//           slc = 1;
        
//         CoarserLevel = MultiGridLevels[i-1];
//         CoarserSol = CoarserLevel->GetSolution();
//         CoarserRhs = CoarserLevel->GetRhs();
      
//     #ifdef _MPI  
//       CoarseParComm = CoarserLevel->GetParComm();   
//     #endif  
//         // smoothing
//         CurrentLevel->Defect(CurrentSol, CurrentRhs, CurrentDefect, oldres);  
//         firstres = initres = oldres;  
//         normsol = sqrt(Ddot(N_DOF, CurrentSol, CurrentSol));
      
//         if(TDatabase::ParamDB->SC_VERBOSE>=2
//     #ifdef _MPI  
//             && rank==TDatabase::ParamDB->Par_P0
//     #endif   
//           )
//           {
//           OutPut("level " << i << " ");
//           OutPut("res before presmoothing: " << oldres << endl);
//           }

//         if (slc)
//         {
//           memcpy(OldSol, CurrentSol, N_DOF*SizeOfDouble);
//           memcpy(OldDefect, CurrentDefect, N_DOF*SizeOfDouble);
//         }
        


//     #ifdef _MPI
//         t1 = MPI_Wtime();
//         Smooth(PreSmooth, CurrentLevel, ParComm, oldres
//         #ifdef _CUDA
//           ,stream_smooth
//         #endif
//         ); 
//         t2 = MPI_Wtime();
//     #else
//         t1 = GetTime();
//         Smooth(PreSmooth, CurrentLevel, oldres);
//         t2 = GetTime();
//     #endif
//         tSmoother += t2-t1 ;
        
//         PUSH_RANGE("defect", 6)
//         // calculate defect
//         CurrentLevel->Defect(CurrentSol, CurrentRhs, CurrentDefect, oldres);
        
//         if (TDatabase::ParamDB->SC_VERBOSE>=2
//     #ifdef _MPI  
//             && rank==TDatabase::ParamDB->Par_P0
//     #endif 
//           )
//           {
//             OutPut("normsol: " << normsol << " oldres: " << oldres << endl);
//             OutPut("level " << i << " ");
//             OutPut("res after presmoothing: " << oldres << endl);
//             OutPut("Smoothing (" << i << "): " << oldres/normsol << endl);
//           }

//           POP_RANGE
//         // restrict defect
//     //     exit(0);

//     PUSH_RANGE("restriction", 2)
//     #ifdef _MPI  
//             ParComm->CommUpdate(CurrentDefect);
      
//       memcpy(CurrentLevel->GetTemp_arr(),CurrentDefect,CurrentLevel->GetN_DOF());
        
//       DefectRestriction(CoarserLevel->GetFESpace(), CurrentLevel->GetFESpace(),
//                               CoarserRhs, CurrentDefect,
//                               CurrentAux);
//     // 	OutPut("restriction "<<endl);
//     // 	exit(0);

//         CoarseParComm->CommUpdateReduce(CoarserRhs);	 

//     // 	OutPut("2.restriction "<<endl);
//       //exit(0);
//     #else
//             DefectRestriction(FESpaces[i-1], FESpaces[i],CoarserRhs, CurrentDefect, CurrentAux);
//     #endif
//     POP_RANGE

//         CoarserLevel->CorrectDefect(CoarserRhs);  //non-active part set to 0
//         CoarserLevel->Reset(CoarserSol);          //all set to 0

//   }
  
//   if(i==0)
//   {

//           CurrentLevel = MultiGridLevels[i];
//       CurrentDefect = CurrentLevel->GetAuxVector(0);
//       CurrentAux = CurrentLevel->GetAuxVector(1);
//       slc =0;
//       if ((TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_ALL_SCALAR)||
//           (TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_FINE_SCALAR))
//         slc = 1;
//       if (slc)
//       {
//         OldSol = CurrentLevel->GetAuxVector(2);
//         OldDefect = CurrentLevel->GetAuxVector(3);
//       }
//       CurrentAux2 = CurrentDefect;
//       CurrentSol = CurrentLevel->GetSolution();
//       CurrentRhs = CurrentLevel->GetRhs();
//       N_DOF = CurrentLevel->GetN_DOF();
      
//     #ifdef _MPI  
//       TParFECommunicator3D *ParComm, *CoarseParComm; 
//       int rank;
    
//       ParComm = CurrentLevel->GetParComm();  

//       MPI_Comm_rank(TDatabase::ParamDB->Comm, &rank); 
//     #endif  
//     // OutPut("Norm of B rhs in cycle " <<  sqrt(Ddot(N_DOF,CurrentRhs,CurrentRhs)) <<"i is:"<<i <<endl); 
      
//       slc =0;

//         if (TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_ALL_SCALAR)
//           slc = 1;
//         else if ((TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_FINE_SCALAR)
//                 &&(i==N_Levels-1))
//           slc = 1;
        
//         CoarserLevel = MultiGridLevels[i-1];
//         CoarserSol = CoarserLevel->GetSolution();
//         CoarserRhs = CoarserLevel->GetRhs();
      
//     #ifdef _MPI  
//       CoarseParComm = CoarserLevel->GetParComm();   
//     #endif  
//         // smoothing
//         CurrentLevel->Defect(CurrentSol, CurrentRhs, CurrentDefect, oldres);  
//         firstres = initres = oldres;  
//         normsol = sqrt(Ddot(N_DOF, CurrentSol, CurrentSol));

//     PUSH_RANGE("coarse_solve",1)
//     // coarse grid
// //     cout << "coarse grid" << endl;
// //     res = 1;
// //     maxit =  TDatabase::ParamDB->SC_COARSE_MAXIT_SCALAR;
// //     it = 0;
// //     CurrentLevel->Defect(CurrentSol, CurrentRhs, CurrentDefect, res);
    
// //    if(TDatabase::ParamDB->SC_VERBOSE>=2 
// // #ifdef _MPI  
// //         && rank==TDatabase::ParamDB->Par_P0
// // #endif  
// //      )
// //      {
// //       OutPut("residual before on coarse "<<res << endl);
// //      }
     
// //     reduction = TDatabase::ParamDB->SC_COARSE_RED_FACTOR_SCALAR*res;
// //     while ((res>reduction)&&(it<maxit))
// //     {
// #ifdef _MPI
//      t1 = MPI_Wtime();
//      Smooth(CoarseSmooth, CurrentLevel, ParComm, res
//      #ifdef _CUDA
//       ,stream_smooth
//      #endif
//      ); 
//      t2 = MPI_Wtime();
// #else
//      t1 = GetTime();
//      Smooth(CoarseSmooth, CurrentLevel, res);
//      t2 = GetTime();
// #endif
//      tSmoother += t2-t1 ;
     
// //       CurrentLevel->Defect(CurrentSol, CurrentRhs, CurrentDefect, res);
// //       it++;
// //       if(TDatabase::ParamDB->SC_VERBOSE>=2
// // #ifdef _MPI  
// //         && rank==TDatabase::ParamDB->Par_P0
// // #endif
// //         )
// //          OutPut("itr no. :: "<<it-1<<"        res on coarse: " << res << endl);
// //     }//end while
// POP_RANGE
//   }
  
//   for(int i=1; i<N_Levels; i++)
//   {

//           CurrentLevel = MultiGridLevels[i];
//       CurrentDefect = CurrentLevel->GetAuxVector(0);
//       CurrentAux = CurrentLevel->GetAuxVector(1);
//       slc =0;
//       if ((TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_ALL_SCALAR)||
//           (TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_FINE_SCALAR))
//         slc = 1;
//       if (slc)
//       {
//         OldSol = CurrentLevel->GetAuxVector(2);
//         OldDefect = CurrentLevel->GetAuxVector(3);
//       }
//       CurrentAux2 = CurrentDefect;
//       CurrentSol = CurrentLevel->GetSolution();
//       CurrentRhs = CurrentLevel->GetRhs();
//       N_DOF = CurrentLevel->GetN_DOF();
      
//     #ifdef _MPI  
//       TParFECommunicator3D *ParComm, *CoarseParComm; 
//       int rank;
    
//       ParComm = CurrentLevel->GetParComm();  

//       MPI_Comm_rank(TDatabase::ParamDB->Comm, &rank); 
//     #endif  
//     // OutPut("Norm of B rhs in cycle " <<  sqrt(Ddot(N_DOF,CurrentRhs,CurrentRhs)) <<"i is:"<<i <<endl); 
      
//       slc =0;

//         if (TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_ALL_SCALAR)
//           slc = 1;
//         else if ((TDatabase::ParamDB->SC_STEP_LENGTH_CONTROL_FINE_SCALAR)
//                 &&(i==N_Levels-1))
//           slc = 1;
        
//         CoarserLevel = MultiGridLevels[i-1];
//         CoarserSol = CoarserLevel->GetSolution();
//         CoarserRhs = CoarserLevel->GetRhs();
      
//     #ifdef _MPI  
//       CoarseParComm = CoarserLevel->GetParComm();   
//     #endif  
//         // smoothing
//         CurrentLevel->Defect(CurrentSol, CurrentRhs, CurrentDefect, oldres);  
//         firstres = initres = oldres;  
//         normsol = sqrt(Ddot(N_DOF, CurrentSol, CurrentSol));
  
//     if(TDatabase::ParamDB->SC_VERBOSE>=2
// #ifdef _MPI  
//         && rank==TDatabase::ParamDB->Par_P0
// #endif   
//       )
//       {
//        OutPut("level " << i << " ");
//        OutPut("res before presmoothing: " << oldres << endl);
//       }

//     if (slc)
//     {
//       memcpy(OldSol, CurrentSol, N_DOF*SizeOfDouble);
//       memcpy(OldDefect, CurrentDefect, N_DOF*SizeOfDouble);
//     }

// PUSH_RANGE("prolongate",3)
//     // prolongate correction
// #ifdef _MPI  
//       Prolongate(CoarserLevel->GetFESpace(), CurrentLevel->GetFESpace(),
//                        CoarserSol, CurrentLevel->GetTemp_arr(),
//                        CurrentLevel->GetAuxVector(1));


// 	ParComm->CommUpdateReduce(CurrentLevel->GetTemp_arr());

      
//       CurrentLevel->Update(CurrentSol, CurrentLevel->GetTemp_arr());
      
//       ParComm->CommUpdate(CurrentSol);
// #else 
//     Prolongate(FESpaces[i-1], FESpaces[i], 
//                    CoarserSol, CurrentAux2, CurrentAux);

//     CurrentLevel->CorrectNodes(CurrentAux2);

//     CurrentLevel->Update(CurrentSol, CurrentAux2);
// #endif  
// POP_RANGE
//     CurrentLevel->Defect(CurrentSol, CurrentRhs, CurrentDefect, oldres);
  
//     initres = oldres;
//     normsol = sqrt(Ddot(N_DOF, CurrentSol, CurrentSol));
    
//     if (TDatabase::ParamDB->SC_VERBOSE>=2
// #ifdef _MPI  
//         && rank==TDatabase::ParamDB->Par_P0
// #endif  
//        )
//       {
//         OutPut("level " << i << " ");
//         OutPut("res before postsmoothing: " << oldres << endl);
//       }
// //        exit(0);
//     // smoothing
// #ifdef _MPI
//      t1 = MPI_Wtime();
//      Smooth(PostSmooth, CurrentLevel, ParComm, oldres
//     #ifdef _CUDA
//       ,stream_smooth
//     #endif
//      ); 
//      t2 = MPI_Wtime();
// #else
//      t1 = GetTime();
//      Smooth(PostSmooth, CurrentLevel, oldres);
//      t2 = GetTime();
// #endif
//      tSmoother += t2-t1 ;
     
//     if (slc)
//     {
//       alpha = CurrentLevel->StepLengthControl(CurrentSol, OldSol,
//                           OldDefect,                  
//                           N_Parameters,Parameters);       
      
//       for (j=0;j<N_DOF;j++)
//         CurrentSol[j] = OldSol[j] + alpha *( CurrentSol[j]-OldSol[j]);
//     }

//     CurrentLevel->Defect(CurrentSol, CurrentRhs, CurrentDefect, res);
    
//     if (TDatabase::ParamDB->SC_VERBOSE>=2
// #ifdef _MPI  
//         && rank==TDatabase::ParamDB->Par_P0
// #endif  
//     )
//       {
//         OutPut("level " << i << " ");
//         OutPut("res after postsmoothing: " << res);
//         OutPut(" rate: " << res/firstres << endl);
//         // OutPut("Smoothing2 (" << i << "): " << initres/normsol << endl);
//       }
//   }
// }

void TMultiGrid3D::SetDirichletNodes(int i)
{
  int HangingNodeBound, N_Dirichlet;
  TMGLevel3D *CurrentLevel;
  double *X, *R;

  if(i>=N_Levels) return;

  CurrentLevel = MultiGridLevels[i];

  X = CurrentLevel->GetSolution();
  R = CurrentLevel->GetRhs();

  HangingNodeBound = CurrentLevel->GetHangingNodeBound();
  N_Dirichlet = CurrentLevel->GetN_Dirichlet();

  memcpy(X+HangingNodeBound, R+HangingNodeBound, SizeOfDouble*N_Dirichlet);
}

/** set recursion for MultiGrid3D */ 
void TMultiGrid3D::SetRecursion(int levels)
{
  int gam = TDatabase::ParamDB->SC_MG_CYCLE_SCALAR,k;

  // coarsest grid 
  mg_recursions[1] = 1;
  if (gam>0)
    for (k=2;k<=levels;k++)        
      mg_recursions[k] = gam;
  else                /* F -- cycle */
    for (k=2;k<=levels;k++)        
      mg_recursions[k] = 2;
}
