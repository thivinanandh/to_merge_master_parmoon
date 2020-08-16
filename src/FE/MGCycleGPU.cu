// =======================================================================
// %W% %G%
//
// Class:       TMultiGrid3D
// Purpose:     overlap matrix data transfer with smoothing
//
// Author:      Iyer Neha Mohan
//              Shah Manan Jayant
//
// History:     10.03.2020 start of implementation
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


double tSmoother1 = 0.0;


void TMultiGrid3D::CycleIterative(double &res, int cycle_index)
{
  double s,t1,t2;
//   cout<<"new cycle"<<endl;
  TMGLevel3D *CurrentLevel, *CoarserLevel;
  double *CurrentSol, *CoarserSol, *CoarserRhs;
  double *CurrentRhs, *CurrentDefect, *CurrentAux;
  double *CurrentAux2, *OldSol, *OldDefect;
  double oldres,reduction, alpha;
  int j, N_DOF, maxit, it, slc,gam;
  double initres, normsol, firstres;

  int i;
  
  #ifdef _MPI  
  int rank;
  MPI_Comm_rank(TDatabase::ParamDB->Comm, &rank); 
#endif  


  if(cycle_index==1){
    PUSH_RANGE("transfer", 12);
    DeviceDataTransfer(N_Levels-1, PreSmooth);
    POP_RANGE
    CUDA_CHECK(cudaStreamSynchronize(stream_transfer));
  }

    // if(rank==0){
        // cout<<cycle_index<<endl;
    //     cout<<"before"<<endl;
    //   cout<<"A1:"<<A1<<endl;
    //   cout<<"A2:"<<A2<<endl;
    // }
  

  for(i=N_Levels-1; i>0; i--){

    Swap_Matrices();

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
          ,d_RowPtr1
          ,d_KCol1
          ,d_Entries1
          ,d_sol1
          ,d_aux1
        #endif
        ); 
        t2 = MPI_Wtime();
    #else
        t1 = GetTime();
        Smooth(PreSmooth, CurrentLevel, oldres, NULL, NULL, NULL, NULL, NULL, NULL );
        t2 = GetTime();
    #endif
        tSmoother1 += t2-t1 ;

        PUSH_RANGE("transfer", 12);
        DeviceDataTransfer(i-1, PreSmooth );
        POP_RANGE
        
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

        CUDA_CHECK(cudaStreamSynchronize(stream_smooth[0]));
        CUDA_CHECK(cudaStreamSynchronize(stream_smooth[1]));
        CUDA_CHECK(cudaStreamSynchronize(stream_transfer));

  }

  CUDA_CHECK(cudaStreamSynchronize(stream_smooth[0]));
  CUDA_CHECK(cudaStreamSynchronize(stream_smooth[1]));
  CUDA_CHECK(cudaStreamSynchronize(stream_transfer));
  
  Swap_Matrices();

  if(i==0)
  {

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
      ,d_RowPtr1
      ,d_KCol1
      ,d_Entries1
      ,d_sol1
      ,d_aux1
     #endif
     ); 
     t2 = MPI_Wtime();
#else
     t1 = GetTime();
     Smooth(CoarseSmooth, CurrentLevel, res, NULL, NULL, NULL, NULL, NULL, NULL );
     t2 = GetTime();
#endif
     tSmoother1 += t2-t1 ;
     
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

  CUDA_CHECK(cudaStreamSynchronize(stream_smooth[0]));
  CUDA_CHECK(cudaStreamSynchronize(stream_smooth[1]));
  
  Swap_Matrices();

  for(int i=1; i<N_Levels; i++)
  {    
    PUSH_RANGE("transfer", 12);
    if(i<N_Levels-1){
      DeviceDataTransfer(i+1, PostSmooth);
    }
    POP_RANGE

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

    PUSH_RANGE("defect", 6)
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

      POP_RANGE

    if (slc)
    {
      memcpy(OldSol, CurrentSol, N_DOF*SizeOfDouble);
      memcpy(OldDefect, CurrentDefect, N_DOF*SizeOfDouble);
    }



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
      ,d_RowPtr1
      ,d_KCol1
      ,d_Entries1
      ,d_sol1
      ,d_aux1
    #endif
     ); 
     t2 = MPI_Wtime();
#else
     t1 = GetTime();
     Smooth(PostSmooth, CurrentLevel, oldres, NULL, NULL, NULL, NULL, NULL, NULL );
     t2 = GetTime();
#endif
     tSmoother1 += t2-t1 ;
     
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

      CUDA_CHECK(cudaStreamSynchronize(stream_transfer));
      CUDA_CHECK(cudaStreamSynchronize(stream_smooth[0]));
      CUDA_CHECK(cudaStreamSynchronize(stream_smooth[1]));

      Swap_Matrices();


      

  }



//   if(rank==0){
//       cout<<"after"<<endl;
//     cout<<"A1:"<<A1<<endl;
//     cout<<"A2:"<<A2<<endl;
//   }


}