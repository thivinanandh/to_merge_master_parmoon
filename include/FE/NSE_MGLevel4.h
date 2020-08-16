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
// @(#)NSE_MGLevel4.h        1.6 07/03/00
//
// Class:       TNSE_MGLevel4
// Purpose:     store all data for one level in a multi grid method
//              for solving a Stokes-/ Navier-Stokes system of
//              type 4 (all Aij, B1, B2, B3)
//
// Author:      Volker John 25.07.2000
//
// History:      25.07.2000start of implementation
//
// =======================================================================

#include <Queue.h>
#ifndef __NSE_MGLEVEL4__
#define __NSE_MGLEVEL4__

#include <NSE_MGLevel.h>
#include <Magma_Solver.h>

class TNSE_MGLevel4 : public TNSE_MGLevel
{
  protected:
#ifdef __2D__
    /** matrix A11 */
    TSquareMatrix2D *A11;

    /** matrix A12 */
    TSquareMatrix2D *A12;

    /** matrix A21 */
    TSquareMatrix2D *A21;

    /** matrix A22 */
    TSquareMatrix2D *A22;

    /** structure of matrix A */
    TSquareStructure2D *StructureA;

    /** matrix B1 */
    TMatrix2D *B1;

    /** matrix B2 */
    TMatrix2D *B2;

    /** matrix B1T */
    TMatrix2D *B1T;

    /** matrix B2 */
    TMatrix2D *B2T;

    /** structure of matrix B */
    TStructure2D *StructureB;

    /** structure of matrix BT */
    TStructure2D *StructureBT;

    /** structure of matrix C */
    TStructure2D *StructureC;

    /** matrix C */
    TMatrix2D *C;
#endif  

#ifdef __3D__
    /** matrix A11 */
    TSquareMatrix3D *A11;

    /** matrix A12 */
    TSquareMatrix3D *A12;

    /** matrix A13 */
    TSquareMatrix3D *A13;

    /** matrix A21 */
    TSquareMatrix3D *A21;

    /** matrix A22 */
    TSquareMatrix3D *A22;

    /** matrix A23 */
    TSquareMatrix3D *A23;

    /** matrix A31 */
    TSquareMatrix3D *A31;

    /** matrix A32 */
    TSquareMatrix3D *A32;

    /** matrix A33 */
    TSquareMatrix3D *A33;

    /** structure of matrix A */
    TSquareStructure3D *StructureA;

    /** matrix B1 */
    TMatrix3D *B1;

    /** matrix B2 */
    TMatrix3D *B2;

    /** matrix B3 */
    TMatrix3D *B3;

    /** matrix B1T */
    TMatrix3D *B1T;

    /** matrix B2 */
    TMatrix3D *B2T;

    /** matrix B3 */
    TMatrix3D *B3T;

    /** structure of matrix B */
    TStructure3D *StructureB;
 
    /** structure of matrix BT */
    TStructure3D *StructureBT;

    /** structure of matrix C */
    TStructure3D *StructureC;

    /** matrix C */
    TMatrix3D *C;
#endif  

    /** row pointer for matrix A */
    int *ARowPtr;

    /** column number vector for matrix A */
    int *AKCol;

    /** matrix entries of matrix A */
    double *A11Entries;

    /** matrix entries of matrix A */
    double *A12Entries;

    /** matrix entries of matrix A */
    double *A21Entries;

    /** matrix entries of matrix A */
    double *A22Entries;

    /** matrix entries of matrix B1 */
    double *B1Entries;

    /** matrix entries of matrix B2 */
    double *B2Entries;

    /** matrix entries of matrix B1 */
    double *B1TEntries;

    /** matrix entries of matrix B2 */
    double *B2TEntries;
#ifdef __3D__
    /** matrix entries of matrix A */
    double *A13Entries;

    /** matrix entries of matrix A */
    double *A23Entries;

    /** matrix entries of matrix A */
    double *A31Entries;

    /** matrix entries of matrix A */
    double *A32Entries;

    /** matrix entries of matrix A */
    double *A33Entries;

    /** matrix entries of matrix B3 */
    double *B3Entries;

    /** matrix entries of matrix BT3 */
    double *B3TEntries;

#endif  

    /** row pointer for matrix B */
    int *BRowPtr;

    /** column number vector for matrix B */
    int *BKCol;

    /** row pointer for matrix BT */
    int *BTRowPtr;

    /** column number vector for matrix BT */
    int *BTKCol;

    /** row pointer for matrix C */
    int *CRowPtr;

    /** column number vector for matrix C */
    int *CKCol;

    /** matrix entries of matrix C */
    double *CEntries;

    /** total local DOFs in a cell */
    int N_U, N_P, N_LocalDOF;

    Magma_Solver *mg_solver;
    
#ifdef _MPI
    
    int* u_clip;
    int *p_clip;
    int *cell_clip;
    queue* cell_queue;
    
    int *re_pdof;
    TParFEMapper3D *ParMapper;
    
#endif
    
#ifdef _HYBRID
    
    int N_CIntCell, *ptrCellColors, *CellReorder, maxCellsPerColor;
    int N_CPDOF, *ptrPDOFColors, *PDOFReorder;
    
#endif
    
  public:
    /** constructor */
#ifdef __2D__
    TNSE_MGLevel4(int level, 
                  TSquareMatrix2D *A11, TSquareMatrix2D *A12, 
                  TSquareMatrix2D *A21, TSquareMatrix2D *A22, 
                  TMatrix2D *B1, TMatrix2D *B2,
                  TMatrix2D *B1T, TMatrix2D *B2T,
                  double *f1, double *u1,
                  int n_aux, double *al, int VelocitySpace, 
                  int PressureSpace, TCollection *coll, int *dw);

    TNSE_MGLevel4(int level,
                  TSquareMatrix2D *a11, TSquareMatrix2D *a12,
                  TSquareMatrix2D *a21, TSquareMatrix2D *a22,
                  TMatrix2D *b1, TMatrix2D *b2,
                  TMatrix2D *b1t, TMatrix2D *b2t,
                  TMatrix2D *c,
                  double *f1, double *u1,
                  int n_aux, double *al, int velocity_space,
                  int pressure_space, TCollection *Coll,
                  int *dw);
#endif  
#ifdef __3D__
    TNSE_MGLevel4(int level, 
                  TSquareMatrix3D *A11, TSquareMatrix3D *A12, 
                  TSquareMatrix3D *A13, 
                  TSquareMatrix3D *A21, TSquareMatrix3D *A22, 
                  TSquareMatrix3D *A23, 
                  TSquareMatrix3D *A31, TSquareMatrix3D *A32, 
                  TSquareMatrix3D *A33, 
                  TMatrix3D *B1, TMatrix3D *B2, TMatrix3D *B3,  
                  TMatrix3D *B1T, TMatrix3D *B2T, TMatrix3D *B3T,
                  double *f1, double *u1,
                  int n_aux, double *al, int VelocitySpace, 
                  int PressureSpace, TCollection *coll, int *dw  
#ifdef _MPI
		  ,TParFECommunicator3D *parComm_U, TParFECommunicator3D *parComm_P, TParFEMapper3D *parMapper_P
#endif
		 );

TNSE_MGLevel4(int level,
                               TSquareMatrix3D *a11, TSquareMatrix3D *a12,
                               TSquareMatrix3D *a13, TSquareMatrix3D *a21,
                               TSquareMatrix3D *a22, TSquareMatrix3D *a23,
                               TSquareMatrix3D *a31, TSquareMatrix3D *a32,
                               TSquareMatrix3D *a33,
                               TMatrix3D *b1, TMatrix3D *b2, TMatrix3D *b3,
                               TMatrix3D *b1t, TMatrix3D *b2t, TMatrix3D *b3t,
                               TMatrix3D *c,
                               double *f1, double *u1,
                               int n_aux, double *al, int velocity_space,
                               int pressure_space, TCollection *Coll,
	      int *dw);

#endif  

    /** destructor */
    ~TNSE_MGLevel4();

    TSquareMatrix3D *GetA11Matrix(){
        return A11;
    }

    TSquareMatrix3D *GetA12Matrix(){
        return A12;
    }

    TSquareMatrix3D *GetA13Matrix(){
        return A13;
    }

    TSquareMatrix3D *GetA21Matrix(){
        return A21;
    }

    TSquareMatrix3D *GetA22Matrix(){
        return A22;
    }

    TSquareMatrix3D *GetA23Matrix(){
        return A23;
    }

    TSquareMatrix3D *GetA31Matrix(){
        return A31;
    }

    TSquareMatrix3D *GetA32Matrix(){
        return A32;
    }

    TSquareMatrix3D *GetA33Matrix(){
        return A33;
    }

    TMatrix3D *GetB1Matrix(){
        return B1;
    }

    TMatrix3D *GetB2Matrix(){
        return B2;
    }

    TMatrix3D *GetB3Matrix(){
        return B3;
    }

    TMatrix3D *GetB1TMatrix(){
        return B1T;
    }

    TMatrix3D *GetB2TMatrix(){
        return B2T;
    }

    TMatrix3D *GetB3TMatrix(){
        return B3T;
    }

    virtual void Defect(double *u1, double *f1, double *d1, double &res);

    /** correct Dirichlet and hanging nodes */
    virtual void CorrectNodes(double *u1);

    /** Vanka smoother */
    virtual void CellVanka(double *u1, double *rhs1, double *aux, 
        int N_Parameters, double *Parameters, int smoother, int N_Levels);

    /** Vanka smoother */
    virtual void NodalVanka(double *u1, double *rhs1, double *aux, 
        int N_Parameters, double *Parameters, int smoother, int N_Levels);
#ifdef _MPI
#ifdef _HYBRID
#ifdef _CUDA
    
    virtual void CellVanka_CPU_GPU(double *u1, double *rhs1, double *aux,
        int N_Parameters, double *Parameters, int smoother, int smooth, cudaStream_t *stream, int* d_ARowPtr, int*  d_AKCol,
        double* d_A11Entries, double* d_A12Entries, double* d_A13Entries,
        double* d_A21Entries, double* d_A22Entries, double* d_A23Entries,
        double* d_A31Entries, double* d_A32Entries, double* d_A33Entries,
        int* d_BTRowPtr, int*  d_BTKCol,
        double* d_B1TEntries, double* d_B2TEntries, double* d_B3TEntries,
        int* d_BRowPtr, int*  d_BKCol,
        double* d_B1Entries, double* d_B2Entries, double* d_B3Entries);
    
    virtual void CellVanka_GPU(double *u1, double *rhs1, double *aux,
        int N_Parameters, double *Parameters, int smoother, int smooth, cudaStream_t *stream, int* d_ARowPtr, int*  d_AKCol,
        double* d_A11Entries, double* d_A12Entries, double* d_A13Entries,
        double* d_A21Entries, double* d_A22Entries, double* d_A23Entries,
        double* d_A31Entries, double* d_A32Entries, double* d_A33Entries,
        int* d_BTRowPtr, int*  d_BTKCol,
        double* d_B1TEntries, double* d_B2TEntries, double* d_B3TEntries,
        int* d_BRowPtr, int*  d_BKCol,
        double* d_B1Entries, double* d_B2Entries, double* d_B3Entries);
    
    // virtual void CellVanka_Level_Split(double *u1, double *rhs1, double *aux,
    //     int N_Parameters, double *Parameters, int smoother, int smooth);
    
    virtual void CellVanka_Combo(double *u1, double *rhs1, double *aux,
        int N_Parameters, double *Parameters, int smoother, int smooth, cudaStream_t *stream, int* d_ARowPtr, int*  d_AKCol,
        double* d_A11Entries, double* d_A12Entries, double* d_A13Entries,
        double* d_A21Entries, double* d_A22Entries, double* d_A23Entries,
        double* d_A31Entries, double* d_A32Entries, double* d_A33Entries,
        int* d_BTRowPtr, int*  d_BTKCol,
        double* d_B1TEntries, double* d_B2TEntries, double* d_B3TEntries,
        int* d_BRowPtr, int*  d_BKCol,
        double* d_B1Entries, double* d_B2Entries, double* d_B3Entries);

    virtual void NodalVanka_CPU_GPU(double *u1, double *rhs1, double *aux,
        int N_Parameters, double *Parameters, int smoother, int smooth);
    
    virtual void NodalVanka_GPU(double *u1, double *rhs1, double *aux,
        int N_Parameters, double *Parameters, int smoother, int smooth);
    
    virtual void NodalVanka_Level_Split(double *u1, double *rhs1, double *aux,
        int N_Parameters, double *Parameters, int smoother, int smooth);
#endif
#endif
#endif    
    /** solve exact on this level */
    virtual void SolveExact(double *u1, double *rhs1);

    /** solve exact on this level */
    virtual void SolveExactUMFPACK(double *u1, double *rhs1, int &umfpack_flag);

    /** Braess Sarazin smoother */
    virtual void BraessSarazin(double *u1, double *rhs1, double *aux,
        int N_Parameters, double *Parameters,int N_Levels);

    /** step length control for Vanka */
    virtual double StepLengthControl(double *u1, double *u1old, double *def1,
                                     int N_Parameters, double *Parameter);

    /** print all matrices and both right hand sides */
    virtual void PrintAll();
    
   // double Dotprod(double *d1, double *d2);
    
#ifdef _MPI
    void Par_Directsolve(double*,double*);
    
    void ComputeOrder();
    
    void gmres_solve(double *sol, double * rhs);

    void Update(TBaseCell* cell);
    
    void NeighUpdate(TBaseCell* cell, int iter);
    
    void Reorder();
    
    virtual void UpdateHaloRhs(double*, double*); 
    

    
#endif
};

#endif
