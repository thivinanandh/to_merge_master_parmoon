#ifndef __SYSTEMHYPERELASTIC__
#define __SYSTEMHYPERELASTIC__

#include <list>
#include <Constants.h>
#include <FESpace3D.h>
#include <FEFunction3D.h>
#include <FEVectFunct3D.h>
#include <AssembleMat3D.h>
#include <Output3D.h>

class TSystemHyperelastic
{
private:
    std::list<std::pair<int, double>> dirichletList;
    TDomain *Domain;
    TFESpace3D *fespace, *fesp[1], *ferhs[3];
    TFEVectFunct3D *fefunction;
    TFEFunction3D *fefct[3];
    double *sol, *sol_exp, *rhs, *lambda, *dummy_buffer, *RHS[3];
    TSquareStructure3D *sqstructure;
    TSquareMatrix3D *SQMATRICES[9];
    TDiscreteForm3D *discreteform, *discreteformAdjoint = nullptr;
    TAuxParam3D *aux;
    TAssembleMat3D *Assembly, *AdjointAssembly = nullptr;
    BoundCondFunct3D **BoundConditions, *NoBoundConditions[3];
    BoundValueFunct3D **BoundValues, *BoundValues_Adjoint[3];

    TOutput3D *Output;

    int *RowptrGlobal, *ColindGlobal;
    double *EntriesGlobal;

public:
    TSystemHyperelastic(TFESpace3D *fespace, TFEVectFunct3D *fefunction, BoundCondFunct3D **BoundConditions, BoundValueFunct3D **BoundValues, double *sol, double *rhs, TDomain *domain = nullptr):fespace(fespace), fefunction(fefunction), BoundConditions(BoundConditions), BoundValues(BoundValues), sol(sol), rhs(rhs), Domain(domain){

    }

    void Init();

    void ImposeDirichletBoundary();

    void GetGlobalStructure();

    void GetGlobalEntries();

    void Assemble();

    void AssembleOnlyRHS();

    void solve(int logLevel = 2);

    void solveQuasiNewton();

    void setAdjointErrorVector(double *);

    double GetObjectiveFunctionValue();

    void GetGradientForParameters(std::vector<double>&, std::vector<std::string>&);

    void WriteOut(const char *);

    int check;

    void checkClass()
    {
        std::cout << "Check done, check val = " << check << std::endl;
    }
};

void paramfct_hyper3d(double *in, double *out)
{
    for (int i = 3; i < 12; i++)
        out[i - 3] = in[i];
}

MultiIndex3D AllDerivatives_Hyperelastic_Single_Field[3] = { D100, D010, D001 };
int N_Terms_Hyperelastic_Single_Field = 3;
int Spacenumbers_Hyperelastic_Single_Field[3] = {0, 0, 0};
int N_SQMATRICES_Hyperelastic_Single_Field = 9;
int N_RHS_Hyperelastic_Single_Field = 3;
int rowspace_Hyperelastic_Single_Field[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
int columnspace_Hyperelastic_Single_Field[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
int rhsspace_Hyperelastic_Single_Field[3] = {0, 0, 0};

int n_fespaces_Hyperelastic_Single_Field = 1, 
    n_fefct_Hyperelastic_Single_Field = 3,
    n_paramfct_Hyperelastic_Single_Field = 1, 
    n_fevalues_Hyperelastic_Single_Field = 9, 
    n_parameters_Hyperelastic_Single_Field = 9,
    fevalind_Hyperelastic_Single_Field[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2},
    begin_index_Hyperelastic_Single_Field[1] = {0};
MultiIndex3D Hyper3D_Hyperelastic_Single_Field[9] = {D100, D100, D100, D010, D010, D010, D001, D001, D001};
ParamFct *paramfctarr_Hyperelastic_Single_Field[1] = {paramfct_hyper3d};

#endif