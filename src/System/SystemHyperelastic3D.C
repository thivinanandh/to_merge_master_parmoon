#include <DiscreteForm3D.h>
#include <SquareStructure3D.h>
#include <SquareMatrix3D.h>
#include <LinAlg.h>
#include <string.h>
#include <chrono>
#include <ctime>
#include <mkl.h>
#include <algorithm>
#include <sstream>

#include <SystemHyperelastic3D.h>
#include <DynamicParamStore.h>

void BoundCondition(int i, double X, double Y, double Z, BoundCond &cond);
void NoBoundCondition(int i, double X, double Y, double Z, BoundCond &cond);


// void TSystemHyperelastic::Init()
// {
//     int N_U = fespace->GetN_DegreesOfFreedom();
//     int N_DOF = 3 * N_U;
//     char Name[] = "discreteform for hyperelasticity";
//     discreteform = new TDiscreteForm3D(Name, Name, N_Terms_Hyperelastic_Single_Field, AllDerivatives_Hyperelastic_Single_Field, 
//                                                         Spacenumbers_Hyperelastic_Single_Field, N_SQMATRICES_Hyperelastic_Single_Field, N_RHS_Hyperelastic_Single_Field, 
//                                                         rowspace_Hyperelastic_Single_Field, columnspace_Hyperelastic_Single_Field, rhsspace_Hyperelastic_Single_Field, Assembly_Hyper, NULL, NULL);

    
//     fesp[0] = fespace;
//     for(int i=0; i<3; i++) fefct[i] = fefunction->GetComponent(i);

//     aux = new TAuxParam3D(n_fespaces_Hyperelastic_Single_Field, n_fefct_Hyperelastic_Single_Field, n_paramfct_Hyperelastic_Single_Field, n_fevalues_Hyperelastic_Single_Field, 
//                                        fesp, fefct, 
//                                        paramfctarr_Hyperelastic_Single_Field, fevalind_Hyperelastic_Single_Field, 
//                                        Hyper3D_Hyperelastic_Single_Field, n_parameters_Hyperelastic_Single_Field, begin_index_Hyperelastic_Single_Field);

//     NoBoundConditions[0] = NoBoundCondition;
//     NoBoundConditions[1] = NoBoundCondition;
//     NoBoundConditions[2] = NoBoundCondition;

//     sqstructure = new TSquareStructure3D(fespace);
//     sqstructure->Sort();
    
//     //int N_SQMATRICES = 9;
//     //int N_RHS = 3;
    
//     for(int i=0; i<N_SQMATRICES_Hyperelastic_Single_Field; i++) SQMATRICES[i] = new TSquareMatrix3D(sqstructure);

//     for(int i=0; i<3; i++)
//     {
//         RHS[i] = rhs + i * N_U;
//         ferhs[i] = fespace;
//     }
    
//     Assembly = new TAssembleMat3D(1, fesp,
//                                   N_SQMATRICES_Hyperelastic_Single_Field, SQMATRICES,
//                                   0, nullptr,
//                                   N_RHS_Hyperelastic_Single_Field, RHS, ferhs,
//                                   discreteform,
//                                   NoBoundConditions, BoundValues,
//                                   aux);

//     Assembly->Init();

//     GetGlobalNumbersForDirichlet(fespace, BoundConditions, BoundValues, dirichletList);

//     int NNZ = 9 * sqstructure->GetRowPtr()[N_U];
//     RowptrGlobal = new int[N_DOF + 1]();
//     ColindGlobal = new int[NNZ]();
//     EntriesGlobal = new double[NNZ]();

//     lambda = new double[N_DOF]();
//     dummy_buffer = new double[N_DOF]();

//     if(Domain != nullptr){
//         Output = new TOutput3D(1, 0, 1, 1, Domain);
//         system("mkdir -p VTK");
//         Output->AddFEVectFunct(fefunction);
//     }
// }

// void TSystemHyperelastic::Assemble()
// {
//     Assembly->Reset();
//     Assembly->Assemble3D();
// }

// void TSystemHyperelastic::ImposeDirichletBoundary()
// {
//     ManipulateMatrices(3, SQMATRICES, 0, NULL, fespace->GetN_DegreesOfFreedom(), rhs, dirichletList);
// }

// void TSystemHyperelastic::GetGlobalStructure()
// {
//     int N_U = fespace->GetN_DegreesOfFreedom();
//     int *rowptr_local = sqstructure->GetRowPtr();
//     int *colind_local = sqstructure->GetKCol();

//     int pos = 0;

//     for (int i_row = 0; i_row < 3; i_row++)
//     {
//         for (int i = 0; i < N_U; i++)
//         {
//             for (int i_col = 0; i_col < 3; i_col++)
//             {
//                 for (int j = rowptr_local[i]; j < rowptr_local[i + 1]; j++)
//                 {
//                     ColindGlobal[pos] = colind_local[j] + i_col * N_U;
//                     pos++;
//                 }
//             }
//             RowptrGlobal[i + i_row * N_U  + 1] = pos;
//         }
//     }
// }

// void TSystemHyperelastic::GetGlobalEntries()
// {
//     int N_U = fespace->GetN_DegreesOfFreedom();
//     int N_Active = fespace->GetN_ActiveDegrees();
//     int pos = 0;
//     int *rowptr_block = SQMATRICES[0]->GetRowPtr(), *colind_block = SQMATRICES[0]->GetKCol();

//     double *entriesA11 = SQMATRICES[0]->GetEntries(), *entriesA12 = SQMATRICES[1]->GetEntries(), *entriesA13 = SQMATRICES[2]->GetEntries(),
//            *entriesA21 = SQMATRICES[3]->GetEntries(), *entriesA22 = SQMATRICES[4]->GetEntries(), *entriesA23 = SQMATRICES[5]->GetEntries(),
//            *entriesA31 = SQMATRICES[6]->GetEntries(), *entriesA32 = SQMATRICES[7]->GetEntries(), *entriesA33 = SQMATRICES[8]->GetEntries();

//     double *entriesA[3][3] = {{entriesA11, entriesA12, entriesA13},
//                               {entriesA21, entriesA22, entriesA23},
//                               {entriesA31, entriesA32, entriesA33}};

//     for (int i_row = 0; i_row < 3; i_row++)
//     {
//         for (int i = 0; i < N_U; i++)
//         {
//             for (int i_col = 0; i_col < 3; i_col++)
//             {
//                 for (int j = rowptr_block[i]; j < rowptr_block[i + 1]; j++)
//                 {
//                     EntriesGlobal[pos] = entriesA[i_row][i_col][j];
//                     pos++;
//                 }
//             }
//         }
//     }

//     // double K[3 * N_U][3 * N_U] = {{0}};
//     // for (int i = 0; i < 3 * N_U; i++)
//     // {
//     //     for (int j = RowptrGlobal[i]; j < RowptrGlobal[i + 1]; j++)
//     //     {
//     //         K[i][ColindGlobal[j]] = EntriesGlobal[j];
//     //     }
//     // }
//     // cout << endl;
//     // for (int i = 0; i < 3 * N_U; i++)
//     // {
//     //     for (int j = 0; j < 3 * N_U; j++)
//     //     {
//     //         cout << K[i][j] << "  ";
//     //     }
//     //     cout << endl;
//     // }
// }

// void TSystemHyperelastic::solve(int logLevel)
// {
//     dynamicParamDB paramStore;

//     int N_U = fespace->GetN_DegreesOfFreedom();
//     int N_DOF = 3 * N_U, iter = 0;
//     double rhs_norm, rhs_norm_0;
//     std::fill_n(sol, N_DOF, 0.0);

//     int loadStepsToApply = paramStore["LOAD_STEPS"];
//     int loadStepsToApplied = 0;
//     int max_iter = paramStore["MAX_ITER"];

//     double boundval_this_step = boundval / loadStepsToApply;
//     boundval = boundval_this_step;

//     double *del_sol = new double[N_DOF](),
//            *sol_0 = new double[N_DOF](),
//            *rhs_0 = new double[N_DOF]();
//     std::stringstream ss;

//     unsigned int i_steps = 0;
//     double tow = paramStore["TOW"];
//     double towCoeff = paramStore["TOWCOEFF"];
//     double towParam = 1;
//     double del_p = 1.0, del_solNorm0 = 0.0, del_solNorm1 = 0.0;

//     double beta0 = 0.0, beta1 = 1.0, r0 = 1.0, r1, beta1n;

//     TSquareMatrix3D *diag[] = {SQMATRICES[0], SQMATRICES[4], SQMATRICES[8]};
//     bool once = true, log = true;
//     GetGlobalStructure();

//     cout << "Forward Solver Started" << endl << endl;
//     while (max_iter--)
//     {
//         auto start = std::chrono::system_clock::now();


//         Assembly->Reset();
//         Assembly->Assemble3D();

//         auto end = std::chrono::system_clock::now();


//         std::chrono::duration<double> elapsed_seconds = end-start;
//         std::time_t end_time = std::chrono::system_clock::to_time_t(end);

//         // cout << endl
//         //      << endl
//         //      << "Time taken to assembly = " << elapsed_seconds.count() << endl
//         //      << endl;

//         ManipulateMatrices(3, SQMATRICES, 0, NULL, N_U, rhs, dirichletList);

//         GetGlobalEntries();
//         solve_pardiso(N_DOF, RowptrGlobal, ColindGlobal, EntriesGlobal, rhs, del_sol);

//         // cout << endl
//         //      << "Enter tow value " << endl;
//         // cin >> tow;
//         // cout << endl;
//         // double dtow = 0.1;
//         // memcpy(sol_0, sol, N_DOF * sizeof(double));
//         // for (double xtow = 0.1; xtow < 1.01; xtow+=dtow)
//         // {
//         //     memcpy(sol, sol_0, N_DOF * sizeof(double));
//         //     cblas_daxpy(N_DOF, xtow, del_sol, 1, sol, 1);
//         //     AssembleOnlyRHS();
//         //     cout << xtow << "\t" << cblas_dnrm2(N_DOF, rhs, 1) << endl;
//         // }
//         // cout << endl;

//         // cout << endl
//         //      << "Enter tow value " << endl;
//         // cin >> tow;
//         // memcpy(sol, sol_0, N_DOF * sizeof(double));
//         // memcpy(sol_0, sol, N_DOF * sizeof(double));
//         // memcpy(rhs_0, rhs, N_DOF * sizeof(double));
//         cblas_daxpy(N_DOF, tow, del_sol, 1, sol, 1);

//         // double s0 = cblas_ddot(N_DOF, del_sol, 1, rhs, 1);
//         // this->AssembleOnlyRHS();
//         // double si = cblas_ddot(N_DOF, del_sol, 1, rhs, 1);

//         // double r1 = si / s0;

//         // cout << endl
//         //      << "r1 = " << r1 << endl
//         //      << "Suggestion for new tow = " << tow * (s0 / (s0 - si)) << endl;

//         // cin >> tow;

//         // memcpy(sol, sol_0, N_DOF * sizeof(double));
//         // memcpy(rhs, rhs_0, N_DOF * sizeof(double));
//         // cblas_daxpy(N_DOF, tow, del_sol, 1, sol, 1);

//         // for (int i = 0; i < N_DOF; i++)
//         // {
//         //     sol[i] += del_sol[i];
//         //     // sol[i] = sol_buffer[i];
//         // }

//         // cout << endl;
//         // if(!iter){
//         //     rhs_norm_0 = Dnorm(N_DOF, rhs);
//         // }

//         // rhs_norm = Dnorm(N_DOF, rhs);

//         // if(log){
//         //     cout << "Force Residual Norm at Iteration " << iter << "/Force Residual Norm at Iteration 0" << " = " << rhs_norm/rhs_norm_0 << endl;
//         //     cout << "RHS Norm = " << rhs_norm << endl; 
//         // }

//         if(logLevel == 2 && i_steps == 0)
//             std::cout << "RHS Norm\r\t\t\tRHS Norm ratio\r\t\t\t\t\t\tdel_sol Norm\r\t\t\t\t\t\t\t\t\tsol Norm" << std::endl << std::endl;

//         i_steps++;

//         if(checkConvergence(N_DOF, rhs, del_sol, sol, loadStepsToApply, logLevel == 2)){            
//             loadStepsToApply--;
//             if (loadStepsToApply)
//             {
//                 loadStepsToApplied++;
//                 // tow = exp(-towCoeff * (loadStepsToApplied / paramStore["LOAD_STEPS"]));
//                 // tow = max(-(1.0 / 11.0) * cblas_dnrm2(N_DOF, sol, 1) + 1, 0.15);
//                 tow = max(exp(-paramStore["TOWCOEFF"] * cblas_dnrm2(N_DOF, sol, 1)), 0.075);
//                 // if (cblas_dnrm2(N_DOF, sol, 1) > 1.0)
//                 //     tow = paramStore["TOW1"];

//                 if(logLevel == 1)
//                     cout << "\rLoad step " << loadStepsToApplied << " of " << paramStore["LOAD_STEPS"] << " converged!" << flush;
                
//                 if (logLevel == 2)
//                     cout << endl
//                          << endl
//                          << "Entering next load step with force = " << boundval + boundval_this_step << endl
//                          << "Entering with tow value " << tow << endl
//                          << endl;

//                 boundval += boundval_this_step;
//                 i_steps = 0;
                
//                 if(logLevel == 2)
//                 {
//                     ostringstream os;
//                     os.seekp(ios::beg);
//                     os << "VTK/unitcube_" << loadStepsToApplied <<".vtk";

//                     WriteOut(os.str().c_str());
//                 }
//             }
//             else
//             {
//                 cout << endl
//                      << endl;
//                 break;
//             }
//         }

//         // memcpy(sol_0, sol, N_DOF * sizeof(double));
//         // cblas_daxpy(N_DOF, tow, del_sol, 1, sol, 1);
//         // if (i_steps != 0)
//         // {
//         //     double s0 = cblas_ddot(N_DOF, del_sol, 1, rhs, 1);

//         //     this->AssembleOnlyRHS();
//         //     double si = cblas_ddot(N_DOF, del_sol, 1, rhs, 1);

//         //     double r1 = fabs(si / s0);
//         //     beta1n = (beta0 * r0 - beta1 * r1) / (r0 - r1);

//         //     cout << endl
//         //          << beta1n << endl
//         //          << endl;

//         //     beta0 = max(beta0, beta1);
//         //     r0 = min(r0, r1);
//         //     beta1 = beta1n;
//         // }
//         // i_steps++;
//     }

//     delete[] del_sol;
//     delete[] rhs_0;
//     delete[] sol_0;
// }

// void TSystemHyperelastic::solveQuasiNewton()
// {
//     dynamicParamDB paramStore;

//     int N_U = fespace->GetN_DegreesOfFreedom();
//     int N_DOF = 3 * N_U, iter = 0;
//     double rhs_norm, rhs_norm_0;
//     std::fill_n(sol, N_DOF, 0.0);

//     int loadStepsToApply = paramStore["LOAD_STEPS"];
//     int max_iter = paramStore["MAX_ITER"];

//     double boundval_this_step = boundval / loadStepsToApply;
//     boundval = boundval_this_step;

//     double *del_sol = new double[N_DOF](),
//            *sol_0 = new double[N_DOF]();
//     std::stringstream ss;

//     unsigned int i_steps = 0;
//     double tow = paramStore["TOW"];
//     double towParam = 1;
//     double del_p = 1.0, del_solNorm0 = 0.0, del_solNorm1 = 0.0;

//     double beta0 = 0.0, beta1 = 1.0, r0 = 1.0, r1, beta1n;

//     TSquareMatrix3D *diag[] = {SQMATRICES[0], SQMATRICES[4], SQMATRICES[8]};
//     bool once = true, log = true;
//     GetGlobalStructure();
//     while (max_iter--)
//     {
//         auto start = std::chrono::system_clock::now();

//         if(!i_steps)
//         {
//             Assemble();
//         }
//         else
//         {
//             AssembleOnlyRHS();
//         }

//         auto end = std::chrono::system_clock::now();


//         std::chrono::duration<double> elapsed_seconds = end-start;
//         std::time_t end_time = std::chrono::system_clock::to_time_t(end);

//         // cout << endl
//         //      << endl
//         //      << "Time taken to assembly = " << elapsed_seconds.count() << endl
//         //      << endl;

//         ManipulateMatrices(3, SQMATRICES, 0, NULL, N_U, rhs, dirichletList);

//         GetGlobalEntries();
//         solve_pardiso(N_DOF, RowptrGlobal, ColindGlobal, EntriesGlobal, rhs, del_sol);

//         // cout << endl
//         //      << "Enter tow value " << endl;
//         // cin >> tow;
//         cblas_daxpy(N_DOF, tow, del_sol, 1, sol, 1);

//         // for (int i = 0; i < N_DOF; i++)
//         // {
//         //     sol[i] += del_sol[i];
//         //     // sol[i] = sol_buffer[i];
//         // }

//         // cout << endl;
//         // if(!iter){
//         //     rhs_norm_0 = Dnorm(N_DOF, rhs);
//         // }

//         // rhs_norm = Dnorm(N_DOF, rhs);

//         // if(log){
//         //     cout << "Force Residual Norm at Iteration " << iter << "/Force Residual Norm at Iteration 0" << " = " << rhs_norm/rhs_norm_0 << endl;
//         //     cout << "RHS Norm = " << rhs_norm << endl; 
//         // }

//         if(checkConvergence(N_DOF, rhs, del_sol, sol, loadStepsToApply)){            
//             loadStepsToApply--;
//             if (loadStepsToApply)
//             {
//                 tow = max(exp(-paramStore["TOWCOEFF"] * cblas_dnrm2(N_DOF, sol, 1)), 0.15);
//                 // if (cblas_dnrm2(N_DOF, sol, 1) > 1.0)
//                 //     tow = paramStore["TOW1"];
//                 cout << endl
//                      << endl
//                      << "Entering next load step with force = " << boundval + boundval_this_step << endl
//                      << "Entering with tow value " << tow << endl
//                      << endl;

//                 boundval += boundval_this_step;
//                 i_steps = 0;
//                 beta0 = 0.0, beta1 = 1.0, r0 = 1.0;
//             }
//             else
//             {
//                 break; 
//             }
//         }

//         // memcpy(sol_0, sol, N_DOF * sizeof(double));
//         // cblas_daxpy(N_DOF, tow, del_sol, 1, sol, 1);
//         // if (i_steps != 0)
//         // {
//         //     double s0 = cblas_ddot(N_DOF, del_sol, 1, rhs, 1);

//         //     this->AssembleOnlyRHS();
//         //     double si = cblas_ddot(N_DOF, del_sol, 1, rhs, 1);

//         //     double r1 = fabs(si / s0);
//         //     beta1n = (beta0 * r0 - beta1 * r1) / (r0 - r1);

//         //     cout << endl
//         //          << beta1n << endl
//         //          << endl;

//         //     beta0 = max(beta0, beta1);
//         //     r0 = min(r0, r1);
//         //     beta1 = beta1n;
//         // }
//         i_steps++;
//     }

//     delete[] del_sol;
//     delete[] sol_0;
// }

// void TSystemHyperelastic::AssembleOnlyRHS()
// {
//     int N_DOF = 3 * fespace->GetN_DegreesOfFreedom();
//     fill_n(rhs, N_DOF, 0.0);
//     onlyRHS = true;
//     Assembly->Assemble3D();
//     onlyRHS = false;
// }

// void TSystemHyperelastic::WriteOut(const char *filename)
// {
//     if(Domain == nullptr)
//         cout << endl
//              << endl
//              << "Cannot write output no domain input given" << endl
//              << endl;
//     else
//         Output->WriteVtk(filename);
// }

// void TSystemHyperelastic::setAdjointErrorVector(double *sol)
// {
//     sol_exp = sol;
// }

// double TSystemHyperelastic::GetObjectiveFunctionValue()
// {
//     int N_DOF = 3 * fespace->GetN_DegreesOfFreedom();
//     memcpy(dummy_buffer, sol, N_DOF * sizeof(double));
//     cblas_daxpy(N_DOF, -1.0, sol_exp, 1, dummy_buffer, 1);

//     return 0.5 * cblas_ddot(N_DOF, dummy_buffer, 1, dummy_buffer, 1);
// }

// void TSystemHyperelastic::GetGradientForParameters(std::vector<double> &grad, std::vector<std::string> &ParameterNames)
// {
//     dynamicParamDB paramStore;
//     int N_U = fespace->GetN_DegreesOfFreedom();
//     int N_DOF = 3 * N_U;
//     int n_params = ParameterNames.size();

//     Assembly->Reset();
//     Assembly->Assemble3D();

//     ManipulateMatrices(3, SQMATRICES, 0, NULL, N_U, rhs, dirichletList);
//     GetGlobalEntries();

//     memcpy(rhs, sol, N_DOF * sizeof(double));
//     cblas_daxpy(N_DOF, -1.0, sol_exp, 1, rhs, 1);

//     /*
//     * SCALE FOR ADJOINT PROBLEM
//     */
//     cblas_dscal(N_DOF, paramStore["ADJOINT_SCALE"], rhs, 1);

//     solve_pardiso(N_DOF, RowptrGlobal, ColindGlobal, EntriesGlobal, rhs, lambda);

//     if(AdjointAssembly == nullptr)
//     {
//         char Name[] = "discreteform for adjoint";

//         BoundValues_Adjoint[0] = BoundValue;
//         BoundValues_Adjoint[1] = BoundValue;
//         BoundValues_Adjoint[2] = BoundValue;

//         discreteformAdjoint = new TDiscreteForm3D(Name, Name, N_Terms_Hyperelastic_Single_Field, AllDerivatives_Hyperelastic_Single_Field,
//                                                   Spacenumbers_Hyperelastic_Single_Field, N_SQMATRICES_Hyperelastic_Single_Field, N_RHS_Hyperelastic_Single_Field,
//                                                   rowspace_Hyperelastic_Single_Field, columnspace_Hyperelastic_Single_Field, rhsspace_Hyperelastic_Single_Field, Assembly_Adjoint, NULL, NULL);

//         AdjointAssembly = new TAssembleMat3D(1, fesp,
//                                              N_SQMATRICES_Hyperelastic_Single_Field, SQMATRICES,
//                                              0, nullptr,
//                                              N_RHS_Hyperelastic_Single_Field, RHS, ferhs,
//                                              discreteformAdjoint,
//                                              NoBoundConditions, BoundValues_Adjoint,
//                                              aux);

//         AdjointAssembly->Init();
//     }

//     for (int i = 0; i < n_params; i++)
//     {
//         /* 
//         ! SET FLAG FOR PARAMETER SOMEHOW 
//         */
//         Material::constantID = ParameterNames[i];
//         fill_n(rhs, N_DOF, 0.0);
//         AdjointAssembly->Assemble3D();
//         grad[i] = -cblas_ddot(N_DOF, lambda, 1, rhs, 1);
//     }
// }