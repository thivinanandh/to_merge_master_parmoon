// #define __UREA__
// #define __SIMPATURS__
//   
// #include <Urea_3d4d.h>
// #include <MacroCell.h>

void ExampleFile()
{
  
  OutPut("Example: newfile.h" << endl); //OutPut("GRID_TYPE set to " << TDatabase::ParamDB->GRID_TYPE << endl);
}

// ========================================================================
// definitions for the temperature
// ========================================================================

void Exact( double x, double y, double z, double *values)
{

}

// initial conditon
void InitialCondition(double x, double y, double z, double *values)
{ 

}

void BoundCondition(int dummy,double x, double y, double z, BoundCond &cond)
{

}

// value of boundary condition
void BoundValue(int dummy,double x, double y, double z, double &value)
{

}

// ========================================================================
// BilinearCoeffs for Heat 
// ========================================================================
void BilinearCoeffs(int n_points, double *X, double *Y, double *Z,
               double **parameters, double **coeffs)
{

}
