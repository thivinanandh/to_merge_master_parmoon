//new problem
// 

void ExampleFile()
{
  TDatabase::ParamDB->INTERNAL_PROBLEM_IDENTITY = OSEEN_PROBLEM;
  OutPut("Example: newfile.h with INTERNAL_PROBLEM_IDENTITY " <<
	 TDatabase::ParamDB->INTERNAL_PROBLEM_IDENTITY <<  endl) ;
}

// ========================================================================
// exact solution
// ========================================================================
void ExactU1(double x, double y, double *values)
{

}

void ExactU2(double x, double y, double *values)
{

}

void ExactP(double x, double y, double *values)
{

}

void InitialU1(double x, double y, double *values)
{

}

void InitialU2(double x, double y, double *values)
{

}

void InitialP(double x, double y, double *values)
{

}

// ========================================================================
// boundary conditions
// ========================================================================
void BoundCondition(int i, double t, BoundCond &cond)
{

}

void U1BoundValue(int BdComp, double Param, double &value)
{

}

void U2BoundValue(int BdComp, double Param, double &value)
{

}

// ========================================================================
// coefficients for Stokes form: A, B1, B2, f1, f2
// ========================================================================
void LinCoeffs(int n_points, double *X, double *Y,
               double **parameters, double **coeffs)
{

}
