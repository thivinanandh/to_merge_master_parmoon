// Navier-Stokes problem, Driven Cavity
// 
// u(x,y) = ?
// p(x,y) = ?

void ExampleFile()
{
  OutPut("Example: newfile.h" << endl) ;
}
// ========================================================================
// initial solution
// ========================================================================
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


