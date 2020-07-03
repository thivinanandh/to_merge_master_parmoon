// new problem


void ExampleFile()
{
  OutPut("Example: newfile.h" << endl) ;
}

// ========================================================================
// exact solution
// ========================================================================
void ExactU1(double x, double y,  double z, double *values)
{

}

void ExactU2(double x, double y,  double z, double *values)
{

}

void ExactU3(double x, double y,  double z, double *values)
{

}

void ExactP(double x, double y,  double z, double *values)
{

}

// kind of boundary condition (for FE space needed)
void BoundCondition(int CompID, double x, double y, double z, BoundCond &cond)
{

}

// value of boundary condition
void U1BoundValue(int CompID, double x, double y, double z, double &value)
{

}

// value of boundary condition
void U2BoundValue(int CompID, double x, double y, double z, double &value)
{

}

// value of boundary condition
void U3BoundValue(int CompID, double x, double y, double z, double &value)
{

}
// ========================================================================
// initial solution
// ========================================================================
void InitialU1(double x, double y,  double z, double *values)
{

}

void InitialU2(double x, double y,  double z, double *values)
{

}

void InitialU3(double x, double y,  double z, double *values)
{

}

void InitialP(double x, double y,  double z, double *values)
{

}

// ========================================================================
// coefficients for Stokes form: A, B1, B2, f1, f2
// ========================================================================
void LinCoeffs(int n_points, double *X, double *Y, double *Z,
               double **parameters, double **coeffs)
{

    
}
