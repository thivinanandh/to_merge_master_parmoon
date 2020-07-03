// ======================================================================
// instationary problem
// ======================================================================

/// ========================================================================
// example file
// ========================================================================

#define __SIN3__

void ExampleFile()
{
  OutPut("Example: newfile.h" << endl);
}

// exact solution
void Exact(double x, double y, double *values)
{

}

// kind of boundary condition (for FE space needed)
void BoundCondition(int BdComp, double t, BoundCond &cond)
{

}

// value of boundary condition
void BoundValue(int BdComp, double Param, double &value)
{

}

// initial conditon
void InitialCondition(double x, double y, double *values)
{

}


void BilinearCoeffs(int n_points, double *X, double *Y,
        double **parameters, double **coeffs)
{

}
