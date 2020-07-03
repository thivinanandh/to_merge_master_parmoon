// ======================================================================
// new problem
// ======================================================================


void ExampleFile()
{
  OutPut("Example: newfile.h" << endl) ;
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

void BilinearCoeffs(int n_points, double *x, double *y,
        double **parameters, double **coeffs)
{

}

