// ======================================================================
// new problem 3D
// ======================================================================
// #include <ConvDiff3D.h>

void ExampleFile()
{
  OutPut("Example: newfile.h" << endl) ;
}

// exact solution
void Exact(double x, double y, double z, double *values)
{

}

// kind of boundary condition (for FE space needed)
void BoundCondition(int BdID, double x, double y, double z, BoundCond &cond)
{

}

// value of boundary condition
void BoundValue(int BdID, double x, double y, double z, double &value)
{

}

void BilinearCoeffs(int n_points, double *x, double *y, double *z,
        double **parameters, double **coeffs)
{

}

