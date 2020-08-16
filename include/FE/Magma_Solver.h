#include "magma_v2.h"
#include "magmasparse.h"

class Magma_Solver{

    protected:
        magma_int_t N, nrhs, ldda, lddb, **dipiv_array, *dinfo_array, *dipiv;
        magma_queue_t queue;

    public:
        /** constructor **/
        Magma_Solver(int N, int nrhs, int ldda, int lddb);

        /** destructor **/
        ~Magma_Solver();

        /** solve batch of linear systems **/
        void Magma_Batch_Solver(magmaDouble_ptr d_A, magmaDouble_ptr d_B, double *h_X, int batchcount);
    
};