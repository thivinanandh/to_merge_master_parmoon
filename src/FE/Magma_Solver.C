#include <magma_v2.h>
#include <magmasparse.h>
#include <Magma_Solver.h>
#include <iostream>

using namespace std;

Magma_Solver::Magma_Solver(int N, int nrhs, int ldda, int lddb){
    this->N = N;
    this->nrhs = nrhs;
    this->ldda = ldda;
    this->lddb = lddb;
    dipiv_array = NULL;
    dinfo_array = NULL;
    dipiv = NULL;
    magma_queue_create( 0, &queue );
}

void Magma_Solver::Magma_Batch_Solver(magmaDouble_ptr d_A, magmaDouble_ptr d_B, double *h_X, int batchCount){

    magma_init();
    magma_imalloc( &dipiv, N * batchCount);
    magma_imalloc( &dinfo_array, batchCount);

    double **dA_array = NULL;
    double **dB_array = NULL;

    magma_malloc( (void**) &dA_array,    batchCount * sizeof(double*) );
    magma_malloc( (void**) &dB_array,    batchCount * sizeof(double*) );
    magma_malloc( (void**) &dipiv_array, batchCount * sizeof(magma_int_t*) );

    // cout<<"here"<<endl;

    magma_dset_pointer( dA_array, d_A, ldda, 0, 0, ldda*N, batchCount, queue );
    magma_dset_pointer( dB_array, d_B, lddb, 0, 0, lddb*nrhs, batchCount, queue );
    magma_iset_pointer( dipiv_array, dipiv, 1, 0, 0, N, batchCount, queue );

    magma_int_t k = magma_dgesv_batched(N, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, dinfo_array, batchCount, queue);

    magma_free(dA_array);
    magma_free(dB_array);
    magma_free(dipiv_array);
    magma_free(dipiv);
    magma_free(dinfo_array);


    // cout<<k<<endl;

    // magma_dgetmatrix( N, nrhs*batchCount, d_B, lddb, h_X, lddb, queue );



}