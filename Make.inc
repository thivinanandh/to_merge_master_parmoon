# ======================================================================
# INTEL64: Linux on INTEL64 using intel compiler
# ======================================================================
CPPFLAG_INTEL64_OPTI  = -O3 -s $(INC) -qopenmp
CPPFLAG_INTEL64_FAST  = -s $(INC) -qopenmp
CPPFLAG_INTEL64_DEBUG = -g $(INC) -qopenmp

ifeq ($(PARALLEL_TYPE),$(filter $(PARALLEL_TYPE),OPENMP SEQUENTIAL))
   CPP_INTEL64           =  icpc
   CC_INTEL64            =  icc
 ifeq ($(PRG_TYPE),SMPI)
  CPP_INTEL64           = mpiicpc
  CC_INTEL64            = mpiicc
 endif   
else
   CPP_INTEL64           = mpiicpc  
   CC_INTEL64            = mpiicc  
endif

DEFINE_INTEL64        = -D__$(GEO)__ -D__LINUX__ $(MORTAR) $(SPECIAL) -DMKL_ILP64 
CCFLAG_INTEL64_OPTI   = -O3 -s -DTRILIBRARY -DREDUCED -DNO_TIMER $(INC) -qopenmp
CCFLAG_INTEL64_FAST   = -s -DTRILIBRARY -DREDUCED -DNO_TIMER $(INC) -qopenmp
CCFLAG_INTEL64_DEBUG  = -g -DTRILIBRARY -DREDUCED -DNO_TIMER $(INC) -qopenmp
LDFLAG_INTEL64        =
BLASFLAG_INTEL64      =  
BLAS_INTEL64          = -L$(MKLROOT)/lib/intel64 -lmkl_rt -lpthread -qopenmp

ifeq ($(PARALLEL_TYPE),MPI)
BLACS_PATH = EXT_LIB/MPIBLACS
MPIBLACS_INTEL64 = -L$(BLACS_PATH) -lscalapack_INTEL64 $(BLACS_PATH)/blacs_MPI_INTEL64-0.a \
                     $(BLACS_PATH)/blacsF77init_MPI_INTEL64-0.a $(BLACS_PATH)/blacs_MPI_INTEL64-0.a -lifcore
endif

ifeq ($(PRG_TYPE), SMPI)

BLACS_PATH = EXT_LIB/MPIBLACS
MPIBLACS_INTEL64 = -L$(BLACS_PATH) -lscalapack_INTEL64 $(BLACS_PATH)/libblacs_MPI_INTEL64-0.a \
                     $(BLACS_PATH)/libblacsF77init_MPI_INTEL64-0.a $(BLACS_PATH)/libblacs_MPI_INTEL64-0.a -lifcore
endif


# ======================================================================
# LINUX64: Linux with gnu
# ======================================================================
CPPFLAG_LINUX64_OPTI  = -O3 -s $(INC) -std=c++11 -fopenmp
CPPFLAG_LINUX64_FAST  =  $(INC) -std=c++11 -fopenmp 
CPPFLAG_LINUX64_DEBUG = -g $(INC) -std=c++11 -fopenmp
DEFINE_LINUX64        = -D__$(GEO)__ -D__LINUX__ $(MORTAR) $(SPECIAL)   -D__$(FLAG)  
CCFLAG_LINUX64_OPTI   = -O3 -s -DTRILIBRARY -DREDUCED -DNO_TIMER $(INC) -DMKL_ILP64 -m64 -fopenmp
CCFLAG_LINUX64_FAST   =  -DTRILIBRARY -DREDUCED -DNO_TIMER $(INC) -DMKL_ILP64 -m64 -fopenmp
CCFLAG_LINUX64_DEBUG  = -g -DTRILIBRARY -DREDUCED -DNO_TIMER $(INC) -DMKL_ILP64 -m64  -fopenmp
 

ifeq ($(PARALLEL_TYPE),$(filter $(PARALLEL_TYPE),OPENMP SEQUENTIAL))
 CPP_LINUX64           = g++
 CC_LINUX64            = gcc
 ifeq ($(PRG_TYPE),SMPI)
  CPP_LINUX64           = mpicxx 
  CC_LINUX64            = mpicc 
 endif
else
  CPP_LINUX64           = mpicxx 
  CC_LINUX64            = mpicc 
endif


#BLAS_LINUX64 = -L$(MKLROOT)/lib/intel64 -lmkl_rt -lgfortran   -lpthread 

ACML_PATH =  EXT_LIB/ACML/gfortran64/lib
#BLAS_LINUX64 =  -L$(ACML_PATH) -lacml  -lacml_mp  -lgfortran
BLAS_LINUX64 =  $(ACML_PATH)/libacml.a  $(ACML_PATH)/libacml_mp.a -lgfortran  -lpthread

ifeq ($(PARALLEL_TYPE), MPI)

BLACS_PATH = EXT_LIB/MPIBLACS
MPI_PATH = /home/raviteja/mpich-install/lib64
MPIBLACS_LINUX64  = -L$(BLACS_PATH) -lscalapack_LINUX64 -lblacs_MPI_LINUX64-0 -lblacsF77init_MPI_LINUX64-0 -lblacs_MPI_LINUX64-0  \
                     -L$(MPI_PATH)  -lmpich -lmpichf90 -lmpichcxx
endif

ifeq ($(PRG_TYPE), SMPI)

BLACS_PATH = EXT_LIB/MPIBLACS
MPI_PATH = /home/raviteja/mpich-install/lib64
MPIBLACS_LINUX64  = -L$(BLACS_PATH) -lscalapack_LINUX64 -lblacs_MPI_LINUX64-0 -lblacsF77init_MPI_LINUX64-0 -lblacs_MPI_LINUX64-0  \
                     -L$(MPI_PATH)  -lmpich -lmpichf90 -lmpichcxx
endif


ifeq ($(PARALLEL_TYPE), HYBRID)

BLACS_PATH = EXT_LIB/MPIBLACS
MPI_PATH = /opt/mpich2/lib64
MPIBLACS_LINUX64  = -L$(BLACS_PATH) -lscalapack_LINUX64 -lblacs_MPI_LINUX64-0 -lblacsF77init_MPI_LINUX64-0 -lblacs_MPI_LINUX64-0  \
                     -L$(MPI_PATH)  -lmpich -lmpichf90 -lmpichcxx
endif

ifeq ($(PARALLEL_TYPE),OMPONLY)

MUMPSLIB = EXT_LIB/MUMPS/lib
PARSOLVER_LIB = -L$(MUMPSLIB)  -ldmumps_$(MACH) -lmumps_common_$(MACH)  \
                -LEXT_LIB/Metis  -lparmetis_$(MACH) -lmetis_$(MACH) \
                -L$(MUMPSLIB) -lpord_$(MACH) -lptesmumps_$(MACH) -lptscotch_$(MACH) -lscotch_$(MACH) -lptscotcherrexit_$(MACH) -lptscotchparmetis_$(MACH) \
                 $(MPIBLACS_$(MACH)) -lpardiso500-GNU481-X86-64_$(MACH)
endif

# ======================================================================
# MAC64: MAC with CLANG
# ======================================================================
CPPFLAG_MAC64_OPTI  = -O3 $(INC) -fapple-pragma-pack 
CPPFLAG_MAC64_FAST  =  $(INC) -fapple-pragma-pack 
CPPFLAG_MAC64_DEBUG = -g $(INC) -fapple-pragma-pack 
DEFINE_MAC64        = -D__$(GEO)__ -D__MAC64__ $(MORTAR) $(SPECIAL)   -D__$(FLAG)  
CCFLAG_MAC64_OPTI   = -O3  -DTRILIBRARY -DREDUCED -DNO_TIMER $(INC) -DMKL_ILP64 -m64 -fapple-pragma-pack  
CCFLAG_MAC64_FAST   = -DTRILIBRARY -DREDUCED -DNO_TIMER $(INC) -DMKL_ILP64 -m64 -fapple-pragma-pack 
CCFLAG_MAC64_DEBUG  = -g -DTRILIBRARY -DREDUCED -DNO_TIMER $(INC) -DMKL_ILP64 -m64  -fapple-pragma-pack  
 
ifeq ($(PARALLEL_TYPE),$(filter $(PARALLEL_TYPE),OPENMP SEQUENTIAL))
  CPP_MAC64           = clang++ 
  CC_MAC64            = clang 
else
  CPP_MAC64           = mpicxx-mpich-mp  
  CC_MAC64            = mpicc-mpich-mp

endif


BLAS_MAC64 =   -framework Accelerate -lpthread
 

ifeq ($(PARALLEL_TYPE), MPI)

MPI_MACPATH = /opt/local/lib/mpich-mp
BLACS_PATH = EXT_LIB/MPIBLACS
MPIBLACS_MAC64  = -L$(BLACS_PATH) -lscalapack_MAC64 -llapack_MAC64 \
                  -L$(MPI_MACPATH)  -lmpi -lmpifort -lmpicxx
                        
endif


# ======================================================================
#TYRONE64: Linux on AMD 64 TYRONE64 cluster
# ======================================================================
CPPFLAG_TYRONE64_OPTI  = -O3 -s $(INC)
CPPFLAG_TYRONE64_FAST  = -s $(INC)
CPPFLAG_TYRONE64_DEBUG = -g $(INC)
DEFINE_TYRONE64        = -D__$(GEO)__ -D__LINUX__ $(MORTAR) $(SPECIAL) -DMPICH_IGNORE_CXX_SEEK -D__$(FLAG)  
CCFLAG_TYRONE64_OPTI   = -O3 -s -DTRILIBRARY -DREDUCED -DNO_TIMER $(INC)
CCFLAG_TYRONE64_FAST   = -s -DTRILIBRARY -DREDUCED -DNO_TIMER $(INC)
CCFLAG_TYRONE64_DEBUG  = -g -DTRILIBRARY -DREDUCED -DNO_TIMER $(INC)
 

ifeq ($(PARALLEL_TYPE),$(filter $(PARALLEL_TYPE),OPENMP SEQUENTIAL))
  CPP_TYRONE64           = g++ 
  CC_TYRONE64            = gcc  
else
  CPP_TYRONE64           = mpicxx  
  CC_TYRONE64            = mpicc
endif

ACML_PATH = EXT_LIB/ACML/gfortran64/lib
BLAS_TYRONE64 =  $(ACML_PATH)/libacml.a  $(ACML_PATH)/libacml_mp.a -lgfortran -lrt -lpthread

ifeq ($(PARALLEL_TYPE), MPI)

  BLACS_PATH = EXT_LIB/MPIBLACS
  MPI_PATH = /opt/mvapich2-1.8-r5423/gcc/lib
  MPIBLACS_TYRONE64  = -L$(BLACS_PATH) -lscalapack_TYRONE64 $(BLACS_PATH)/blacs_MPI_TYRONE64-0.a \
                      $(BLACS_PATH)/blacsF77init_MPI_TYRONE64-0.a $(BLACS_PATH)/blacs_MPI_TYRONE64-0.a \
		      -L$(MPI_PATH)  -lmpich \
                      -L$(MPI_PATH) -lmpichf90 \
                      -L$(MPI_PATH) -lmpichcxx
 
 else

ACML_PATH = EXT_LIB/ACML/gfortran64/lib
BLAS_TYRONE64 = $(ACML_PATH)/libacml.a  $(ACML_PATH)/libacml_mp.a -lgfortran -lrt -lpthread
PARADISO_TYRONE64 =  -LEXT_LIB/PARDISO/TYRONE64/OpenMP -lpardiso400_GNU430_IA64
 OMP_COMPILE_FLAGS_TYRONE64 = -fopenmp

 endif

# ======================================================================
#CRAY64: Linux on SUSE Linux 64 CRAY64 cluster
# ======================================================================
CPPFLAG_CRAY64_OPTI  = -O3 $(INC)
CPPFLAG_CRAY64_FAST  =  $(INC)
CPPFLAG_CRAY64_DEBUG = -g $(INC)
DEFINE_CRAY64        = -D__$(GEO)__ -D__LINUX__ $(MORTAR) $(SPECIAL) -DMPICH_IGNORE_CXX_SEEK -D__$(FLAG)
CCFLAG_CRAY64_OPTI   = -O3 -DTRILIBRARY -DREDUCED -DNO_TIMER $(INC)
CCFLAG_CRAY64_FAST   = -DTRILIBRARY -DREDUCED -DNO_TIMER $(INC)
CCFLAG_CRAY64_DEBUG  = -g -DTRILIBRARY -DREDUCED -DNO_TIMER $(INC)


ifeq ($(PARALLEL_TYPE),$(filter $(PARALLEL_TYPE),OPENMP SEQUENTIAL))
  CPP_CRAY64           = CC
  CC_CRAY64            = cc
else
  CPP_CRAY64           = CC
  CC_CRAY64            = cc
endif

ifeq ($(PARALLEL_TYPE), MPI)

 # BLACS_PATH =
 # MPIBLACS_CRAY64  =
 else

 BLAS_CRAY64 =
 PARADISO_CRAY64 =  -LEXT_LIB/PARDISO/CRAY64/OpenMP -lpardiso400_GNU430_IA64
 OMP_COMPILE_FLAGS_CRAY64 = -fopenmp

 endif

# ======================================================================
# LINUX: Linux
# ======================================================================
CPPFLAG_LINUX_OPTI  = -s -O3 -fexceptions -pipe  -fomit-frame-pointer\
    -funroll-loops -fstrict-aliasing \
          -fsched-interblock \
          -falign-loops=16 \
          -falign-jumps=16 \
          -falign-functions=16 \
          -ffast-math \
          -fstrict-aliasing \
          -fexpensive-optimizations \
          -m32  -mmmx -msse -msse2 -msse3 -march=nocona\
          $(INC)
#CPPFLAG_LINUX_OPTI  =  -O3 -m32 -ffast-math -fexpensive-optimizations \
#           -s $(INC)
CPPFLAG_LINUX_FAST  = -s $(INC)
CPPFLAG_LINUX_DEBUG = -g $(INC)
CPP_LINUX           = g++
DEFINE_LINUX        = -D__$(GEO)__ -D__LINUX__ $(MORTAR) $(SPECIAL)
CC_LINUX            = gcc
CCFLAG_LINUX_OPTI  = -O3 -s -fexceptions -pipe -fomit-frame-pointer \
    -funroll-loops -fstrict-aliasing \
          -fsched-interblock \
          -falign-loops=16 \
          -falign-jumps=16 \
          -falign-functions=16 \
          -ffast-math \
          -fstrict-aliasing \
          -fexpensive-optimizations \
          -m32  -mmmx -msse -msse2 -msse3 -march=nocona\
          $(INC)
#CCFLAG_LINUX_OPTI   = -O3 -s -DTRILIBRARY -DREDUCED -DNO_TIMER -fexceptions $(INC)
CCFLAG_LINUX_FAST   = -s -DTRILIBRARY -DREDUCED -DNO_TIMER $(INC)
CCFLAG_LINUX_DEBUG  = -g -DTRILIBRARY -DREDUCED -DNO_TIMER $(INC)
LDFLAG_LINUX        =
BLASFLAG_LINUX      = -lg2c
#BLASFLAG_LINUX      = -lg2c -static
BLASFLAG_LINUX      = EXT_LIB/g2c/libg2c_$(MACH).so.0

#ACML_PATH = EXT_LIB/ACML/gfortran32/lib
#BLAS_LINUX =  $(ACML_PATH) -lacml.a -lgfortran  -lpthread

BLAS_LINUX =   -LEXT_LIB/BLAS  -llapack_LINUX -lblas_LINUX -lgfortran -lpthread

# ======================================================================
# SiCortex:  Gentoo Linux, kautz.urz.uni-magdeburg.de (141.44.8.32), Parallel Machine in Uni_MD
# MPICH2, gnuc, pathscale compiler
# scmpi, blas, lapack, blacs, scalapack
# ======================================================================
CPPFLAG_SiCortex_OPTI  = -O3  $(INC) -mips64 -march=5kf -mtune=5kf -mabi=64 
CPPFLAG_SiCortex_FAST  = -O0    $(INC) 
CPPFLAG_SiCortex_DEBUG = -g2 -trapuv -Wall $(INC) 
DEFINE_SiCortex        = -D__$(GEO)__  -D__SiCortex__ $(MORTAR) $(SPECIAL) -DMPICH_IGNORE_CXX_SEEK
CCFLAG_SiCortex_OPTI   = -O3  -mips64 -march=5kf -mtune=5kf -mabi=64 -s -DTRILIBRARY -DREDUCED -DNO_TIMER $(INC)
CCFLAG_SiCortex_FAST   = -s    -DTRILIBRARY -DREDUCED -DNO_TIMER $(INC)
CCFLAG_SiCortex_DEBUG  = -g2  -trapuv  -Wall -DTRILIBRARY -DREDUCED -DNO_TIMER $(INC)
LDFLAG_SiCortex        = -lpathfortran -lscm -lmpath -lm
BLASFLAG_SiCortex      =  


ifeq ($(PARALLEL_TYPE),MPI)
 ifeq ($(USE_MARMOT),YES)
  CPP_SiCortex           = marmotcxx
  CC_SiCortex            = marmotcc
 else
  CPP_SiCortex           = mpicxx --gnu
  CC_SiCortex            = mpicc --gnu
 endif
else
 CPP_SiCortex           = pathCC --gnu
 CC_SiCortex            = pathcc --gnu
endif

ifeq ($(PRG_TYPE), SEQUENTIAL)
 BLAS_SiCortex = -llapack -lf77blas -lcblas -latlas
else

 ifeq ($(PARALLEL_TYPE), MPI)
  BLAS_SiCortex =   -llapack -lf77blas -lcblas -latlas
  MPIBLACS_SiCortex  = -lscalapack -lblacs -lblacsF77init -lblacs  -L/usr/lib64/ -lscmpi
 else
 BLAS_SiCortex = -llapack -lf77blas -lcblas -latlas
 PARADISO_SiCortex =  -LEXT_LIB/PARDISO/SiCortex/OpenMP -lpardiso_GNU43_AMD64_INT64_P
 OMP_COMPILE_FLAGS_SiCortex = -openmp
 endif
endif


# ======================================================================
# general settings
# ======================================================================
 ifeq ($(PRG_TYPE),PARALLEL)
  ifeq ($(PARALLEL_TYPE),MPI)
   PRG_DEFINE = -D_PAR -D_MPIONLY -D_MPI
  endif
  ifeq ($(PARALLEL_TYPE),OPENMP)
   PRG_DEFINE = -D_PAR -D_OMPONLY -D_OMP
   OMP_COMPILE_FLAGS = $(OMP_COMPILE_FLAGS_$(MACH))
  endif
  ifeq ($(PARALLEL_TYPE),HYBRID)
   PRG_DEFINE = -D_PAR -D_MPI -D_HYBRID
   OMP_COMPILE_FLAGS = $(OMP_COMPILE_FLAGS_$(MACH))
  endif
  ifeq ($(PARALLEL_TYPE),OMPONLY)
   PRG_DEFINE = -D_PAR -D_OMPONLY
   OMP_COMPILE_FLAGS = $(OMP_COMPILE_FLAGS_$(MACH))
  endif
 else  
  PRG_DEFINE = -D_SEQ
  ifeq ($(PRG_TYPE),SMPI)
   PRG_DEFINE = -D_SEQ -D_SMPI
  endif
 endif

RM            = rm -rf
CPP           = $(CPP_$(MACH))
CPPFLAG       = $(CPPFLAG_$(MACH)_$(FLAG)) $(OMP_COMPILE_FLAGS) 
DEFINE        = $(PRG_DEFINE) $(DEFINE_$(MACH)) -DTETLIBRARY -D__PRIVATE__
CC            = $(CC_$(MACH))
CCFLAG        = $(CCFLAG_$(MACH)_$(FLAG)) $(OMP_COMPILE_FLAGS) 
LIB_GEOxD     = $(LIB_GEO$(GEO))
MEMBERS_GEOxD = $(MEMBERS_GEO$(GEO))
LDFLAG        = $(BLASFLAG_$(MACH)) $(LDFLAG_$(MACH))

UMFPACK       = -LEXT_LIB/UMFPACK/Lib -lumfpack_$(MACH) -lamd_$(MACH)   -lsuitesparseconfig_$(MACH)  
        
TECPLOT       = -LEXT_LIB/tecplot/lib -ltecio_$(MACH)
BLAS          = $(BLAS_$(MACH))


ifeq ($(MACH),MAC64)
UMFPACK       = -LEXT_LIB/UMFPACK/Lib -lumfpack_$(MACH) \
                -lamd_$(MACH) -lsuitesparseconfig_$(MACH)
endif

ifeq ($(MACH),TYRONE64)
UMFPACK       = -LEXT_LIB/UMFPACK/Lib -lumfpack_$(MACH) -lamd_$(MACH) -lsuitesparseconfig_$(MACH) -lcholmod_$(MACH) -lcolamd_$(MACH) -lrt
endif 

ifeq ($(MACH),CRAY64)
UMFPACK       = -LEXT_LIB/UMFPACK/Lib -lumfpack_$(MACH) -lamd_$(MACH) -lsuitesparseconfig_$(MACH) -lcholmod_$(MACH) -lcolamd_$(MACH) -lrt
endif


ifeq ($(PARALLEL_TYPE),MPI)

MUMPSLIB = EXT_LIB/MUMPS/lib
PARSOLVER_LIB = -L$(MUMPSLIB)  -ldmumps_$(MACH) -lmumps_common_$(MACH)  \
                -LEXT_LIB/Metis  -lmetis_$(MACH) -lparmetis_$(MACH) \
                -L$(MUMPSLIB) -lpord_$(MACH) -lptesmumps_$(MACH) -lptscotch_$(MACH) -lscotch_$(MACH) -lptscotcherrexit_$(MACH) -lptscotchparmetis_$(MACH) \
                 $(MPIBLACS_$(MACH))
                 
endif

ifeq ($(PRG_TYPE),SMPI)

MUMPSLIB = EXT_LIB/MUMPS/lib
PARSOLVER_LIB = -L$(MUMPSLIB)  -ldmumps_$(MACH) -lmumps_common_$(MACH)  \
                -LEXT_LIB/Metis -lparmetis_$(MACH)  -lmetis_$(MACH) \
                -L$(MUMPSLIB) -lpord_$(MACH) -lptesmumps_$(MACH) -lptscotch_$(MACH) -lscotch_$(MACH) -lptscotcherrexit_$(MACH) -lptscotchparmetis_$(MACH) \
                 $(MPIBLACS_$(MACH))
                 
endif


ifeq ($(PARALLEL_TYPE),HYBRID)

MUMPSLIB = EXT_LIB/MUMPS/lib
PARSOLVER_LIB = -L$(MUMPSLIB)  -ldmumps_$(MACH) -lmumps_common_$(MACH)  \
                -LEXT_LIB/Metis  -lparmetis_$(MACH) -lmetis_$(MACH) \
                -L$(MUMPSLIB) -lpord_$(MACH) -lptesmumps_$(MACH) -lptscotch_$(MACH) -lscotch_$(MACH) -lptscotcherrexit_$(MACH) -lptscotchparmetis_$(MACH) \
                 $(MPIBLACS_$(MACH))
endif

 

ifeq ($(MACH),SUN)
AR            = CC -xar -o
else
AR            = ar -r
endif

# ======================================================================
# general rules
# ======================================================================
%.o : ../%.C
	@echo $<
	@$(CPP) $(CPPFLAG) $(DEFINE)  -c $< -o $@

%.o : ../%.c
	@echo $<
	@$(CC) $(CCFLAG) $(PARMS_FLAG)    -c $< -o $@

