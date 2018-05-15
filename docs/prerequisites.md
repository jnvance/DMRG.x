Prerequisites
=============

The current version requires the following libraries to be installed
 -  [PETSc 3.8.4](https://www.mcs.anl.gov/petsc/)
    [[download]](http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-3.8.4.tar.gz)
    [[installation]](https://www.mcs.anl.gov/petsc/documentation/installation.html)
 -  [SLEPc 3.8.3](http://slepc.upv.es/)
    [[download]](http://slepc.upv.es/download/distrib/slepc-3.8.3.tar.gz)
    [[installation]](http://slepc.upv.es/documentation/instal.htm)

These will also require a working installation of MPI and a suitable BLAS/LAPACK library.
The program may work with more recent versions of PETSc and SLEPc but it has not yet been tested for compatibility.

To properly use the Makefiles for generating executables, the following environment variables must also be defined with the correct values:
 - `PETSC_DIR`
 - `PETSC_ARCH`
 - `SLEPC_DIR`
