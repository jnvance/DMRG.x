## testing the kronecker product

import os
import sys

os.environ['PETSC_DIR']="/Users/jnvance/Source/petsc-3.7.6"
os.environ['PETSC_ARCH']="arch-darwin-complex-opt"
sys.path.append(os.environ['PETSC_DIR']+"/bin")

import PetscBinaryIO as pet
import numpy as np
import scipy

import matplotlib.pyplot as plt


def run_and_check(nprocs, plot=True):
    ## create empty test_kron folder
    os.system('rm -rf test_kron; mkdir test_kron; make test_kron.x')
    ## run executable
    os.system('mpirun -np {} ./test_kron.x'.format(nprocs))
    io = pet.PetscBinaryIO()
    with open('test_kron/A.dat','r') as fh:
        A = io.readBinaryFile(fh,'scipy.sparse')[0]
    with open('test_kron/B.dat','r') as fh:
        B = io.readBinaryFile(fh,'scipy.sparse')[0]
    with open('test_kron/C.dat','r') as fh:
        C = io.readBinaryFile(fh,'scipy.sparse')[0]
    spC = scipy.sparse.kron(A,B)

    if plot:
        plt.imshow(np.absolute(A.toarray()))
        plt.savefig("test_kron/A.png")
        plt.clf()
        plt.imshow(np.absolute(B.toarray()))
        plt.savefig("test_kron/B.png")
        plt.clf()
        plt.imshow(np.absolute(C.toarray()))
        plt.savefig("test_kron/C.png")
        plt.clf()
        plt.imshow(np.absolute(spC.toarray()))
        plt.savefig("test_kron/spC.png")
        plt.clf()


    return np.all(C.toarray() == spC.toarray())

if __name__ == '__main__':

    ## make executable
    if not os.path.exists('test_kron.x'):
        os.system('make test_kron.x')

    ## read and check results
    if run_and_check(1, True):
        print("PASSED")
    else:
        print("FAILED")
