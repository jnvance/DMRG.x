#!/usr/bin/env python

from __future__ import print_function

from sys import argv, exit
from os.path import join, exists
from numpy import array,genfromtxt,atleast_2d
from numpy.linalg import norm

def extract_eigvals(folder):
    '''
    Reads eigvals.dat in folder and extracts raw data as a
    two-dimensional array of double-precision floating point numbers
    '''
    filename = join(folder,"eigvals.dat")
    assert exists(filename), "File does not exist {}".format(filename)
    rawdata = atleast_2d(genfromtxt(filename,dtype="d"))
    return rawdata

if __name__ == '__main__':
    eigvals1=extract_eigvals(argv[1])[:,3]
    eigvals2=extract_eigvals(argv[2])[:,3]
    tol = 1.0e-16
    if (len(argv) >= 4): tol = argv[3]
    err = norm(eigvals1-eigvals2) > tol
    exit(int(err))
