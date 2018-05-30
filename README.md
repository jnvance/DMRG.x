<!--start01-->
DMRG.x
======

A distributed-memory implementation of the DMRG algorithm based on the [PETSc](https://www.mcs.anl.gov/petsc/) and
[SLEPc](http://slepc.upv.es/) libraries.

<!--end01-->

__Source code__: [https://github.com/jnvance/DMRG.x](https://github.com/jnvance/DMRG.x)

__Documentation__: [https://dmrgx.readthedocs.io](http://dmrgx.readthedocs.io)

[![Documentation Status](https://readthedocs.org/projects/dmrgx/badge/?version=master)](https://dmrgx.readthedocs.io/en/master/?badge=master)

Abstract
--------

The Density Matrix Renormalization Group (DMRG) algorithm is a numerical technique used in the study of low-dimensional
strongly correlated quantum systems. With this implementation, one can study two-dimensional spin systems using a one-dimensional
traversal of the lattice.

In particular, the current version deals with a __square-type lattice__ with longitudinal dimension \f$ L_x \f$ and transverse
dimension \f$ L_y \f$ shown in the following figure.
The spin sites can interact with their nearest neighbor (NN) and next-nearest neighbors (NNN), and different boundary conditions may be implemented on the two directions (such as the cylindrical boundary conditions illustrated below).

![](./assets/img/lattice-j1-j2-square.png)

The Hamiltonian for this implementation takes the form:

![](./assets/img/equation-j1-j2.png)

which maps to a Heisenberg model when \f$ J_1 = 1/2 \f$ and \f$ J_2 = \Delta_2 = 0 \f$, and to the J1-J2 XY model when \f$ \Delta_1 = \Delta_2 = 0 \f$.

To reduce the computational cost, we exploit \f$ U(1) \f$ symmetry through conservation of the total magnetization (\f$ S_z \f$).
We also implement a matrix-free approach in the diagonalization of the superblock Hamiltonian.

Documentation
-------------

The full documentation generated with doxygen may be viewed in the link above.
To generate the documentation yourself, go to the root directory and execute

    $ make docs-default

which places the documentation in `docs/html/index.html`.

<!-- __NOTE:__
Since the same markdown files were used in generating the Doxygen documentation, some links may be available only for the
markdown/github version (links marked as __[md]__) and others may work only for the doxygen version (marked as __[dox]__). -->

References
----------

This application was developed as part of the following thesis for the [Master in High Performance Computing Programme](http://mhpc.it):
 - J. Vance. "Large-Scale Implementation of the Density Matrix Renormalization Group Algorithm." (2017). [[link]](http://hdl.handle.net/20.500.11767/68070
)

For more information on the DMRG algorithm, we recommend the following reading materials:
 - U. Schollwöck. "The density-matrix renormalization group." Rev. Mod. Phys. 77, 259 – Published 26 April 2005
    [[doi]](https://doi.org/10.1103/RevModPhys.77.259)
 - A E Feiguin. "The Density Matrix Renormalization Group". In: Strongly Correlated Systems. Berlin, Heidelberg: Springer Berlin Heidelberg, Apr. 2013, pp. 31–65. [[link]](https://www.springer.com/cda/content/document/cda_downloaddocument/9783642351051-c2.pdf?SGWID=0-0-45-1391718-p174727662)

To learn DMRG through a simpler implementation, we suggest starting from the following Python code:
 - James R. Garrison, & Ryan V. Mishmash. (2017, November 29). simple-dmrg/simple-dmrg: Simple DMRG 1.0 (Version v1.0.0). Zenodo.
    [[link]](https://simple-dmrg.readthedocs.io)
    [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1068359.svg)](https://doi.org/10.5281/zenodo.1068359)


<!--
Table of Contents
-----------------
 - [Prerequisites](#prerequisites)
 - [Installation](#installation)
 - [Usage](#usage)
 - [Contributing](#contributing)
 - [Credits](#credits)
 - [License](#license)

Prerequisites
-------------

Installation
------------

Usage
-----

Contributing
------------
-->

Credits
-------

Author:
 - [James Vance](https://www.github.com/jnvance)

This application was developed in collaboration with the
[Condensed Matter and Statistical Physics Section](https://www.ictp.it/research/cmsp.aspx) of the
[The Abdus Salam International Centre for Theoretical Physics](http://www.ictp.it), under the supervision of:
 - [Marcello Dalmonte](https://www.ictp.it/research/cmsp/members/long-term-visiting-researchers/marcello-dalmonte.aspx)
 - [Ivan Girotto](https://www.mhpc.it/people/ivan-girotto)

<!-- ![](https://www.ictp.it/img/ictp_head_logo.png =100x) -->

License
-------

MIT License. Copyright (c) 2018 James Vance.

See full text of LICENSE (view on [github](https://github.com/jnvance/DMRG/blob/master/LICENSE)).
