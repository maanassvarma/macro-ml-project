# Robust Makefile: compute SRCDIR relative to this Makefile
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SRCDIR       := $(abspath $(MAKEFILE_DIR)/../src)

CXX=g++
FC=gfortran
CXXFLAGS=-O3 -std=c++17 -march=native -Wall -Wextra
FFLAGS=-O3

all: bellman_demo bellman_fortran

bellman_demo: $(SRCDIR)/dsge_rbc_cpp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

bellman_fortran: $(SRCDIR)/dsge_rbc_fortran.f90
	$(FC) $(FFLAGS) -o $@ $<

clean:
	rm -f bellman_demo bellman_fortran
