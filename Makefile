# Robust Makefile: compute SRCDIR relative to this Makefile
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SRCDIR       := $(abspath $(MAKEFILE_DIR)/../src)

CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -march=native -Wall -Wextra
FC       ?= gfortran
FFLAGS   ?= -O3

# Enable OpenMP by passing OMP=1 to make
ifeq ($(OMP),1)
  CXXFLAGS += -fopenmp
  FFLAGS   += -fopenmp
endif

.PHONY: all clean

all: bellman_demo bellman_fortran

bellman_demo: $(SRCDIR)/dsge_rbc_cpp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Build Fortran if gfortran exists; otherwise print skip message
GFORT := $(shell command -v $(FC) 2>/dev/null)
ifeq ($(GFORT),)
bellman_fortran:
	@echo "gfortran not found; skipping Fortran build"
else
bellman_fortran: $(SRCDIR)/dsge_rbc_fortran.f90
	$(FC) $(FFLAGS) -o $@ $<
endif

clean:
	rm -f bellman_demo bellman_fortran
bench:
	. .venv/bin/activate && python3 benchmarks/run_benchmarks.py
