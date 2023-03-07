#!/bin/sh
set -e -u

# No arg => regular simulation. Otherwise, it's modal
if [ $# -eq 0 ]; then
    ccx_preCICE -i flap -precice-participant SolidSolver
fi