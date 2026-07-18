/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 */
#ifndef GMX_MDLIB_COUPLINGTESTING_H
#define GMX_MDLIB_COUPLINGTESTING_H

namespace gmx
{

//! Parse the opt-in LAMMPS FixNH drag value used by the PCFF MTTK path.
double pcffLammpsMttkDragFromText(const char* value);

//! Return FixNH's per-subcycle drag multiplier.
double pcffLammpsMttkDragFactor(double outerDtPs,
                                double frequencyPerPs,
                                double drag,
                                int    chainSubcycles);

//! Apply one chain kick in either LAMMPS FixNH or legacy GROMACS order.
double pcffLammpsMttkChainVelocityAfterKick(double velocity,
                                            double forceIncrement,
                                            double exponentialFactor,
                                            bool   useLammpsFixNhOrder,
                                            double dragFactor);

//! Apply the FixNH pressure kick followed by pdrag to the box velocity.
double pcffLammpsMttkBarostatVelocityAfterKick(double velocity,
                                               double forceIncrement,
                                               bool   useLammpsDragPath,
                                               double dragFactor);

} // namespace gmx

#endif
