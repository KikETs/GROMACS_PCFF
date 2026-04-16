/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 */
#ifndef GMX_MDTYPES_EXACTRESPAPARAMETERS_H
#define GMX_MDTYPES_EXACTRESPAPARAMETERS_H

#include <vector>

#include "gromacs/mdtypes/mtstypes.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"

struct t_inputrec;

namespace gmx
{

//! Exact LAMMPS-style r-RESPA settings stored in the inputrec.
struct LammpsRespaParameters
{
    bool enabled = false;
    int  bondLevel = 0;
    int  angleLevel = 0;
    int  dihedralLevel = 0;
    int  improperLevel = 0;
    int  pair14Level = 0;
    int  pairLevel = 0;
    int  kspaceLevel = 0;
    int  innerLevel = -1;
    int  middleLevel = -1;
    int  outerLevel = -1;
    real innerOff = 0;
    real innerOn = 0;
    real outerOn = 0;
    real outerOff = 0;

    bool hasPairSplitting() const { return innerLevel >= 0 || middleLevel >= 0 || outerLevel >= 0; }
    bool hasMiddle() const { return middleLevel >= 0; }
};

//! Standalone exact r-RESPA metadata, kept separate from generic MTS execution settings.
struct ExactRespaParameters
{
    //! Fast-to-slow level step factors; level 0 is included and is typically 1.
    std::vector<int> levelStepFactors;
    //! Force-class to level mapping and pair-splitting metadata.
    LammpsRespaParameters forceLayout;

    bool enabled() const { return forceLayout.enabled; }
    void clear()
    {
        levelStepFactors.clear();
        forceLayout = {};
    }
};

//! Builds standalone exact-r-RESPA metadata from the current legacy MTS-backed representation.
ExactRespaParameters exactRespaParametersFromLegacyMts(MtsMode                        mtsMode,
                                                       ArrayRef<const MtsLevel>      mtsLevels,
                                                       const LammpsRespaParameters&  lammpsRespa);

//! Returns whether exact r-RESPA semantics are enabled, independent of legacy useMts.
bool useExactRespa(const t_inputrec& ir);

//! Returns whether legacy multiple time stepping is enabled.
bool useMtsSubstepping(const t_inputrec& ir);

//! Returns whether an exact r-RESPA inputrec still retains legacy MTS runtime state.
bool exactRespaRetainsLegacyMtsState(const t_inputrec& ir);

//! Clears legacy MTS runtime state after canonicalizing a LAMMPS-style input into exact r-RESPA.
void clearLegacyMtsStateForExactRespa(t_inputrec* ir);

//! Asserts that exact r-RESPA owns the runtime state and no legacy MTS path can be taken.
void assertExactRespaOwnsNoLegacyMtsState(const t_inputrec& ir);

} // namespace gmx

#endif
