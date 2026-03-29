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

#include "gmxpre.h"

#include "gromacs/mdtypes/exactrespaparameters.h"

#include "gromacs/mdtypes/inputrec.h"

namespace gmx
{

ExactRespaParameters exactRespaParametersFromLegacyMts(const MtsMode                  mtsMode,
                                                       const ArrayRef<const MtsLevel> mtsLevels,
                                                       const LammpsRespaParameters&   lammpsRespa)
{
    ExactRespaParameters parameters;
    if (mtsMode != MtsMode::LammpsRespa || !lammpsRespa.enabled || mtsLevels.empty())
    {
        return parameters;
    }

    parameters.levelStepFactors.reserve(mtsLevels.size());
    for (const auto& mtsLevel : mtsLevels)
    {
        parameters.levelStepFactors.push_back(mtsLevel.stepFactor);
    }
    parameters.forceLayout = lammpsRespa;
    return parameters;
}

bool useExactRespa(const t_inputrec& ir)
{
    return ir.exactRespa.enabled();
}

bool useMtsSubstepping(const t_inputrec& ir)
{
    return ir.useMts;
}

} // namespace gmx
