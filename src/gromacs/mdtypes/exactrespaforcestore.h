/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2020- The GROMACS Authors
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
#ifndef GMX_MDTYPES_EXACTRESPAFORCESTORE_H
#define GMX_MDTYPES_EXACTRESPAFORCESTORE_H

#include <array>
#include <vector>

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/vectypes.h"

namespace gmx
{

class ExactRespaForceStore
{
public:
    static constexpr int c_numStoredLevels = 3;

    void clear();

    void update(ArrayRef<const RVec> physicalTotal,
                ArrayRef<const RVec> recomputedLevel1Total,
                ArrayRef<const RVec> recomputedLevel2Total,
                int                  numLevels);

    void updateFromLevelTotals(ArrayRef<const RVec> level0Total,
                               ArrayRef<const RVec> recomputedLevel1Total,
                               ArrayRef<const RVec> recomputedLevel2Total,
                               int                  numLevels);

    bool hasLevel(int level) const;

    ArrayRef<const RVec> levelTotal(int level) const;

    int numLevels() const { return numLevels_; }

private:
    void assignLevelTotal(int level, ArrayRef<const RVec> total);

    std::array<std::vector<RVec>, c_numStoredLevels> levelTotals_;
    std::array<bool, c_numStoredLevels>              haveLevel_ = { false, false, false };
    int                                              numLevels_ = 0;
};

} // namespace gmx

#endif
