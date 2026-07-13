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

#include "gmxpre.h"

#include "gromacs/mdtypes/exactrespaforcestore.h"

#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "gromacs/utility/gmxassert.h"

namespace gmx
{

namespace
{

bool exactRespaForceStoreMemcpyCopyRequested()
{
    static const bool requested = [] {
        const char* env = std::getenv("GMX_PCFF_EXACT_RESPA_FORCESTORE_MEMCPY_COPY");
        return env == nullptr || std::strcmp(env, "0") != 0;
    }();
    return requested;
}

} // namespace

void ExactRespaForceStore::clear()
{
    for (auto& levelTotal : levelTotals_)
    {
        levelTotal.clear();
    }
    haveLevel_.fill(false);
    numLevels_ = 0;
}

void ExactRespaForceStore::assignLevelTotal(const int level, const ArrayRef<const RVec> total)
{
    static_assert(std::is_trivially_copyable_v<RVec>,
                  "Exact r-RESPA force-store assumes RVec can be copied as contiguous storage");
    GMX_RELEASE_ASSERT(level >= 0 && level < c_numStoredLevels, "Exact r-RESPA force-store level should be valid");
    auto& levelTotal = levelTotals_[level];
    if (exactRespaForceStoreMemcpyCopyRequested())
    {
        levelTotal.resize(total.size());
        if (!total.empty())
        {
            std::memcpy(levelTotal.data(), total.data(), total.size() * sizeof(RVec));
        }
    }
    else
    {
        levelTotal.assign(total.begin(), total.end());
    }
    haveLevel_[level] = true;
}

void ExactRespaForceStore::update(const ArrayRef<const RVec> physicalTotal,
                                  const ArrayRef<const RVec> recomputedLevel1Total,
                                  const ArrayRef<const RVec> recomputedLevel2Total,
                                  const int                  numLevels)
{
    GMX_RELEASE_ASSERT(numLevels >= 1 && numLevels <= c_numStoredLevels,
                       "Exact r-RESPA force-store supports 1-3 total levels");

    numLevels_ = numLevels;

    for (int level = numLevels; level < c_numStoredLevels; ++level)
    {
        levelTotals_[level].clear();
        haveLevel_[level] = false;
    }

    if (numLevels > 1 && !recomputedLevel1Total.empty())
    {
        GMX_RELEASE_ASSERT(recomputedLevel1Total.size() == physicalTotal.size(),
                           "Exact r-RESPA level-1 totals should match the physical total size");
        assignLevelTotal(1, recomputedLevel1Total);
    }
    else if (numLevels > 1)
    {
        GMX_RELEASE_ASSERT(haveLevel_[1] && levelTotals_[1].size() == physicalTotal.size(),
                           "Exact r-RESPA level-1 total should persist across fast steps");
    }

    if (numLevels > 2 && !recomputedLevel2Total.empty())
    {
        GMX_RELEASE_ASSERT(recomputedLevel2Total.size() == physicalTotal.size(),
                           "Exact r-RESPA level-2 totals should match the physical total size");
        assignLevelTotal(2, recomputedLevel2Total);
    }
    else if (numLevels > 2)
    {
        GMX_RELEASE_ASSERT(haveLevel_[2] && levelTotals_[2].size() == physicalTotal.size(),
                           "Exact r-RESPA level-2 total should persist across fast steps");
    }

    assignLevelTotal(0, physicalTotal);
    for (int atom = 0; atom < static_cast<int>(physicalTotal.size()); ++atom)
    {
        if (numLevels > 1)
        {
            levelTotals_[0][atom] -= levelTotals_[1][atom];
        }
        if (numLevels > 2)
        {
            levelTotals_[0][atom] -= levelTotals_[2][atom];
        }
    }
}

void ExactRespaForceStore::updateFromLevelTotals(const ArrayRef<const RVec> level0Total,
                                                 const ArrayRef<const RVec> recomputedLevel1Total,
                                                 const ArrayRef<const RVec> recomputedLevel2Total,
                                                 const int                  numLevels)
{
    GMX_RELEASE_ASSERT(numLevels >= 1 && numLevels <= c_numStoredLevels,
                       "Exact r-RESPA force-store supports 1-3 total levels");

    numLevels_ = numLevels;

    for (int level = numLevels; level < c_numStoredLevels; ++level)
    {
        levelTotals_[level].clear();
        haveLevel_[level] = false;
    }

    assignLevelTotal(0, level0Total);

    if (numLevels > 1 && !recomputedLevel1Total.empty())
    {
        GMX_RELEASE_ASSERT(recomputedLevel1Total.size() == level0Total.size(),
                           "Exact r-RESPA level-1 totals should match the level-0 total size");
        assignLevelTotal(1, recomputedLevel1Total);
    }
    else if (numLevels > 1)
    {
        GMX_RELEASE_ASSERT(haveLevel_[1] && levelTotals_[1].size() == level0Total.size(),
                           "Exact r-RESPA level-1 total should persist across fast steps");
    }

    if (numLevels > 2 && !recomputedLevel2Total.empty())
    {
        GMX_RELEASE_ASSERT(recomputedLevel2Total.size() == level0Total.size(),
                           "Exact r-RESPA level-2 totals should match the level-0 total size");
        assignLevelTotal(2, recomputedLevel2Total);
    }
    else if (numLevels > 2)
    {
        GMX_RELEASE_ASSERT(haveLevel_[2] && levelTotals_[2].size() == level0Total.size(),
                           "Exact r-RESPA level-2 total should persist across fast steps");
    }
}

bool ExactRespaForceStore::hasLevel(const int level) const
{
    GMX_RELEASE_ASSERT(level >= 0 && level < c_numStoredLevels, "Exact r-RESPA force-store level should be valid");
    return haveLevel_[level];
}

ArrayRef<const RVec> ExactRespaForceStore::levelTotal(const int level) const
{
    GMX_RELEASE_ASSERT(hasLevel(level), "Requested exact r-RESPA force-store level should be available");
    return makeConstArrayRef(levelTotals_[level]);
}

} // namespace gmx
