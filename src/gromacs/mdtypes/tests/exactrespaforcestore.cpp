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

#include <gtest/gtest.h>

namespace gmx
{
namespace
{

void expectRVecEq(const RVec& actual, const RVec& expected)
{
    EXPECT_EQ(actual[XX], expected[XX]);
    EXPECT_EQ(actual[YY], expected[YY]);
    EXPECT_EQ(actual[ZZ], expected[ZZ]);
}

TEST(ExactRespaForceStore, ReconstructsLevel0FromPhysicalAndSlowTotals)
{
    ExactRespaForceStore store;
    const std::vector<RVec> physicalTotal = { RVec{ 10, 20, 30 }, RVec{ 6, 7, 8 } };
    const std::vector<RVec> level1Total   = { RVec{ 3, 4, 5 }, RVec{ 1, 1, 1 } };
    const std::vector<RVec> level2Total   = { RVec{ 2, 3, 4 }, RVec{ 2, 2, 2 } };

    store.update(makeConstArrayRef(physicalTotal),
                 makeConstArrayRef(level1Total),
                 makeConstArrayRef(level2Total),
                 3);

    ASSERT_TRUE(store.hasLevel(0));
    ASSERT_TRUE(store.hasLevel(1));
    ASSERT_TRUE(store.hasLevel(2));
    expectRVecEq(store.levelTotal(0)[0], RVec{ 5, 13, 21 });
    expectRVecEq(store.levelTotal(0)[1], RVec{ 3, 4, 5 });
    expectRVecEq(store.levelTotal(1)[0], level1Total[0]);
    expectRVecEq(store.levelTotal(2)[0], level2Total[0]);
}

TEST(ExactRespaForceStore, PreservesInactiveSlowLevelsAcrossFastSteps)
{
    ExactRespaForceStore store;
    const std::vector<RVec> outerPhysical = { RVec{ 10, 0, 0 } };
    const std::vector<RVec> level1Total   = { RVec{ 3, 0, 0 } };
    const std::vector<RVec> level2Total   = { RVec{ 2, 0, 0 } };
    store.update(makeConstArrayRef(outerPhysical),
                 makeConstArrayRef(level1Total),
                 makeConstArrayRef(level2Total),
                 3);

    const std::vector<RVec> fastPhysical = { RVec{ 11, 0, 0 } };
    store.update(makeConstArrayRef(fastPhysical), {}, {}, 3);

    ASSERT_TRUE(store.hasLevel(1));
    ASSERT_TRUE(store.hasLevel(2));
    expectRVecEq(store.levelTotal(1)[0], level1Total[0]);
    expectRVecEq(store.levelTotal(2)[0], level2Total[0]);
    expectRVecEq(store.levelTotal(0)[0], RVec{ 6, 0, 0 });
}

} // namespace
} // namespace gmx
