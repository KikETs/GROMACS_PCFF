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
/*! \internal \file
 * \brief Tests for standalone exact-r-RESPA scheduling helpers.
 *
 * \ingroup module_mdtypes
 */
#include "gmxpre.h"

#include "gromacs/mdtypes/exactrespaschedule.h"
#include "gromacs/mdtypes/inputrec.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace gmx
{

namespace test
{

namespace
{

ExactRespaParameters threeLevelExactRespa()
{
    ExactRespaParameters exactRespa;
    exactRespa.levelStepFactors   = { 1, 2, 4 };
    exactRespa.forceLayout.enabled = true;
    exactRespa.forceLayout.pairLevel = 1;
    exactRespa.forceLayout.kspaceLevel = 2;
    return exactRespa;
}

ExactRespaParameters threeLevelSplitExactRespa()
{
    ExactRespaParameters exactRespa = threeLevelExactRespa();
    exactRespa.forceLayout.innerLevel = 0;
    exactRespa.forceLayout.middleLevel = 1;
    exactRespa.forceLayout.outerLevel  = 2;
    exactRespa.forceLayout.innerOff    = 0.2;
    exactRespa.forceLayout.innerOn     = 0.3;
    exactRespa.forceLayout.outerOn     = 0.6;
    exactRespa.forceLayout.outerOff    = 0.8;
    return exactRespa;
}

} // namespace

TEST(ExactRespaSchedule, ReportsHighestActiveLevelForNestedSchedule)
{
    const auto exactRespa = threeLevelExactRespa();

    EXPECT_EQ(highestActiveExactRespaLevel(exactRespa, 0), 2);
    EXPECT_EQ(highestActiveExactRespaLevel(exactRespa, 1), 0);
    EXPECT_EQ(highestActiveExactRespaLevel(exactRespa, 2), 1);
    EXPECT_EQ(highestActiveExactRespaLevel(exactRespa, 3), 0);
    EXPECT_EQ(highestActiveExactRespaLevel(exactRespa, 4), 2);
}

TEST(ExactRespaSchedule, ReconstructsRestartStoreOnlyAwayFromOuterBoundary)
{
    const auto exactRespa = threeLevelExactRespa();

    EXPECT_FALSE(exactRespaRestartRequiresForceStoreReconstruction(exactRespa, 0));
    EXPECT_TRUE(exactRespaRestartRequiresForceStoreReconstruction(exactRespa, 1));
    EXPECT_TRUE(exactRespaRestartRequiresForceStoreReconstruction(exactRespa, 2));
    EXPECT_TRUE(exactRespaRestartRequiresForceStoreReconstruction(exactRespa, 3));
    EXPECT_FALSE(exactRespaRestartRequiresForceStoreReconstruction(exactRespa, 4));

    EXPECT_FALSE(exactRespaRestartRequiresForceStoreReconstruction(ExactRespaParameters{}, 1));
}

TEST(ExactRespaSchedule, ReportsBaseStepTraceForThreeLevelSchedule)
{
    const auto exactRespa = threeLevelExactRespa();

    const auto trace0 = exactRespaBaseStepTrace(exactRespa, 0);
    EXPECT_THAT(trace0.initialKickLevels, ::testing::ElementsAre(2, 1, 0));
    EXPECT_THAT(trace0.refreshedForceLevels, ::testing::ElementsAre(0));
    EXPECT_THAT(trace0.finalKickLevels, ::testing::ElementsAre(0));

    const auto trace1 = exactRespaBaseStepTrace(exactRespa, 1);
    EXPECT_THAT(trace1.initialKickLevels, ::testing::ElementsAre(0));
    EXPECT_THAT(trace1.refreshedForceLevels, ::testing::ElementsAre(0, 1));
    EXPECT_THAT(trace1.finalKickLevels, ::testing::ElementsAre(0, 1));

    const auto trace2 = exactRespaBaseStepTrace(exactRespa, 2);
    EXPECT_THAT(trace2.initialKickLevels, ::testing::ElementsAre(1, 0));
    EXPECT_THAT(trace2.refreshedForceLevels, ::testing::ElementsAre(0));
    EXPECT_THAT(trace2.finalKickLevels, ::testing::ElementsAre(0));

    const auto trace3 = exactRespaBaseStepTrace(exactRespa, 3);
    EXPECT_THAT(trace3.initialKickLevels, ::testing::ElementsAre(0));
    EXPECT_THAT(trace3.refreshedForceLevels, ::testing::ElementsAre(0, 1, 2));
    EXPECT_THAT(trace3.finalKickLevels, ::testing::ElementsAre(0, 1, 2));
}

TEST(ExactRespaSchedule, ReportsSlowestStepFactor)
{
    t_inputrec ir;
    EXPECT_EQ(exactRespaSlowestStepFactor(ir), 1);

    ir.exactRespa = threeLevelExactRespa();
    EXPECT_EQ(exactRespaSlowestStepFactor(ir), 4);
}

TEST(ExactRespaSchedule, ReportsNonbondedSplitLevels)
{
    t_inputrec ir;
    ir.exactRespa = threeLevelSplitExactRespa();

    EXPECT_TRUE(exactRespaHasPairSplitting(ir));
    EXPECT_EQ(exactRespaNonbondedInnerLevel(ir), 0);
    EXPECT_EQ(exactRespaNonbondedMiddleLevel(ir), 1);
    EXPECT_EQ(exactRespaNonbondedOuterLevel(ir), 2);
    EXPECT_EQ(exactRespaNonbondedMtsFactor(ir), 4);
    EXPECT_EQ(exactRespaLongrangeNonbondedLevel(ir), 2);
    EXPECT_EQ(exactRespaPullLevel(ir), 0);
    EXPECT_EQ(exactRespaAwhLevel(ir), 0);
}

} // namespace test

} // namespace gmx
