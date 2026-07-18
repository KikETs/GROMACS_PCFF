/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 */

#include "gmxpre.h"

#include <gtest/gtest.h>

#include "gromacs/mdlib/couplingtesting.h"

namespace gmx
{
namespace test
{
namespace
{

TEST(PcffMttkLammpsDrag, UnsetAndExplicitZeroDisableTheOptIn)
{
    EXPECT_DOUBLE_EQ(pcffLammpsMttkDragFromText(nullptr), 0.0);
    EXPECT_DOUBLE_EQ(pcffLammpsMttkDragFromText(""), 0.0);
    EXPECT_DOUBLE_EQ(pcffLammpsMttkDragFromText("0"), 0.0);
    EXPECT_DOUBLE_EQ(pcffLammpsMttkDragFromText("off"), 0.0);
    EXPECT_DOUBLE_EQ(pcffLammpsMttkDragFromText("false"), 0.0);
    EXPECT_DOUBLE_EQ(pcffLammpsMttkDragFromText("FALSE"), 0.0);
    EXPECT_DOUBLE_EQ(pcffLammpsMttkDragFromText("4"), 4.0);
}

TEST(PcffMttkLammpsDrag, Eq09FactorsMatchFixNhOuterTimestepFormula)
{
    // Successful 1474101 Eq09: dt=0.25 fs, Tdamp=100 fs,
    // Pdamp=5000 fs, drag=4, and default tloop=ploop=1.
    constexpr double c_outerDtPs = 0.00025;
    constexpr double c_drag      = 4.0;

    EXPECT_NEAR(pcffLammpsMttkDragFactor(c_outerDtPs, 1.0 / 0.1, c_drag, 1),
                0.99,
                1.0e-15);
    EXPECT_NEAR(pcffLammpsMttkDragFactor(c_outerDtPs, 1.0 / 5.0, c_drag, 1),
                0.9998,
                1.0e-15);
}

TEST(PcffMttkLammpsDrag, ChainDragIsAppliedBetweenKickAndSecondExponential)
{
    constexpr double c_velocity          = 2.0;
    constexpr double c_forceIncrement    = 0.5;
    constexpr double c_exponentialFactor = 0.8;
    constexpr double c_dragFactor        = 0.9;

    const double expected = ((c_velocity * c_exponentialFactor + c_forceIncrement)
                             * c_dragFactor)
                            * c_exponentialFactor;
    EXPECT_DOUBLE_EQ(pcffLammpsMttkChainVelocityAfterKick(c_velocity,
                                                          c_forceIncrement,
                                                          c_exponentialFactor,
                                                          true,
                                                          c_dragFactor),
                     expected);
}

TEST(PcffMttkLammpsDrag, ZeroDragThermostatChainStillUsesFixNhOperationOrder)
{
    constexpr double c_velocity          = 2.0;
    constexpr double c_forceIncrement    = 0.5;
    constexpr double c_exponentialFactor = 0.8;
    const double expected = (c_velocity * c_exponentialFactor + c_forceIncrement)
                            * c_exponentialFactor;

    EXPECT_DOUBLE_EQ(pcffLammpsMttkChainVelocityAfterKick(c_velocity,
                                                          c_forceIncrement,
                                                          c_exponentialFactor,
                                                          true,
                                                          1.0),
                     expected);
}

TEST(PcffMttkLammpsDrag, ZeroDragPressureChainStillUsesFixNhOperationOrder)
{
    constexpr double c_velocity          = -1.25;
    constexpr double c_forceIncrement    = 0.375;
    constexpr double c_exponentialFactor = 0.95;
    const double expected = (c_velocity * c_exponentialFactor + c_forceIncrement)
                            * c_exponentialFactor;

    EXPECT_DOUBLE_EQ(pcffLammpsMttkChainVelocityAfterKick(c_velocity,
                                                          c_forceIncrement,
                                                          c_exponentialFactor,
                                                          true,
                                                          1.0),
                     expected);
}

TEST(PcffMttkLammpsDrag, NonLammpsChainPathPreservesLegacyOperationOrder)
{
    constexpr double c_velocity          = 2.0;
    constexpr double c_forceIncrement    = 0.5;
    constexpr double c_exponentialFactor = 0.8;
    const double legacy = c_exponentialFactor * (c_velocity + c_forceIncrement)
                          * c_exponentialFactor;

    EXPECT_DOUBLE_EQ(pcffLammpsMttkChainVelocityAfterKick(c_velocity,
                                                          c_forceIncrement,
                                                          c_exponentialFactor,
                                                          false,
                                                          1.0),
                     legacy);
}

TEST(PcffMttkLammpsDrag, BarostatDragFollowsThePressureKick)
{
    EXPECT_DOUBLE_EQ(pcffLammpsMttkBarostatVelocityAfterKick(2.0, 0.5, true, 0.9), 2.25);
    EXPECT_DOUBLE_EQ(pcffLammpsMttkBarostatVelocityAfterKick(2.0, 0.5, false, 1.0), 2.5);
}

} // namespace
} // namespace test
} // namespace gmx
