/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2022- The GROMACS Authors
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
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 * \brief
 * Tests for velocity generation.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 */

#include "gmxpre.h"

#include "gromacs/gmxpreprocess/gen_maxwell_velocities.h"

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

#include "gromacs/math/units.h"
#include "gromacs/topology/atoms.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/textreader.h"
#include "gromacs/utility/vectypes.h"

#include "testutils/cmdlinetest.h"
#include "testutils/conftest.h"
#include "testutils/refdata.h"
#include "testutils/testasserts.h"
#include "testutils/testfilemanager.h"
#include "testutils/textblockmatchers.h"
#include "testutils/topologyhelpers.h"

namespace gmx
{
namespace test
{
namespace
{

/*! \brief
 * Test params for testing velocity generation.
 *
 * Order is: temperature, seed, numWaters
 */
using MaxwellTestParams = std::tuple<real, int, int>;

class MaxwellTest : public ::testing::Test, public ::testing::WithParamInterface<MaxwellTestParams>
{

public:
    MaxwellTest() : checker_(data_.rootChecker()) {}

    ~MaxwellTest() override;
    //! Initialize topology with \p numWaters.
    void initMtop(int numWaters);
    //! Run test with specific \p temp and \p seed.
    void runTest(real temp, int seed);

private:
    //! System topology.
    gmx_mtop_t mtop_;
    //! Velocity vector.
    std::vector<gmx::RVec> v;
    //! Storage for reference data.
    TestReferenceData data_;
    //! Checker for reference data.
    TestReferenceChecker checker_;
};

MaxwellTest::~MaxwellTest()
{
    done_atom(&mtop_.moltype[0].atoms);
}

void MaxwellTest::initMtop(int numWaters)
{
    addNWaterMolecules(&mtop_, numWaters);
    mtop_.finalize();
    v.resize(mtop_.natoms);
}

void MaxwellTest::runTest(real temp, int seed)
{
    MDLogger logger;
    maxwell_speed(temp, seed, &mtop_, as_rvec_array(v.data()), logger);
    TestReferenceChecker compound(checker_.checkCompound("Velocities", nullptr));
    const auto           tolerance = relativeToleranceAsPrecisionDependentUlp(1.0, 40, 20);
    compound.setDefaultTolerance(tolerance);
    compound.checkSequence(v.begin(), v.end(), "Velocity values");
}

TEST_P(MaxwellTest, CreationWorks)
{
    const auto& params    = GetParam();
    const real  temp      = std::get<0>(params);
    const int   seed      = std::get<1>(params);
    const int   numWaters = std::get<2>(params);
    initMtop(numWaters);

    runTest(temp, seed);
}

INSTANTIATE_TEST_SUITE_P(CorrectVelocity,
                         MaxwellTest,
                         ::testing::Combine(::testing::Values(150, 298, 313, 350),
                                            ::testing::Values(1, 42),
                                            ::testing::Values(23, 42)));

TEST(LammpsMomRotVelocityScaleTest, RemovesLinearAndAngularMomentumAndRestoresTargetTemperature)
{
    constexpr int  c_numAtoms   = 4;
    constexpr real c_targetTemp = 353.0_real;
    std::array<real, c_numAtoms> mass = { 1.0_real, 2.0_real, 3.0_real, 4.0_real };
    std::array<RVec, c_numAtoms> position = { RVec{ 0.0, 0.0, 0.0 },
                                               RVec{ 1.0, 0.0, 0.0 },
                                               RVec{ 0.0, 2.0, 0.0 },
                                               RVec{ 0.0, 0.0, 3.0 } };
    std::array<RVec, c_numAtoms> velocity = { RVec{ 1.0, 2.0, 3.0 },
                                               RVec{ -2.0, 0.5, 1.0 },
                                               RVec{ 0.25, -1.0, 2.5 },
                                               RVec{ 3.0, -2.0, 0.75 } };

    MDLogger logger;
    lammps_mom_rot_velocity_scale(c_targetTemp,
                                  c_numAtoms,
                                  mass.data(),
                                  as_rvec_array(position.data()),
                                  as_rvec_array(velocity.data()),
                                  logger);

    double totalMass = 0;
    RVec   centerOfMass{ 0, 0, 0 };
    RVec   linearMomentum{ 0, 0, 0 };
    for (int atom = 0; atom < c_numAtoms; ++atom)
    {
        totalMass += mass[atom];
        for (int dimension = 0; dimension < DIM; ++dimension)
        {
            centerOfMass[dimension] += mass[atom] * position[atom][dimension];
            linearMomentum[dimension] += mass[atom] * velocity[atom][dimension];
        }
    }
    for (int dimension = 0; dimension < DIM; ++dimension)
    {
        centerOfMass[dimension] /= totalMass;
    }

    RVec   angularMomentum{ 0, 0, 0 };
    double kineticEnergy = 0;
    for (int atom = 0; atom < c_numAtoms; ++atom)
    {
        const double rx = position[atom][XX] - centerOfMass[XX];
        const double ry = position[atom][YY] - centerOfMass[YY];
        const double rz = position[atom][ZZ] - centerOfMass[ZZ];
        const double vx = velocity[atom][XX];
        const double vy = velocity[atom][YY];
        const double vz = velocity[atom][ZZ];
        angularMomentum[XX] += mass[atom] * (ry * vz - rz * vy);
        angularMomentum[YY] += mass[atom] * (rz * vx - rx * vz);
        angularMomentum[ZZ] += mass[atom] * (rx * vy - ry * vx);
        kineticEnergy += 0.5 * mass[atom] * (vx * vx + vy * vy + vz * vz);
    }
    const double temperature =
            2.0 * kineticEnergy / ((DIM * c_numAtoms - DIM) * c_boltz);
    const double momentumTolerance =
            5000.0 * std::numeric_limits<real>::epsilon();
    for (int dimension = 0; dimension < DIM; ++dimension)
    {
        EXPECT_NEAR(linearMomentum[dimension], 0.0, momentumTolerance);
        EXPECT_NEAR(angularMomentum[dimension], 0.0, momentumTolerance);
    }
    EXPECT_NEAR(temperature,
                c_targetTemp,
                5000.0 * std::numeric_limits<real>::epsilon() * c_targetTemp);
}

TEST(LammpsMomRotVelocityScaleTest, PreservesAngularMomentumRemovalForImageUnwrappedCoordinates)
{
    constexpr int  c_numAtoms   = 4;
    constexpr real c_targetTemp = 353.0_real;
    std::array<real, c_numAtoms> mass = { 1.0_real, 2.0_real, 3.0_real, 4.0_real };
    // These are periodic-image-unwrapped coordinates reconstructed from a
    // 1 nm orthorhombic box. Atoms 1, 2, and 3 carry ix=-1, iy=+1, and iz=-1.
    std::array<RVec, c_numAtoms> position = { RVec{ 0.1, 0.2, 0.3 },
                                               RVec{ -0.1, 0.25, 0.4 },
                                               RVec{ 0.2, 1.85, 0.45 },
                                               RVec{ 0.3, 0.4, -0.05 } };
    std::array<RVec, c_numAtoms> velocity = { RVec{ 1.0, 2.0, 3.0 },
                                               RVec{ -2.0, 0.5, 1.0 },
                                               RVec{ 0.25, -1.0, 2.5 },
                                               RVec{ 3.0, -2.0, 0.75 } };

    MDLogger logger;
    lammps_mom_rot_velocity_scale(c_targetTemp,
                                  c_numAtoms,
                                  mass.data(),
                                  as_rvec_array(position.data()),
                                  as_rvec_array(velocity.data()),
                                  logger);

    double totalMass = 0;
    RVec   centerOfMass{ 0, 0, 0 };
    RVec   linearMomentum{ 0, 0, 0 };
    for (int atom = 0; atom < c_numAtoms; ++atom)
    {
        totalMass += mass[atom];
        for (int dimension = 0; dimension < DIM; ++dimension)
        {
            centerOfMass[dimension] += mass[atom] * position[atom][dimension];
            linearMomentum[dimension] += mass[atom] * velocity[atom][dimension];
        }
    }
    for (int dimension = 0; dimension < DIM; ++dimension)
    {
        centerOfMass[dimension] /= totalMass;
    }

    RVec angularMomentum{ 0, 0, 0 };
    for (int atom = 0; atom < c_numAtoms; ++atom)
    {
        const double rx = position[atom][XX] - centerOfMass[XX];
        const double ry = position[atom][YY] - centerOfMass[YY];
        const double rz = position[atom][ZZ] - centerOfMass[ZZ];
        const double vx = velocity[atom][XX];
        const double vy = velocity[atom][YY];
        const double vz = velocity[atom][ZZ];
        angularMomentum[XX] += mass[atom] * (ry * vz - rz * vy);
        angularMomentum[YY] += mass[atom] * (rz * vx - rx * vz);
        angularMomentum[ZZ] += mass[atom] * (rx * vy - ry * vx);
    }

    const double momentumTolerance = 5000.0 * std::numeric_limits<real>::epsilon();
    for (int dimension = 0; dimension < DIM; ++dimension)
    {
        EXPECT_NEAR(linearMomentum[dimension], 0.0, momentumTolerance);
        EXPECT_NEAR(angularMomentum[dimension], 0.0, momentumTolerance);
    }
}

TEST(LammpsMomRotVelocityScaleTest, RejectsSingularInertiaTensor)
{
    constexpr int c_numAtoms = 3;
    std::array<real, c_numAtoms> mass = { 1.0_real, 1.0_real, 1.0_real };
    std::array<RVec, c_numAtoms> position = {
        RVec{ 0.0, 0.0, 0.0 }, RVec{ 1.0, 0.0, 0.0 }, RVec{ 2.0, 0.0, 0.0 }
    };
    std::array<RVec, c_numAtoms> velocity = {
        RVec{ 1.0, 2.0, 3.0 }, RVec{ -1.0, 0.5, 1.0 }, RVec{ 0.25, -1.0, 2.5 }
    };
    MDLogger logger;

    EXPECT_DEATH_IF_SUPPORTED(lammps_mom_rot_velocity_scale(353.0_real,
                                                            c_numAtoms,
                                                            mass.data(),
                                                            as_rvec_array(position.data()),
                                                            as_rvec_array(velocity.data()),
                                                            logger),
                              "singular inertia tensor");
}

} // namespace
} // namespace test
} // namespace gmx
