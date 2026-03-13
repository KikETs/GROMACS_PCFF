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
/*! \internal \file
 * \brief
 * Tests for PCFF/Class2 bonded kernels against the frozen M1 golden corpus.
 */

#include "gmxpre.h"

#include "gromacs/listed_forces/bonded.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/math/paddedvector.h"
#include "gromacs/math/units.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/vec.h"
#include "gromacs/utility/vectypes.h"

#include "testutils/testasserts.h"
#include "testutils/testfilemanager.h"

namespace gmx
{
namespace test
{
namespace
{

constexpr int  c_maxAtoms                      = 4;
constexpr real c_angstromToNm                 = 0.1_real;
constexpr real c_kcalToKj                     = 4.184_real;
constexpr real c_forceKjPerNmToKcalPerAngstrom = 1.0_real / 41.84_real;

struct OutputQuantities
{
    real energy = 0;
    real dvdlambda = 0;
    rvec fshift[c_numShiftVectors] = { { 0 } };
    alignas(GMX_REAL_MAX_SIMD_WIDTH * sizeof(real)) rvec4 f[c_maxAtoms] = { { 0 } };
};

struct ReferenceSummary
{
    std::unordered_map<std::string, double> energies;
    std::vector<RVec>                       forces;
};

struct PcffSystemDefinition
{
    std::string           systemId;
    int                   numAtoms = 0;
    PaddedVector<RVec>    coordinates;
    std::vector<t_iatom>  bondIatoms;
    std::vector<t_iatom>  angleIatoms;
    std::vector<t_iatom>  improperIatoms;
    t_iparams             bondParams = { { 0 } };
    t_iparams             angleParams = { { 0 } };
    t_iparams             improperParams = { { 0 } };
};

real kcalToKj(const real value)
{
    return value * c_kcalToKj;
}

real kjToKcal(const real value)
{
    return value / c_kcalToKj;
}

real angstromToNm(const real value)
{
    return value * c_angstromToNm;
}

real bondK2ToGromacs(const real value)
{
    return kcalToKj(value) / gmx::square(c_angstromToNm);
}

real bondK3ToGromacs(const real value)
{
    return kcalToKj(value) / (gmx::square(c_angstromToNm) * c_angstromToNm);
}

real bondK4ToGromacs(const real value)
{
    return kcalToKj(value) / gmx::square(gmx::square(c_angstromToNm));
}

real bondBondKToGromacs(const real value)
{
    return kcalToKj(value) / gmx::square(c_angstromToNm);
}

real bondAngleKToGromacs(const real value)
{
    return kcalToKj(value) / c_angstromToNm;
}

real forceToLammpsUnits(const real value)
{
    return value * c_forceKjPerNmToKcalPerAngstrom;
}

std::filesystem::path referenceResultsRoot()
{
    std::filesystem::path root = TestFileManager::getInputDataDirectory();
    for (int i = 0; i < 4; ++i)
    {
        root = root.parent_path();
    }
    return root / "tests" / "reference_results" / "m2";
}

ReferenceSummary loadReferenceSummary(const std::string& systemId)
{
    ReferenceSummary summary;
    std::ifstream    input(referenceResultsRoot() / (systemId + ".tsv"));
    if (!input.is_open())
    {
        ADD_FAILURE() << "Could not open reference summary for " << systemId;
        return summary;
    }

    std::string line;
    while (std::getline(input, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        std::istringstream stream(line);
        std::string        kind;
        stream >> kind;
        if (kind == "energy")
        {
            std::string name;
            double      value;
            EXPECT_TRUE(static_cast<bool>(stream >> name >> value));
            summary.energies[name] = value;
        }
        else if (kind == "force")
        {
            int    atomId;
            double fx, fy, fz;
            EXPECT_TRUE(static_cast<bool>(stream >> atomId >> fx >> fy >> fz));
            if (summary.forces.size() < static_cast<std::size_t>(atomId))
            {
                summary.forces.resize(atomId);
            }
            summary.forces[atomId - 1] =
                    RVec{ static_cast<real>(fx), static_cast<real>(fy), static_cast<real>(fz) };
        }
    }

    return summary;
}

OutputQuantities evaluateSingleType(InteractionFunction       ftype,
                                    const std::vector<t_iatom>& iatoms,
                                    const t_iparams&           iparams,
                                    const PaddedVector<RVec>&  coordinates,
                                    const int                  numAtoms)
{
    OutputQuantities output;
    std::vector<int> globalAtomIndex(numAtoms);
    std::iota(globalAtomIndex.begin(), globalAtomIndex.end(), 0);
    std::vector<real> charge(numAtoms, 0);

    matrix box;
    t_pbc  pbc;
    clear_mat(box);
    box[XX][XX] = 10.0;
    box[YY][YY] = 10.0;
    box[ZZ][ZZ] = 10.0;
    set_pbc(&pbc, PbcType::Xyz, box);

    output.energy = calculateSimpleBond(ftype,
                                        iatoms.size(),
                                        iatoms.data(),
                                        &iparams,
                                        as_rvec_array(coordinates.data()),
                                        output.f,
                                        output.fshift,
                                        &pbc,
                                        0.0,
                                        &output.dvdlambda,
                                        charge,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        globalAtomIndex.data(),
                                        BondedKernelFlavor::ForcesAndVirialAndEnergy);
    return output;
}

void accumulateOutput(const OutputQuantities& source, OutputQuantities* destination)
{
    destination->energy += source.energy;
    destination->dvdlambda += source.dvdlambda;
    for (int i = 0; i < c_numShiftVectors; ++i)
    {
        rvec_inc(destination->fshift[i], source.fshift[i]);
    }
    for (int atom = 0; atom < c_maxAtoms; ++atom)
    {
        for (int d = 0; d < DIM; ++d)
        {
            destination->f[atom][d] += source.f[atom][d];
        }
    }
}

OutputQuantities evaluateSystem(const PcffSystemDefinition& system)
{
    OutputQuantities total;
    if (!system.bondIatoms.empty())
    {
        accumulateOutput(
                evaluateSingleType(InteractionFunction::BondClass2,
                                   system.bondIatoms,
                                   system.bondParams,
                                   system.coordinates,
                                   system.numAtoms),
                &total);
    }
    if (!system.angleIatoms.empty())
    {
        accumulateOutput(
                evaluateSingleType(InteractionFunction::AngleClass2,
                                   system.angleIatoms,
                                   system.angleParams,
                                   system.coordinates,
                                   system.numAtoms),
                &total);
    }
    if (!system.improperIatoms.empty())
    {
        accumulateOutput(
                evaluateSingleType(InteractionFunction::ImproperClass2,
                                   system.improperIatoms,
                                   system.improperParams,
                                   system.coordinates,
                                   system.numAtoms),
                &total);
    }
    return total;
}

PcffSystemDefinition makeBondToy()
{
    PcffSystemDefinition system;
    system.systemId    = "bond_toy";
    system.numAtoms    = 2;
    system.coordinates = { { -0.08, 0.0, 0.0 }, { 0.08, 0.0, 0.0 } };
    system.bondIatoms  = { 0, 0, 1 };

    system.bondParams.bond_class2.r0 = angstromToNm(1.53_real);
    system.bondParams.bond_class2.k2 = bondK2ToGromacs(250.0_real);
    system.bondParams.bond_class2.k3 = bondK3ToGromacs(-35.0_real);
    system.bondParams.bond_class2.k4 = bondK4ToGromacs(8.0_real);
    return system;
}

PcffSystemDefinition makeAngleToy()
{
    PcffSystemDefinition system;
    system.systemId    = "angle_toy";
    system.numAtoms    = 3;
    system.coordinates = { { -0.15, 0.0, 0.0 }, { 0.0, 0.0, 0.0 }, { 0.075, 0.1299038, 0.0 } };
    system.bondIatoms  = { 0, 0, 1, 0, 1, 2 };
    system.angleIatoms = { 0, 0, 1, 2 };

    system.bondParams.bond_class2.r0 = angstromToNm(1.42_real);
    system.bondParams.bond_class2.k2 = bondK2ToGromacs(200.0_real);
    system.bondParams.bond_class2.k3 = bondK3ToGromacs(-25.0_real);
    system.bondParams.bond_class2.k4 = bondK4ToGromacs(6.0_real);

    system.angleParams.angle_class2.theta0 = 109.5_real * gmx::c_deg2Rad;
    system.angleParams.angle_class2.k2     = kcalToKj(35.0_real);
    system.angleParams.angle_class2.k3     = kcalToKj(-4.0_real);
    system.angleParams.angle_class2.k4     = kcalToKj(1.2_real);
    system.angleParams.angle_class2.bb_k   = bondBondKToGromacs(8.0_real);
    system.angleParams.angle_class2.bb_r1  = angstromToNm(1.42_real);
    system.angleParams.angle_class2.bb_r2  = angstromToNm(1.42_real);
    system.angleParams.angle_class2.ba_k1  = bondAngleKToGromacs(3.0_real);
    system.angleParams.angle_class2.ba_k2  = bondAngleKToGromacs(2.5_real);
    system.angleParams.angle_class2.ba_r1  = angstromToNm(1.42_real);
    system.angleParams.angle_class2.ba_r2  = angstromToNm(1.42_real);
    return system;
}

PcffSystemDefinition makeImproperToy()
{
    PcffSystemDefinition system;
    system.systemId    = "improper_toy";
    system.numAtoms    = 4;
    system.coordinates = { { -0.1, 0.0, 0.015 }, { 0.0, 0.0, 0.0 }, { 0.105, 0.0, -0.02 }, { 0.0, 0.11, 0.035 } };
    system.bondIatoms  = { 0, 0, 1, 0, 1, 2, 0, 1, 3 };
    system.angleIatoms = { 0, 0, 1, 2, 0, 0, 1, 3, 0, 2, 1, 3 };
    system.improperIatoms = { 0, 0, 1, 2, 3 };

    system.bondParams.bond_class2.r0 = angstromToNm(1.45_real);
    system.bondParams.bond_class2.k2 = bondK2ToGromacs(210.0_real);
    system.bondParams.bond_class2.k3 = bondK3ToGromacs(-28.0_real);
    system.bondParams.bond_class2.k4 = bondK4ToGromacs(6.0_real);

    system.angleParams.angle_class2.theta0 = 109.0_real * gmx::c_deg2Rad;
    system.angleParams.angle_class2.k2     = kcalToKj(28.0_real);
    system.angleParams.angle_class2.k3     = kcalToKj(-3.0_real);
    system.angleParams.angle_class2.k4     = kcalToKj(0.8_real);
    system.angleParams.angle_class2.bb_k   = bondBondKToGromacs(4.5_real);
    system.angleParams.angle_class2.bb_r1  = angstromToNm(1.45_real);
    system.angleParams.angle_class2.bb_r2  = angstromToNm(1.45_real);
    system.angleParams.angle_class2.ba_k1  = bondAngleKToGromacs(1.3_real);
    system.angleParams.angle_class2.ba_k2  = bondAngleKToGromacs(1.1_real);
    system.angleParams.angle_class2.ba_r1  = angstromToNm(1.45_real);
    system.angleParams.angle_class2.ba_r2  = angstromToNm(1.45_real);

    system.improperParams.improper_class2.k0          = kcalToKj(25.0_real);
    system.improperParams.improper_class2.chi0        = 0.0_real;
    system.improperParams.improper_class2.aa_k1       = kcalToKj(1.2_real);
    system.improperParams.improper_class2.aa_k2       = kcalToKj(1.0_real);
    system.improperParams.improper_class2.aa_k3       = kcalToKj(0.9_real);
    system.improperParams.improper_class2.aa_theta0_1 = 110.0_real * gmx::c_deg2Rad;
    system.improperParams.improper_class2.aa_theta0_2 = 109.0_real * gmx::c_deg2Rad;
    system.improperParams.improper_class2.aa_theta0_3 = 108.0_real * gmx::c_deg2Rad;
    return system;
}

real bondClass2AnalyticEnergy(const PcffSystemDefinition& system)
{
    const rvec* coordinates = as_rvec_array(system.coordinates.data());
    rvec        dx;
    rvec_sub(coordinates[0], coordinates[1], dx);
    const real dr = norm(dx) - system.bondParams.bond_class2.r0;
    return system.bondParams.bond_class2.k2 * dr * dr + system.bondParams.bond_class2.k3 * dr * dr * dr
           + system.bondParams.bond_class2.k4 * dr * dr * dr * dr;
}

real angleClass2AnalyticEnergy(const PcffSystemDefinition& system)
{
    const rvec* coordinates = as_rvec_array(system.coordinates.data());
    rvec        r_ij;
    rvec        r_kj;
    real        costh;
    int         t1, t2;
    const real  theta = bond_angle(
            coordinates[0], coordinates[1], coordinates[2], nullptr, r_ij, r_kj, &costh, &t1, &t2);
    const real dtheta = theta - system.angleParams.angle_class2.theta0;
    const real r1     = norm(r_ij);
    const real r2     = norm(r_kj);
    const real dr1    = r1 - system.angleParams.angle_class2.bb_r1;
    const real dr2    = r2 - system.angleParams.angle_class2.bb_r2;

    return system.angleParams.angle_class2.k2 * dtheta * dtheta
           + system.angleParams.angle_class2.k3 * dtheta * dtheta * dtheta
           + system.angleParams.angle_class2.k4 * dtheta * dtheta * dtheta * dtheta
           + system.angleParams.angle_class2.bb_k * dr1 * dr2
           + system.angleParams.angle_class2.ba_k1 * dr1 * dtheta
           + system.angleParams.angle_class2.ba_k2 * dr2 * dtheta;
}

real improperClass2AnalyticEnergy(const PcffSystemDefinition& system)
{
    const rvec* coordinates = as_rvec_array(system.coordinates.data());
    rvec        delr[3];
    rvec_sub(coordinates[0], coordinates[1], delr[0]);
    rvec_sub(coordinates[2], coordinates[1], delr[1]);
    rvec_sub(coordinates[3], coordinates[1], delr[2]);

    real rmag[3];
    for (int i = 0; i < 3; ++i)
    {
        rmag[i] = norm(delr[i]);
    }

    real costheta[3];
    costheta[0] = std::clamp(iprod(delr[0], delr[1]) / (rmag[0] * rmag[1]), -1.0_real, 1.0_real);
    costheta[1] = std::clamp(iprod(delr[1], delr[2]) / (rmag[1] * rmag[2]), -1.0_real, 1.0_real);
    costheta[2] = std::clamp(iprod(delr[0], delr[2]) / (rmag[0] * rmag[2]), -1.0_real, 1.0_real);

    real theta[3];
    real invstheta[3];
    for (int i = 0; i < 3; ++i)
    {
        theta[i] = std::acos(costheta[i]);
        invstheta[i] = 1.0_real / std::sin(theta[i]);
    }

    rvec rABxrCB, rDBxrAB, rCBxrDB;
    cprod(delr[0], delr[1], rABxrCB);
    cprod(delr[2], delr[0], rDBxrAB);
    cprod(delr[1], delr[2], rCBxrDB);

    const real inv3r = 1.0_real / (rmag[0] * rmag[1] * rmag[2]);
    const real chiABCD =
            std::asin(std::clamp(iprod(rCBxrDB, delr[0]) * invstheta[1] * inv3r, -1.0_real, 1.0_real));
    const real chiCBDA =
            std::asin(std::clamp(iprod(rDBxrAB, delr[1]) * invstheta[2] * inv3r, -1.0_real, 1.0_real));
    const real chiDBAC =
            std::asin(std::clamp(iprod(rABxrCB, delr[2]) * invstheta[0] * inv3r, -1.0_real, 1.0_real));
    const real chi = (chiABCD + chiCBDA + chiDBAC) / 3.0_real;

    const real dchi   = chi - system.improperParams.improper_class2.chi0;
    const real dthABC = theta[0] - system.improperParams.improper_class2.aa_theta0_1;
    const real dthCBD = theta[1] - system.improperParams.improper_class2.aa_theta0_3;
    const real dthABD = theta[2] - system.improperParams.improper_class2.aa_theta0_2;

    return system.improperParams.improper_class2.k0 * dchi * dchi
           + system.improperParams.improper_class2.aa_k2 * dthABC * dthABD
           + system.improperParams.improper_class2.aa_k1 * dthABC * dthCBD
           + system.improperParams.improper_class2.aa_k3 * dthABD * dthCBD;
}

void expectForcesMatchGolden(const OutputQuantities& output,
                             const ReferenceSummary& reference,
                             const int               numAtoms,
                             const real              tolerance)
{
    ASSERT_EQ(reference.forces.size(), static_cast<std::size_t>(numAtoms));
    for (int atom = 0; atom < numAtoms; ++atom)
    {
        for (int d = 0; d < DIM; ++d)
        {
            EXPECT_NEAR(forceToLammpsUnits(output.f[atom][d]), reference.forces[atom][d], tolerance)
                    << "Force mismatch for atom " << atom + 1 << " component " << d;
        }
    }
}

void expectFiniteDifferenceMatches(const PcffSystemDefinition& system,
                                   const int                  atomIndex,
                                   const int                  dimension,
                                   const real                 deltaNm,
                                   const real                 tolerance)
{
    PcffSystemDefinition plus = system;
    PcffSystemDefinition minus = system;
    plus.coordinates[atomIndex][dimension] += deltaNm;
    minus.coordinates[atomIndex][dimension] -= deltaNm;

    const real numericalForce =
            -(evaluateSystem(plus).energy - evaluateSystem(minus).energy) / (2 * deltaNm);
    const real analyticForce = evaluateSystem(system).f[atomIndex][dimension];
    EXPECT_NEAR(analyticForce, numericalForce, tolerance);
}

TEST(PcffClass2FormulaTest, BondEnergyMatchesAnalyticExpression)
{
    const auto system = makeBondToy();
    const auto output = evaluateSingleType(InteractionFunction::BondClass2,
                                           system.bondIatoms,
                                           system.bondParams,
                                           system.coordinates,
                                           system.numAtoms);

    EXPECT_NEAR(output.energy, bondClass2AnalyticEnergy(system), 1e-6);
}

TEST(PcffClass2FormulaTest, AngleEnergyMatchesAnalyticExpression)
{
    const auto system = makeAngleToy();
    const auto output = evaluateSingleType(InteractionFunction::AngleClass2,
                                           system.angleIatoms,
                                           system.angleParams,
                                           system.coordinates,
                                           system.numAtoms);

    EXPECT_NEAR(output.energy, angleClass2AnalyticEnergy(system), 1e-5);
}

TEST(PcffClass2FormulaTest, ImproperEnergyMatchesAnalyticExpression)
{
    const auto system = makeImproperToy();
    const auto output = evaluateSingleType(InteractionFunction::ImproperClass2,
                                           system.improperIatoms,
                                           system.improperParams,
                                           system.coordinates,
                                           system.numAtoms);

    EXPECT_NEAR(output.energy, improperClass2AnalyticEnergy(system), 1e-4);
}

TEST(PcffClass2GoldenTest, BondToyMatchesGoldenSummary)
{
    const auto system     = makeBondToy();
    const auto reference  = loadReferenceSummary(system.systemId);
    const auto bondOutput = evaluateSingleType(InteractionFunction::BondClass2,
                                               system.bondIatoms,
                                               system.bondParams,
                                               system.coordinates,
                                               system.numAtoms);

    EXPECT_NEAR(kjToKcal(bondOutput.energy), reference.energies.at("pe"), 2e-5);
    EXPECT_NEAR(kjToKcal(bondOutput.energy), reference.energies.at("ebond"), 2e-5);
    expectForcesMatchGolden(bondOutput, reference, system.numAtoms, 5e-3);
}

TEST(PcffClass2GoldenTest, AngleToyMatchesGoldenSummary)
{
    const auto system      = makeAngleToy();
    const auto reference   = loadReferenceSummary(system.systemId);
    const auto bondOutput  = evaluateSingleType(InteractionFunction::BondClass2,
                                               system.bondIatoms,
                                               system.bondParams,
                                               system.coordinates,
                                               system.numAtoms);
    const auto angleOutput = evaluateSingleType(InteractionFunction::AngleClass2,
                                                system.angleIatoms,
                                                system.angleParams,
                                                system.coordinates,
                                                system.numAtoms);
    const auto totalOutput = evaluateSystem(system);

    EXPECT_NEAR(kjToKcal(totalOutput.energy), reference.energies.at("pe"), 5e-5);
    EXPECT_NEAR(kjToKcal(bondOutput.energy), reference.energies.at("ebond"), 5e-5);
    EXPECT_NEAR(kjToKcal(angleOutput.energy), reference.energies.at("eangle"), 5e-5);
    expectForcesMatchGolden(totalOutput, reference, system.numAtoms, 2e-2);
}

TEST(PcffClass2GoldenTest, ImproperToyMatchesGoldenSummary)
{
    const auto system          = makeImproperToy();
    const auto reference       = loadReferenceSummary(system.systemId);
    const auto bondOutput      = evaluateSingleType(InteractionFunction::BondClass2,
                                               system.bondIatoms,
                                               system.bondParams,
                                               system.coordinates,
                                               system.numAtoms);
    const auto angleOutput     = evaluateSingleType(InteractionFunction::AngleClass2,
                                                system.angleIatoms,
                                                system.angleParams,
                                                system.coordinates,
                                                system.numAtoms);
    const auto improperOutput  = evaluateSingleType(InteractionFunction::ImproperClass2,
                                                   system.improperIatoms,
                                                   system.improperParams,
                                                   system.coordinates,
                                                   system.numAtoms);
    const auto totalOutput     = evaluateSystem(system);

    EXPECT_NEAR(kjToKcal(totalOutput.energy), reference.energies.at("pe"), 5e-4);
    EXPECT_NEAR(kjToKcal(bondOutput.energy), reference.energies.at("ebond"), 5e-4);
    EXPECT_NEAR(kjToKcal(angleOutput.energy), reference.energies.at("eangle"), 5e-4);
    EXPECT_NEAR(kjToKcal(improperOutput.energy), reference.energies.at("eimp"), 5e-4);
    expectForcesMatchGolden(totalOutput, reference, system.numAtoms, 6e-2);
}

TEST(PcffClass2ForceValidationTest, BondToyFiniteDifferenceMatchesAnalyticForce)
{
    expectFiniteDifferenceMatches(makeBondToy(), 0, XX, 1e-4, 2e-1);
    expectFiniteDifferenceMatches(makeBondToy(), 1, XX, 1e-4, 2e-1);
}

TEST(PcffClass2ForceValidationTest, AngleToyFiniteDifferenceMatchesAnalyticForce)
{
    expectFiniteDifferenceMatches(makeAngleToy(), 1, XX, 1e-4, 5e-1);
    expectFiniteDifferenceMatches(makeAngleToy(), 1, YY, 1e-4, 5e-1);
    expectFiniteDifferenceMatches(makeAngleToy(), 2, XX, 1e-4, 5e-1);
    expectFiniteDifferenceMatches(makeAngleToy(), 2, YY, 1e-4, 5e-1);
}

TEST(PcffClass2ForceValidationTest, ImproperToyFiniteDifferenceMatchesAnalyticForce)
{
    expectFiniteDifferenceMatches(makeImproperToy(), 3, ZZ, 1e-4, 5.0);
}

} // namespace
} // namespace test
} // namespace gmx
