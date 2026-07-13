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
 * Tests for PCFF/Class2 dihedral kernels against frozen M1/M3 reference data.
 */

#include "gmxpre.h"

#include "gromacs/listed_forces/bonded.h"

#include <algorithm>
#include <array>
#include <cmath>
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

constexpr int  c_maxAtoms                       = 4;
constexpr real c_small                         = 1.0e-7_real;
constexpr real c_angstromToNm                  = 0.1_real;
constexpr real c_kcalToKj                      = 4.184_real;
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

struct DihedralGeometry
{
    real r1 = 0;
    real r2 = 0;
    real r3 = 0;
    real theta12 = 0;
    real theta23 = 0;
    real phi = 0;
    real cosphi = 0;
};

struct PcffDihedralSystemDefinition
{
    std::string          systemId;
    int                  numAtoms = 0;
    PaddedVector<RVec>   coordinates;
    std::vector<t_iatom> bondIatoms;
    std::vector<t_iatom> angleIatoms;
    std::vector<t_iatom> dihedralIatoms;
    t_iparams            bondParams = { { 0 } };
    t_iparams            angleParams = { { 0 } };
    t_iparams            dihedralParams = { { 0 } };
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

real forceToLammpsUnits(const real value)
{
    return value * c_forceKjPerNmToKcalPerAngstrom;
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

real dihedralBondTorsionKToGromacs(const real value)
{
    return kcalToKj(value) / c_angstromToNm;
}

std::filesystem::path referenceResultsRoot()
{
    std::filesystem::path root = TestFileManager::getInputDataDirectory();
    for (int i = 0; i < 4; ++i)
    {
        root = root.parent_path();
    }
    return root / "tests" / "reference_results" / "m3";
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

OutputQuantities evaluateSingleType(InteractionFunction        ftype,
                                    const std::vector<t_iatom>& iatoms,
                                    const t_iparams&            iparams,
                                    const PaddedVector<RVec>&   coordinates,
                                    const int                   numAtoms)
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

OutputQuantities evaluateSystem(const PcffDihedralSystemDefinition& system)
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
    if (!system.dihedralIatoms.empty())
    {
        accumulateOutput(
                evaluateSingleType(InteractionFunction::DihedralClass2,
                                   system.dihedralIatoms,
                                   system.dihedralParams,
                                   system.coordinates,
                                   system.numAtoms),
                &total);
    }
    return total;
}

t_iparams zeroDihedralClass2Parameters()
{
    t_iparams params = { { 0 } };
    return params;
}

PcffDihedralSystemDefinition makeDihedralToy()
{
    PcffDihedralSystemDefinition system;
    system.systemId       = "dihedral_toy";
    system.numAtoms       = 4;
    system.coordinates    = { { 0.0, 0.0, 0.0 }, { 0.15, 0.0, 0.0 }, { 0.285, 0.1, 0.0 }, { 0.41, 0.12, 0.11 } };
    system.bondIatoms     = { 0, 0, 1, 0, 1, 2, 0, 2, 3 };
    system.angleIatoms    = { 0, 0, 1, 2, 0, 1, 2, 3 };
    system.dihedralIatoms = { 0, 0, 1, 2, 3 };

    system.bondParams.bond_class2.r0 = angstromToNm(1.50_real);
    system.bondParams.bond_class2.k2 = bondK2ToGromacs(220.0_real);
    system.bondParams.bond_class2.k3 = bondK3ToGromacs(-30.0_real);
    system.bondParams.bond_class2.k4 = bondK4ToGromacs(6.0_real);

    system.angleParams.angle_class2.theta0 = 112.0_real * gmx::c_deg2Rad;
    system.angleParams.angle_class2.k2     = kcalToKj(30.0_real);
    system.angleParams.angle_class2.k3     = kcalToKj(-3.5_real);
    system.angleParams.angle_class2.k4     = kcalToKj(1.0_real);
    system.angleParams.angle_class2.bb_k   = bondBondKToGromacs(4.0_real);
    system.angleParams.angle_class2.bb_r1  = angstromToNm(1.50_real);
    system.angleParams.angle_class2.bb_r2  = angstromToNm(1.50_real);
    system.angleParams.angle_class2.ba_k1  = bondAngleKToGromacs(1.5_real);
    system.angleParams.angle_class2.ba_k2  = bondAngleKToGromacs(1.1_real);
    system.angleParams.angle_class2.ba_r1  = angstromToNm(1.50_real);
    system.angleParams.angle_class2.ba_r2  = angstromToNm(1.50_real);

    system.dihedralParams = zeroDihedralClass2Parameters();
    auto& dih             = system.dihedralParams.dihedral_class2;
    dih.k1                = kcalToKj(0.8_real);
    dih.phi1              = 0.0_real;
    dih.k2                = kcalToKj(0.6_real);
    dih.phi2              = 180.0_real * gmx::c_deg2Rad;
    dih.k3                = kcalToKj(0.4_real);
    dih.phi3              = 0.0_real;
    dih.mbt_f1            = dihedralBondTorsionKToGromacs(0.12_real);
    dih.mbt_f2            = dihedralBondTorsionKToGromacs(-0.08_real);
    dih.mbt_f3            = dihedralBondTorsionKToGromacs(0.04_real);
    dih.mbt_r0            = angstromToNm(1.50_real);
    dih.ebt_f1_1          = dihedralBondTorsionKToGromacs(0.10_real);
    dih.ebt_f2_1          = dihedralBondTorsionKToGromacs(-0.05_real);
    dih.ebt_f3_1          = dihedralBondTorsionKToGromacs(0.02_real);
    dih.ebt_f1_2          = dihedralBondTorsionKToGromacs(0.11_real);
    dih.ebt_f2_2          = dihedralBondTorsionKToGromacs(-0.03_real);
    dih.ebt_f3_2          = dihedralBondTorsionKToGromacs(0.01_real);
    dih.ebt_r0_1          = angstromToNm(1.50_real);
    dih.ebt_r0_2          = angstromToNm(1.50_real);
    dih.at_f1_1           = kcalToKj(0.06_real);
    dih.at_f2_1           = kcalToKj(-0.03_real);
    dih.at_f3_1           = kcalToKj(0.015_real);
    dih.at_f1_2           = kcalToKj(0.05_real);
    dih.at_f2_2           = kcalToKj(-0.025_real);
    dih.at_f3_2           = kcalToKj(0.01_real);
    dih.at_theta0_1       = 112.0_real * gmx::c_deg2Rad;
    dih.at_theta0_2       = 112.0_real * gmx::c_deg2Rad;
    dih.aat_k             = kcalToKj(0.20_real);
    dih.aat_theta0_1      = 112.0_real * gmx::c_deg2Rad;
    dih.aat_theta0_2      = 112.0_real * gmx::c_deg2Rad;
    dih.bb13t_k           = bondBondKToGromacs(0.18_real);
    dih.bb13t_r10         = angstromToNm(1.50_real);
    dih.bb13t_r30         = angstromToNm(1.50_real);

    return system;
}

PcffDihedralSystemDefinition makeDihedralOnlyToy()
{
    auto system        = makeDihedralToy();
    system.bondIatoms  = {};
    system.angleIatoms = {};
    return system;
}

DihedralGeometry computeDihedralGeometry(const PcffDihedralSystemDefinition& system)
{
    const rvec* coordinates = as_rvec_array(system.coordinates.data());
    rvec        vb1, vb2, vb3;
    rvec_sub(coordinates[0], coordinates[1], vb1);
    rvec_sub(coordinates[2], coordinates[1], vb2);
    rvec_sub(coordinates[3], coordinates[2], vb3);

    const real r1     = norm(vb1);
    const real r2     = norm(vb2);
    const real r3     = norm(vb3);
    const real rb1    = 1.0_real / r1;
    const real rb2    = 1.0_real / r2;
    const real rb3    = 1.0_real / r3;
    const real costh12 = std::clamp(iprod(vb1, vb2) * rb1 * rb2, -1.0_real, 1.0_real);
    const real costh23 = std::clamp((-vb2[XX] * vb3[XX] - vb2[YY] * vb3[YY] - vb2[ZZ] * vb3[ZZ]) * rb2 * rb3,
                                    -1.0_real,
                                    1.0_real);
    const real costh13 = std::clamp(iprod(vb1, vb3) * rb1 * rb3, -1.0_real, 1.0_real);

    real sc1 = std::sqrt(std::max(1.0_real - costh12 * costh12, 0.0_real));
    if (sc1 < c_small)
    {
        sc1 = c_small;
    }
    sc1 = 1.0_real / sc1;

    real sc2 = std::sqrt(std::max(1.0_real - costh23 * costh23, 0.0_real));
    if (sc2 < c_small)
    {
        sc2 = c_small;
    }
    sc2 = 1.0_real / sc2;

    const real c = std::clamp((costh13 + costh12 * costh23) * sc1 * sc2, -1.0_real, 1.0_real);
    real       phi = std::acos(c);

    rvec n123;
    cprod(vb1, vb2, n123);
    if (iprod(n123, vb3) > 0.0_real)
    {
        phi = -phi;
    }

    return DihedralGeometry{ r1, r2, r3, std::acos(costh12), std::acos(costh23), phi, c };
}

real primaryDihedralEnergy(const t_iparams& params, const DihedralGeometry& geometry)
{
    return params.dihedral_class2.k1 * (1.0_real - std::cos(geometry.phi - params.dihedral_class2.phi1))
           + params.dihedral_class2.k2 * (1.0_real - std::cos(2.0_real * geometry.phi - params.dihedral_class2.phi2))
           + params.dihedral_class2.k3 * (1.0_real - std::cos(3.0_real * geometry.phi - params.dihedral_class2.phi3));
}

real middleBondTorsionEnergy(const t_iparams& params, const DihedralGeometry& geometry)
{
    return (geometry.r2 - params.dihedral_class2.mbt_r0)
           * (params.dihedral_class2.mbt_f1 * std::cos(geometry.phi)
              + params.dihedral_class2.mbt_f2 * std::cos(2.0_real * geometry.phi)
              + params.dihedral_class2.mbt_f3 * std::cos(3.0_real * geometry.phi));
}

real endBondTorsionEnergy(const t_iparams& params, const DihedralGeometry& geometry)
{
    return (geometry.r1 - params.dihedral_class2.ebt_r0_1)
                   * (params.dihedral_class2.ebt_f1_1 * std::cos(geometry.phi)
                      + params.dihedral_class2.ebt_f2_1 * std::cos(2.0_real * geometry.phi)
                      + params.dihedral_class2.ebt_f3_1 * std::cos(3.0_real * geometry.phi))
           + (geometry.r3 - params.dihedral_class2.ebt_r0_2)
                     * (params.dihedral_class2.ebt_f1_2 * std::cos(geometry.phi)
                        + params.dihedral_class2.ebt_f2_2 * std::cos(2.0_real * geometry.phi)
                        + params.dihedral_class2.ebt_f3_2 * std::cos(3.0_real * geometry.phi));
}

real angleTorsionEnergy(const t_iparams& params, const DihedralGeometry& geometry)
{
    return (geometry.theta12 - params.dihedral_class2.at_theta0_1)
                   * (params.dihedral_class2.at_f1_1 * std::cos(geometry.phi)
                      + params.dihedral_class2.at_f2_1 * std::cos(2.0_real * geometry.phi)
                      + params.dihedral_class2.at_f3_1 * std::cos(3.0_real * geometry.phi))
           + (geometry.theta23 - params.dihedral_class2.at_theta0_2)
                     * (params.dihedral_class2.at_f1_2 * std::cos(geometry.phi)
                        + params.dihedral_class2.at_f2_2 * std::cos(2.0_real * geometry.phi)
                        + params.dihedral_class2.at_f3_2 * std::cos(3.0_real * geometry.phi));
}

real angleAngleTorsionEnergy(const t_iparams& params, const DihedralGeometry& geometry)
{
    return params.dihedral_class2.aat_k * (geometry.theta12 - params.dihedral_class2.aat_theta0_1)
           * (geometry.theta23 - params.dihedral_class2.aat_theta0_2) * std::cos(geometry.phi);
}

real bondBond13Energy(const t_iparams& params, const DihedralGeometry& geometry)
{
    return params.dihedral_class2.bb13t_k * (geometry.r1 - params.dihedral_class2.bb13t_r10)
           * (geometry.r3 - params.dihedral_class2.bb13t_r30);
}

real totalDihedralEnergyAnalytic(const t_iparams& params, const DihedralGeometry& geometry)
{
    return primaryDihedralEnergy(params, geometry) + middleBondTorsionEnergy(params, geometry)
           + endBondTorsionEnergy(params, geometry) + angleTorsionEnergy(params, geometry)
           + angleAngleTorsionEnergy(params, geometry) + bondBond13Energy(params, geometry);
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

void expectFiniteDifferenceMatches(const PcffDihedralSystemDefinition& system,
                                   const int                          atomIndex,
                                   const int                          dimension,
                                   const real                         deltaNm,
                                   const real                         tolerance)
{
    PcffDihedralSystemDefinition plus = system;
    PcffDihedralSystemDefinition minus = system;
    plus.coordinates[atomIndex][dimension] += deltaNm;
    minus.coordinates[atomIndex][dimension] -= deltaNm;

    const real numericalForce =
            -(evaluateSystem(plus).energy - evaluateSystem(minus).energy) / (2 * deltaNm);
    const real analyticForce = evaluateSystem(system).f[atomIndex][dimension];
    EXPECT_NEAR(analyticForce, numericalForce, tolerance);
}

TEST(PcffClass2DihedralFormulaTest, PrimaryContributionMatchesAnalyticExpression)
{
    auto system       = makeDihedralOnlyToy();
    auto params       = zeroDihedralClass2Parameters();
    params.dihedral_class2.k1   = system.dihedralParams.dihedral_class2.k1;
    params.dihedral_class2.phi1 = system.dihedralParams.dihedral_class2.phi1;
    params.dihedral_class2.k2   = system.dihedralParams.dihedral_class2.k2;
    params.dihedral_class2.phi2 = system.dihedralParams.dihedral_class2.phi2;
    params.dihedral_class2.k3   = system.dihedralParams.dihedral_class2.k3;
    params.dihedral_class2.phi3 = system.dihedralParams.dihedral_class2.phi3;
    system.dihedralParams       = params;

    const auto geometry = computeDihedralGeometry(system);
    const auto output   = evaluateSingleType(InteractionFunction::DihedralClass2,
                                           system.dihedralIatoms,
                                           system.dihedralParams,
                                           system.coordinates,
                                           system.numAtoms);
    EXPECT_NEAR(output.energy, primaryDihedralEnergy(system.dihedralParams, geometry), 1e-5);
}

TEST(PcffClass2DihedralFormulaTest, MiddleBondTorsionContributionMatchesAnalyticExpression)
{
    auto system       = makeDihedralOnlyToy();
    auto params       = zeroDihedralClass2Parameters();
    params.dihedral_class2.mbt_f1 = system.dihedralParams.dihedral_class2.mbt_f1;
    params.dihedral_class2.mbt_f2 = system.dihedralParams.dihedral_class2.mbt_f2;
    params.dihedral_class2.mbt_f3 = system.dihedralParams.dihedral_class2.mbt_f3;
    params.dihedral_class2.mbt_r0 = system.dihedralParams.dihedral_class2.mbt_r0;
    system.dihedralParams         = params;

    const auto geometry = computeDihedralGeometry(system);
    const auto output   = evaluateSingleType(InteractionFunction::DihedralClass2,
                                           system.dihedralIatoms,
                                           system.dihedralParams,
                                           system.coordinates,
                                           system.numAtoms);
    EXPECT_NEAR(output.energy, middleBondTorsionEnergy(system.dihedralParams, geometry), 1e-5);
}

TEST(PcffClass2DihedralFormulaTest, EndBondTorsionContributionMatchesAnalyticExpression)
{
    auto system             = makeDihedralOnlyToy();
    auto params             = zeroDihedralClass2Parameters();
    params.dihedral_class2.ebt_f1_1 = system.dihedralParams.dihedral_class2.ebt_f1_1;
    params.dihedral_class2.ebt_f2_1 = system.dihedralParams.dihedral_class2.ebt_f2_1;
    params.dihedral_class2.ebt_f3_1 = system.dihedralParams.dihedral_class2.ebt_f3_1;
    params.dihedral_class2.ebt_f1_2 = system.dihedralParams.dihedral_class2.ebt_f1_2;
    params.dihedral_class2.ebt_f2_2 = system.dihedralParams.dihedral_class2.ebt_f2_2;
    params.dihedral_class2.ebt_f3_2 = system.dihedralParams.dihedral_class2.ebt_f3_2;
    params.dihedral_class2.ebt_r0_1 = system.dihedralParams.dihedral_class2.ebt_r0_1;
    params.dihedral_class2.ebt_r0_2 = system.dihedralParams.dihedral_class2.ebt_r0_2;
    system.dihedralParams           = params;

    const auto geometry = computeDihedralGeometry(system);
    const auto output   = evaluateSingleType(InteractionFunction::DihedralClass2,
                                           system.dihedralIatoms,
                                           system.dihedralParams,
                                           system.coordinates,
                                           system.numAtoms);
    EXPECT_NEAR(output.energy, endBondTorsionEnergy(system.dihedralParams, geometry), 1e-5);
}

TEST(PcffClass2DihedralFormulaTest, AngleTorsionContributionMatchesAnalyticExpression)
{
    auto system            = makeDihedralOnlyToy();
    auto params            = zeroDihedralClass2Parameters();
    params.dihedral_class2.at_f1_1     = system.dihedralParams.dihedral_class2.at_f1_1;
    params.dihedral_class2.at_f2_1     = system.dihedralParams.dihedral_class2.at_f2_1;
    params.dihedral_class2.at_f3_1     = system.dihedralParams.dihedral_class2.at_f3_1;
    params.dihedral_class2.at_f1_2     = system.dihedralParams.dihedral_class2.at_f1_2;
    params.dihedral_class2.at_f2_2     = system.dihedralParams.dihedral_class2.at_f2_2;
    params.dihedral_class2.at_f3_2     = system.dihedralParams.dihedral_class2.at_f3_2;
    params.dihedral_class2.at_theta0_1 = system.dihedralParams.dihedral_class2.at_theta0_1;
    params.dihedral_class2.at_theta0_2 = system.dihedralParams.dihedral_class2.at_theta0_2;
    system.dihedralParams              = params;

    const auto geometry = computeDihedralGeometry(system);
    const auto output   = evaluateSingleType(InteractionFunction::DihedralClass2,
                                           system.dihedralIatoms,
                                           system.dihedralParams,
                                           system.coordinates,
                                           system.numAtoms);
    EXPECT_NEAR(output.energy, angleTorsionEnergy(system.dihedralParams, geometry), 1e-5);
}

TEST(PcffClass2DihedralFormulaTest, AngleAngleTorsionContributionMatchesAnalyticExpression)
{
    auto system            = makeDihedralOnlyToy();
    auto params            = zeroDihedralClass2Parameters();
    params.dihedral_class2.aat_k        = system.dihedralParams.dihedral_class2.aat_k;
    params.dihedral_class2.aat_theta0_1 = system.dihedralParams.dihedral_class2.aat_theta0_1;
    params.dihedral_class2.aat_theta0_2 = system.dihedralParams.dihedral_class2.aat_theta0_2;
    system.dihedralParams               = params;

    const auto geometry = computeDihedralGeometry(system);
    const auto output   = evaluateSingleType(InteractionFunction::DihedralClass2,
                                           system.dihedralIatoms,
                                           system.dihedralParams,
                                           system.coordinates,
                                           system.numAtoms);
    EXPECT_NEAR(output.energy, angleAngleTorsionEnergy(system.dihedralParams, geometry), 1e-5);
}

TEST(PcffClass2DihedralFormulaTest, BondBond13ContributionMatchesAnalyticExpression)
{
    auto system          = makeDihedralOnlyToy();
    auto params          = zeroDihedralClass2Parameters();
    params.dihedral_class2.bb13t_k   = system.dihedralParams.dihedral_class2.bb13t_k;
    params.dihedral_class2.bb13t_r10 = system.dihedralParams.dihedral_class2.bb13t_r10;
    params.dihedral_class2.bb13t_r30 = system.dihedralParams.dihedral_class2.bb13t_r30;
    system.dihedralParams            = params;

    const auto geometry = computeDihedralGeometry(system);
    const auto output   = evaluateSingleType(InteractionFunction::DihedralClass2,
                                           system.dihedralIatoms,
                                           system.dihedralParams,
                                           system.coordinates,
                                           system.numAtoms);
    EXPECT_NEAR(output.energy, bondBond13Energy(system.dihedralParams, geometry), 1e-5);
}

TEST(PcffClass2DihedralFormulaTest, TotalContributionMatchesAnalyticSum)
{
    const auto system   = makeDihedralOnlyToy();
    const auto geometry = computeDihedralGeometry(system);
    const auto output   = evaluateSingleType(InteractionFunction::DihedralClass2,
                                           system.dihedralIatoms,
                                           system.dihedralParams,
                                           system.coordinates,
                                           system.numAtoms);
    EXPECT_NEAR(output.energy, totalDihedralEnergyAnalytic(system.dihedralParams, geometry), 2e-5);
}

TEST(PcffClass2DihedralGoldenTest, DihedralToyMatchesGoldenSummary)
{
    const auto system         = makeDihedralToy();
    const auto reference      = loadReferenceSummary(system.systemId);
    const auto bondOutput     = evaluateSingleType(InteractionFunction::BondClass2,
                                               system.bondIatoms,
                                               system.bondParams,
                                               system.coordinates,
                                               system.numAtoms);
    const auto angleOutput    = evaluateSingleType(InteractionFunction::AngleClass2,
                                                system.angleIatoms,
                                                system.angleParams,
                                                system.coordinates,
                                                system.numAtoms);
    const auto dihedralOutput = evaluateSingleType(InteractionFunction::DihedralClass2,
                                                   system.dihedralIatoms,
                                                   system.dihedralParams,
                                                   system.coordinates,
                                                   system.numAtoms);
    const auto totalOutput    = evaluateSystem(system);

    EXPECT_NEAR(kjToKcal(totalOutput.energy), reference.energies.at("pe"), 5e-4);
    EXPECT_NEAR(kjToKcal(bondOutput.energy), reference.energies.at("ebond"), 5e-4);
    EXPECT_NEAR(kjToKcal(angleOutput.energy), reference.energies.at("eangle"), 5e-4);
    EXPECT_NEAR(kjToKcal(dihedralOutput.energy), reference.energies.at("edihed"), 5e-4);
    expectForcesMatchGolden(totalOutput, reference, system.numAtoms, 5e-2);
}

TEST(PcffClass2DihedralForceValidationTest, DihedralToyFiniteDifferenceMatchesAnalyticForce)
{
    const auto system = makeDihedralToy();
    expectFiniteDifferenceMatches(system, 1, XX, 1e-5, 3.0);
    expectFiniteDifferenceMatches(system, 1, ZZ, 1e-5, 3.0);
    expectFiniteDifferenceMatches(system, 3, XX, 1e-5, 3.0);
    expectFiniteDifferenceMatches(system, 3, ZZ, 1e-5, 3.0);
}

TEST(PcffClass2DihedralStressTest, NearLinearGeometryRemainsFinite)
{
    auto system         = makeDihedralOnlyToy();
    system.coordinates  = { { 0.0, 0.0, 0.0 }, { 0.15, 0.0, 0.0 }, { 0.30, 1e-7, 0.0 }, { 0.45, 2e-7, 1e-7 } };
    const auto output   = evaluateSingleType(InteractionFunction::DihedralClass2,
                                           system.dihedralIatoms,
                                           system.dihedralParams,
                                           system.coordinates,
                                           system.numAtoms);

    EXPECT_TRUE(std::isfinite(output.energy));
    for (int atom = 0; atom < system.numAtoms; ++atom)
    {
        for (int d = 0; d < DIM; ++d)
        {
            EXPECT_TRUE(std::isfinite(output.f[atom][d]));
        }
    }
}

TEST(PcffClass2DihedralStressTest, MirroredGeometryFlipsPhiSignButPreservesEnergy)
{
    auto system              = makeDihedralOnlyToy();
    auto mirrored            = system;
    mirrored.coordinates[3][ZZ] *= -1;

    const auto geometryOriginal = computeDihedralGeometry(system);
    const auto geometryMirrored = computeDihedralGeometry(mirrored);
    const auto outputOriginal   = evaluateSingleType(InteractionFunction::DihedralClass2,
                                                   system.dihedralIatoms,
                                                   system.dihedralParams,
                                                   system.coordinates,
                                                   system.numAtoms);
    const auto outputMirrored   = evaluateSingleType(InteractionFunction::DihedralClass2,
                                                   mirrored.dihedralIatoms,
                                                   mirrored.dihedralParams,
                                                   mirrored.coordinates,
                                                   mirrored.numAtoms);

    EXPECT_NEAR(geometryOriginal.phi, -geometryMirrored.phi, 1e-5);
    EXPECT_NEAR(outputOriginal.energy, outputMirrored.energy, 1e-5);
}

TEST(PcffClass2DihedralStressTest, PhasePeriodicityAtPlusMinus180DegreesMatches)
{
    auto plusPiSystem        = makeDihedralOnlyToy();
    auto minusPiSystem       = plusPiSystem;
    plusPiSystem.dihedralParams = zeroDihedralClass2Parameters();
    minusPiSystem.dihedralParams = zeroDihedralClass2Parameters();
    plusPiSystem.dihedralParams.dihedral_class2.k2   = kcalToKj(1.0_real);
    minusPiSystem.dihedralParams.dihedral_class2.k2  = kcalToKj(1.0_real);
    plusPiSystem.dihedralParams.dihedral_class2.phi2 = 180.0_real * gmx::c_deg2Rad;
    minusPiSystem.dihedralParams.dihedral_class2.phi2 = -180.0_real * gmx::c_deg2Rad;

    const auto plusOutput  = evaluateSingleType(InteractionFunction::DihedralClass2,
                                              plusPiSystem.dihedralIatoms,
                                              plusPiSystem.dihedralParams,
                                              plusPiSystem.coordinates,
                                              plusPiSystem.numAtoms);
    const auto minusOutput = evaluateSingleType(InteractionFunction::DihedralClass2,
                                               minusPiSystem.dihedralIatoms,
                                               minusPiSystem.dihedralParams,
                                               minusPiSystem.coordinates,
                                               minusPiSystem.numAtoms);

    EXPECT_NEAR(plusOutput.energy, minusOutput.energy, 1e-6);
    for (int atom = 0; atom < plusPiSystem.numAtoms; ++atom)
    {
        for (int d = 0; d < DIM; ++d)
        {
            EXPECT_NEAR(plusOutput.f[atom][d], minusOutput.f[atom][d], GMX_DOUBLE ? 1e-10 : 2e-5);
        }
    }
}

} // namespace
} // namespace test
} // namespace gmx
