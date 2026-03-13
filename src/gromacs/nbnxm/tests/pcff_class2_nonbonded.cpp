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

#include "config.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/gpu_utils/hostallocator.h"
#include "gromacs/math/functions.h"
#include "gromacs/mdlib/forcerec.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdtypes/atominfo.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/locality.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/nbnxm_simd.h"
#include "gromacs/nbnxm/pairlistparams.h"
#include "gromacs/nbnxm/pairlistset.h"
#include "gromacs/nbnxm/pairlistsets.h"
#include "gromacs/nbnxm/pairsearch.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/listoflists.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/vec.h"

namespace gmx
{

namespace test
{

namespace
{

struct PcffNonbondedSystem
{
    int                  numAtomTypes = 1;
    std::vector<real>    nonbondedParameters;
    std::vector<int>     atomTypes;
    std::vector<real>    charges;
    std::vector<int32_t> atomInfo;
    ListOfLists<int>     exclusions;
    std::vector<RVec>    coordinates;
    matrix               box = { { { 0 } } };
};

struct PcffNonbondedOutput
{
    std::vector<RVec> forces;
    real              vdwEnergy     = 0;
    real              coulombEnergy = 0;
};

CoulombInteractionType coulombInteractionType(CoulombKernelType coulombKernelType)
{
    switch (coulombKernelType)
    {
        case CoulombKernelType::Table:
        case CoulombKernelType::TableTwin:
        case CoulombKernelType::Ewald:
        case CoulombKernelType::EwaldTwin: return CoulombInteractionType::Pme;
        case CoulombKernelType::ReactionField: return CoulombInteractionType::RF;
        default: GMX_RELEASE_ASSERT(false, "Unsupported CoulombKernelType for PCFF test");
    }
    return CoulombInteractionType::Count;
}

interaction_const_t makeInteractionConst(CoulombKernelType coulombKernelType, const double repulsionPower, const real cutoff)
{
    t_inputrec ir;

    ir.vdwtype      = VanDerWaalsType::Cut;
    ir.vdw_modifier = InteractionModifiers::None;
    ir.rvdw         = cutoff;

    ir.coulombtype      = coulombInteractionType(coulombKernelType);
    ir.coulomb_modifier = InteractionModifiers::None;
    ir.rcoulomb         = cutoff;
    ir.ewald_rtol       = 1e-6;
    ir.epsilon_r        = 1;
    ir.epsilon_rf       = 0;

    gmx_mtop_t mtop;
    mtop.ffparams.reppow = repulsionPower;
    mtop.ffparams.functype.resize(1);
    mtop.ffparams.functype[0] = InteractionFunction::LennardJonesShortRange;

    interaction_const_t ic = init_interaction_const(nullptr, ir, mtop, false, std::nullopt);
    init_interaction_const_tables(nullptr, &ic, cutoff, 0);
    return ic;
}

interaction_const_t makeCutoffInteractionConst(const double repulsionPower, const real cutoff)
{
    t_inputrec ir;

    ir.vdwtype      = VanDerWaalsType::Cut;
    ir.vdw_modifier = InteractionModifiers::None;
    ir.rvdw         = cutoff;

    ir.coulombtype      = CoulombInteractionType::Cut;
    ir.coulomb_modifier = InteractionModifiers::None;
    ir.rcoulomb         = cutoff;
    ir.ewald_rtol       = 1e-6;
    ir.epsilon_r        = 1;
    ir.epsilon_rf       = 0;

    gmx_mtop_t mtop;
    mtop.ffparams.reppow = repulsionPower;
    mtop.ffparams.functype.resize(1);
    mtop.ffparams.functype[0] = InteractionFunction::LennardJonesShortRange;

    interaction_const_t ic = init_interaction_const(nullptr, ir, mtop, false, std::nullopt);
    init_interaction_const_tables(nullptr, &ic, cutoff, 0);
    return ic;
}

std::vector<NbnxmKernelType> cpuKernelTypesToValidate()
{
    std::vector<NbnxmKernelType> kernelTypes = { NbnxmKernelType::Cpu1x1_PlainC, NbnxmKernelType::Cpu4x4_PlainC };
    if (sc_haveNbnxmSimd4xmKernels)
    {
        kernelTypes.push_back(NbnxmKernelType::Cpu4xN_Simd_4xN);
    }
    if (sc_haveNbnxmSimd2xmmKernels)
    {
        kernelTypes.push_back(NbnxmKernelType::Cpu4xN_Simd_2xNN);
    }
    return kernelTypes;
}

std::unique_ptr<nonbonded_verlet_t> setupNbnxm(const PcffNonbondedSystem& system,
                                               const real                 cutoff,
                                               const CoulombKernelType    coulombKernelType,
                                               const NbnxmKernelType      kernelType)
{
    const PinningPolicy pinPolicy  = PinningPolicy::CannotBePinned;
    const int           numThreads = 1;

    gmx_omp_nthreads_set(ModuleMultiThread::Pairsearch, numThreads);
    gmx_omp_nthreads_set(ModuleMultiThread::Nonbonded, numThreads);

    NbnxmKernelSetup kernelSetup{ kernelType,
                                  coulombKernelType == CoulombKernelType::Table ? EwaldExclusionType::Table
                                                                                 : EwaldExclusionType::Analytical };

    PairlistParams pairlistParams(kernelSetup.kernelType, {}, false, cutoff, false);

    auto pairlistSets = std::make_unique<PairlistSets>(pairlistParams, false, 0, pinPolicy);
    auto pairSearch   = std::make_unique<PairSearch>(PbcType::Xyz,
                                                   false,
                                                   nullptr,
                                                   nullptr,
                                                   pairlistParams.pairlistType,
                                                   false,
                                                   false,
                                                   numThreads,
                                                   pinPolicy);
    auto atomData     = std::make_unique<nbnxn_atomdata_t>(pinPolicy,
                                                       MDLogger(),
                                                       kernelSetup.kernelType,
                                                       LJCombinationRule::None,
                                                       LJCombinationRule::None,
                                                       system.nonbondedParameters,
                                                       true,
                                                       1,
                                                       numThreads);

    auto nbv = std::make_unique<nonbonded_verlet_t>(
            std::move(pairlistSets), std::move(pairSearch), std::move(atomData), kernelSetup, nullptr);

    const rvec lowerCorner = { 0, 0, 0 };
    const rvec upperCorner = { system.box[XX][XX], system.box[YY][YY], system.box[ZZ][ZZ] };
    nbv->putAtomsOnGrid(system.box,
                        0,
                        lowerCorner,
                        upperCorner,
                        nullptr,
                        { 0, int(system.coordinates.size()) },
                        system.coordinates.size(),
                        system.coordinates.size() / det(system.box),
                        system.atomInfo,
                        system.coordinates,
                        nullptr);
    nbv->setAtomProperties(system.atomTypes, system.charges, system.atomInfo);
    nbv->constructPairlist(InteractionLocality::Local, system.exclusions, false, 0, nullptr);

    return nbv;
}

PcffNonbondedOutput evaluateSystem(const PcffNonbondedSystem& system,
                                   const interaction_const_t& ic,
                                   const real                 cutoff,
                                   const CoulombKernelType    coulombKernelType,
                                   const NbnxmKernelType      kernelType)
{
    auto                    nbv = setupNbnxm(system, cutoff, coulombKernelType, kernelType);

    std::vector<RVec> shiftVectors(c_numShiftVectors);
    calc_shifts(system.box, shiftVectors);

    StepWorkload stepWork;
    stepWork.computeForces = true;
    stepWork.computeEnergy = true;

    std::vector<real> vVdw(1, 0.0_real);
    std::vector<real> vCoulomb(1, 0.0_real);
    nbv->dispatchNonbondedKernel(
            InteractionLocality::Local, ic, stepWork, enbvClearFYes, shiftVectors, vVdw, vCoulomb, nullptr);

    PcffNonbondedOutput output;
    output.vdwEnergy     = vVdw[0];
    output.coulombEnergy = vCoulomb[0];

    ArrayRef<const int> atomIndices = nbv->getLocalAtomOrder();
    std::vector<RVec> nbnxmForces(nbv->localAtomOrderMatchesNbnxmOrder() ? atomIndices.size()
                                                                         : system.coordinates.size(),
                                  { 0.0_real, 0.0_real, 0.0_real });
    nbv->atomdata_add_nbat_f_to_f(AtomLocality::All, nbnxmForces);

    output.forces.resize(system.coordinates.size(), { 0.0_real, 0.0_real, 0.0_real });
    if (nbv->localAtomOrderMatchesNbnxmOrder())
    {
        for (Index i = 0; i < atomIndices.ssize(); ++i)
        {
            const int atom = atomIndices[i];
            if (nonbonded_verlet_t::isValidLocalAtom(atom))
            {
                output.forces[atom] = nbnxmForces[i];
            }
        }
    }
    else
    {
        output.forces = std::move(nbnxmForces);
    }

    return output;
}

std::pair<real, real> class2Coefficients(const real sigma, const real epsilon)
{
    return { 18.0_real * epsilon * gmx::power6(sigma), 18.0_real * epsilon * std::pow(sigma, 9) };
}

real class2Energy(const real sigma, const real epsilon, const real r)
{
    const real sigmaOverR = sigma / r;
    return epsilon * (2.0_real * std::pow(sigmaOverR, 9) - 3.0_real * gmx::power6(sigmaOverR));
}

real class2ForceOnFirstAtomAlongX(const real sigma, const real epsilon, const real r)
{
    return 18.0_real * epsilon * (gmx::power6(sigma) / std::pow(r, 7) - std::pow(sigma, 9) / std::pow(r, 10));
}

PcffNonbondedSystem makeTwoAtomSystem(const real distance, const real c6, const real c9, const real q0, const real q1, const bool excludePair)
{
    PcffNonbondedSystem system;
    system.numAtomTypes         = 1;
    system.nonbondedParameters  = { c6, c9 };
    system.atomTypes            = { 0, 0 };
    system.charges              = { q0, q1 };
    system.atomInfo.resize(2, sc_atomInfo_HasVdw);
    if (q0 != 0)
    {
        system.atomInfo[0] |= sc_atomInfo_HasCharge;
    }
    if (q1 != 0)
    {
        system.atomInfo[1] |= sc_atomInfo_HasCharge;
    }

    std::array<int, 2> excluded = { 0, 1 };
    std::array<int, 1> self0    = { 0 };
    std::array<int, 1> self1    = { 1 };
    if (excludePair)
    {
        system.exclusions.pushBack(excluded);
        system.exclusions.pushBack(excluded);
    }
    else
    {
        system.exclusions.pushBack(self0);
        system.exclusions.pushBack(self1);
    }

    system.coordinates = { { 1.0_real, 1.0_real, 1.0_real }, { 1.0_real + distance, 1.0_real, 1.0_real } };
    clear_mat(system.box);
    system.box[XX][XX] = 3.0_real;
    system.box[YY][YY] = 3.0_real;
    system.box[ZZ][ZZ] = 3.0_real;
    return system;
}

PcffNonbondedSystem makeSmallOligomerChargeOnlySystem(const bool includeOneFourPairs)
{
    PcffNonbondedSystem system;
    system.numAtomTypes        = 2;
    system.nonbondedParameters = {
        0.0_real, 0.0_real,
        0.0_real, 0.0_real,
        0.0_real, 0.0_real,
        0.0_real, 0.0_real
    };
    system.atomTypes = { 0, 1, 0, 1, 0, 1 };
    system.charges   = { 0.25_real, -0.25_real, 0.25_real, -0.25_real, 0.25_real, -0.25_real };
    system.atomInfo.resize(6, sc_atomInfo_HasCharge);

    system.exclusions.pushBack(std::array<int, 4>{ 0, 1, 2, 3 });
    system.exclusions.pushBack(std::array<int, 5>{ 0, 1, 2, 3, 4 });
    system.exclusions.pushBack(std::array<int, 6>{ 0, 1, 2, 3, 4, 5 });
    system.exclusions.pushBack(std::array<int, 6>{ 0, 1, 2, 3, 4, 5 });
    system.exclusions.pushBack(std::array<int, 5>{ 1, 2, 3, 4, 5 });
    system.exclusions.pushBack(std::array<int, 4>{ 2, 3, 4, 5 });

    if (includeOneFourPairs)
    {
        system.exclusions.clear();
        system.exclusions.pushBack(std::array<int, 3>{ 0, 1, 2 });
        system.exclusions.pushBack(std::array<int, 4>{ 0, 1, 2, 3 });
        system.exclusions.pushBack(std::array<int, 5>{ 0, 1, 2, 3, 4 });
        system.exclusions.pushBack(std::array<int, 5>{ 1, 2, 3, 4, 5 });
        system.exclusions.pushBack(std::array<int, 4>{ 2, 3, 4, 5 });
        system.exclusions.pushBack(std::array<int, 3>{ 3, 4, 5 });
    }

    system.coordinates = {
        { 0.65_real, 1.02_real, 1.00_real },
        { 0.79_real, 0.96_real, 1.01_real },
        { 0.92_real, 1.05_real, 0.98_real },
        { 1.06_real, 0.97_real, 1.03_real },
        { 1.19_real, 1.06_real, 0.99_real },
        { 1.33_real, 0.98_real, 1.02_real }
    };
    clear_mat(system.box);
    system.box[XX][XX] = 2.0_real;
    system.box[YY][YY] = 2.0_real;
    system.box[ZZ][ZZ] = 2.0_real;
    return system;
}

PcffNonbondedSystem makeSmallOligomerNoPairsSystem(const bool withLennardJones, const bool withCharges)
{
    PcffNonbondedSystem system = makeSmallOligomerChargeOnlySystem(false);
    system.nonbondedParameters = {
        withLennardJones ? 2.32684636e-03_real : 0.0_real, withLennardJones ? 6.09695853e-05_real : 0.0_real,
        withLennardJones ? 1.30511553e-03_real : 0.0_real, withLennardJones ? 2.93372341e-05_real : 0.0_real,
        withLennardJones ? 1.30511553e-03_real : 0.0_real, withLennardJones ? 2.93372341e-05_real : 0.0_real,
        withLennardJones ? 7.32032757e-04_real : 0.0_real, withLennardJones ? 1.31765919e-05_real : 0.0_real
    };
    system.atomInfo.assign(6, 0);
    if (withCharges)
    {
        system.atomInfo.assign(6, sc_atomInfo_HasCharge);
    }
    if (withLennardJones)
    {
        for (auto& atomInfo : system.atomInfo)
        {
            atomInfo |= sc_atomInfo_HasVdw;
        }
    }
    if (!withCharges)
    {
        std::fill(system.charges.begin(), system.charges.end(), 0.0_real);
    }
    return system;
}

real minimumImage(const real delta, const real boxLength)
{
    return delta - boxLength * std::round(delta / boxLength);
}

std::vector<std::vector<bool>> excludedMatrix(const PcffNonbondedSystem& system)
{
    std::vector<std::vector<bool>> excluded(system.coordinates.size(),
                                            std::vector<bool>(system.coordinates.size(), false));
    for (Index i = 0; i < system.exclusions.ssize(); ++i)
    {
        for (const int j : system.exclusions[i])
        {
            excluded[i][j] = true;
        }
    }
    return excluded;
}

void tabulatedPmePairContribution(const interaction_const_t& ic,
                                  const real                 qq,
                                  const real                 r,
                                  const bool                 excluded,
                                  real*                      energy,
                                  real*                      forceScale)
{
    const real rinv       = gmx::invsqrt(r * r);
    const real rinvsq     = rinv * rinv;
    const real scaledR    = r * ic.coulombEwaldTables->scale;
    const int  tableIndex = static_cast<int>(scaledR);
    const real frac       = scaledR - tableIndex;
    const real halfsp     = 0.5_real / ic.coulombEwaldTables->scale;

#if !GMX_DOUBLE
    const real* table = ic.coulombEwaldTables->tableFDV0.data();
    const real  fexcl = table[tableIndex * 4] + frac * table[tableIndex * 4 + 1];
    const real  vcorr =
            table[tableIndex * 4 + 2] - halfsp * frac * (table[tableIndex * 4] + fexcl);
#else
    const real* tableF = ic.coulombEwaldTables->tableF.data();
    const real* tableV = ic.coulombEwaldTables->tableV.data();
    const real  fexcl  = (1 - frac) * tableF[tableIndex] + frac * tableF[tableIndex + 1];
    const real  vcorr  = tableV[tableIndex] - halfsp * frac * (tableF[tableIndex] + fexcl);
#endif

    const real interact = excluded ? 0.0_real : 1.0_real;
    *energy             = qq * (interact * (rinv - ic.coulomb.ewaldShift) - vcorr);
    *forceScale         = qq * (interact * rinvsq - fexcl) * rinv;
}

PcffNonbondedOutput directSpaceCoulombReference(const PcffNonbondedSystem& system, const real cutoff, const real epsfac)
{
    PcffNonbondedOutput output;
    output.forces.resize(system.coordinates.size(), { 0.0_real, 0.0_real, 0.0_real });

    const auto excluded = excludedMatrix(system);

    for (Index i = 0; i < ssize(system.coordinates); ++i)
    {
        for (Index j = i + 1; j < ssize(system.coordinates); ++j)
        {
            if (excluded[i][j] || excluded[j][i])
            {
                continue;
            }

            const real dx = minimumImage(system.coordinates[i][XX] - system.coordinates[j][XX], system.box[XX][XX]);
            const real dy = minimumImage(system.coordinates[i][YY] - system.coordinates[j][YY], system.box[YY][YY]);
            const real dz = minimumImage(system.coordinates[i][ZZ] - system.coordinates[j][ZZ], system.box[ZZ][ZZ]);
            const real r2 = dx * dx + dy * dy + dz * dz;
            const real r  = std::sqrt(r2);
            if (r >= cutoff)
            {
                continue;
            }

            const real qq   = epsfac * system.charges[i] * system.charges[j];
            const real vcoul = qq / r;
            const real fscal = qq / (r2 * r);
            output.coulombEnergy += vcoul;
            output.forces[i][XX] += fscal * dx;
            output.forces[i][YY] += fscal * dy;
            output.forces[i][ZZ] += fscal * dz;
            output.forces[j][XX] -= fscal * dx;
            output.forces[j][YY] -= fscal * dy;
            output.forces[j][ZZ] -= fscal * dz;
        }
    }

    return output;
}

PcffNonbondedOutput pmeTableCoulombReference(const PcffNonbondedSystem& system,
                                             const interaction_const_t&  ic,
                                             const real                  cutoff)
{
    PcffNonbondedOutput output;
    output.forces.resize(system.coordinates.size(), { 0.0_real, 0.0_real, 0.0_real });

    const auto excluded = excludedMatrix(system);

    for (Index i = 0; i < ssize(system.coordinates); ++i)
    {
        for (Index j = i + 1; j < ssize(system.coordinates); ++j)
        {
            const real dx = minimumImage(system.coordinates[i][XX] - system.coordinates[j][XX], system.box[XX][XX]);
            const real dy = minimumImage(system.coordinates[i][YY] - system.coordinates[j][YY], system.box[YY][YY]);
            const real dz = minimumImage(system.coordinates[i][ZZ] - system.coordinates[j][ZZ], system.box[ZZ][ZZ]);
            const real r2 = dx * dx + dy * dy + dz * dz;
            const real r  = std::sqrt(r2);
            if (r >= cutoff)
            {
                continue;
            }

            const real qq = ic.coulomb.epsfac * system.charges[i] * system.charges[j];
            real       energy = 0;
            real       forceScale = 0;
            tabulatedPmePairContribution(
                    ic, qq, r, excluded[i][j] || excluded[j][i], &energy, &forceScale);

            output.coulombEnergy += energy;
            output.forces[i][XX] += forceScale * dx;
            output.forces[i][YY] += forceScale * dy;
            output.forces[i][ZZ] += forceScale * dz;
            output.forces[j][XX] -= forceScale * dx;
            output.forces[j][YY] -= forceScale * dy;
            output.forces[j][ZZ] -= forceScale * dz;
        }
    }

    const real selfEnergy = 0.5_real
#if !GMX_DOUBLE
                            * ic.coulombEwaldTables->tableFDV0[2]
#else
                            * ic.coulombEwaldTables->tableV[0]
#endif
            ;
    for (Index i = 0; i < ssize(system.coordinates); ++i)
    {
        output.coulombEnergy -= ic.coulomb.epsfac * system.charges[i] * system.charges[i] * selfEnergy;
    }

    return output;
}

TEST(PcffClass2NonbondedCurveTest, NineSixPairCurveMatchesAnalyticEnergyAndForce)
{
    constexpr real sigma   = 0.34_real;
    constexpr real epsilon = 0.50208_real;
    constexpr real cutoff  = 1.2_real;

    const auto [c6, c9] = class2Coefficients(sigma, epsilon);
    const std::vector<real> distances = { 0.38_real, 0.45_real, 0.60_real, 0.80_real };

    for (const real distance : distances)
    {
        SCOPED_TRACE(testing::Message() << "distance=" << distance);
        for (const NbnxmKernelType kernelType : cpuKernelTypesToValidate())
        {
            SCOPED_TRACE(testing::Message() << "kernel=" << static_cast<int>(kernelType));
            const auto system = makeTwoAtomSystem(distance, c6, c9, 0.0_real, 0.0_real, false);
            const auto output =
                    evaluateSystem(system, makeInteractionConst(CoulombKernelType::ReactionField, 9.0, cutoff), cutoff, CoulombKernelType::ReactionField, kernelType);

            EXPECT_NEAR(output.vdwEnergy, class2Energy(sigma, epsilon, distance), 2e-4);
            EXPECT_NEAR(output.coulombEnergy, 0.0_real, 1e-8);
            EXPECT_NEAR(output.forces[0][XX], class2ForceOnFirstAtomAlongX(sigma, epsilon, distance), 2e-3);
            EXPECT_NEAR(output.forces[1][XX], -output.forces[0][XX], 2e-6);
            EXPECT_NEAR(output.forces[0][YY], 0.0_real, 1e-8);
            EXPECT_NEAR(output.forces[0][ZZ], 0.0_real, 1e-8);
        }
    }
}

TEST(PcffClass2NonbondedCurveTest, TabulatedPmeCoulombPathKeepsCoulombIndependentFromNineSixVdw)
{
    constexpr real sigma   = 0.34_real;
    constexpr real epsilon = 0.50208_real;
    constexpr real cutoff  = 1.2_real;
    constexpr real distance = 0.52_real;

    const auto [c6, c9]     = class2Coefficients(sigma, epsilon);
    for (const NbnxmKernelType kernelType : cpuKernelTypesToValidate())
    {
        SCOPED_TRACE(testing::Message() << "kernel=" << static_cast<int>(kernelType));
        const auto withVdw = evaluateSystem(makeTwoAtomSystem(distance, c6, c9, 0.35_real, -0.40_real, false),
                                            makeInteractionConst(CoulombKernelType::Table, 9.0, cutoff),
                                            cutoff,
                                            CoulombKernelType::Table,
                                            kernelType);
        const auto withoutVdw = evaluateSystem(makeTwoAtomSystem(distance, 0.0_real, 0.0_real, 0.35_real, -0.40_real, false),
                                               makeInteractionConst(CoulombKernelType::Table, 9.0, cutoff),
                                               cutoff,
                                               CoulombKernelType::Table,
                                               kernelType);
        const real expectedVdwForce = class2ForceOnFirstAtomAlongX(sigma, epsilon, distance);

        EXPECT_NEAR(withVdw.vdwEnergy, class2Energy(sigma, epsilon, distance), 2e-4);
        EXPECT_NEAR(withVdw.coulombEnergy, withoutVdw.coulombEnergy, 1e-7);
        EXPECT_NEAR(withVdw.forces[0][XX] - withoutVdw.forces[0][XX], expectedVdwForce, 2e-3);
        EXPECT_NEAR(withVdw.forces[1][XX] - withoutVdw.forces[1][XX], -expectedVdwForce, 2e-3);
    }
}

TEST(PcffClass2NonbondedCurveTest, ExclusionsSuppressNineSixAndCoulombInteractions)
{
    constexpr real sigma   = 0.34_real;
    constexpr real epsilon = 0.50208_real;
    constexpr real cutoff  = 1.2_real;

    const auto [c6, c9] = class2Coefficients(sigma, epsilon);
    for (const NbnxmKernelType kernelType : cpuKernelTypesToValidate())
    {
        SCOPED_TRACE(testing::Message() << "kernel=" << static_cast<int>(kernelType));
        const auto output = evaluateSystem(makeTwoAtomSystem(0.45_real, c6, c9, 0.0_real, 0.0_real, true),
                                           makeInteractionConst(CoulombKernelType::ReactionField, 9.0, cutoff),
                                           cutoff,
                                           CoulombKernelType::ReactionField,
                                           kernelType);

        EXPECT_NEAR(output.vdwEnergy, 0.0_real, 1e-8);
        EXPECT_NEAR(output.coulombEnergy, 0.0_real, 1e-8);
        for (const auto& force : output.forces)
        {
            EXPECT_NEAR(force[XX], 0.0_real, 1e-8);
            EXPECT_NEAR(force[YY], 0.0_real, 1e-8);
            EXPECT_NEAR(force[ZZ], 0.0_real, 1e-8);
        }
    }
}

TEST(PcffClass2NonbondedCurveTest, CombinedNineSixAndCoulombExclusionsStayZero)
{
    constexpr real sigma   = 0.34_real;
    constexpr real epsilon = 0.50208_real;
    constexpr real cutoff  = 1.2_real;

    const auto [c6, c9] = class2Coefficients(sigma, epsilon);
    const auto ic       = makeCutoffInteractionConst(9.0, cutoff);
    for (const NbnxmKernelType kernelType : cpuKernelTypesToValidate())
    {
        SCOPED_TRACE(testing::Message() << "kernel=" << static_cast<int>(kernelType));
        const auto output = evaluateSystem(makeTwoAtomSystem(0.45_real, c6, c9, 0.35_real, -0.40_real, true),
                                           ic,
                                           cutoff,
                                           CoulombKernelType::ReactionField,
                                           kernelType);

        EXPECT_NEAR(output.vdwEnergy, 0.0_real, 1e-8);
        EXPECT_NEAR(output.coulombEnergy, 0.0_real, 1e-8);
        for (const auto& force : output.forces)
        {
            EXPECT_NEAR(force[XX], 0.0_real, 1e-8);
            EXPECT_NEAR(force[YY], 0.0_real, 1e-8);
            EXPECT_NEAR(force[ZZ], 0.0_real, 1e-8);
        }
    }
}

TEST(PcffClass2NonbondedCurveTest, MakeNonBondedParameterListsUsesRepulsionPowerForNineSix)
{
    constexpr real sigma   = 0.34_real;
    constexpr real epsilon = 0.50208_real;

    std::array<t_iparams, 1> parameters = {};
    parameters[0].lj.c6                 = 3.0_real * epsilon * gmx::power6(sigma);
    parameters[0].lj.c12                = 2.0_real * epsilon * std::pow(sigma, 9);

    const auto nbfp = makeNonBondedParameterLists(1, false, parameters, false, 9.0);

    ASSERT_EQ(nbfp.size(), 2);
    EXPECT_NEAR(nbfp[0], 18.0_real * epsilon * gmx::power6(sigma), 1e-7);
    EXPECT_NEAR(nbfp[1], 18.0_real * epsilon * std::pow(sigma, 9), 1e-7);
}

TEST(PcffClass2NonbondedCurveTest, SmallOligomerChargeOnlyCutoffMatchesDirectCoulombWithExclusions)
{
    constexpr real cutoff = 0.9_real;
    const auto     system = makeSmallOligomerChargeOnlySystem(false);
    const auto     ic     = makeCutoffInteractionConst(9.0, cutoff);
    const auto     reference = directSpaceCoulombReference(system, cutoff, ic.coulomb.epsfac);

    for (const NbnxmKernelType kernelType : cpuKernelTypesToValidate())
    {
        SCOPED_TRACE(testing::Message() << "kernel=" << static_cast<int>(kernelType));
        const auto output = evaluateSystem(system, ic, cutoff, CoulombKernelType::ReactionField, kernelType);

        EXPECT_NEAR(output.coulombEnergy, reference.coulombEnergy, 1e-4);
        for (Index atom = 0; atom < ssize(system.coordinates); ++atom)
        {
            for (int d = 0; d < DIM; ++d)
            {
                EXPECT_NEAR(output.forces[atom][d], reference.forces[atom][d], 2e-4)
                        << "atom=" << atom << " dim=" << d;
            }
        }
    }
}

TEST(PcffClass2NonbondedCurveTest, SmallOligomerNoPairsCombinedKernelIsAdditive)
{
    constexpr real cutoff = 0.9_real;
    const auto     ic     = makeCutoffInteractionConst(9.0, cutoff);
    const auto     fullSystem = makeSmallOligomerNoPairsSystem(true, true);
    const auto     ljOnlySystem = makeSmallOligomerNoPairsSystem(true, false);
    const auto     chargeOnlySystem = makeSmallOligomerNoPairsSystem(false, true);

    for (const NbnxmKernelType kernelType : cpuKernelTypesToValidate())
    {
        SCOPED_TRACE(testing::Message() << "kernel=" << static_cast<int>(kernelType));
        const auto fullOutput = evaluateSystem(fullSystem, ic, cutoff, CoulombKernelType::ReactionField, kernelType);
        const auto ljOutput = evaluateSystem(ljOnlySystem, ic, cutoff, CoulombKernelType::ReactionField, kernelType);
        const auto chargeOutput = evaluateSystem(chargeOnlySystem, ic, cutoff, CoulombKernelType::ReactionField, kernelType);

        EXPECT_NEAR(fullOutput.vdwEnergy, ljOutput.vdwEnergy, 1e-4);
        EXPECT_NEAR(fullOutput.coulombEnergy, chargeOutput.coulombEnergy, 1e-4);
        for (Index atom = 0; atom < ssize(fullSystem.coordinates); ++atom)
        {
            for (int d = 0; d < DIM; ++d)
            {
                EXPECT_NEAR(fullOutput.forces[atom][d], ljOutput.forces[atom][d] + chargeOutput.forces[atom][d], 2e-4)
                        << "atom=" << atom << " dim=" << d;
            }
        }
    }
}

TEST(PcffClass2NonbondedCurveTest, ExcludedPmePairMatchesTabulatedReference)
{
    constexpr real cutoff   = 1.2_real;
    constexpr real distance = 0.52_real;
    constexpr real q0       = 0.35_real;
    constexpr real q1       = -0.40_real;

    const auto system    = makeTwoAtomSystem(distance, 0.0_real, 0.0_real, q0, q1, true);
    const auto ic        = makeInteractionConst(CoulombKernelType::Table, 9.0, cutoff);
    const auto reference = pmeTableCoulombReference(system, ic, cutoff);

    for (const NbnxmKernelType kernelType : cpuKernelTypesToValidate())
    {
        SCOPED_TRACE(testing::Message() << "kernel=" << static_cast<int>(kernelType));
        const auto output = evaluateSystem(system, ic, cutoff, CoulombKernelType::Table, kernelType);

        EXPECT_NEAR(output.vdwEnergy, 0.0_real, 1e-8);
        EXPECT_NEAR(output.coulombEnergy, reference.coulombEnergy, 2e-4);
        for (int d = 0; d < DIM; ++d)
        {
            EXPECT_NEAR(output.forces[0][d], reference.forces[0][d], 2e-4) << "dim=" << d;
            EXPECT_NEAR(output.forces[1][d], reference.forces[1][d], 2e-4) << "dim=" << d;
        }
    }
}

TEST(PcffClass2NonbondedCurveTest, SmallOligomerChargeOnlyPmeMatchesTabulatedReference)
{
    constexpr real cutoff = 0.9_real;
    const auto     system = makeSmallOligomerChargeOnlySystem(false);
    const auto     ic     = makeInteractionConst(CoulombKernelType::Table, 9.0, cutoff);
    const auto     reference = pmeTableCoulombReference(system, ic, cutoff);

    for (const NbnxmKernelType kernelType : cpuKernelTypesToValidate())
    {
        SCOPED_TRACE(testing::Message() << "kernel=" << static_cast<int>(kernelType));
        const auto output = evaluateSystem(system, ic, cutoff, CoulombKernelType::Table, kernelType);

        EXPECT_NEAR(output.vdwEnergy, 0.0_real, 1e-8);
        EXPECT_NEAR(output.coulombEnergy, reference.coulombEnergy, 4e-4);
        for (Index atom = 0; atom < ssize(system.coordinates); ++atom)
        {
            for (int d = 0; d < DIM; ++d)
            {
                EXPECT_NEAR(output.forces[atom][d], reference.forces[atom][d], 5e-4)
                        << "atom=" << atom << " dim=" << d;
            }
        }
    }
}

} // namespace

} // namespace test

} // namespace gmx
