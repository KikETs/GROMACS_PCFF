/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2024- The GROMACS Authors
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
 * Tests for convparm.cpp
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 */

#include "gmxpre.h"

#include "gromacs/gmxpreprocess/convparm.h"

#include <cmath>

#include <array>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gromacs/gmxpreprocess/grompp_impl.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/units.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/stringutil.h"

#include "testutils/naming.h"

namespace gmx
{
namespace test
{
namespace
{

TEST(ConvertInteractionsTest, DoingNothingWorks)
{
    const int                                                      numAtomTypes = 0;
    gmx::EnumerationArray<InteractionFunction, InteractionsOfType> nonBondedInteractions;
    std::vector<MoleculeInformation>                               moleculesInformation;
    const MoleculeInformation* intermolecularInteractions = nullptr;
    CombinationRule            combinationRule            = CombinationRule::Geometric;
    const double               repulsionPower             = 12.0;
    const real                 fudgeQQ                    = 1.0;
    gmx_mtop_t                 mtop;

    convertInteractionsOfType(numAtomTypes,
                              nonBondedInteractions,
                              moleculesInformation,
                              intermolecularInteractions,
                              combinationRule,
                              repulsionPower,
                              fudgeQQ,
                              &mtop);
}

using testing::Eq;
using testing::Pointwise;

//! Fill a vector with size \c size with integer values increasing from 0
std::vector<int> iotaVector(const std::size_t size)
{
    std::vector<int> v(size);
    std::iota(v.begin(), v.end(), 0);
    return v;
}

/*! \brief Fill a vector with iota-style dummy interaction parameter
 * values identical for state A and B
 *
 * Any unused array entries are filled with NaN (on platforms where
 * this is supported). */
std::array<real, MAXFORCEPARAM> iotaParams(const std::size_t sizeA, const std::size_t sizeB)
{
    GMX_RELEASE_ASSERT(sizeA + sizeB <= MAXFORCEPARAM, "Sizes too big for interaction parameter array");
    std::array<real, MAXFORCEPARAM> a;
    std::fill(a.begin(), a.end(), std::nan("0"));
    std::iota(a.begin(), a.begin() + sizeA, 5._real);
    std::iota(a.begin() + sizeA, a.begin() + sizeA + sizeB, 5._real);
    return a;
}

void convertSingleInteractionWithParameters(InteractionFunction    ftype,
                                            const std::vector<real>& parameters,
                                            gmx_mtop_t*            mtop,
                                            CombinationRule        combinationRule = CombinationRule::Geometric,
                                            double                 repulsionPower = 12.0)
{
    const int                                                      numAtomTypes = 0;
    gmx::EnumerationArray<InteractionFunction, InteractionsOfType> nonBondedInteractions;
    std::vector<MoleculeInformation>                               moleculesInformation;
    char**                                                         dummyName = nullptr;

    gmx::EnumerationArray<InteractionFunction, InteractionsOfType> moleculeInteractions;
    std::vector<real>                                              forceParameters(MAXFORCEPARAM, 0);
    std::copy(parameters.begin(), parameters.end(), forceParameters.begin());
    moleculeInteractions[ftype].interactionTypes.emplace_back(
            InteractionOfType{ iotaVector(NRAL(ftype)), forceParameters, "name" });
    moleculesInformation.emplace_back(MoleculeInformation{
            dummyName, 0, false, t_atoms{}, t_block{}, ListOfLists<int>{}, moleculeInteractions });

    const MoleculeInformation* intermolecularInteractions = nullptr;
    const real                 fudgeQQ                    = 1.0;
    mtop->moltype.resize(1);

    convertInteractionsOfType(numAtomTypes,
                              nonBondedInteractions,
                              moleculesInformation,
                              intermolecularInteractions,
                              combinationRule,
                              repulsionPower,
                              fudgeQQ,
                              mtop);
    mtop->molblock.emplace_back(gmx_molblock_t{ 0, 1 });
}

void convertSingleNonbondedInteractionWithParameters(const InteractionFunction ftype,
                                                     const std::vector<real>&  parameters,
                                                     gmx_mtop_t*               mtop,
                                                     CombinationRule           combinationRule = CombinationRule::Geometric,
                                                     double                    repulsionPower = 12.0)
{
    const int                                                      numAtomTypes = 1;
    gmx::EnumerationArray<InteractionFunction, InteractionsOfType> nonBondedInteractions;
    std::vector<MoleculeInformation>                               moleculesInformation;
    std::vector<real>                                              forceParameters(MAXFORCEPARAM, 0);
    std::copy(parameters.begin(), parameters.end(), forceParameters.begin());
    nonBondedInteractions[ftype].interactionTypes.emplace_back(InteractionOfType{ {}, forceParameters, "name" });

    const MoleculeInformation* intermolecularInteractions = nullptr;
    const real                 fudgeQQ                    = 1.0;

    convertInteractionsOfType(numAtomTypes,
                              nonBondedInteractions,
                              moleculesInformation,
                              intermolecularInteractions,
                              combinationRule,
                              repulsionPower,
                              fudgeQQ,
                              mtop);
}

const t_iparams& convertedParameters(const gmx_mtop_t& mtop, const InteractionFunction ftype)
{
    const auto parameterIndex = mtop.moltype[0].ilist[ftype].iatoms[0];
    return mtop.ffparams.iparams[parameterIndex];
}

//! Define a test fixture class taking an integer paramter for ftype
using ConvertInteractionsTest = ::testing::TestWithParam<std::tuple<int>>;

TEST_P(ConvertInteractionsTest, Works)
{
    const int                                                      numAtomTypes = 0;
    gmx::EnumerationArray<InteractionFunction, InteractionsOfType> nonBondedInteractions;
    std::vector<MoleculeInformation>                               moleculesInformation;
    char**                                                         dummyName = nullptr;

    const InteractionFunction ftype = static_cast<InteractionFunction>(std::get<0>(GetParam()));

    // Ensure this function type is handled by convertInteractionsOfType.
    if (!shouldConvertInteractionType(ftype))
    {
        GTEST_SKIP() << "Skipping interaction type that does not represent a interaction with "
                        "parameters converted in grompp";
    }
    for (const auto unsupportedFunctionType :
         { InteractionFunction::GeneralizedBorn12PolarizationUnused,
           InteractionFunction::GeneralizedBorn13PolarizationUnused,
           InteractionFunction::GeneralizedBorn14PolarizationUnused,
           InteractionFunction::GeneralizedBornPolarizationUnused,
           InteractionFunction::NonpolarSolvationUnused })
    {
        if (ftype == unsupportedFunctionType)
        {
            GTEST_SKIP() << "Skipping bonded function type no longer supported";
        }
    }

    // Define a molecule type with a single interaction of type ftype and add it
    // to the molecules information object
    {
        gmx::EnumerationArray<InteractionFunction, InteractionsOfType> moleculeInteractions;
        // For function types with no parameters, assign_param()
        // assumes the parameters are all zero, which leads to not
        // appending all-zero parameter sest to the parameter list
        if (interaction_function[ftype].nrfpA == 0 && interaction_function[ftype].nrfpB == 0)
        {
            std::vector<real> zeroParams(MAXFORCEPARAM, 0);
            moleculeInteractions[ftype].interactionTypes.emplace_back(
                    InteractionOfType{ iotaVector(NRAL(ftype)), zeroParams, "name" });
        }
        else
        {
            // Note force parameters end up defined for both FEP states
            moleculeInteractions[ftype].interactionTypes.emplace_back(
                    InteractionOfType{ iotaVector(NRAL(ftype)), iotaParams(NRFPA(ftype), NRFPB(ftype)), "name" });
        }
        moleculesInformation.emplace_back(MoleculeInformation{
                dummyName, 0, false, t_atoms{}, t_block{}, ListOfLists<int>{}, moleculeInteractions });
    }

    const MoleculeInformation* intermolecularInteractions = nullptr;
    const double               repulsionPower             = 12.0;
    const real                 fudgeQQ                    = 1.0;
    gmx_mtop_t                 mtop;
    // Add molecule type with index 0
    mtop.moltype.resize(1);
    // Fill the molecule type
    convertInteractionsOfType(numAtomTypes,
                              nonBondedInteractions,
                              moleculesInformation,
                              intermolecularInteractions,
                              CombinationRule::Geometric,
                              repulsionPower,
                              fudgeQQ,
                              &mtop);
    // Add a molecule block with 1 molecule of type 0 which was just filled
    mtop.molblock.emplace_back(gmx_molblock_t{ 0, 1 });

    if (interaction_function[ftype].flags & IF_BOND)
    {
        EXPECT_EQ(gmx_mtop_interaction_count(mtop, IF_BOND), 1)
                << "topology has one bonded interaction";
        ASSERT_EQ(gmx_mtop_ftype_count(mtop, static_cast<InteractionFunction>(ftype)), 1)
                << "topology has one kind of interaction";
        EXPECT_EQ(mtop.moltype[0].ilist[ftype].iatoms[0], 0)
                << "the first interaction of the first molecule type uses the first "
                   "interaction function parameters";
    }
    std::vector<int> expected = iotaVector(NRAL(ftype));

    EXPECT_EQ(mtop.moltype[0].ilist[ftype].iatoms[0], 0)
            << "first interaction has index zero when there is only one interaction added";
    EXPECT_THAT(makeArrayRef(mtop.moltype[0].ilist[ftype].iatoms).subArray(1, NRAL(ftype)), Pointwise(Eq(), expected))
            << "the first interaction of the first molecule type has the expected atom indices";
    ASSERT_EQ(mtop.ffparams.numTypes(), 1)
            << "topology has one set of interaction function parameters for function types "
               "that have parameters";
    // It would be nice to check that the contents of mtop.ffparams.iparams[0] has the expected
    // relationship with moleculesInformation[0].interactions[ftype].interactionTypes[0].forceParam().subArray(0, NRFP(ftype)),
    // but t_iparams is a union that contains a variety of data types and numbers of logical units *and*
    // assign_param() in convparm.cpp sometimes converts units or precomputes squares for the convenience of mdrun.
    // Our code is not yet flexible enough to make that an easy job.
}

TEST(ConvertInteractionsPcffClass2Test, BondClass2ParametersAreStoredVerbatim)
{
    const std::vector<real> parameters = { 1.53_real, 250.0_real, -35.0_real, 8.0_real };
    gmx_mtop_t             mtop;
    convertSingleInteractionWithParameters(InteractionFunction::BondClass2, parameters, &mtop);
    const auto& ip = convertedParameters(mtop, InteractionFunction::BondClass2).bond_class2;

    EXPECT_EQ(mtop.ffparams.functype[0], InteractionFunction::BondClass2);
    EXPECT_EQ(ip.r0, parameters[0]);
    EXPECT_EQ(ip.k2, parameters[1]);
    EXPECT_EQ(ip.k3, parameters[2]);
    EXPECT_EQ(ip.k4, parameters[3]);
}

TEST(ConvertInteractionsPcffClass2Test, AngleClass2ConvertsAnglesToRadians)
{
    const std::vector<real> parameters = { 109.5_real, 35.0_real, -4.0_real, 1.2_real, 8.0_real, 1.42_real,
                                           1.42_real, 3.0_real, 2.5_real, 1.42_real, 1.42_real };
    gmx_mtop_t             mtop;
    convertSingleInteractionWithParameters(InteractionFunction::AngleClass2, parameters, &mtop);
    const auto& ip = convertedParameters(mtop, InteractionFunction::AngleClass2).angle_class2;

    EXPECT_EQ(mtop.ffparams.functype[0], InteractionFunction::AngleClass2);
    EXPECT_NEAR(ip.theta0, parameters[0] * gmx::c_deg2Rad, 1e-6);
    EXPECT_EQ(ip.k2, parameters[1]);
    EXPECT_EQ(ip.k3, parameters[2]);
    EXPECT_EQ(ip.k4, parameters[3]);
    EXPECT_EQ(ip.bb_k, parameters[4]);
    EXPECT_EQ(ip.bb_r1, parameters[5]);
    EXPECT_EQ(ip.bb_r2, parameters[6]);
    EXPECT_EQ(ip.ba_k1, parameters[7]);
    EXPECT_EQ(ip.ba_k2, parameters[8]);
    EXPECT_EQ(ip.ba_r1, parameters[9]);
    EXPECT_EQ(ip.ba_r2, parameters[10]);
}

TEST(ConvertInteractionsPcffClass2Test, ImproperClass2ConvertsAnglesToRadians)
{
    const std::vector<real> parameters = { 25.0_real, 0.0_real, 1.2_real, 1.0_real, 0.9_real,
                                           110.0_real, 109.0_real, 108.0_real };
    gmx_mtop_t             mtop;
    convertSingleInteractionWithParameters(InteractionFunction::ImproperClass2, parameters, &mtop);
    const auto& ip = convertedParameters(mtop, InteractionFunction::ImproperClass2).improper_class2;

    EXPECT_EQ(mtop.ffparams.functype[0], InteractionFunction::ImproperClass2);
    EXPECT_EQ(ip.k0, parameters[0]);
    EXPECT_NEAR(ip.chi0, parameters[1] * gmx::c_deg2Rad, 1e-6);
    EXPECT_EQ(ip.aa_k1, parameters[2]);
    EXPECT_EQ(ip.aa_k2, parameters[3]);
    EXPECT_EQ(ip.aa_k3, parameters[4]);
    EXPECT_NEAR(ip.aa_theta0_1, parameters[5] * gmx::c_deg2Rad, 1e-6);
    EXPECT_NEAR(ip.aa_theta0_2, parameters[6] * gmx::c_deg2Rad, 1e-6);
    EXPECT_NEAR(ip.aa_theta0_3, parameters[7] * gmx::c_deg2Rad, 1e-6);
}

TEST(ConvertInteractionsPcffClass2Test, DihedralClass2ConvertsAnglesToRadians)
{
    const std::vector<real> parameters = { 0.8_real,   0.0_real,   0.6_real,   180.0_real, 0.4_real,   0.0_real,
                                           0.12_real,  -0.08_real, 0.04_real,  1.50_real,  0.10_real,  -0.05_real,
                                           0.02_real,  0.11_real,  -0.03_real, 0.01_real,  1.50_real,  1.50_real,
                                           0.06_real,  -0.03_real, 0.015_real, 0.05_real,  -0.025_real, 0.01_real,
                                           112.0_real, 112.0_real, 0.20_real,  112.0_real, 112.0_real, 0.18_real,
                                           1.50_real,  1.50_real };
    gmx_mtop_t             mtop;
    convertSingleInteractionWithParameters(InteractionFunction::DihedralClass2, parameters, &mtop);
    const auto& ip = convertedParameters(mtop, InteractionFunction::DihedralClass2).dihedral_class2;

    EXPECT_EQ(mtop.ffparams.functype[0], InteractionFunction::DihedralClass2);
    EXPECT_EQ(ip.k1, parameters[0]);
    EXPECT_NEAR(ip.phi1, parameters[1] * gmx::c_deg2Rad, 1e-6);
    EXPECT_EQ(ip.k2, parameters[2]);
    EXPECT_NEAR(ip.phi2, parameters[3] * gmx::c_deg2Rad, 1e-6);
    EXPECT_EQ(ip.k3, parameters[4]);
    EXPECT_NEAR(ip.phi3, parameters[5] * gmx::c_deg2Rad, 1e-6);
    EXPECT_EQ(ip.mbt_f1, parameters[6]);
    EXPECT_EQ(ip.ebt_f1_2, parameters[13]);
    EXPECT_EQ(ip.at_f3_2, parameters[23]);
    EXPECT_NEAR(ip.at_theta0_1, parameters[24] * gmx::c_deg2Rad, 1e-6);
    EXPECT_NEAR(ip.at_theta0_2, parameters[25] * gmx::c_deg2Rad, 1e-6);
    EXPECT_EQ(ip.aat_k, parameters[26]);
    EXPECT_NEAR(ip.aat_theta0_1, parameters[27] * gmx::c_deg2Rad, 1e-6);
    EXPECT_NEAR(ip.aat_theta0_2, parameters[28] * gmx::c_deg2Rad, 1e-6);
    EXPECT_EQ(ip.bb13t_k, parameters[29]);
    EXPECT_EQ(ip.bb13t_r10, parameters[30]);
    EXPECT_EQ(ip.bb13t_r30, parameters[31]);
}

TEST(ConvertInteractionsPcffClass2Test, LennardJonesShortRangeSixthPowerConvertsToClass2NineSixCoefficients)
{
    const std::vector<real> parameters = { 0.34_real, 0.12_real };
    gmx_mtop_t              mtop;
    convertSingleNonbondedInteractionWithParameters(
            InteractionFunction::LennardJonesShortRange, parameters, &mtop, CombinationRule::SixthPower, 9.0);
    const auto& ip = mtop.ffparams.iparams[0].lj;

    ASSERT_EQ(mtop.ffparams.functype.size(), 1);
    EXPECT_EQ(mtop.ffparams.functype[0], InteractionFunction::LennardJonesShortRange);
    EXPECT_NEAR(ip.c6, 3.0_real * parameters[1] * gmx::power6(parameters[0]), 1e-6);
    EXPECT_NEAR(ip.c12, 2.0_real * parameters[1] * std::pow(parameters[0], 9.0_real), 1e-6);
}

TEST(ConvertInteractionsPcffClass2Test, LennardJones14SixthPowerConvertsToDirectNineSixPairCoefficients)
{
    const std::vector<real> parameters = { 0.34_real, 0.12_real, 0.34_real, 0.12_real };
    gmx_mtop_t              mtop;
    convertSingleInteractionWithParameters(
            InteractionFunction::LennardJones14, parameters, &mtop, CombinationRule::SixthPower, 9.0);
    const auto& ip = convertedParameters(mtop, InteractionFunction::LennardJones14).lj14;

    EXPECT_EQ(mtop.ffparams.functype[0], InteractionFunction::LennardJones14);
    EXPECT_NEAR(ip.c6A, 3.0_real * parameters[1] * gmx::power6(parameters[0]), 1e-6);
    EXPECT_NEAR(ip.c12A, 2.0_real * parameters[1] * std::pow(parameters[0], 9.0_real), 1e-6);
    EXPECT_NEAR(ip.c6B, 3.0_real * parameters[3] * gmx::power6(parameters[2]), 1e-6);
    EXPECT_NEAR(ip.c12B, 2.0_real * parameters[3] * std::pow(parameters[2], 9.0_real), 1e-6);
}

std::string ftypeToName(const int ftype)
{
    GMX_RELEASE_ASSERT(ftype < static_cast<int>(InteractionFunction::Count),
                       "Must have valid kind of interaction function");
    return interaction_function[ftype].longname;
}

const NameOfTestFromTuple<std::tuple<int>> sc_testNamer{ std::make_tuple(ftypeToName) };

using testing::Combine;
using testing::Range;
INSTANTIATE_TEST_SUITE_P(InteractionFunctionKind,
                         ConvertInteractionsTest,
                         Combine(Range(0, static_cast<int>(InteractionFunction::Count))),
                         sc_testNamer);

} // namespace
} // namespace test
} // namespace gmx
