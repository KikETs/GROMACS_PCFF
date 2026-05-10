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
 * \brief Tests for bonded GPU input gating relevant to PCFF/class2 support.
 */

#include "gmxpre.h"

#include "gromacs/listed_forces/listed_forces_gpu.h"

#include <gtest/gtest.h>

#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/topology/topology.h"

#include "testutils/setenv.h"

namespace gmx
{
namespace test
{
namespace
{

void initializeDynamicsInputrec(t_inputrec* ir)
{
    ir->eI          = IntegrationAlgorithm::MD;
    ir->opts.ngener = 1;
    ir->nwall       = 0;
    ir->useMts      = false;
}

void initializeListedPairTopology(gmx_mtop_t* mtop, const double repulsionPower)
{
    mtop->ffparams.reppow = repulsionPower;
    mtop->moltype.resize(1);
    mtop->moltype[0].ilist[InteractionFunction::LennardJones14].iatoms = { 0, 0, 1 };
}

void initializeHarmonicBondTopology(gmx_mtop_t* mtop, const double repulsionPower)
{
    mtop->ffparams.reppow = repulsionPower;
    mtop->moltype.resize(1);
    mtop->moltype[0].ilist[InteractionFunction::Bonds].iatoms = { 0, 0, 1 };
}

void initializeExactLammpsRespaInputrec(t_inputrec* ir)
{
    initializeDynamicsInputrec(ir);
    ir->useMts   = true;
    ir->mtsMode  = MtsMode::LammpsRespa;
    ir->lammpsRespa.enabled       = true;
    ir->lammpsRespa.bondLevel     = 0;
    ir->lammpsRespa.angleLevel    = 1;
    ir->lammpsRespa.dihedralLevel = 1;
    ir->lammpsRespa.improperLevel = 1;
    ir->lammpsRespa.pair14Level   = 1;
    ir->lammpsRespa.kspaceLevel   = 2;
    ir->exactRespa.levelStepFactors = { 1, 2, 4 };
    ir->exactRespa.forceLayout      = ir->lammpsRespa;
}

void initializeMixedClass2BondTopology(gmx_mtop_t* mtop)
{
    mtop->ffparams.reppow = 12.0;
    mtop->moltype.resize(1);
    mtop->moltype[0].ilist[InteractionFunction::BondClass2].iatoms = { 0, 0, 1 };
}

TEST(ListedForcesGpuInputSupportTest, RejectsSixthPowerRepulsionForBondedGpu)
{
    t_inputrec  ir;
    gmx_mtop_t  mtop;
    std::string error;
    initializeDynamicsInputrec(&ir);
    initializeListedPairTopology(&mtop, 9.0);

    EXPECT_FALSE(inputSupportsListedForcesGpu(ir, mtop, &error));
    EXPECT_NE(error.find("only supports 12-6 listed 1-4 Lennard-Jones semantics outside the exact"),
              std::string::npos);
}

TEST(ListedForcesGpuInputSupportTest, AcceptsTwelveSixListedPairsForBondedGpu)
{
    t_inputrec  ir;
    gmx_mtop_t  mtop;
    std::string error;
    initializeDynamicsInputrec(&ir);
    initializeListedPairTopology(&mtop, 12.0);

    EXPECT_TRUE(inputSupportsListedForcesGpu(ir, mtop, &error)) << error;
}

TEST(ListedForcesGpuInputSupportTest, ExactRespaAcceptsNineSixListedPairsForBondedGpu)
{
    t_inputrec  ir;
    gmx_mtop_t  mtop;
    std::string error;
    initializeExactLammpsRespaInputrec(&ir);
    initializeListedPairTopology(&mtop, 9.0);

    EXPECT_TRUE(inputSupportsListedForcesGpu(ir, mtop, &error)) << error;
}

TEST(ListedForcesGpuInputSupportTest, DoesNotRejectNinthPowerRepulsionWithoutListedPairs)
{
    t_inputrec  ir;
    gmx_mtop_t  mtop;
    std::string error;
    initializeDynamicsInputrec(&ir);
    initializeHarmonicBondTopology(&mtop, 9.0);

    EXPECT_TRUE(inputSupportsListedForcesGpu(ir, mtop, &error)) << error;
}

TEST(ListedForcesGpuInputSupportTest, ExactRespaPcffClass2WithLennardJones14PairsIsAdmitted)
{
    t_inputrec  ir;
    gmx_mtop_t  mtop;
    std::string error;
    initializeExactLammpsRespaInputrec(&ir);
    initializeMixedClass2BondTopology(&mtop);
    mtop.moltype[0].ilist[InteractionFunction::LennardJones14].iatoms = { 0, 0, 1 };

    EXPECT_TRUE(inputSupportsListedForcesGpu(ir, mtop, &error)) << error;
}

TEST(ListedForcesGpuInputSupportTest, ExactRespaRejectsGpuCapableBondTypesOutsidePair14OnlyMode)
{
    t_inputrec  ir;
    gmx_mtop_t  mtop;
    std::string error;
    initializeExactLammpsRespaInputrec(&ir);
    initializeListedPairTopology(&mtop, 9.0);
    mtop.moltype[0].ilist[InteractionFunction::Bonds].iatoms = { 0, 0, 1 };

    EXPECT_FALSE(inputSupportsListedForcesGpu(ir, mtop, &error));
    EXPECT_NE(error.find("pair14-only narrow mode"), std::string::npos);
}

TEST(ListedForcesGpuInputSupportTest, ExactRespaWideGpuBondedEnvAdmitsGpuCapableBondTypes)
{
    t_inputrec  ir;
    gmx_mtop_t  mtop;
    std::string error;
    initializeExactLammpsRespaInputrec(&ir);
    initializeListedPairTopology(&mtop, 9.0);
    mtop.moltype[0].ilist[InteractionFunction::Bonds].iatoms = { 0, 0, 1 };

    gmxSetenv("GMX_PCFF_EXACT_RESPA_GPU_BONDED_FTYPES", "all", true);
    EXPECT_TRUE(inputSupportsListedForcesGpu(ir, mtop, &error)) << error;
    gmxUnsetenv("GMX_PCFF_EXACT_RESPA_GPU_BONDED_FTYPES");
}

} // namespace
} // namespace test
} // namespace gmx
