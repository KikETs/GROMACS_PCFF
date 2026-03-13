/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2021- The GROMACS Authors
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
 * Tests for grompp directives parsing
 *
 * \author Eliane Briand <eliane@br.iand.fr>
 */

#include "gmxpre.h"

#include <array>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/fileio/tpxio.h"
#include "gromacs/gmxpreprocess/grompp.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/units.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/textreader.h"
#include "gromacs/utility/textwriter.h"

#include "testutils/cmdlinetest.h"
#include "testutils/conftest.h"
#include "testutils/refdata.h"
#include "testutils/testasserts.h"
#include "testutils/testfilemanager.h"
#include "testutils/textblockmatchers.h"

namespace gmx
{
namespace test
{
namespace
{

using gmx::test::CommandLine;
using gmx::test::TestFileManager;

enum class ExpectedResult
{
    Success,
    Death
};

class GromppDirectiveTest :
    public ::testing::TestWithParam<std::tuple<std::string, std::array<std::array<int, 4>, 2>>>
{
public:
    GromppDirectiveTest() = default;

protected:
    gmx::test::TestFileManager fileManager_;
    std::string                mdpContentString_ =
            "title                   = Directive edge case test \n"
            "integrator              = md \n"
            "nsteps                  = 1 \n"
            "dt                      = 0.002 \n"
            "vdwtype                 = cutoff \n"
            "coulombtype             = cutoff \n"
            "tcoupl                  = no \n"
            "pcoupl                  = no \n"
            "pbc                     = xyz \n"
            "gen_vel                 = yes \n";
};

class PcffClass2DirectiveTest : public ::testing::Test
{
protected:
    TestFileManager fileManager_;
    std::string     mdpContentString_ =
            "title                   = PCFF class2 directive test \n"
            "integrator              = md \n"
            "nsteps                  = 1 \n"
            "dt                      = 0.002 \n"
            "vdwtype                 = cutoff \n"
            "coulombtype             = cutoff \n"
            "tcoupl                  = no \n"
            "pcoupl                  = no \n"
            "pbc                     = xyz \n"
            "gen_vel                 = no \n";
    std::string groContentString_ =
            "pcff class2 directives\n"
            "4\n"
            "    1RES     C1    1   0.000   0.000   0.000\n"
            "    1RES     C2    2   0.150   0.000   0.000\n"
            "    1RES     C3    3   0.225   0.130   0.000\n"
            "    1RES     C4    4   0.150   0.110   0.035\n"
            "   3.00000   3.00000   3.00000\n";

    std::string makeTopology(const std::string& bondedDirectives) const
    {
        return std::string(
                       "[ defaults ]\n"
                       "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ\n"
                       "1 2 no 1.0 1.0\n"
                       "\n"
                       "[ atomtypes ]\n"
                       "; name mass charge ptype sigma epsilon\n"
                       "C 12.011 0.0 A 0.34 0.0\n"
                       "\n"
                       "[ moleculetype ]\n"
                       "; Name nrexcl\n"
                       "MOL 3\n"
                       "\n"
                       "[ atoms ]\n"
                       "; nr type resnr residue atom cgnr charge mass\n"
                       "1 C 1 RES C1 1 0.0 12.011\n"
                       "2 C 1 RES C2 1 0.0 12.011\n"
                       "3 C 1 RES C3 1 0.0 12.011\n"
                       "4 C 1 RES C4 1 0.0 12.011\n"
                       "\n")
               + bondedDirectives
               + std::string(
                       "\n[ system ]\n"
                       "PCFF class2 directive test\n"
                       "\n"
                       "[ molecules ]\n"
                       "MOL 1\n");
    }

    CommandLine makeCommandLine(const std::string& topContent, std::string* outTprFilename)
    {
        CommandLine cmdline;
        cmdline.addOption("grompp");

        const std::string mdpInputFileName = fileManager_.getTemporaryFilePath("pcff-class2.mdp").string();
        const std::string groInputFileName = fileManager_.getTemporaryFilePath("pcff-class2.gro").string();
        const std::string topInputFileName = fileManager_.getTemporaryFilePath("pcff-class2.top").string();
        *outTprFilename                    = fileManager_.getTemporaryFilePath("pcff-class2.tpr").string();

        gmx::TextWriter::writeFileFromString(mdpInputFileName, mdpContentString_);
        gmx::TextWriter::writeFileFromString(groInputFileName, groContentString_);
        gmx::TextWriter::writeFileFromString(topInputFileName, topContent);

        cmdline.addOption("-f", mdpInputFileName);
        cmdline.addOption("-c", groInputFileName);
        cmdline.addOption("-p", topInputFileName);
        cmdline.addOption("-o", *outTprFilename);
        return cmdline;
    }
};

class PcffClass2NonbondedDirectiveTest : public ::testing::Test
{
protected:
    TestFileManager fileManager_;
    std::string     mdpContentString_ =
            "title                   = PCFF class2 nonbonded directive test \n"
            "integrator              = md \n"
            "nsteps                  = 1 \n"
            "dt                      = 0.002 \n"
            "vdwtype                 = cutoff \n"
            "coulombtype             = cutoff \n"
            "tcoupl                  = no \n"
            "pcoupl                  = no \n"
            "pbc                     = xyz \n"
            "gen_vel                 = no \n";
    std::string groContentString_ =
            "pcff class2 nonbonded directives\n"
            "4\n"
            "    1RES     A1    1   0.000   0.000   0.000\n"
            "    1RES     A2    2   0.150   0.000   0.000\n"
            "    1RES     A3    3   0.300   0.000   0.000\n"
            "    1RES     A4    4   0.450   0.000   0.000\n"
            "   3.00000   3.00000   3.00000\n";

    std::string makeTopology(const std::string& defaultsLine) const
    {
        return std::string("[ defaults ]\n"
                           "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ rep-pow\n")
               + defaultsLine
               + std::string("\n"
                             "[ atomtypes ]\n"
                             "; name mass charge ptype sigma epsilon\n"
                             "A 12.011 0.0 A 0.34 0.12\n"
                             "B 12.011 0.0 A 0.30 0.08\n"
                             "\n"
                             "[ moleculetype ]\n"
                             "; Name nrexcl\n"
                             "MOL 3\n"
                             "\n"
                             "[ atoms ]\n"
                             "; nr type resnr residue atom cgnr charge mass\n"
                             "1 A 1 RES A1 1 0.0 12.011\n"
                             "2 B 1 RES A2 1 0.0 12.011\n"
                             "3 B 1 RES A3 1 0.0 12.011\n"
                             "4 B 1 RES A4 1 0.0 12.011\n"
                             "\n"
                             "[ bonds ]\n"
                             "1 2 1 0.153 1000\n"
                             "2 3 1 0.153 1000\n"
                             "3 4 1 0.153 1000\n"
                             "\n"
                             "[ pairs ]\n"
                             "1 4 1\n"
                             "\n"
                             "[ system ]\n"
                             "PCFF class2 nonbonded directive test\n"
                             "\n"
                             "[ molecules ]\n"
                             "MOL 1\n");
    }

    CommandLine makeCommandLine(const std::string& topContent, std::string* outTprFilename)
    {
        CommandLine cmdline;
        cmdline.addOption("grompp");

        const std::string mdpInputFileName = fileManager_.getTemporaryFilePath("pcff-class2-nb.mdp").string();
        const std::string groInputFileName = fileManager_.getTemporaryFilePath("pcff-class2-nb.gro").string();
        const std::string topInputFileName = fileManager_.getTemporaryFilePath("pcff-class2-nb.top").string();
        *outTprFilename                    = fileManager_.getTemporaryFilePath("pcff-class2-nb.tpr").string();

        gmx::TextWriter::writeFileFromString(mdpInputFileName, mdpContentString_);
        gmx::TextWriter::writeFileFromString(groInputFileName, groContentString_);
        gmx::TextWriter::writeFileFromString(topInputFileName, topContent);

        cmdline.addOption("-f", mdpInputFileName);
        cmdline.addOption("-c", groInputFileName);
        cmdline.addOption("-p", topInputFileName);
        cmdline.addOption("-o", *outTprFilename);
        return cmdline;
    }
};

TEST_F(GromppDirectiveTest, edgeCaseAtomTypeNames)
{
    CommandLine cmdline;
    cmdline.addOption("grompp");

    const std::string mdpInputFileName = fileManager_.getTemporaryFilePath("directives.mdp").string();
    gmx::TextWriter::writeFileFromString(mdpInputFileName, mdpContentString_);
    cmdline.addOption("-f", mdpInputFileName);


    cmdline.addOption("-c", TestFileManager::getInputFilePath("directives.gro").string());
    cmdline.addOption("-p", TestFileManager::getInputFilePath("directives.top").string());

    std::string outTprFilename = fileManager_.getTemporaryFilePath("directives.tpr").string();
    cmdline.addOption("-o", outTprFilename);

    ASSERT_EQ(0, gmx_grompp(cmdline.argc(), cmdline.argv()));
    {
        gmx_mtop_t top_after;
        t_inputrec ir_after;
        t_state    state;
        read_tpx_state(outTprFilename, &ir_after, &state, &top_after);

        int indexInMoltype = top_after.molblock[0].type;

        // Check atomic numbers (or lack thereof coded as -1)
        ASSERT_EQ(top_after.moltype[indexInMoltype].atoms.nr, 8);
        EXPECT_EQ(top_after.moltype[indexInMoltype].atoms.atom[0].atomnumber, -1);
        EXPECT_EQ(top_after.moltype[indexInMoltype].atoms.atom[1].atomnumber, 6);
        EXPECT_EQ(top_after.moltype[indexInMoltype].atoms.atom[2].atomnumber, 7);
        EXPECT_EQ(top_after.moltype[indexInMoltype].atoms.atom[3].atomnumber, -1);
        EXPECT_EQ(top_after.moltype[indexInMoltype].atoms.atom[4].atomnumber, -1);
        EXPECT_EQ(top_after.moltype[indexInMoltype].atoms.atom[5].atomnumber, 6);
        EXPECT_EQ(top_after.moltype[indexInMoltype].atoms.atom[6].atomnumber, 7);
        EXPECT_EQ(top_after.moltype[indexInMoltype].atoms.atom[7].atomnumber, -1);
    }
}

TEST_F(GromppDirectiveTest, NoteOnDihedralNotSumToZero)
{
    CommandLine cmdline;
    cmdline.addOption("grompp");

    std::string mdpString = mdpContentString_;
    mdpString += "define = -DDIHEDRAL_SUM_NOT_ZERO";

    const std::string mdpInputFileName = fileManager_.getTemporaryFilePath("directives.mdp").string();
    gmx::TextWriter::writeFileFromString(mdpInputFileName, mdpString);
    cmdline.addOption("-f", mdpInputFileName);


    cmdline.addOption("-c", TestFileManager::getInputFilePath("directives.gro").string());
    cmdline.addOption("-p", TestFileManager::getInputFilePath("directives.top").string());

    std::string outTprFilename = fileManager_.getTemporaryFilePath("directives.tpr").string();
    cmdline.addOption("-o", outTprFilename);

    // We cannot directly check printing of a note, but we at least check that it terminates
    // successfully.
    EXPECT_EQ(gmx_grompp(cmdline.argc(), cmdline.argv()), 0);
}

TEST_F(PcffClass2DirectiveTest, ParsesBondAngleDihedralAndImproperClass2Directives)
{
    const std::string bondedDirectives =
            "[ bonds ]\n"
            "; ai aj funct c0 c1 c2 c3\n"
            "1 2 11 0.153 250.0 -35.0 8.0\n"
            "\n"
            "[ angles ]\n"
            "; ai aj ak funct c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10\n"
            "1 2 3 11 109.5 35.0 -4.0 1.2 8.0 0.142 0.142 3.0 2.5 0.142 0.142\n"
            "\n"
            "[ dihedrals ]\n"
            "; ai aj ak al funct c0..c31 or c0..c7\n"
            "1 2 3 4 13 0.8 0.0 0.6 180.0 0.4 0.0 0.12 -0.08 0.04 1.50 0.10 -0.05 0.02 "
            "0.11 -0.03 0.01 1.50 1.50 0.06 -0.03 0.015 0.05 -0.025 0.01 112.0 112.0 0.20 "
            "112.0 112.0 0.18 1.50 1.50\n"
            "1 2 3 4 12 25.0 0.0 1.2 1.0 0.9 110.0 109.0 108.0\n";

    std::string outTprFilename;
    auto        cmdline = makeCommandLine(makeTopology(bondedDirectives), &outTprFilename);

    ASSERT_EQ(gmx_grompp(cmdline.argc(), cmdline.argv()), 0);

    gmx_mtop_t topAfter;
    t_inputrec irAfter;
    t_state    state;
    read_tpx_state(outTprFilename, &irAfter, &state, &topAfter);

    const int moltypeIndex = topAfter.molblock[0].type;
    const auto& moltype = topAfter.moltype[moltypeIndex];

    ASSERT_EQ(moltype.ilist[InteractionFunction::BondClass2].size(), 1 + NRAL(InteractionFunction::BondClass2));
    ASSERT_EQ(moltype.ilist[InteractionFunction::AngleClass2].size(), 1 + NRAL(InteractionFunction::AngleClass2));
    ASSERT_EQ(moltype.ilist[InteractionFunction::DihedralClass2].size(), 1 + NRAL(InteractionFunction::DihedralClass2));
    ASSERT_EQ(moltype.ilist[InteractionFunction::ImproperClass2].size(), 1 + NRAL(InteractionFunction::ImproperClass2));

    const int bondParamIndex = moltype.ilist[InteractionFunction::BondClass2].iatoms[0];
    ASSERT_EQ(topAfter.ffparams.functype[bondParamIndex], InteractionFunction::BondClass2);
    EXPECT_EQ(topAfter.ffparams.iparams[bondParamIndex].bond_class2.r0, 0.153_real);
    EXPECT_EQ(topAfter.ffparams.iparams[bondParamIndex].bond_class2.k2, 250.0_real);
    EXPECT_EQ(topAfter.ffparams.iparams[bondParamIndex].bond_class2.k3, -35.0_real);
    EXPECT_EQ(topAfter.ffparams.iparams[bondParamIndex].bond_class2.k4, 8.0_real);

    const int angleParamIndex = moltype.ilist[InteractionFunction::AngleClass2].iatoms[0];
    ASSERT_EQ(topAfter.ffparams.functype[angleParamIndex], InteractionFunction::AngleClass2);
    EXPECT_NEAR(topAfter.ffparams.iparams[angleParamIndex].angle_class2.theta0, 109.5_real * gmx::c_deg2Rad, 1e-6);
    EXPECT_EQ(topAfter.ffparams.iparams[angleParamIndex].angle_class2.k2, 35.0_real);
    EXPECT_EQ(topAfter.ffparams.iparams[angleParamIndex].angle_class2.bb_k, 8.0_real);
    EXPECT_EQ(topAfter.ffparams.iparams[angleParamIndex].angle_class2.ba_k1, 3.0_real);

    const int dihedralParamIndex = moltype.ilist[InteractionFunction::DihedralClass2].iatoms[0];
    ASSERT_EQ(topAfter.ffparams.functype[dihedralParamIndex], InteractionFunction::DihedralClass2);
    EXPECT_EQ(topAfter.ffparams.iparams[dihedralParamIndex].dihedral_class2.k1, 0.8_real);
    EXPECT_NEAR(topAfter.ffparams.iparams[dihedralParamIndex].dihedral_class2.phi2,
                180.0_real * gmx::c_deg2Rad,
                1e-6);
    EXPECT_EQ(topAfter.ffparams.iparams[dihedralParamIndex].dihedral_class2.mbt_f1, 0.12_real);
    EXPECT_EQ(topAfter.ffparams.iparams[dihedralParamIndex].dihedral_class2.ebt_f1_2, 0.11_real);
    EXPECT_NEAR(topAfter.ffparams.iparams[dihedralParamIndex].dihedral_class2.at_theta0_1,
                112.0_real * gmx::c_deg2Rad,
                1e-6);
    EXPECT_NEAR(topAfter.ffparams.iparams[dihedralParamIndex].dihedral_class2.aat_theta0_2,
                112.0_real * gmx::c_deg2Rad,
                1e-6);
    EXPECT_EQ(topAfter.ffparams.iparams[dihedralParamIndex].dihedral_class2.bb13t_k, 0.18_real);

    const int improperParamIndex = moltype.ilist[InteractionFunction::ImproperClass2].iatoms[0];
    ASSERT_EQ(topAfter.ffparams.functype[improperParamIndex], InteractionFunction::ImproperClass2);
    EXPECT_EQ(topAfter.ffparams.iparams[improperParamIndex].improper_class2.k0, 25.0_real);
    EXPECT_NEAR(topAfter.ffparams.iparams[improperParamIndex].improper_class2.chi0, 0.0_real, 1e-6);
    EXPECT_NEAR(topAfter.ffparams.iparams[improperParamIndex].improper_class2.aa_theta0_1,
                110.0_real * gmx::c_deg2Rad,
                1e-6);
    EXPECT_NEAR(topAfter.ffparams.iparams[improperParamIndex].improper_class2.aa_theta0_2,
                109.0_real * gmx::c_deg2Rad,
                1e-6);
    EXPECT_NEAR(topAfter.ffparams.iparams[improperParamIndex].improper_class2.aa_theta0_3,
                108.0_real * gmx::c_deg2Rad,
                1e-6);
}

TEST_F(PcffClass2DirectiveTest, ParsesBondAngleAndImproperClass2DirectivesWithoutDihedralClass2)
{
    const std::string bondedDirectives =
            "[ bonds ]\n"
            "; ai aj funct c0 c1 c2 c3\n"
            "1 2 11 0.153 250.0 -35.0 8.0\n"
            "\n"
            "[ angles ]\n"
            "; ai aj ak funct c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10\n"
            "1 2 3 11 109.5 35.0 -4.0 1.2 8.0 0.142 0.142 3.0 2.5 0.142 0.142\n"
            "\n"
            "[ dihedrals ]\n"
            "; ai aj ak al funct c0..c7\n"
            "1 2 3 4 12 25.0 0.0 1.2 1.0 0.9 110.0 109.0 108.0\n";

    std::string outTprFilename;
    auto        cmdline = makeCommandLine(makeTopology(bondedDirectives), &outTprFilename);

    ASSERT_EQ(gmx_grompp(cmdline.argc(), cmdline.argv()), 0);

    gmx_mtop_t topAfter;
    t_inputrec irAfter;
    t_state    state;
    read_tpx_state(outTprFilename, &irAfter, &state, &topAfter);

    const int   moltypeIndex = topAfter.molblock[0].type;
    const auto& moltype      = topAfter.moltype[moltypeIndex];

    ASSERT_EQ(moltype.ilist[InteractionFunction::BondClass2].size(), 1 + NRAL(InteractionFunction::BondClass2));
    ASSERT_EQ(moltype.ilist[InteractionFunction::AngleClass2].size(), 1 + NRAL(InteractionFunction::AngleClass2));
    EXPECT_EQ(moltype.ilist[InteractionFunction::DihedralClass2].size(), 0);
    ASSERT_EQ(moltype.ilist[InteractionFunction::ImproperClass2].size(), 1 + NRAL(InteractionFunction::ImproperClass2));

    const int bondParamIndex = moltype.ilist[InteractionFunction::BondClass2].iatoms[0];
    ASSERT_EQ(topAfter.ffparams.functype[bondParamIndex], InteractionFunction::BondClass2);
    EXPECT_EQ(topAfter.ffparams.iparams[bondParamIndex].bond_class2.r0, 0.153_real);
    EXPECT_EQ(topAfter.ffparams.iparams[bondParamIndex].bond_class2.k2, 250.0_real);
    EXPECT_EQ(topAfter.ffparams.iparams[bondParamIndex].bond_class2.k3, -35.0_real);
    EXPECT_EQ(topAfter.ffparams.iparams[bondParamIndex].bond_class2.k4, 8.0_real);

    const int angleParamIndex = moltype.ilist[InteractionFunction::AngleClass2].iatoms[0];
    ASSERT_EQ(topAfter.ffparams.functype[angleParamIndex], InteractionFunction::AngleClass2);
    EXPECT_NEAR(topAfter.ffparams.iparams[angleParamIndex].angle_class2.theta0, 109.5_real * gmx::c_deg2Rad, 1e-6);
    EXPECT_EQ(topAfter.ffparams.iparams[angleParamIndex].angle_class2.k2, 35.0_real);
    EXPECT_EQ(topAfter.ffparams.iparams[angleParamIndex].angle_class2.bb_k, 8.0_real);
    EXPECT_EQ(topAfter.ffparams.iparams[angleParamIndex].angle_class2.ba_k1, 3.0_real);

    const int improperParamIndex = moltype.ilist[InteractionFunction::ImproperClass2].iatoms[0];
    ASSERT_EQ(topAfter.ffparams.functype[improperParamIndex], InteractionFunction::ImproperClass2);
    EXPECT_EQ(topAfter.ffparams.iparams[improperParamIndex].improper_class2.k0, 25.0_real);
    EXPECT_NEAR(topAfter.ffparams.iparams[improperParamIndex].improper_class2.chi0, 0.0_real, 1e-6);
    EXPECT_NEAR(topAfter.ffparams.iparams[improperParamIndex].improper_class2.aa_theta0_1,
                110.0_real * gmx::c_deg2Rad,
                1e-6);
    EXPECT_NEAR(topAfter.ffparams.iparams[improperParamIndex].improper_class2.aa_theta0_2,
                109.0_real * gmx::c_deg2Rad,
                1e-6);
    EXPECT_NEAR(topAfter.ffparams.iparams[improperParamIndex].improper_class2.aa_theta0_3,
                108.0_real * gmx::c_deg2Rad,
                1e-6);
}

TEST_F(PcffClass2DirectiveTest, RejectsMalformedBondClass2Parameters)
{
    const std::string bondedDirectives =
            "[ bonds ]\n"
            "1 2 11 0.153 250.0 -35.0\n";

    std::string outTprFilename;
    auto        cmdline = makeCommandLine(makeTopology(bondedDirectives), &outTprFilename);

    GMX_EXPECT_DEATH_IF_SUPPORTED(gmx_grompp(cmdline.argc(), cmdline.argv()), "Incorrect number of parameters");
}

TEST_F(PcffClass2DirectiveTest, RejectsMalformedAngleClass2Parameters)
{
    const std::string bondedDirectives =
            "[ angles ]\n"
            "1 2 3 11 109.5 35.0 -4.0 1.2 8.0 0.142 0.142 3.0 2.5 0.142\n";

    std::string outTprFilename;
    auto        cmdline = makeCommandLine(makeTopology(bondedDirectives), &outTprFilename);

    GMX_EXPECT_DEATH_IF_SUPPORTED(gmx_grompp(cmdline.argc(), cmdline.argv()), "Incorrect number of parameters");
}

TEST_F(PcffClass2DirectiveTest, RejectsMalformedImproperClass2Parameters)
{
    const std::string bondedDirectives =
            "[ dihedrals ]\n"
            "1 2 3 4 12 25.0 0.0 1.2 1.0 0.9 110.0 109.0\n";

    std::string outTprFilename;
    auto        cmdline = makeCommandLine(makeTopology(bondedDirectives), &outTprFilename);

    GMX_EXPECT_DEATH_IF_SUPPORTED(gmx_grompp(cmdline.argc(), cmdline.argv()), "Incorrect number of parameters");
}

TEST_F(PcffClass2DirectiveTest, RejectsMalformedDihedralClass2Parameters)
{
    const std::string bondedDirectives =
            "[ dihedrals ]\n"
            "1 2 3 4 13 0.8 0.0 0.6 180.0 0.4 0.0 0.12 -0.08 0.04 1.50 0.10 -0.05 0.02 "
            "0.11 -0.03 0.01 1.50 1.50 0.06 -0.03 0.015 0.05 -0.025 0.01 112.0 112.0 0.20 "
            "112.0 112.0 0.18 1.50\n";

    std::string outTprFilename;
    auto        cmdline = makeCommandLine(makeTopology(bondedDirectives), &outTprFilename);

    GMX_EXPECT_DEATH_IF_SUPPORTED(gmx_grompp(cmdline.argc(), cmdline.argv()), "Incorrect number of parameters");
}

TEST_F(PcffClass2NonbondedDirectiveTest, ParsesSixthPowerMixingAndGeneratedOneFourPairs)
{
    std::string outTprFilename;
    auto        cmdline = makeCommandLine(makeTopology("1 4 yes 1.0 1.0 9.0\n"), &outTprFilename);

    ASSERT_EQ(gmx_grompp(cmdline.argc(), cmdline.argv()), 0);

    gmx_mtop_t topAfter;
    t_inputrec irAfter;
    t_state    state;
    read_tpx_state(outTprFilename, &irAfter, &state, &topAfter);

    EXPECT_DOUBLE_EQ(topAfter.ffparams.reppow, 9.0);

    const real sigmaA = 0.34_real;
    const real sigmaB = 0.30_real;
    const real epsilonA = 0.12_real;
    const real epsilonB = 0.08_real;
    const real sigmaMix = std::pow(0.5_real * (gmx::power6(sigmaA) + gmx::power6(sigmaB)), 1.0_real / 6.0_real);
    const real epsilonMix =
            2.0_real * std::sqrt(epsilonA * epsilonB) * std::sqrt(gmx::power6(sigmaA) * gmx::power6(sigmaB))
            / (gmx::power6(sigmaA) + gmx::power6(sigmaB));

    const int moltypeIndex = topAfter.molblock[0].type;
    const auto& moltype = topAfter.moltype[moltypeIndex];

    ASSERT_EQ(moltype.ilist[InteractionFunction::LennardJones14].size(),
              1 + NRAL(InteractionFunction::LennardJones14));
    EXPECT_EQ(moltype.ilist[InteractionFunction::LennardJones14].iatoms[1], 0);
    EXPECT_EQ(moltype.ilist[InteractionFunction::LennardJones14].iatoms[2], 3);
    EXPECT_EQ(moltype.ilist[InteractionFunction::LennardJonesCoulomb14Q].size(), 0);

    const int pairParamIndex = moltype.ilist[InteractionFunction::LennardJones14].iatoms[0];
    ASSERT_EQ(topAfter.ffparams.functype[pairParamIndex], InteractionFunction::LennardJones14);

    const auto& ip = topAfter.ffparams.iparams[pairParamIndex].lj14;
    EXPECT_NEAR(ip.c6A, 3.0_real * epsilonMix * gmx::power6(sigmaMix), 1e-6);
    EXPECT_NEAR(ip.c12A, 2.0_real * epsilonMix * std::pow(sigmaMix, 9.0_real), 1e-6);
    EXPECT_NEAR(ip.c6B, ip.c6A, 1e-6);
    EXPECT_NEAR(ip.c12B, ip.c12A, 1e-6);
}

TEST_F(PcffClass2NonbondedDirectiveTest, RejectsSixthPowerWithWrongRepulsionPower)
{
    std::string outTprFilename;
    auto        cmdline = makeCommandLine(makeTopology("1 4 yes 1.0 1.0 12.0\n"), &outTprFilename);

    GMX_EXPECT_DEATH_IF_SUPPORTED(gmx_grompp(cmdline.argc(), cmdline.argv()),
                                  "Combination rule SixthPower requires repulsion power 9");
}

TEST_F(GromppDirectiveTest, WarnOnDihedralSumDifferentForFreeEnergy)
{
    CommandLine cmdline;
    cmdline.addOption("grompp");

    std::string mdpString = mdpContentString_;
    mdpString +=
            "define = -DDIHEDRAL_SUM_DIFFERENT_STATEA_STATEB\n"
            "free-energy = yes\n"
            "init-lambda = 0.5";

    const std::string mdpInputFileName = fileManager_.getTemporaryFilePath("directives.mdp").string();
    gmx::TextWriter::writeFileFromString(mdpInputFileName, mdpString);
    cmdline.addOption("-f", mdpInputFileName);


    cmdline.addOption("-c", TestFileManager::getInputFilePath("directives.gro").string());
    cmdline.addOption("-p", TestFileManager::getInputFilePath("directives.top").string());

    std::string outTprFilename = fileManager_.getTemporaryFilePath("directives.tpr").string();
    cmdline.addOption("-o", outTprFilename);

    GMX_EXPECT_DEATH_IF_SUPPORTED(gmx_grompp(cmdline.argc(), cmdline.argv()),
                                  "undesired offset in dHdl values");
}

TEST_P(GromppDirectiveTest, LEaPImproperDihedralAtomReordering)
{
    CommandLine cmdline;
    cmdline.addOption("grompp");

    auto        testParam = GetParam();
    std::string mdpString = mdpContentString_;
    mdpString += std::get<0>(testParam);

    const std::string mdpInputFileName = fileManager_.getTemporaryFilePath("directives.mdp").string();
    gmx::TextWriter::writeFileFromString(mdpInputFileName, mdpString);
    cmdline.addOption("-f", mdpInputFileName);

    cmdline.addOption("-c", TestFileManager::getInputFilePath("directives.gro").string());
    cmdline.addOption("-p", TestFileManager::getInputFilePath("directives.top").string());

    std::string outTprFilename = fileManager_.getTemporaryFilePath("directives.tpr").string();
    cmdline.addOption("-o", outTprFilename);

    EXPECT_EQ(gmx_grompp(cmdline.argc(), cmdline.argv()), 0);
    {
        gmx_mtop_t top_after;
        t_inputrec ir_after;
        t_state    state;
        read_tpx_state(outTprFilename, &ir_after, &state, &top_after);

        int indexInMoltype = top_after.molblock[0].type;

        const int nDihedrals = 2;
        ASSERT_EQ(top_after.moltype[indexInMoltype]
                          .ilist[InteractionFunction::PeriodicImproperDihedrals]
                          .size(),
                  (1 + NRAL(InteractionFunction::PeriodicImproperDihedrals)) * nDihedrals);

        auto expectedAtomIndexes = std::get<1>(testParam);
        ASSERT_EQ(top_after.ffparams.functype[top_after.moltype[indexInMoltype]
                                                      .ilist[InteractionFunction::PeriodicImproperDihedrals]
                                                      .iatoms[0]],
                  InteractionFunction::PeriodicImproperDihedrals);
        ASSERT_EQ(top_after.moltype[indexInMoltype]
                          .ilist[InteractionFunction::PeriodicImproperDihedrals]
                          .iatoms[1],
                  expectedAtomIndexes[0][0]);
        ASSERT_EQ(top_after.moltype[indexInMoltype]
                          .ilist[InteractionFunction::PeriodicImproperDihedrals]
                          .iatoms[2],
                  expectedAtomIndexes[0][1]);
        ASSERT_EQ(top_after.moltype[indexInMoltype]
                          .ilist[InteractionFunction::PeriodicImproperDihedrals]
                          .iatoms[3],
                  expectedAtomIndexes[0][2]);
        ASSERT_EQ(top_after.moltype[indexInMoltype]
                          .ilist[InteractionFunction::PeriodicImproperDihedrals]
                          .iatoms[4],
                  expectedAtomIndexes[0][3]);

        ASSERT_EQ(top_after.ffparams.functype[top_after.moltype[indexInMoltype]
                                                      .ilist[InteractionFunction::PeriodicImproperDihedrals]
                                                      .iatoms[5]],
                  InteractionFunction::PeriodicImproperDihedrals);
        ASSERT_EQ(top_after.moltype[indexInMoltype]
                          .ilist[InteractionFunction::PeriodicImproperDihedrals]
                          .iatoms[6],
                  expectedAtomIndexes[1][0]);
        ASSERT_EQ(top_after.moltype[indexInMoltype]
                          .ilist[InteractionFunction::PeriodicImproperDihedrals]
                          .iatoms[7],
                  expectedAtomIndexes[1][1]);
        ASSERT_EQ(top_after.moltype[indexInMoltype]
                          .ilist[InteractionFunction::PeriodicImproperDihedrals]
                          .iatoms[8],
                  expectedAtomIndexes[1][2]);
        ASSERT_EQ(top_after.moltype[indexInMoltype]
                          .ilist[InteractionFunction::PeriodicImproperDihedrals]
                          .iatoms[9],
                  expectedAtomIndexes[1][3]);
    }
}

std::vector<std::tuple<std::string, std::array<std::array<int, 4>, 2>>> dihedralAtomOrderings = {
    { "define = -DIMPROPER_DIHEDRALS -DIMPROPER_DIHEDRAL_TYPE_NOX", { { { 0, 2, 1, 3 }, { 4, 6, 5, 7 } } } },
    { "define = -DIMPROPER_DIHEDRALS -DIMPROPER_DIHEDRAL_TYPE_X1", { { { 0, 2, 1, 3 }, { 4, 6, 5, 7 } } } },
    { "define = -DIMPROPER_DIHEDRALS -DIMPROPER_DIHEDRAL_TYPE_X2", { { { 0, 2, 1, 3 }, { 4, 6, 5, 7 } } } },
    { "define = -DIMPROPER_DIHEDRALS -DIMPROPER_DIHEDRAL_TYPE_X4", { { { 0, 2, 1, 3 }, { 4, 6, 5, 7 } } } },
    { "define = -DIMPROPER_DIHEDRALS -DIMPROPER_DIHEDRAL_TYPE_X12", { { { 0, 2, 1, 3 }, { 4, 6, 5, 7 } } } },
    { "define = -DIMPROPER_DIHEDRALS -DIMPROPER_DIHEDRAL_TYPE_X14", { { { 0, 2, 1, 3 }, { 4, 6, 5, 7 } } } },
    { "define = -DIMPROPER_DIHEDRALS -DIMPROPER_DIHEDRAL_TYPE_X24", { { { 0, 2, 1, 3 }, { 4, 6, 5, 7 } } } },
    { "define = -DIMPROPER_DIHEDRALS -DIMPROPER_DIHEDRAL_TYPE_X124", { { { 0, 2, 1, 3 }, { 4, 6, 5, 7 } } } },
    { "define = -DPRETEND_TO_BE_FF19SB -DIMPROPER_DIHEDRALS -DIMPROPER_DIHEDRAL_TYPE_NOX",
      { { { 0, 2, 1, 3 }, { 4, 6, 5, 7 } } } },
    { "define = -DPRETEND_TO_BE_FF19SB -DIMPROPER_DIHEDRALS -DIMPROPER_DIHEDRAL_TYPE_X1",
      { { { 0, 2, 1, 3 }, { 4, 6, 5, 7 } } } },
    { "define = -DPRETEND_TO_BE_FF19SB -DIMPROPER_DIHEDRALS -DIMPROPER_DIHEDRAL_TYPE_X2",
      { { { 0, 2, 1, 3 }, { 6, 4, 5, 7 } } } },
    { "define = -DPRETEND_TO_BE_FF19SB -DSTOP_PRETENDING -DIMPROPER_DIHEDRALS "
      "-DIMPROPER_DIHEDRAL_TYPE_X2",
      { { { 0, 2, 1, 3 }, { 4, 6, 5, 7 } } } },
    { "define = -DPRETEND_TO_BE_FF19SB -DIMPROPER_DIHEDRALS -DIMPROPER_DIHEDRAL_TYPE_X4",
      { { { 3, 1, 2, 0 }, { 7, 4, 6, 5 } } } },
    { "define = -DPRETEND_TO_BE_FF19SB -DSTOP_PRETENDING -DIMPROPER_DIHEDRALS "
      "-DIMPROPER_DIHEDRAL_TYPE_X4",
      { { { 3, 1, 2, 0 }, { 4, 6, 5, 7 } } } },
    { "define = -DPRETEND_TO_BE_FF19SB -DIMPROPER_DIHEDRALS -DIMPROPER_DIHEDRAL_TYPE_X12",
      { { { 0, 2, 1, 3 }, { 4, 6, 5, 7 } } } },
    { "define = -DPRETEND_TO_BE_FF19SB -DIMPROPER_DIHEDRALS -DIMPROPER_DIHEDRAL_TYPE_X14",
      { { { 0, 2, 1, 3 }, { 4, 7, 5, 6 } } } },
    { "define = -DPRETEND_TO_BE_FF19SB -DIMPROPER_DIHEDRALS -DIMPROPER_DIHEDRAL_TYPE_X24",
      { { { 3, 1, 2, 0 }, { 7, 4, 6, 5 } } } },
    { "define = -DPRETEND_TO_BE_FF19SB -DSTOP_PRETENDING -DIMPROPER_DIHEDRALS "
      "-DIMPROPER_DIHEDRAL_TYPE_X24",
      { { { 3, 1, 2, 0 }, { 4, 6, 5, 7 } } } },
    { "define = -DPRETEND_TO_BE_FF19SB -DIMPROPER_DIHEDRALS -DIMPROPER_DIHEDRAL_TYPE_X124",
      { { { 0, 2, 1, 3 }, { 4, 6, 5, 7 } } } },
};

INSTANTIATE_TEST_SUITE_P(DihedralAtomOrdering, GromppDirectiveTest, testing::ValuesIn(dihedralAtomOrderings));

class GromppCmapDirectiveTest :
    public ::testing::TestWithParam<std::tuple<std::string, ExpectedResult, std::string>>
{
public:
    GromppCmapDirectiveTest() = default;

protected:
    gmx::test::TestFileManager fileManager_;
    std::string                mdpContentString_ =
            "title                   = Directive edge case test \n"
            "integrator              = md \n"
            "nsteps                  = 1 \n"
            "dt                      = 0.002 \n"
            "vdwtype                 = cutoff \n"
            "coulombtype             = cutoff \n"
            "tcoupl                  = no \n"
            "pcoupl                  = no \n"
            "pbc                     = xyz \n"
            "gen_vel                 = yes \n";
};

TEST_P(GromppCmapDirectiveTest, AcceptValidAndErrorOnInvalidCMAP)
{
    auto testParam = GetParam();

    CommandLine cmdline;
    cmdline.addOption("grompp");

    std::string mdpString = mdpContentString_;
    mdpString += std::get<0>(testParam);

    const std::string mdpInputFileName =
            fileManager_.getTemporaryFilePath("directives-cmap.mdp").string();
    gmx::TextWriter::writeFileFromString(mdpInputFileName, mdpString);
    cmdline.addOption("-f", mdpInputFileName);


    cmdline.addOption("-c", TestFileManager::getInputFilePath("directives-cmap.gro").string());
    cmdline.addOption("-p", TestFileManager::getInputFilePath("directives-cmap.top").string());

    std::string outTprFilename = fileManager_.getTemporaryFilePath("directives-cmap.tpr").string();
    cmdline.addOption("-o", outTprFilename);

    switch (std::get<1>(testParam))
    {
        case ExpectedResult::Success:
            EXPECT_EQ(gmx_grompp(cmdline.argc(), cmdline.argv()), 0);
            break;
        case ExpectedResult::Death:
            GMX_EXPECT_DEATH_IF_SUPPORTED(gmx_grompp(cmdline.argc(), cmdline.argv()),
                                          std::get<2>(testParam));
            break;
        default: FAIL();
    }
}

std::vector<std::tuple<std::string, ExpectedResult, std::string>> cmapValidInputOutput = {
    { "",
      ExpectedResult::Death,
      "Unable to assign a cmap type to torsion between atoms 1 2 3 4 and 5" },
    { "define = -DNOT_A_CMAPTYPE",
      ExpectedResult::Death,
      "Unknown atomtype 1 found at position 5 in cmap type" },
    { "define = -DMATCHING_CMAPTYPE", ExpectedResult::Success, "" },
    { "define = -DMATCHING_CMAPTYPE_DOUBLESPACED", ExpectedResult::Success, "" },
    { "define = -DMATCHING_CMAPTYPE_PADDED", ExpectedResult::Success, "" },
    { "define = -DMATCHING_CMAPTYPE_TABBED", ExpectedResult::Success, "" },
    { "define = -DUNKNOWN_ATOMTYPE_IN_CMAPTYPE",
      ExpectedResult::Death,
      "Unknown bond_atomtype for Z in cmap atomtypes X Y X X Z" },
    { "define = -DTOO_MANY_ATOMTYPES_IN_CMAPTYPE",
      ExpectedResult::Death,
      "Invalid function type for cmap type: must be a number, found Y" },
    { "define = -DTOO_FEW_ATOMTYPES_IN_CMAPTYPE",
      ExpectedResult::Death,
      "Invalid function type for cmap type: must be 1" },
    { "define = -DINVALID_FUNCTYPE_IN_CMAPTYPE",
      ExpectedResult::Death,
      "Invalid function type for cmap type: must be 1" },
    { "define = -DRECTANGULAR_GRID_IN_CMAPTYPE",
      ExpectedResult::Death,
      "Not the same grid extent in x and y for cmap grid: x=2, y=3" },
    { "define = -DUNREAL_GRID_EXTENT_IN_CMAPTYPE",
      ExpectedResult::Death,
      "Invalid cmap-type grid extents in x and y dimensions: must be numbers,\n  found Tarydium" },
    { "define = -DTOO_FEW_GRID_PARAMETERS_IN_CMAPTYPE",
      ExpectedResult::Death,
      "Error in reading cmap parameter for atomtypes X Y X X Y: found 3,\n  expected 4" },
    { "define = -DTOO_MANY_GRID_PARAMETERS_IN_CMAPTYPE",
      ExpectedResult::Death,
      "One or more unread cmap parameters exist for atomtypes X Y X X Y" },
    { "define = -DUNREAL_GRID_PARAMETER_IN_CMAPTYPE",
      ExpectedResult::Death,
      "Invalid cmap parameters for atomtypes X Y X X Y: must be real numbers,\n  found Tarydium" },
    { "define = -DSOME_RESIDUE_NAMES_IN_CMAPTYPE",
      ExpectedResult::Death,
      "Incorrect format for cmap atomtypes X Y X X Y, residuetypes are required\n  for all 5 "
      "atomtypes or none" },
    { "define = -DMATCHING_RESIDUE_STARS_IN_CMAPTYPE", ExpectedResult::Success, "" },
    { "define = -DMATCHING_RESIDUE_NAMES_IN_CMAPTYPE", ExpectedResult::Success, "" },
    { "define = -DNONMATCHING_RESIDUE_NAMES_IN_CMAPTYPE",
      ExpectedResult::Death,
      "Unable to assign a cmap type to torsion between atoms 1 2 3 4 and 5" },
    { "define = -DNOT_A_CMAP_TORSION", ExpectedResult::Death, "Too few parameters on line" },
    { "define = -DINVALID_FUNCTYPE_IN_CMAP_TORSION",
      ExpectedResult::Death,
      "Invalid function type for cmap torsion: must be 1" },
    { "define = -DUSER_SPECIFIED_CMAPTYPE", ExpectedResult::Success, "" },
    { "define = -DUSER_SPECIFIED_CMAPTYPE_OUT_OF_BOUNDS",
      ExpectedResult::Death,
      "Unable to assign a cmap type to torsion between atoms 1 2 3 4 and 5" },
    { "define = -DALL_CMAP_TYPES_MUST_USE_SAME_GRID_EXTENT",
      ExpectedResult::Death,
      "each CMAP must have the same grid extent" },
};

INSTANTIATE_TEST_SUITE_P(CMAPDefinesAndErrors, GromppCmapDirectiveTest, testing::ValuesIn(cmapValidInputOutput));

} // namespace
} // namespace test
} // namespace gmx
