/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 1991- The GROMACS Authors
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
/* This file is completely threadsafe - keep it that way! */
#include "gmxpre.h"

#include "gpp_nextnb.h"

#include <cstdlib>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "gromacs/gmxpreprocess/grompp_impl.h"
#include "gromacs/gmxpreprocess/toputil.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/listoflists.h"
#include "gromacs/utility/smalloc.h"

/* #define DEBUG_NNB */

typedef struct
{
    int ai, aj;
} sortable;

static int bond_sort(const void* a, const void* b)
{
    const sortable *sa, *sb;

    sa = reinterpret_cast<const sortable*>(a);
    sb = reinterpret_cast<const sortable*>(b);

    if (sa->ai == sb->ai)
    {
        return (sa->aj - sb->aj);
    }
    else
    {
        return (sa->ai - sb->ai);
    }
}

static int compare_int(const void* a, const void* b)
{
    return (*reinterpret_cast<const int*>(a) - *reinterpret_cast<const int*>(b));
}

namespace
{

constexpr int c_targetAtomI  = 0;
constexpr int c_targetAtomJ  = 1;
constexpr int c_controlAtomI = 0;
constexpr int c_controlAtomJ = 4;

void appendOwnershipTraceLine(const char* traceDirPath, const std::string& filename, const std::string& line)
{
    GMX_RELEASE_ASSERT(traceDirPath != nullptr && *traceDirPath != '\0',
                       "Need a valid ownership handoff trace directory");

    std::filesystem::path traceDir(traceDirPath);
    std::filesystem::create_directories(traceDir);
    std::filesystem::path outputPath = traceDir / filename;

    FILE* dumpFile = std::fopen(outputPath.string().c_str(), "a");
    if (dumpFile == nullptr)
    {
        gmx_fatal(FARGS, "Could not open ownership handoff trace output '%s'", outputPath.string().c_str());
    }
    std::fprintf(dumpFile, "%s\n", line.c_str());
    std::fclose(dumpFile);
}

void appendOwnershipHandoffTraceLine(const char* traceDirPath, const std::string& line)
{
    appendOwnershipTraceLine(traceDirPath, "step0_grompp_generate_excl_trace.txt", line);
}

std::string joinExclusionList(const gmx::ArrayRef<const int> exclusionsForAtom)
{
    std::string result;
    for (gmx::Index index = 0; index < exclusionsForAtom.ssize(); ++index)
    {
        if (!result.empty())
        {
            result += ",";
        }
        result += std::to_string(exclusionsForAtom[index]);
    }
    return result.empty() ? std::string("none") : result;
}

bool listContains(const gmx::ArrayRef<const int> values, const int target)
{
    return std::find(values.begin(), values.end(), target) != values.end();
}

bool matchesDirectedPair(const int ai, const int aj, const int expectedAi, const int expectedAj)
{
    return ai == expectedAi && aj == expectedAj;
}

int countDirectedPairEntries(const sortable* sortables, const int count, const int expectedAi, const int expectedAj)
{
    int matches = 0;
    for (int index = 0; index < count; ++index)
    {
        if (matchesDirectedPair(sortables[index].ai, sortables[index].aj, expectedAi, expectedAj))
        {
            matches++;
        }
    }
    return matches;
}

std::string joinNnbBucket(const t_nextnb* nnb, const int atom, const int order)
{
    std::string result;
    for (int index = 0; index < nnb->nrexcl[atom][order]; ++index)
    {
        if (!result.empty())
        {
            result += ",";
        }
        result += std::to_string(nnb->a[atom][order][index]);
    }
    return result.empty() ? std::string("none") : result;
}

bool nnbBucketContains(const t_nextnb* nnb, const int atom, const int order, const int target)
{
    for (int index = 0; index < nnb->nrexcl[atom][order]; ++index)
    {
        if (nnb->a[atom][order][index] == target)
        {
            return true;
        }
    }
    return false;
}

} // namespace


#ifdef DEBUG
#    define prints(str, n, s) __prints(str, n, s)
static void __prints(char* str, int n, sortable* s)
{
    int i;

    if (debug)
    {
        fprintf(debug, "%s\n", str);
        fprintf(debug, "Sortables \n");
        for (i = 0; (i < n); i++)
        {
            fprintf(debug, "%d\t%d\n", s[i].ai, s[i].aj);
        }

        std::fflush(debug);
    }
}
#else
#    define prints(str, n, s)
#endif

void init_nnb(t_nextnb* nnb, int nr, int nrex)
{
    int i;

    /* initiate nnb */
    nnb->nr   = nr;
    nnb->nrex = nrex;

    snew(nnb->a, nr);
    snew(nnb->nrexcl, nr);
    for (i = 0; (i < nr); i++)
    {
        snew(nnb->a[i], nrex + 1);
        snew(nnb->nrexcl[i], nrex + 1);
    }
}

static void add_nnb(t_nextnb* nnb, int nre, int i, int j)
{
    srenew(nnb->a[i][nre], nnb->nrexcl[i][nre] + 1);
    nnb->a[i][nre][nnb->nrexcl[i][nre]] = j;
    nnb->nrexcl[i][nre]++;
}

void done_nnb(t_nextnb* nnb)
{
    int i, nre;

    for (i = 0; (i < nnb->nr); i++)
    {
        for (nre = 0; (nre <= nnb->nrex); nre++)
        {
            if (nnb->nrexcl[i][nre] > 0)
            {
                sfree(nnb->a[i][nre]);
            }
        }
        sfree(nnb->nrexcl[i]);
        sfree(nnb->a[i]);
    }
    sfree(nnb->a);
    sfree(nnb->nrexcl);
    nnb->nr   = 0;
    nnb->nrex = 0;
}

#ifdef DEBUG_NNB
void __print_nnb(t_nextnb* nnb, char* s)
{
    int i, j, k;

    if (debug)
    {
        fprintf(debug, "%s\n", s);
        fprintf(debug, "nnb->nr: %d\n", nnb->nr);
        fprintf(debug, "nnb->nrex: %d\n", nnb->nrex);
        for (i = 0; (i < nnb->nr); i++)
        {
            for (j = 0; (j <= nnb->nrex); j++)
            {
                fprintf(debug, "nrexcl[%d][%d]: %d, excl: ", i, j, nnb->nrexcl[i][j]);
                for (k = 0; (k < nnb->nrexcl[i][j]); k++)
                {
                    fprintf(debug, "%d, ", nnb->a[i][j][k]);
                }
                fprintf(debug, "\n");
            }
        }
    }
}
#endif

static void nnb2excl(t_nextnb* nnb, gmx::ListOfLists<int>* excls)
{
    int       i, j, j_index;
    int       nre, nrx, nrs, nr_of_sortables;
    sortable* s;
    const char* traceDirPath = std::getenv("GMX_PCFF_RESPA_OWNERSHIP_HANDOFF_TRACE_DIR");
    static bool dumpedTargetFlattenTrace = false;
    static bool dumpedTargetEmitTrace    = false;
    static bool dumpedControlFlattenTrace = false;

    excls->clear();

    for (i = 0; (i < nnb->nr); i++)
    {
        /* calculate the total number of exclusions for atom i */
        nr_of_sortables = 0;
        for (nre = 0; (nre <= nnb->nrex); nre++)
        {
            nr_of_sortables += nnb->nrexcl[i][nre];
        }

        /* make space for sortable array */
        snew(s, nr_of_sortables);

        /* fill the sortable array and sort it */
        nrs = 0;
        for (nre = 0; (nre <= nnb->nrex); nre++)
        {
            for (nrx = 0; (nrx < nnb->nrexcl[i][nre]); nrx++)
            {
                s[nrs].ai = i;
                s[nrs].aj = nnb->a[i][nre][nrx];
                nrs++;
            }
        }
        if (nrs != nr_of_sortables)
        {
            gmx_incons("Generating exclusions");
        }
        if (traceDirPath != nullptr && *traceDirPath != '\0' && i == c_targetAtomI)
        {
            const int targetCount  = countDirectedPairEntries(s, nr_of_sortables, c_targetAtomI, c_targetAtomJ);
            const int controlCount = countDirectedPairEntries(s, nr_of_sortables, c_controlAtomI, c_controlAtomJ);
            if (!dumpedTargetFlattenTrace)
            {
                appendOwnershipTraceLine(
                        traceDirPath,
                        "step0_generate_excl_internal_trace.txt",
                        "stage=nnb2excl_flattened atom=0 nr_of_sortables=" + std::to_string(nr_of_sortables)
                                + " target_present=" + std::string(targetCount > 0 ? "true" : "false")
                                + " target_count=" + std::to_string(targetCount) + " control_present="
                                + std::string(controlCount > 0 ? "true" : "false") + " control_count="
                                + std::to_string(controlCount));
                dumpedTargetFlattenTrace = true;
            }
            if (!dumpedControlFlattenTrace)
            {
                appendOwnershipTraceLine(
                        traceDirPath,
                        "step0_generate_excl_internal_trace.txt",
                        "stage=nnb2excl_control_check atom=0 control_present="
                                + std::string(controlCount > 0 ? "true" : "false") + " control_count="
                                + std::to_string(controlCount));
                dumpedControlFlattenTrace = true;
            }
        }
        prints("nnb2excl before qsort", nr_of_sortables, s);
        if (nr_of_sortables > 1)
        {
            std::qsort(s, nr_of_sortables, static_cast<size_t>(sizeof(s[0])), bond_sort);
            prints("nnb2excl after qsort", nr_of_sortables, s);
        }

        /* remove duplicate entries from the list */
        j_index = 0;
        if (nr_of_sortables > 0)
        {
            for (j = 1; (j < nr_of_sortables); j++)
            {
                if ((s[j].ai != s[j - 1].ai) || (s[j].aj != s[j - 1].aj))
                {
                    s[j_index++] = s[j - 1];
                }
            }
            s[j_index++] = s[j - 1];
        }
        nr_of_sortables = j_index;
        prints("after rm-double", j_index, s);

        /* put the sorted exclusions in the target list */
        excls->pushBackListOfSize(nr_of_sortables);
        gmx::ArrayRef<int> exclusionsForAtom = excls->back();
        for (nrs = 0; (nrs < nr_of_sortables); nrs++)
        {
            exclusionsForAtom[nrs] = s[nrs].aj;
        }
        if (!dumpedTargetEmitTrace && traceDirPath != nullptr && *traceDirPath != '\0' && i == c_targetAtomI)
        {
            appendOwnershipTraceLine(
                    traceDirPath,
                    "step0_generate_excl_internal_trace.txt",
                    "stage=nnb2excl_emit atom=0 emitted=" + joinExclusionList(exclusionsForAtom)
                            + " target_output_index="
                            + std::to_string(
                                    static_cast<int>(std::find(exclusionsForAtom.begin(),
                                                               exclusionsForAtom.end(),
                                                               c_targetAtomJ)
                                                     - exclusionsForAtom.begin()))
                            + " control_output_index="
                            + std::to_string(
                                    std::find(exclusionsForAtom.begin(),
                                              exclusionsForAtom.end(),
                                              c_controlAtomJ)
                                            == exclusionsForAtom.end()
                                            ? -1
                                            : static_cast<int>(std::find(exclusionsForAtom.begin(),
                                                                         exclusionsForAtom.end(),
                                                                         c_controlAtomJ)
                                                               - exclusionsForAtom.begin()))
                            + " target_present="
                            + std::string(listContains(exclusionsForAtom, c_targetAtomJ) ? "true" : "false")
                            + " control_present="
                            + std::string(listContains(exclusionsForAtom, c_controlAtomJ) ? "true" : "false"));
            dumpedTargetEmitTrace = true;
        }

        /* cleanup temporary space */
        sfree(s);
    }
}

/*! \brief Return true of neighbor is already present in some exclusion level
 *
 * To avoid exploding complexity when processing exclusions for highly
 * connected molecules with lots of exclusions, this routine is used to
 * check whether a particular neighbor has already been excluded at any lower
 * bond distance, in which case we should not add it to avoid creating loops.
 *
 * \param nnb            Valid initialized next-neighbor structure
 * \param atom           The host atom whose neighbors we are searching
 * \param highest_order  The highest-rank neighbor list to search.
 * \param query          Atom index to look for
 *
 * \return True if query is present as an exclusion of up to highest_order
 *         (inclusive) from atom. For instance, if highest_order is 2,
 *         the routine will return true if the query atom is already listed as
 *         first or second neighbor (exclusion) in nnb.
 */
static bool atom_is_present_in_nnb(const t_nextnb* nnb, int atom, int highest_order, int query)
{
    GMX_RELEASE_ASSERT(highest_order < nnb->nrex, "Inconsistent nnb search parameters");

    for (int order = 0; order <= highest_order; order++)
    {
        for (int m = 0; m < nnb->nrexcl[atom][order]; m++)
        {
            if (nnb->a[atom][order][m] == query)
            {
                return true;
            }
        }
    }
    return false;
}

static void do_gen(int       nrbonds, /* total number of bonds in s	*/
                   sortable* s,       /* bidirectional list of bonds    */
                   t_nextnb* nnb)     /* the tmp storage for excl     */
/* Assume excl is initialised and s[] contains all bonds bidirectional */
{
    int i, j, k, n, nb;
    const char* traceDirPath = std::getenv("GMX_PCFF_RESPA_OWNERSHIP_HANDOFF_TRACE_DIR");
    static bool dumpedTargetBondRuleFire  = false;
    static bool dumpedControlBondRuleFire = false;
    bool        targetDirectBondRuleFired = false;
    bool        controlDirectBondRuleFired = false;
    bool        targetHigherOrderRuleFired = false;
    bool        controlHigherOrderRuleFired = false;

    /* exclude self */
    for (i = 0; (i < nnb->nr); i++)
    {
        add_nnb(nnb, 0, i, i);
    }
    print_nnb(nnb, "After exclude self");

    /* exclude all the bonded atoms */
    if (nnb->nrex > 0)
    {
        for (i = 0; (i < nrbonds); i++)
        {
            if (matchesDirectedPair(s[i].ai, s[i].aj, c_targetAtomI, c_targetAtomJ))
            {
                targetDirectBondRuleFired = true;
                if (!dumpedTargetBondRuleFire && traceDirPath != nullptr && *traceDirPath != '\0')
                {
                    appendOwnershipTraceLine(
                            traceDirPath,
                            "step0_generate_excl_internal_trace.txt",
                            "stage=do_gen_direct_bond_rule_fire ai=0 aj=1 source=exclude_all_bonded_atoms "
                                    "condition_nrex_positive=true bond_list_index="
                                    + std::to_string(i) + " nre_bucket=1 add_nnb_called=true");
                    dumpedTargetBondRuleFire = true;
                }
            }
            if (matchesDirectedPair(s[i].ai, s[i].aj, c_controlAtomI, c_controlAtomJ))
            {
                controlDirectBondRuleFired = true;
                if (!dumpedControlBondRuleFire && traceDirPath != nullptr && *traceDirPath != '\0')
                {
                    appendOwnershipTraceLine(
                            traceDirPath,
                            "step0_generate_excl_internal_trace.txt",
                            "stage=do_gen_direct_bond_control_fire ai=0 aj=4 source=exclude_all_bonded_atoms "
                                    "condition_nrex_positive=true bond_list_index="
                                    + std::to_string(i) + " nre_bucket=1 add_nnb_called=true");
                    dumpedControlBondRuleFire = true;
                }
            }
            add_nnb(nnb, 1, s[i].ai, s[i].aj);
        }
    }
    print_nnb(nnb, "After exclude bonds");

    /* for the nr of exclusions per atom */
    for (n = 1; (n < nnb->nrex); n++)
    {
        /* now for all atoms */
        for (i = 0; (i < nnb->nr); i++)
        {
            /* for all directly bonded atoms of atom i */
            for (j = 0; (j < nnb->nrexcl[i][1]); j++)
            {

                /* store the 1st neighbour in nb */
                nb = nnb->a[i][1][j];

                /* store all atoms in nb's n-th list into i's n+1-th list */
                for (k = 0; (k < nnb->nrexcl[nb][n]); k++)
                {
                    if (i == c_targetAtomI && nnb->a[nb][n][k] == c_targetAtomJ)
                    {
                        targetHigherOrderRuleFired = true;
                    }
                    if (i == c_controlAtomI && nnb->a[nb][n][k] == c_controlAtomJ)
                    {
                        controlHigherOrderRuleFired = true;
                    }
                    // Only add if it is not already present as a closer neighbor
                    // to avoid exploding complexity for highly connected molecules
                    // with high exclusion order
                    if (!atom_is_present_in_nnb(nnb, i, n, nnb->a[nb][n][k]))
                    {
                        add_nnb(nnb, n + 1, i, nnb->a[nb][n][k]);
                    }
                }
            }
        }
    }
    print_nnb(nnb, "After exclude rest");
    if (traceDirPath != nullptr && *traceDirPath != '\0')
    {
        appendOwnershipTraceLine(
                traceDirPath,
                "step0_generate_excl_internal_trace.txt",
                "stage=do_gen_summary target_direct_bond_rule_fired="
                        + std::string(targetDirectBondRuleFired ? "true" : "false")
                        + " target_higher_order_rule_fired="
                        + std::string(targetHigherOrderRuleFired ? "true" : "false")
                        + " control_direct_bond_rule_fired="
                        + std::string(controlDirectBondRuleFired ? "true" : "false")
                        + " control_higher_order_rule_fired="
                        + std::string(controlHigherOrderRuleFired ? "true" : "false"));
    }
}

static void add_b(InteractionsOfType* bonds, int* nrf, sortable* s)
{
    int i = 0;
    for (const auto& bond : bonds->interactionTypes)
    {
        int ai = bond.ai();
        int aj = bond.aj();
        if ((ai < 0) || (aj < 0))
        {
            gmx_fatal(FARGS, "Impossible atom numbers in bond %d: ai=%d, aj=%d", i, ai, aj);
        }
        /* Add every bond twice */
        s[(*nrf)].ai   = ai;
        s[(*nrf)++].aj = aj;
        s[(*nrf)].aj   = ai;
        s[(*nrf)++].ai = aj;
        i++;
    }
}

void gen_nnb(t_nextnb* nnb, gmx::EnumerationArray<InteractionFunction, InteractionsOfType>& plist)
{
    sortable* s;
    int       nrbonds, nrf;
    const char* traceDirPath = std::getenv("GMX_PCFF_RESPA_OWNERSHIP_HANDOFF_TRACE_DIR");
    static bool dumpedGenNnbBondMembership = false;

    nrbonds = 0;
    for (const auto i : gmx::EnumerationWrapper<InteractionFunction>{})
    {
        if (IS_CHEMBOND(i))
        {
            /* we need every bond twice (bidirectional) */
            nrbonds += 2 * plist[i].size();
        }
    }

    snew(s, nrbonds);

    nrf = 0;
    for (const auto i : gmx::EnumerationWrapper<InteractionFunction>{})
    {
        if (IS_CHEMBOND(i))
        {
            add_b(&plist[i], &nrf, s);
        }
    }

    /* now sort the bonds */
    prints("gen_excl before qsort", nrbonds, s);
    if (nrbonds > 1)
    {
        std::qsort(s, nrbonds, static_cast<size_t>(sizeof(sortable)), bond_sort);
        prints("gen_excl after qsort", nrbonds, s);
    }
    if (!dumpedGenNnbBondMembership && traceDirPath != nullptr && *traceDirPath != '\0')
    {
        appendOwnershipTraceLine(
                traceDirPath,
                "step0_generate_excl_internal_trace.txt",
                "stage=gen_nnb_bond_membership nrbonds=" + std::to_string(nrbonds)
                        + " target_present="
                        + std::string(countDirectedPairEntries(s, nrbonds, c_targetAtomI, c_targetAtomJ) > 0
                                              ? "true"
                                              : "false")
                        + " target_count="
                        + std::to_string(countDirectedPairEntries(s, nrbonds, c_targetAtomI, c_targetAtomJ))
                        + " control_present="
                        + std::string(countDirectedPairEntries(s, nrbonds, c_controlAtomI, c_controlAtomJ) > 0
                                              ? "true"
                                              : "false")
                        + " control_count="
                        + std::to_string(countDirectedPairEntries(s, nrbonds, c_controlAtomI, c_controlAtomJ)));
        dumpedGenNnbBondMembership = true;
    }

    do_gen(nrbonds, s, nnb);
    sfree(s);
}

static void sort_and_purge_nnb(t_nextnb* nnb)
{
    int  i, j, k, m, n, cnt, prev, idx;
    bool found;
    const char* traceDirPath = std::getenv("GMX_PCFF_RESPA_OWNERSHIP_HANDOFF_TRACE_DIR");
    static bool dumpedSortAndPurgeTrace = false;

    for (i = 0; (i < nnb->nr); i++)
    {
        for (n = 0; (n <= nnb->nrex); n++)
        {
            /* Sort atoms in this list */
            if (nnb->nrexcl[i][n] > 0)
            {
                std::qsort(nnb->a[i][n], nnb->nrexcl[i][n], sizeof(int), compare_int);
            }
            cnt  = 0;
            prev = -1;
            for (j = 0; j < nnb->nrexcl[i][n]; j++)
            {
                idx = nnb->a[i][n][j];

                found = false;
                for (m = 0; m < n && !found; m++)
                {
                    for (k = 0; k < nnb->nrexcl[i][m] && !found; k++)
                    {
                        found = idx == nnb->a[i][m][k];
                    }
                }

                if (!found && nnb->a[i][n][j] != prev)
                {
                    nnb->a[i][n][cnt] = nnb->a[i][n][j];
                    prev              = nnb->a[i][n][cnt];
                    cnt++;
                }
            }
            nnb->nrexcl[i][n] = cnt;
        }
        if (!dumpedSortAndPurgeTrace && traceDirPath != nullptr && *traceDirPath != '\0' && i == c_targetAtomI)
        {
            appendOwnershipTraceLine(
                    traceDirPath,
                    "step0_generate_excl_internal_trace.txt",
                    "stage=sort_and_purge_output atom=0 level0=" + joinNnbBucket(nnb, i, 0)
                            + " level1=" + joinNnbBucket(nnb, i, 1) + " level2="
                            + joinNnbBucket(nnb, i, 2) + " level3="
                            + joinNnbBucket(nnb, i, 3) + " target_in_level1="
                            + std::string(nnbBucketContains(nnb, i, 1, c_targetAtomJ) ? "true" : "false")
                            + " control_in_any_level="
                            + std::string((nnbBucketContains(nnb, i, 0, c_controlAtomJ)
                                           || nnbBucketContains(nnb, i, 1, c_controlAtomJ)
                                           || nnbBucketContains(nnb, i, 2, c_controlAtomJ)
                                           || nnbBucketContains(nnb, i, 3, c_controlAtomJ))
                                                  ? "true"
                                                  : "false"));
            dumpedSortAndPurgeTrace = true;
        }
    }
}


void generate_excl(int                                                             nrexcl,
                   int                                                             nratoms,
                   gmx::EnumerationArray<InteractionFunction, InteractionsOfType>& plist,
                   gmx::ListOfLists<int>*                                          excls)
{
    t_nextnb nnb;
    if (nrexcl < 0)
    {
        gmx_fatal(FARGS, "Can't have %d exclusions...", nrexcl);
    }
    init_nnb(&nnb, nratoms, nrexcl);
    gen_nnb(&nnb, plist);
    sort_and_purge_nnb(&nnb);
    nnb2excl(&nnb, excls);
    const char* ownershipTraceDirPath = std::getenv("GMX_PCFF_RESPA_OWNERSHIP_HANDOFF_TRACE_DIR");
    static bool dumpedOwnershipTrace  = false;
    if (!dumpedOwnershipTrace && ownershipTraceDirPath != nullptr && *ownershipTraceDirPath != '\0'
        && excls->ssize() > 0)
    {
        const auto    exclusionsForAtom0 = (*excls)[c_targetAtomI];
        appendOwnershipHandoffTraceLine(
                ownershipTraceDirPath,
                "stage=generate_excl_output atom=0 nrexcl=" + std::to_string(nrexcl) + " exclusions="
                        + joinExclusionList(exclusionsForAtom0) + " contains_target="
                        + std::string(listContains(exclusionsForAtom0, c_targetAtomJ) ? "true" : "false")
                        + " contains_control="
                        + std::string(listContains(exclusionsForAtom0, c_controlAtomJ) ? "true" : "false"));
        dumpedOwnershipTrace = true;
    }
    done_nnb(&nnb);
}
