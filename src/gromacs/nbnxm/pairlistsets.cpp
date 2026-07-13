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
 * Implements functionality for PairlistSets.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_nbnxm
 */

#include "gmxpre.h"

#include "pairlistsets.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "gromacs/utility/gmxassert.h"

#include "atomdata.h"
#include "pairlistset.h"

namespace gmx
{

namespace
{

static bool plainPairlistCacheDisabledForTrace()
{
    static const bool disabled = [] {
        const char* pairWriteProofDir = std::getenv("GMX_PCFF_RESPA_PAIR_WRITE_PROOF_DIR");
        const char* ownershipTraceDir = std::getenv("GMX_PCFF_RESPA_OWNERSHIP_HANDOFF_TRACE_DIR");
        return (pairWriteProofDir != nullptr && *pairWriteProofDir != '\0')
               || (ownershipTraceDir != nullptr && *ownershipTraceDir != '\0');
    }();
    return disabled;
}

static bool plainPairlistCacheRequested()
{
    static const bool requested = [] {
        const char* value = std::getenv("GMX_PCFF_EXACT_RESPA_PLAIN_PAIRLIST_CACHE");
        return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
    }();
    return requested;
}

template<typename Value, typename Values>
static bool bitwiseMatchesSnapshot(const Values& values, const std::vector<Value>& snapshot)
{
    if (values.size() != snapshot.size())
    {
        return false;
    }
    if (values.size() == 0)
    {
        return true;
    }
    return std::memcmp(values.data(), snapshot.data(), values.size() * sizeof(Value)) == 0;
}

} // namespace

const PlainPairlist& PairlistSets::plainPairlist(const real              range,
                                                 const nbnxn_atomdata_t& nbat,
                                                 ArrayRef<const int>     atomIndices)
{
    GMX_RELEASE_ASSERT(includesAllPairs_ == true,
                       "We should have all pairs when getting a plain pairlist");

    const bool canUseCache = plainPairlistCacheRequested() && !plainPairlistCacheDisabledForTrace();
    if (canUseCache)
    {
        const auto x = nbat.x();
        if (plainPairlistCacheValid_ && range == plainPairlistRange_
            && bitwiseMatchesSnapshot<real>(x, plainPairlistCoordinateSnapshot_)
            && bitwiseMatchesSnapshot<RVec>(nbat.shift_vec, plainPairlistShiftVectorSnapshot_)
            && bitwiseMatchesSnapshot<int>(atomIndices, plainPairlistAtomIndicesSnapshot_))
        {
            return plainPairlist_;
        }
    }

    plainPairlist_.pairs.clear();
    plainPairlist_.excludedPairs.clear();

    localSet_->appendPlainPairlist(&plainPairlist_, range, nbat, atomIndices);

    if (nonlocalSet_)
    {
        nonlocalSet_->appendPlainPairlist(&plainPairlist_, range, nbat, atomIndices);
    }

    if (canUseCache)
    {
        const auto x       = nbat.x();
        plainPairlistRange_ = range;
        plainPairlistCoordinateSnapshot_.assign(x.begin(), x.end());
        plainPairlistShiftVectorSnapshot_.assign(nbat.shift_vec.begin(), nbat.shift_vec.end());
        plainPairlistAtomIndicesSnapshot_.assign(atomIndices.begin(), atomIndices.end());
        plainPairlistCacheValid_ = true;
    }
    else
    {
        invalidatePlainPairlistCache();
    }

    return plainPairlist_;
}

} // namespace gmx
