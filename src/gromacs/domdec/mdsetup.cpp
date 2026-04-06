/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2016- The GROMACS Authors
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
#include "gmxpre.h"

#include "mdsetup.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/domdec/localtopology.h"
#include "gromacs/ewald/pme.h"
#include "gromacs/listed_forces/listed_forces.h"
#include "gromacs/mdlib/constr.h"
#include "gromacs/mdlib/mdatoms.h"
#include "gromacs/mdlib/vsite.h"
#include "gromacs/mdlib/wholemoleculetransform.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/forcebuffers.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/gmxassert.h"

namespace gmx
{

static void appendOwnershipTraceLineMdsetup(const std::string& filePath, const std::string& line)
{
    if (FILE* file = std::fopen(filePath.c_str(), "a"))
    {
        std::fputs(line.c_str(), file);
        std::fputc('\n', file);
        std::fclose(file);
    }
}

/* TODO: Add a routine that collects the initial setup of the algorithms.
 *
 * The final solution should be an MD algorithm base class with methods
 * for initialization and atom-data setup.
 */
void mdAlgorithmsSetupAtomData(const gmx_domdec_t*  dd,
                               const t_inputrec&    inputrec,
                               const gmx_mtop_t&    top_global,
                               gmx_localtop_t*      top,
                               t_forcerec*          fr,
                               ForceBuffers*        force,
                               MDAtoms*             mdAtoms,
                               Constraints*         constr,
                               VirtualSitesHandler* vsite,
                               gmx_shellfc_t*       shellfc)
{
    int numAtomIndex;
    int numHomeAtoms;
    int numTotalAtoms;

    if (dd)
    {
        numAtomIndex  = dd_natoms_mdatoms(*dd);
        numHomeAtoms  = dd_numHomeAtoms(*dd);
        numTotalAtoms = dd_natoms_mdatoms(*dd);
    }
    else
    {
        numAtomIndex  = -1;
        numHomeAtoms  = top_global.natoms;
        numTotalAtoms = top_global.natoms;
    }

    if (force != nullptr)
    {
        force->resize(numTotalAtoms);
    }

    ArrayRef<const int> globalAtomIndices;
    if (dd)
    {
        globalAtomIndices = dd->globalAtomIndices;
    }
    atoms2md(top_global, inputrec, numAtomIndex, globalAtomIndices, numHomeAtoms, mdAtoms);

    t_mdatoms* mdatoms = mdAtoms->mdatoms();
    if (dd == nullptr)
    {
        gmx_mtop_generate_local_top(top_global, top, inputrec.efep != FreeEnergyPerturbationType::No);

        const char* ownershipTraceDirPath = std::getenv("GMX_PCFF_RESPA_OWNERSHIP_HANDOFF_TRACE_DIR");
        if (ownershipTraceDirPath != nullptr && *ownershipTraceDirPath != '\0')
        {
            auto joinIndexList = [](const auto& values)
            {
                std::string joined;
                for (int value : values)
                {
                    if (!joined.empty())
                    {
                        joined += ",";
                    }
                    joined += std::to_string(value);
                }
                if (joined.empty())
                {
                    joined = "none";
                }
                return joined;
            };

            const std::string globalMolTypeExclusions =
                    (top_global.moltype.empty() || top_global.moltype.front().excls.empty())
                            ? "none"
                            : joinIndexList(top_global.moltype.front().excls[0]);
            const std::string localTopExclusions =
                    top->excls.empty() ? "none" : joinIndexList(top->excls.front());
            const std::string intermolecularExclusions =
                    joinIndexList(top_global.intermolecularExclusionGroup);

            appendOwnershipTraceLineMdsetup(
                    std::string(ownershipTraceDirPath) + "/step0_mdsetup_local_topology_trace.txt",
                    "stage=post_generate_local_top dd=false moltype0_atom0_exclusions="
                            + globalMolTypeExclusions + " intermolecular_exclusion_group="
                            + intermolecularExclusions + " local_top_atom0_exclusions="
                            + localTopExclusions + " use_gpu_nonbonded=pending");
        }
    }

    if (fr->wholeMoleculeTransform && dd)
    {
        fr->wholeMoleculeTransform->updateAtomOrder(dd->globalAtomIndices, *dd->ga2la);
    }

    if (vsite)
    {
        vsite->setVirtualSites(top->idef.il, mdatoms->nr, mdatoms->homenr, mdatoms->ptype);
    }

    /* Note that with DD only flexible constraints, not shells, are supported
     * and these don't require setup in make_local_shells().
     *
     * TODO: This should only happen in ShellFCElement (it is called directly by the modular
     *       simulator ShellFCElement already, but still used here by legacy simulators)
     */
    if (dd == nullptr && shellfc)
    {
        make_local_shells(dd, *mdatoms, shellfc);
    }

    // TODO: warning/error if posresCom and posresComB do not have the same size
    for (auto& listedForces : fr->listedForces)
    {
        listedForces.setup(top->idef, fr->natoms_force, fr->listedForcesGpu != nullptr, mdatoms->cVCM);
    }

    if ((usingPme(fr->ic->coulomb.type) || usingLJPme(fr->ic->vdw.type)) && thisRankHasPmeDuty(dd))
    {
        /* This handles the PP+PME rank case where fr->pmedata is valid.
         * For PME-only ranks, gmx_pmeonly() has its own call to gmx_pme_reinit_atoms().
         */
        const int numPmeAtoms = numHomeAtoms - fr->n_tpi;
        gmx_pme_reinit_atoms(fr->pmedata, numPmeAtoms, mdatoms->chargeA, mdatoms->chargeB);
    }

    if (constr)
    {
        constr->setConstraints(top,
                               mdatoms->nr,
                               mdatoms->homenr,
                               mdatoms->massT,
                               mdatoms->invmass,
                               mdatoms->nMassPerturbed != 0,
                               mdatoms->lambda,
                               mdatoms->cFREEZE);
    }
}

} // namespace gmx
