/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2013- The GROMACS Authors
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

#include "mdrun/mdrun_main.h"

#include <cstdlib>
#include <cstring>

#if !defined(_WIN32)
#    include <unistd.h>
#endif

#include "gromacs/commandline/cmdlinemodule.h"
#include "gromacs/commandline/cmdlinemodulemanager.h"

namespace
{

//! Initializer for a module that defaults to nice level zero.
void initSettingsNoNice(gmx::CommandLineModuleSettings* settings)
{
    settings->setDefaultNiceLevel(0);
}

bool envValueIsFalse(const char* value)
{
    return value != nullptr
           && (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0
               || std::strcmp(value, "FALSE") == 0 || std::strcmp(value, "off") == 0);
}

bool commandLineRequestsAtLeast16OpenmpThreads(int argc, char* argv[])
{
    for (int i = 1; i + 1 < argc; ++i)
    {
        if (std::strcmp(argv[i], "-ntomp") == 0)
        {
            return std::atoi(argv[i + 1]) >= 16;
        }
    }

    const char* ompNumThreads = std::getenv("OMP_NUM_THREADS");
    return ompNumThreads != nullptr && std::atoi(ompNumThreads) >= 16;
}

bool setEnvDefault(const char* name, const char* value)
{
    if (std::getenv(name) != nullptr)
    {
        return false;
    }
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 0);
#endif
    return true;
}

void tuneActual16OpenmpWaitPolicyBeforeRuntimeInit(int argc, char* argv[])
{
    if (!commandLineRequestsAtLeast16OpenmpThreads(argc, argv))
    {
        return;
    }
    if (!envValueIsFalse(std::getenv("GMX_PCFF_EXACT_RESPA_GPU_CAP_EXPLICIT_NTOMP"))
        && !envValueIsFalse(std::getenv("GMX_PCFF_EXACT_RESPA_GPU_NTOMP_MAX")))
    {
        return;
    }

    bool changed = false;
    changed = setEnvDefault("GOMP_SPINCOUNT", "7000") || changed;
    changed = setEnvDefault("OMP_WAIT_POLICY", "active") || changed;

#if !defined(_WIN32)
    if (changed && std::getenv("GMX_PCFF_ACTUAL16_GOMP_REEXEC") == nullptr)
    {
        setenv("GMX_PCFF_ACTUAL16_GOMP_REEXEC", "1", 1);
        execv(argv[0], argv);
    }
#else
    GMX_UNUSED_VALUE(changed);
#endif
}

} // namespace

int main(int argc, char* argv[])
{
    tuneActual16OpenmpWaitPolicyBeforeRuntimeInit(argc, argv);
    return gmx::CommandLineModuleManager::runAsMainCMainWithSettings(
            argc, argv, &gmx::gmx_mdrun, &initSettingsNoNice);
}
