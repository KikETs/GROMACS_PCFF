#ifndef GMX_MDLIB_EXACTRESPA_NONBONDED_GPU_H
#define GMX_MDLIB_EXACTRESPA_NONBONDED_GPU_H

#include <array>
#include <cstdint>

#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/vectypes.h"

struct gmx_enerdata_t;
struct t_forcerec;
struct t_inputrec;
struct t_mdatoms;

namespace gmx
{

class StepWorkload;

struct ExactRespaGpuLevelOutput
{
    bool             active              = false;
    ArrayRef<RVec>   force;
    ArrayRef<RVec>   shift;
    ForceWithVirial* directVirialOutput = nullptr;
};

struct ExactRespaGpuOutputView
{
    static constexpr int c_numLevels = 3;
    std::array<ExactRespaGpuLevelOutput, c_numLevels> levels;
};

bool exactRespaNonbondedGpuSupported(const t_inputrec& inputrec, const t_forcerec& fr);

void computeExactRespaNonbondedGpu(const t_inputrec&             inputrec,
                                   t_forcerec*                   fr,
                                   const t_mdatoms&              mdatoms,
                                   ArrayRef<const RVec>          coordinates,
                                   const ExactRespaGpuOutputView& outputView,
                                   gmx_enerdata_t*               enerd,
                                   const StepWorkload&           stepWork,
                                   int64_t                       step);

} // namespace gmx

#endif
