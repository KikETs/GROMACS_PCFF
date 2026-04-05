#ifndef GMX_MDLIB_EXACTRESPA_NONBONDED_GPU_INTERNAL_H
#define GMX_MDLIB_EXACTRESPA_NONBONDED_GPU_INTERNAL_H

#include "config.h"

#if GMX_GPU_CUDA

#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/gpu_utils/gputraits.h"

namespace gmx
{

struct ExactRespaGpuPairEntry
{
    int ai         = 0;
    int aj         = 0;
    int shiftIndex = 0;
    int excluded   = 0;
};

struct ExactRespaGpuRuntimeParams
{
    int   numAtoms                = 0;
    int   numPairs                = 0;
    int   numTypes                = 0;
    int   ntype2                  = 0;
    int   centralShiftIndex       = 0;
    int   activeLevelMask         = 0;
    int   shiftLevelMask          = 0;
    int   directVirialLevelMask   = 0;
    int   accumulateEnergyMask    = 0;
    int   innerLevel              = -1;
    int   middleLevel             = -1;
    int   outerLevel              = -1;
    int   hasMiddle               = 0;
    int   coulombTableElementCount = 0;
    float innerOff                = 0.0F;
    float innerOn                 = 0.0F;
    float outerOn                 = 0.0F;
    float outerOff                = 0.0F;
    float coulombCutoff2          = 0.0F;
    float vdwCutoff2              = 0.0F;
    float repulsionPower          = 0.0F;
    float invRepulsionPower       = 0.0F;
    float epsfac                  = 0.0F;
    float ewaldShift              = 0.0F;
    float coulombTableScale       = 0.0F;
};

void launchExactRespaNonbondedGpuKernel(const ExactRespaGpuRuntimeParams&      params,
                                        const DeviceBuffer<ExactRespaGpuPairEntry>& d_pairEntries,
                                        const DeviceBuffer<Float3>&             d_coordinates,
                                        const DeviceBuffer<Float3>&             d_shiftVectors,
                                        const DeviceBuffer<int>&                d_atomTypes,
                                        const DeviceBuffer<float>&              d_atomCharges,
                                        const DeviceBuffer<float>&              d_nbfp,
                                        const DeviceBuffer<float>&              d_coulombTable,
                                        DeviceBuffer<Float3>                    d_levelForces,
                                        DeviceBuffer<Float3>                    d_levelShiftForces,
                                        DeviceBuffer<float>                     d_levelLjEnergies,
                                        DeviceBuffer<float>                     d_levelCoulombEnergies,
                                        DeviceBuffer<float>                     d_levelExcludedCoulombEnergies,
                                        DeviceBuffer<float>                     d_levelVirials,
                                        const DeviceStream&                     deviceStream);

} // namespace gmx

#endif

#endif
