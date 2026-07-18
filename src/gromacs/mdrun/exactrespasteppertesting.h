/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2025- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 */
#ifndef GMX_MDRUN_EXACTRESPASTEPPERTESTING_H
#define GMX_MDRUN_EXACTRESPASTEPPERTESTING_H

#include <cstdint>

namespace gmx
{

enum class ExactRespaRuntimeEventType : int
{
    InitialKick,
    Drift,
    RefreshForce,
    FinalKick
};

struct ExactRespaRuntimeEvent
{
    int64_t                    baseStep = 0;
    ExactRespaRuntimeEventType type     = ExactRespaRuntimeEventType::Drift;
    int                        level    = 0;
};

class ExactRespaRuntimeEventSink
{
public:
    virtual ~ExactRespaRuntimeEventSink() = default;
    virtual void recordEvent(const ExactRespaRuntimeEvent& event) = 0;
};

void setExactRespaRuntimeEventSinkForTesting(ExactRespaRuntimeEventSink* sink);

//! Returns whether a post-Trotter replay setting requires the next-step virial.
bool exactRespaPostTrotterReplayNeedsNextVirialForTesting(const char* value);

} // namespace gmx

#endif
