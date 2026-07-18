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
#include "gmxpre.h"

#include "gen_maxwell_velocities.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "gromacs/math/units.h"
#include "gromacs/random/seed.h"
#include "gromacs/random/tabulatednormaldistribution.h"
#include "gromacs/random/threefry.h"
#include "gromacs/topology/atoms.h"
#include "gromacs/topology/mtop_atomloops.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/vec.h"
#include "gromacs/utility/vectypes.h"

static void low_mspeed(real tempi, gmx_mtop_t* mtop, rvec v[], gmx::ThreeFry2x64<>* rng, const gmx::MDLogger& logger)
{
    int                                    nrdf;
    real                                   ekin, temp;
    gmx::TabulatedNormalDistribution<real> normalDist;

    ekin = 0.0;
    nrdf = 0;
    for (const AtomProxy atomP : AtomRange(*mtop))
    {
        const t_atom& local = atomP.atom();
        int           i     = atomP.globalAtomNumber();
        real          mass  = local.m;
        if (mass > 0)
        {
            rng->restart(i, 0);
            real sd = std::sqrt(gmx::c_boltz * tempi / mass);
            for (int m = 0; (m < DIM); m++)
            {
                v[i][m] = sd * normalDist(*rng);
                ekin += 0.5 * mass * v[i][m] * v[i][m];
            }
            nrdf += DIM;
        }
    }
    temp = (2.0 * ekin) / (nrdf * gmx::c_boltz);
    if (temp > 0)
    {
        real scal = std::sqrt(tempi / temp);
        for (int i = 0; (i < mtop->natoms); i++)
        {
            for (int m = 0; (m < DIM); m++)
            {
                v[i][m] *= scal;
            }
        }
    }
    GMX_LOG(logger.info)
            .asParagraph()
            .appendTextFormatted("Velocities were taken from a Maxwell distribution at %g K", tempi);
    if (debug)
    {
        fprintf(debug,
                "Velocities were taken from a Maxwell distribution\n"
                "Initial generated temperature: %12.5e (scaled to: %12.5e)\n",
                temp,
                tempi);
    }
}

void maxwell_speed(real tempi, int seed, gmx_mtop_t* mtop, rvec v[], const gmx::MDLogger& logger)
{

    if (seed == -1)
    {
        seed = static_cast<int>(gmx::makeRandomSeed());
        GMX_LOG(logger.info)
                .asParagraph()
                .appendTextFormatted("Using random seed %d for generating velocities", seed);
    }
    gmx::ThreeFry2x64<> rng(seed, gmx::RandomDomain::MaxwellVelocities);

    low_mspeed(tempi, mtop, v, &rng, logger);
}

static real calc_cm(int natoms, const real mass[], rvec x[], rvec v[], rvec xcm, rvec vcm, rvec acm, matrix L)
{
    rvec dx, a0;
    real tm, m0;
    int  i, m;

    clear_rvec(xcm);
    clear_rvec(vcm);
    clear_rvec(acm);
    tm = 0.0;
    for (i = 0; (i < natoms); i++)
    {
        m0 = mass[i];
        tm += m0;
        cprod(x[i], v[i], a0);
        for (m = 0; (m < DIM); m++)
        {
            xcm[m] += m0 * x[i][m]; /* c.o.m. position */
            vcm[m] += m0 * v[i][m]; /* c.o.m. velocity */
            acm[m] += m0 * a0[m];   /* rotational velocity around c.o.m. */
        }
    }
    cprod(xcm, vcm, a0);
    for (m = 0; (m < DIM); m++)
    {
        xcm[m] /= tm;
        vcm[m] /= tm;
        acm[m] -= a0[m] / tm;
    }

    clear_mat(L);
    for (i = 0; (i < natoms); i++)
    {
        m0 = mass[i];
        for (m = 0; (m < DIM); m++)
        {
            dx[m] = x[i][m] - xcm[m];
        }
        L[XX][XX] += dx[XX] * dx[XX] * m0;
        L[XX][YY] += dx[XX] * dx[YY] * m0;
        L[XX][ZZ] += dx[XX] * dx[ZZ] * m0;
        L[YY][YY] += dx[YY] * dx[YY] * m0;
        L[YY][ZZ] += dx[YY] * dx[ZZ] * m0;
        L[ZZ][ZZ] += dx[ZZ] * dx[ZZ] * m0;
    }

    return tm;
}

void stop_cm(const gmx::MDLogger gmx_unused& logger, int natoms, real mass[], rvec x[], rvec v[])
{
    rvec   xcm, vcm, acm;
    tensor L;
    int    i, m;

#ifdef DEBUG
    GMX_LOG(logger.info).asParagraph().appendTextFormatted("stopping center of mass motion...");
#endif
    (void)calc_cm(natoms, mass, x, v, xcm, vcm, acm, L);

    /* Subtract center of mass velocity */
    for (i = 0; (i < natoms); i++)
    {
        for (m = 0; (m < DIM); m++)
        {
            v[i][m] -= vcm[m];
        }
    }
}

void lammps_mom_rot_velocity_scale(const real           tempi,
                                   const int            natoms,
                                   const real           mass[],
                                   const rvec           x[],
                                   rvec                 v[],
                                   const gmx::MDLogger& logger)
{
    if (!(tempi > 0) || !std::isfinite(tempi))
    {
        gmx_fatal(FARGS, "LAMMPS-style velocity creation requires a finite positive temperature");
    }
    if (natoms <= 1 || mass == nullptr || x == nullptr || v == nullptr)
    {
        gmx_fatal(FARGS, "LAMMPS-style mom/rot velocity creation requires at least two atoms");
    }

    std::array<double, DIM> centerOfMass         = { 0, 0, 0 };
    std::array<double, DIM> centerOfMassVelocity = { 0, 0, 0 };
    double                  totalMass            = 0;
    int                     numMobileAtoms       = 0;
    for (int atom = 0; atom < natoms; ++atom)
    {
        const double atomMass = mass[atom];
        if (!(atomMass > 0))
        {
            continue;
        }
        ++numMobileAtoms;
        totalMass += atomMass;
        for (int dimension = 0; dimension < DIM; ++dimension)
        {
            centerOfMass[dimension] += atomMass * x[atom][dimension];
            centerOfMassVelocity[dimension] += atomMass * v[atom][dimension];
        }
    }
    if (numMobileAtoms <= 1 || !(totalMass > 0) || !std::isfinite(totalMass))
    {
        gmx_fatal(FARGS,
                  "LAMMPS-style mom/rot velocity creation requires at least two mobile atoms");
    }
    for (int dimension = 0; dimension < DIM; ++dimension)
    {
        centerOfMass[dimension] /= totalMass;
        centerOfMassVelocity[dimension] /= totalMass;
    }
    for (int atom = 0; atom < natoms; ++atom)
    {
        if (mass[atom] <= 0)
        {
            continue;
        }
        for (int dimension = 0; dimension < DIM; ++dimension)
        {
            v[atom][dimension] -= static_cast<real>(centerOfMassVelocity[dimension]);
        }
    }

    // Symmetric inertia tensor [a b c; b d e; c e f] and L=sum m*(r x v).
    double a = 0;
    double b = 0;
    double c = 0;
    double d = 0;
    double e = 0;
    double f = 0;
    std::array<double, DIM> angularMomentum = { 0, 0, 0 };
    for (int atom = 0; atom < natoms; ++atom)
    {
        const double atomMass = mass[atom];
        if (!(atomMass > 0))
        {
            continue;
        }
        const double rx = static_cast<double>(x[atom][XX]) - centerOfMass[XX];
        const double ry = static_cast<double>(x[atom][YY]) - centerOfMass[YY];
        const double rz = static_cast<double>(x[atom][ZZ]) - centerOfMass[ZZ];
        const double vx = v[atom][XX];
        const double vy = v[atom][YY];
        const double vz = v[atom][ZZ];

        a += atomMass * (ry * ry + rz * rz);
        b -= atomMass * rx * ry;
        c -= atomMass * rx * rz;
        d += atomMass * (rx * rx + rz * rz);
        e -= atomMass * ry * rz;
        f += atomMass * (rx * rx + ry * ry);
        angularMomentum[XX] += atomMass * (ry * vz - rz * vy);
        angularMomentum[YY] += atomMass * (rz * vx - rx * vz);
        angularMomentum[ZZ] += atomMass * (rx * vy - ry * vx);
    }

    const double determinant = a * (d * f - e * e) - b * (b * f - c * e)
                               + c * (b * e - c * d);
    const double inertiaScale = std::max({ std::abs(a),
                                           std::abs(b),
                                           std::abs(c),
                                           std::abs(d),
                                           std::abs(e),
                                           std::abs(f) });
    const double singularTolerance = 256.0 * std::numeric_limits<double>::epsilon()
                                     * inertiaScale * inertiaScale * inertiaScale;
    if (!(inertiaScale > 0) || !std::isfinite(determinant)
        || std::abs(determinant) <= singularTolerance)
    {
        gmx_fatal(FARGS,
                  "LAMMPS-style mom/rot velocity creation has a singular inertia tensor "
                  "(determinant=%g, scale=%g)",
                  determinant,
                  inertiaScale);
    }

    const double cofactor00 = d * f - e * e;
    const double cofactor01 = c * e - b * f;
    const double cofactor02 = b * e - c * d;
    const double cofactor11 = a * f - c * c;
    const double cofactor12 = b * c - a * e;
    const double cofactor22 = a * d - b * b;
    const std::array<double, DIM> omega = {
        (cofactor00 * angularMomentum[XX] + cofactor01 * angularMomentum[YY]
         + cofactor02 * angularMomentum[ZZ])
                / determinant,
        (cofactor01 * angularMomentum[XX] + cofactor11 * angularMomentum[YY]
         + cofactor12 * angularMomentum[ZZ])
                / determinant,
        (cofactor02 * angularMomentum[XX] + cofactor12 * angularMomentum[YY]
         + cofactor22 * angularMomentum[ZZ])
                / determinant
    };

    for (int atom = 0; atom < natoms; ++atom)
    {
        if (mass[atom] <= 0)
        {
            continue;
        }
        const double rx = static_cast<double>(x[atom][XX]) - centerOfMass[XX];
        const double ry = static_cast<double>(x[atom][YY]) - centerOfMass[YY];
        const double rz = static_cast<double>(x[atom][ZZ]) - centerOfMass[ZZ];
        v[atom][XX] -= static_cast<real>(omega[YY] * rz - omega[ZZ] * ry);
        v[atom][YY] -= static_cast<real>(omega[ZZ] * rx - omega[XX] * rz);
        v[atom][ZZ] -= static_cast<real>(omega[XX] * ry - omega[YY] * rx);
    }

    double kineticEnergy = 0;
    for (int atom = 0; atom < natoms; ++atom)
    {
        if (mass[atom] <= 0)
        {
            continue;
        }
        kineticEnergy += 0.5 * static_cast<double>(mass[atom])
                         * (static_cast<double>(v[atom][XX]) * v[atom][XX]
                            + static_cast<double>(v[atom][YY]) * v[atom][YY]
                            + static_cast<double>(v[atom][ZZ]) * v[atom][ZZ]);
    }
    const int    degreesOfFreedom = DIM * numMobileAtoms - DIM;
    const double targetKineticEnergy =
            0.5 * degreesOfFreedom * static_cast<double>(gmx::c_boltz) * tempi;
    if (!(kineticEnergy > 0) || !std::isfinite(kineticEnergy))
    {
        gmx_fatal(FARGS,
                  "LAMMPS-style mom/rot velocity creation cannot rescale zero kinetic energy");
    }
    const real scale = static_cast<real>(std::sqrt(targetKineticEnergy / kineticEnergy));
    for (int atom = 0; atom < natoms; ++atom)
    {
        if (mass[atom] <= 0)
        {
            continue;
        }
        for (int dimension = 0; dimension < DIM; ++dimension)
        {
            v[atom][dimension] *= scale;
        }
    }

    GMX_LOG(logger.info)
            .asParagraph()
            .appendTextFormatted("Applied LAMMPS velocity-create mom/rot removal and rescaled to %g K",
                                 tempi);
}
