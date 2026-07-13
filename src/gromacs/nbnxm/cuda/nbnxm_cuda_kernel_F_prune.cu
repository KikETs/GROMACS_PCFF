/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2015- The GROMACS Authors
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

#include "gromacs/gpu_utils/cudautils.cuh"

#include "nbnxm_cuda_kernel_utils.cuh"
#include "nbnxm_cuda_types.h"

/* Top-level kernel generation: will generate through multiple
 * inclusion the following flavors for all kernel:
 * force-only output with pair list pruning;
 */
#define PRUNE_NBL
#define FUNCTION_DECLARATION_ONLY
#include "nbnxm_cuda_kernels.cuh"
#define EXACT_RESPA_NATIVE_MULTI_NB_KERNEL
#define EXACT_RESPA_NATIVE_MULTI_NB_COUNT 2
#define NB_KERNEL_FUNC_VARIANT_SUFFIX _nativeMulti2
#include "nbnxm_cuda_kernels.cuh"
#undef NB_KERNEL_FUNC_VARIANT_SUFFIX
#undef EXACT_RESPA_NATIVE_MULTI_NB_COUNT
#define EXACT_RESPA_NATIVE_MULTI_NB_COUNT 3
#define NB_KERNEL_FUNC_VARIANT_SUFFIX _nativeMulti
#include "nbnxm_cuda_kernels.cuh"
#undef NB_KERNEL_FUNC_VARIANT_SUFFIX
#define EXACT_RESPA_NATIVE_MULTI_NB_MIN_BLOCKS_PER_MP 16
#define NB_KERNEL_FUNC_VARIANT_SUFFIX _nativeMultiLowReg
#include "nbnxm_cuda_kernels.cuh"
#undef NB_KERNEL_FUNC_VARIANT_SUFFIX
#undef EXACT_RESPA_NATIVE_MULTI_NB_MIN_BLOCKS_PER_MP
#undef EXACT_RESPA_NATIVE_MULTI_NB_COUNT
#undef EXACT_RESPA_NATIVE_MULTI_NB_KERNEL
#undef FUNCTION_DECLARATION_ONLY
#include "nbnxm_cuda_kernels.cuh"
#define EXACT_RESPA_NATIVE_MULTI_NB_KERNEL
#define EXACT_RESPA_NATIVE_MULTI_NB_COUNT 2
#define NB_KERNEL_FUNC_VARIANT_SUFFIX _nativeMulti2
#include "nbnxm_cuda_kernels.cuh"
#undef NB_KERNEL_FUNC_VARIANT_SUFFIX
#undef EXACT_RESPA_NATIVE_MULTI_NB_COUNT
#define EXACT_RESPA_NATIVE_MULTI_NB_COUNT 3
#define NB_KERNEL_FUNC_VARIANT_SUFFIX _nativeMulti
#include "nbnxm_cuda_kernels.cuh"
#undef NB_KERNEL_FUNC_VARIANT_SUFFIX
#define EXACT_RESPA_NATIVE_MULTI_NB_MIN_BLOCKS_PER_MP 16
#define NB_KERNEL_FUNC_VARIANT_SUFFIX _nativeMultiLowReg
#include "nbnxm_cuda_kernels.cuh"
#undef NB_KERNEL_FUNC_VARIANT_SUFFIX
#undef EXACT_RESPA_NATIVE_MULTI_NB_MIN_BLOCKS_PER_MP
#undef EXACT_RESPA_NATIVE_MULTI_NB_COUNT
#undef EXACT_RESPA_NATIVE_MULTI_NB_KERNEL
#undef PRUNE_NBL
