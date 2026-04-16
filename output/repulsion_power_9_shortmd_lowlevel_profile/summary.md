# Repulsion-Power-9 Short-MD Low-Level CPU Profile

| layout | ns/day | wall s | IPC | cache miss rate | cache MPKI | backend bound | memory-bound backend | PME FFT s | PME spread s | PME gather s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| omp6 | 172.847 | 2.500 | 3.27 | 8.32% | 3.388 | 39.2% | 20.8% | 0.921 | 0.151 | 0.359 |

Top DSOs for `omp6`:
- `libgromacs.so.12.0.0`: `50.88%`
- `libfftw3f.so.3.6.10`: `27.90%`
- `libc.so.6`: `10.53%`
- `libgomp.so.1.0.0`: `8.29%`
- `libm.so.6`: `2.07%`

Top symbols for `omp6`:
- `18.76%` `void gmx::nbnxmKernelSimd<(KernelLayout)1, (gmx::KernelCoulombType)1, (VdwCutoffCheck)0, (gmx::LJCombinationRule)2, (InteractionModifiers)1, (LJEwald)0, (EnergyOutput)0>(gmx::NbnxnPairlistCpu const&, gmx::nbnxn_atomdata_t const&, interaction_const_t const&, float const (*) [3], gmx::nbnxn_atomdata_output_t*)` in `libgromacs.so.12.0.0`
- `9.21%` `fft5d_execute(fft5d_plan_t*, int, gmx_wallcycle*)` in `libgromacs.so.12.0.0`
- `7.56%` `getenv` in `libc.so.6`
- `4.84%` `float (anonymous namespace)::dihedral_class2<(BondedKernelFlavor)0>(int, int const*, t_iparams const*, float const (*) [3], float (*) [4], float (*) [3], t_pbc const*, float, float*, gmx::ArrayRef<float const>, t_fcdata*, t_disresdata*, t_oriresdata*, int*)` in `libgromacs.so.12.0.0`
- `4.45%` `0x00000000000258a0` in `libgomp.so.1.0.0`
- `3.24%` `0x00000000000256c0` in `libgomp.so.1.0.0`

| omp12 | 155.522 | 2.778 | 1.55 | 18.12% | 7.087 | 55.1% | 41.7% | 1.086 | 0.375 | 0.424 |

Top DSOs for `omp12`:
- `libgromacs.so.12.0.0`: `35.04%`
- `libgomp.so.1.0.0`: `29.37%`
- `libfftw3f.so.3.6.10`: `26.31%`
- `libc.so.6`: `7.74%`
- `libm.so.6`: `0.96%`

Top symbols for `omp12`:
- `19.48%` `0x00000000000258a0` in `libgomp.so.1.0.0`
- `9.03%` `void gmx::nbnxmKernelSimd<(KernelLayout)1, (gmx::KernelCoulombType)1, (VdwCutoffCheck)0, (gmx::LJCombinationRule)2, (InteractionModifiers)1, (LJEwald)0, (EnergyOutput)0>(gmx::NbnxnPairlistCpu const&, gmx::nbnxn_atomdata_t const&, interaction_const_t const&, float const (*) [3], gmx::nbnxn_atomdata_output_t*)` in `libgromacs.so.12.0.0`
- `8.56%` `0x00000000000256c0` in `libgomp.so.1.0.0`
- `8.33%` `fft5d_execute(fft5d_plan_t*, int, gmx_wallcycle*)` in `libgromacs.so.12.0.0`
- `3.78%` `getenv` in `libc.so.6`
- `2.90%` `spread_on_grid(gmx_pme_t const*, PmeAtomComm*, PmeAndFftGrids*, bool, bool, bool) [clone ._omp_fn.2]` in `libgromacs.so.12.0.0`

| split12_pp6_pme6 | 240.140 | 1.799 | 2.29 | 8.20% | 3.318 | 44.7% | 27.6% | 0.969 | 0.172 | 0.396 |

Top DSOs for `split12_pp6_pme6`:
- `libgromacs.so.12.0.0`: `40.69%`
- `libgomp.so.1.0.0`: `29.76%`
- `libfftw3f.so.3.6.10`: `20.44%`
- `libc.so.6`: `7.18%`
- `libm.so.6`: `1.38%`

Top symbols for `split12_pp6_pme6`:
- `23.77%` `0x00000000000256c0` in `libgomp.so.1.0.0`
- `12.85%` `void gmx::nbnxmKernelSimd<(KernelLayout)1, (gmx::KernelCoulombType)1, (VdwCutoffCheck)0, (gmx::LJCombinationRule)2, (InteractionModifiers)1, (LJEwald)0, (EnergyOutput)0>(gmx::NbnxnPairlistCpu const&, gmx::nbnxn_atomdata_t const&, interaction_const_t const&, float const (*) [3], gmx::nbnxn_atomdata_output_t*)` in `libgromacs.so.12.0.0`
- `5.68%` `fft5d_execute(fft5d_plan_t*, int, gmx_wallcycle*)` in `libgromacs.so.12.0.0`
- `5.63%` `tMPI_Event_wait(tMPI_Event_t*)` in `libgromacs.so.12.0.0`
- `4.93%` `0x00000000000258a0` in `libgomp.so.1.0.0`
- `4.81%` `getenv` in `libc.so.6`
