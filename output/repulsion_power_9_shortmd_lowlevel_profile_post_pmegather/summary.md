# Repulsion-Power-9 Short-MD Post-PME-Gather Low-Level Profile

Representative specialized-layout profile after caching PME trace env lookups and per-atom trace gating.

| layout | ns/day | wall s | IPC | cache miss rate | cache MPKI | backend bound | memory-bound backend | PME FFT s | PME spread s | PME gather s | getenv % | gather % | fft % | gomp 256c0 % | gomp 258a0 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| omp6 | 187.258 | 2.307 | 3.20 | 8.49% | 3.880 | 43.4% | 22.0% | 0.927 | 0.152 | 0.162 | 0.18% | 4.10% | 9.28% | 3.20% | 5.40% |
| omp12 | 160.828 | 2.686 | 1.46 | 18.14% | 7.810 | 58.0% | 43.6% | 1.105 | 0.380 | 0.305 | 0.08% | 2.37% | 9.12% | 9.34% | 20.82% |
| split12_pp6_pme6 | 263.032 | 1.643 | 2.31 | 7.78% | 3.475 | 47.7% | 28.5% | 1.000 | 0.175 | 0.195 | 0.12% | 3.28% | 6.70% | 22.21% | 6.12% |
