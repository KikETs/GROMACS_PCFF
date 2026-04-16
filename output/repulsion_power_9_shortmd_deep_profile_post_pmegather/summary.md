# Repulsion-Power-9 Short-MD Deep Low-Level Profile

Representative specialized-layout deep profile after enabling broader PMU access.

| layout | ns/day | wall s | IPC | branch miss | cache miss | L1 miss | dTLB miss | iTLB miss | backend bound | mem bound | frontend bound | front bw | PME FFT s | PME spread s | PME gather s | libfftw3f | libgomp | affinity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| omp6 | 187.882 | 2.300 | 3.03 | 0.41% | 8.37% | 5.71% | 0.31% | 0.09% | 43.3% | 21.8% | 10.2% | 7.1% | 0.926 | 0.154 | 0.162 | 30.37% | 9.73% | 0-5 |
| omp12 | 162.094 | 2.665 | 1.42 | 0.34% | 18.30% | 6.16% | 0.90% | 0.29% | 58.2% | 43.6% | 20.1% | 18.3% | 1.087 | 0.376 | 0.307 | 28.40% | 31.07% | 0-11 |
| split12_pp6_pme6 | 264.561 | 1.633 | 2.32 | 0.24% | 7.83% | 5.66% | 0.07% | 0.47% | 47.6% | 28.3% | 17.7% | 15.5% | 0.974 | 0.173 | 0.210 | 22.04% | 28.88% | 0-5 / 6-11 |
