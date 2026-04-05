# Gate H Neutral Transport Pilot

- Pilot status: `PASS`
- Replica count per layout: `2`
- Equilibration / production: `10.0 ps / 100.0 ps`
- Coord stride: `0.5 ps`
- MSD estimator: `gmx msd -mol on the all-oligomer atom group, split into molecules by topology.`
- Recommendation: Neutral MSD pilot is internally consistent enough to justify longer neutral Gate H scaling, but this is still not TP0 production sign-off.

## Observable
- `oligomer_diffusivity_cm2_s`: `stochastic` / passes=`True`
