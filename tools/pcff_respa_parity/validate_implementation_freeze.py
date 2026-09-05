"""Bounded implementation checks. Each mdrun is a fresh process; originals are read-only.

Run with the existing MD environment (Python 3.9, numpy, pandas).
No transport runs. Raw inputs, commands, environment, hashes and failures are kept.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import xdrlib
import numpy as np
import pandas as pd

ROOT = Path('/home/kiket/Desktop/test')
OUT = ROOT / 'GROMACS_PCFF/output/implementation_closeout_20260905/final_validation'
CAM = ROOT / 'GROMACS_PCFF/output/polygen_multisystem_validation_20260512_m1p50'
DIAG = CAM / 'analysis/parity_root_cause_20260905'
RUN = CAM / 'runs_batch/remote_mid_b_zero_bond_ranked_27068_26452_538b2ba1d7_20260828/remote_mid_b'
BIN = {'double_cpu': ROOT / 'ab_installs/pcff_implementation_freeze_double_r2_20260905/bin/gmx_d',
       'mixed_cpu': ROOT / 'ab_installs/pcff_implementation_freeze_cuda_r2_20260905/bin/gmx',
       'mixed_gpu': ROOT / 'ab_installs/pcff_implementation_freeze_cuda_r2_20260905/bin/gmx'}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def command(cmd, dest, env, label, stdin=None):
    start = time.monotonic()
    p = subprocess.run(list(map(str, cmd)), cwd=dest, env=env, input=stdin,
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (dest / (label + '.log')).write_text(p.stdout)
    record = dict(command=list(map(str, cmd)), returncode=p.returncode,
                  wall_seconds=time.monotonic()-start)
    with (dest / 'commands.jsonl').open('a') as h:
        h.write(json.dumps(record) + '\n')
    if p.returncode:
        raise RuntimeError(str(dest) + '/' + label + '.log')
    return p.stdout

def environment(lane, threads=1, trajectory='2706801'):
    env = {k:v for k,v in os.environ.items() if not k.startswith(('GMX_', 'GROMACS_BATCH_', 'OMP_'))}
    context = RUN / 'gmx_cpu' / ('Traj_' + trajectory) / 'MD_GMX/gromacs_runtime_context.json'
    # The stage-specific context only holds beta. The preserved common-state
    # environment also contains the explicit COM admission and NHC conventions.
    env.update(json.loads((DIAG/'common_state_short_dynamics/environment.json').read_text()))
    env.update(json.loads(context.read_text())['gromacs_stage_layouts']['prod_nvt']['env'])
    env.update(OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1', GMX_MAXBACKUP='-1',
               GMX_DISABLE_MODULAR_SIMULATOR='1', GMX_PCFF_LAMMPS_DISPERSION_CORRECTION='1')
    return env

def edit(text, changes):
    for k,v in changes.items():
        pattern = r'^' + re.escape(k) + r'\s*=.*$'
        text, n = re.subn(pattern, k + ' = ' + str(v), text, flags=re.M)
        if n == 0: text += '\n' + k + ' = ' + str(v) + '\n'
        assert n <= 1
    return text

def execute(label, lane, mdp, state, topology, threads=1, trajectory='2706801', overrides=None, allow_no_com_warning=False):
    dest = OUT / label
    dest.mkdir(parents=True, exist_ok=True)
    env = environment(lane, threads, trajectory)
    for key,value in (overrides or {}).items():
        if value is None: env.pop(key,None)
        else: env[key]=str(value)
    binary = BIN[lane]
    suffix = '_d' if lane == 'double_cpu' else ''
    lib = binary.parent.parent / ('lib/libgromacs' + suffix + '.so.12.0.0')
    signatures = {str(p):sha(p) for p in [binary,lib,state,topology]}
    # The campaign topologies include local ITPs. Preserve their hashes as well.
    signatures.update({str(p):sha(p) for p in topology.parent.glob('*.itp')})
    signature = dict(files=signatures, mdp=mdp, lane=lane, threads=threads,
                     environment={k:v for k,v in env.items() if k.startswith(('GMX_', 'OMP_', 'OPENBLAS_'))})
    meta = dest / 'completed.json'
    if meta.exists():
        assert json.loads(meta.read_text())['signature'] == signature, dest
        return dest
    assert not (dest/'eval.edr').exists(), 'Do not overwrite a partial run; investigate first.'
    (dest/'configuration.json').write_text(json.dumps(signature,indent=2))
    (dest/'eval.mdp').write_text(mdp)
    prep=command([binary,'grompp','-f','eval.mdp','-c',state,'-p',topology,'-o','eval.tpr',
                  '-maxwarn',str(int(allow_no_com_warning))],dest,env,'grompp')
    if allow_no_com_warning:
        assert len(re.findall(r'^WARNING \d+',prep,re.M))==1
        assert 'You are not using center of mass motion removal' in prep
    backend = 'gpu' if lane == 'mixed_gpu' else 'cpu'
    args = [binary,'mdrun','-deffnm','eval','-ntmpi','1','-ntomp',str(threads),
            '-nb',backend,'-pme',backend,'-bonded',backend,'-update','cpu',
            '-pin','off','-dlb','no','-notunepme']
    if lane != 'mixed_gpu': args += ['-reprod']
    command(args,dest,env,'mdrun')
    assert all(sha(p)==h for p,h in signatures.items())
    meta.write_text(json.dumps(dict(signature=signature, tpr_sha256=sha(dest/'eval.tpr')),
                               indent=2))
    print('completed',label,flush=True)
    return dest

def frames(path):
    data=path.read_bytes(); u=xdrlib.Unpacker(data); result=[]
    while u.get_position()<len(data):
        assert u.unpack_int()==1993
        u.unpack_int(); assert u.unpack_string()==b'GMX_trn_file'
        sizes=[u.unpack_int() for _ in range(10)]
        n=u.unpack_int();step=u.unpack_int();u.unpack_int()
        width=sizes[2]//9; assert width in [4,8]
        val=u.unpack_double if width==8 else u.unpack_float
        t=val();val();offset=u.get_position();arrays={}
        for name,size in zip(['ir','e','box','vir','pres','top','sym','x','v','f'],sizes):
            if size: arrays[name]=np.frombuffer(data,dtype='>f'+str(width),count=size//width,offset=offset).astype(float)
            offset+=size
        u.set_position(offset);result.append((step,t,n,arrays))
    return result

def energies(dest,lane,terms):
    command([BIN[lane],'energy','-f','eval.edr','-o','energy.xvg','-xvg','xmgrace','-dp'],
            dest,environment(lane),'energy','\n'.join(terms)+'\n0\n')
    text=(dest/'energy.xvg').read_text()
    names=re.findall(r'@\s+s\d+\s+legend\s+"([^"]+)"',text)
    a=np.array([[float(v) for v in line.split()] for line in text.splitlines()
                if line.strip() and not line.startswith(('#','@'))])
    assert a.shape[1]==len(terms)+1==len(names)+1
    # gmx energy writes EDR ordering, not the user's selection ordering.
    key=lambda s: re.sub(r'[^a-z0-9]','',s.lower())
    mapping={key(n):n for n in names}
    df=pd.DataFrame(a,columns=['time_ps']+names)
    return df[['time_ps']+[mapping[key(t)] for t in terms]].set_axis(['time_ps']+terms,axis=1)

def static():
    rows=[]
    for tid in ['2706801','2645201']:
        project=RUN/'gmx_cpu'/('Traj_'+tid)/'MD_GMX'
        base=DIAG/'force_probes'/tid
        refdir=DIAG/'same_state_pressure'/tid
        raw=json.loads((refdir/'raw_values.json').read_text())['lammps_real_units']
        lines=(refdir/'lammps/forces.dump').read_text().splitlines()
        i=next(i for i,l in enumerate(lines) if l.startswith('ITEM: ATOMS'))+1
        a=np.array([[float(v) for v in l.split()] for l in lines[i:]])
        ref=a[np.argsort(a[:,0]),1:4]*41.84
        for lane in ['double_cpu','mixed_cpu','mixed_gpu']:
            mdp=edit((base/'cpu5_fine/eval.mdp').read_text(),{'pme-order':4,'nstvout':0})
            dest=execute('static_matched_runtime/'+tid+'/'+lane,lane,mdp,base/'state.gro',project/'topol.top',trajectory=tid)
            f=frames(dest/'eval.trr')[0][3]['f'].reshape(-1,3)
            assert f.shape==ref.shape
            e=energies(dest,lane,['Potential','Pressure','Pres-XX','Pres-YY','Pres-ZZ','Pres-XY','Pres-XZ','Pres-YZ'])
            tensor=np.array([raw[k]*1.01325 for k in ['Pxx','Pyy','Pzz','Pxy','Pxz','Pyz']])
            row=dict(trajectory=tid,lane=lane,force_relative_rms_percent=float(np.sqrt(np.mean((f-ref)**2)/np.mean(ref**2))*100),
                     force_max_kj_mol_nm=float(abs(f-ref).max()),potential_difference_kj_mol=float(e.Potential.iloc[0]-raw['PotEng']*4.184),
                     pressure_difference_bar=float(e.Pressure.iloc[0]-raw['Press']*1.01325),
                     pressure_tensor_max_difference_bar=float(abs(e.iloc[0,3:].to_numpy()-tensor).max()))
            rows.append(row);print(row,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'static_summary.csv',index=False)

def dynamics():
    project=RUN/'gmx_cpu/Traj_2706801/MD_GMX'
    base=DIAG/'common_state_short_dynamics'
    template=(base/'fine_dof_matched_nvt/gromacs/eval.mdp').read_text()
    rows=[]
    for lane in ['double_cpu','mixed_gpu']:
        for dt,factor in [(.0005,4),(.0005,2),(.00025,4)]:
            nsteps=round(5/dt); interval=round(.01/dt)
            changes={'dt':dt,'nsteps':nsteps,'tcoupl':'no','nsttcouple':-1,'nstcomm':1000000000,
                     'exact-respa-level2-factor':factor,'nstcalcenergy':interval,'nstenergy':interval,
                     'nstlog':nsteps,'nstxout':nsteps,'nstvout':nsteps,'nstfout':0,
                     'pme-order':4,'fourierspacing':.12}
            dest=execute('nve/'+lane+'_dt'+str(dt)+'_factor'+str(factor),lane,edit(template,changes),
                         base/'state.gro',project/'topol.top',threads=8)
            e=energies(dest,lane,['Potential','Kinetic-En.','Total-Energy','Temperature'])
            assert len(e)==501 and abs(e.time_ps.iloc[-1]-5)<1e-8 and np.isfinite(e.to_numpy()).all()
            total=e['Total-Energy'].to_numpy(); t=e.time_ps.to_numpy()
            slope=float(np.polyfit(t,total,1)[0]); n=frames(dest/'eval.trr')[0][2]
            row=dict(lane=lane,dt_ps=dt,outer_dt_ps=dt*factor,natoms=n,duration_ps=5,
                     drift_kj_mol_ps=slope,drift_kj_mol_atom_ps=slope/n,
                     energy_span_kj_mol=float(np.ptp(total)),endpoint_energy_change_kj_mol=float(total[-1]-total[0]),
                     kinetic_mean_kj_mol=float(e['Kinetic-En.'].mean()),
                     energy_span_fraction_mean_ke=float(np.ptp(total)/e['Kinetic-En.'].mean()))
            rows.append(row);pd.DataFrame(rows).to_csv(OUT/'nve_summary.csv',index=False)
            print(row,flush=True)

def benchmark():
    project=RUN/'gmx_cpu/Traj_2706801/MD_GMX';base=DIAG/'common_state_short_dynamics'
    template=(base/'fine_dof_matched_nvt/gromacs/eval.mdp').read_text()
    # Same system/protocol and common order/grid on all lanes. Single-host measurements.
    mdp=edit(template,{'nsteps':40000,'nstcomm':400,'nstcalcenergy':400,'nstenergy':40000,
                      'nstlog':40000,'nstxout':0,'nstvout':0,'nstfout':0,'pme-order':4,'fourierspacing':.12})
    rows=[]
    # Interleave lanes to avoid putting all repetitions of one lane at one time.
    for rep in range(3):
        for lane in ['double_cpu','mixed_cpu','mixed_gpu']:
            dest=execute('benchmark/'+lane+'_r'+str(rep),lane,mdp,base/'state.gro',project/'topol.top',threads=12)
            log=(dest/'mdrun.log').read_text()
            m=re.search(r'^Performance:\s+([0-9.eE+-]+)',log,re.M);assert m
            row=dict(lane=lane,repeat=rep,ns_day=float(m[1]),simulated_ps=20,ntomp=12,update='cpu',pin='off')
            rows.append(row);pd.DataFrame(rows).to_csv(OUT/'benchmark_summary.csv',index=False)
            print(row,flush=True)

def micro():
    import sys
    src=ROOT/'ab_worktrees/GROMACS_PCFF_com_removal_538b2ba1d7_20260905'
    sys.path.insert(0,str(src/'tools/pcff_respa_parity'))
    from force_compare import exact_respa_mdp
    rows=[]
    for system in ['small_oligomer','small_salt_polymer_box']:
        fixture=src/'tests/reference_results/m6_respa'/system
        runs={}
        mdp=edit(exact_respa_mdp(5,1),{'nstfout':4,'nstvout':4})
        for name,threads,extra in [('cpu1',1,{}),('cpu2',2,{}),('cpu12',12,{}),
                                   ('plain',1,{'GMX_DISABLE_SIMD_KERNELS':'1'}),
                                   ('scalar_specialized',1,{'GMX_PCFF_EXACT_RESPA_DISABLE_NBNXM_NARROW':'1'}),
                                   ('scalar_generic',1,{'GMX_PCFF_EXACT_RESPA_DISABLE_NBNXM_NARROW':'1',
                                                       'GMX_DISABLE_REPULSION_POWER_9_EXACT_RESPA_CPU_SPECIALIZATION':'1'})]:
            label='micro/'+system+'/'+name;dest=OUT/label
            override={'GMX_PCFF_EWALD_BETA_INV_A':None,'GMX_PCFF_LAMMPS_DISPERSION_CORRECTION':None,
                      'GMX_EXACT_RESPA_TOTAL_FORCE_DUMP_FILE':str(dest/'total.tsv'),
                      'GMX_EXACT_RESPA_PER_LEVEL_FORCE_DUMP_FILE':str(dest/'levels.tsv'),**extra}
            execute(label,'double_cpu',mdp,fixture/'initial_nve.gro',fixture/'topol.top',threads,overrides=override,allow_no_com_warning=True)
            runs[name]=frames(dest/'eval.trr')
            total=np.loadtxt(dest/'total.tsv',ndmin=2);levels=np.loadtxt(dest/'levels.tsv',ndmin=2)
            assert total.shape[1]==8 and levels.shape[1]==9
            totalmap={(int(r[0]),int(r[2]),int(r[4])):r[5:] for r in total}
            assert len(totalmap)==len(total)
            residual=[]
            for key,force in totalmap.items():
                keep=(levels[:,0]==key[0])&(levels[:,2]==key[1])&(levels[:,5]==key[2])
                group=levels[keep]
                assert len(group)==key[1]+1
                assert sorted(group[:,3].astype(int))==list(range(key[1]+1))
                residual.extend(group[:,6:].sum(axis=0)-force)
            closure=float(np.max(np.abs(residual)));assert closure<5e-4
            log=(dest/'mdrun.log').read_text()+(dest/'eval.log').read_text()
            if name=='scalar_generic': assert 'keep the generic repulsion-power-9 scalar patch path' in log
            if name=='scalar_specialized': assert 'specialized exact repulsion-power-9 scalar patch path' in log
            if name=='cpu1': assert 'narrow per-contribution NBNXM path' in log
            reference=runs['cpu1'];candidate=runs[name]
            assert len(reference)==len(candidate)==6
            metrics={}
            for field in ['x','v','f']:
                delta=[]
                for a,b in zip(reference,candidate):
                    assert a[:3]==b[:3]
                    delta.extend(b[3][field]-a[3][field])
                metrics[field+'_max_abs_difference']=float(np.max(np.abs(delta)))
            row=dict(system=system,variant=name,closure_max_kj_mol_nm=closure,**metrics)
            rows.append(row);print(row,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'micro_summary.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['static','dynamics','benchmark','micro'])
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--cpu',type=Path,default=BIN['double_cpu'])
    p.add_argument('--gpu',type=Path,default=BIN['mixed_gpu'])
    args=p.parse_args();OUT=args.out.resolve();OUT.mkdir(parents=True,exist_ok=True)
    BIN.update(double_cpu=args.cpu.resolve(),mixed_cpu=args.gpu.resolve(),mixed_gpu=args.gpu.resolve())
    globals()[args.mode]()
