"""Run a frozen SAM 3 or MarineInst encoder across the eight existing benchmarks.

Five benchmarks use explicit-v1. DeepFish, Biota and CoralMask retain their
existing legacy protocols and are saved separately; see docs/encoder-baselines.md.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_suite.sam3_backbone import SAM3_REVISION, SAM31_REVISION

EXPLICIT = ('german_bank', 'substrate', 'fathomnet', 'brackish', 'coralscapes')
LEGACY = ('deepfish', 'biota', 'coralmask')


def build_jobs(args):
    output = Path(args.output_dir or f'results/{args.baseline}_encoder').resolve()
    common = [sys.executable, '-u', str(ROOT/'run_eval.py'), '--baselines', args.baseline+'-encoder',
              '--batch_size', str(args.batch_size), '--num_workers', str(args.num_workers),
              '--seed', str(args.seed), '--image_size', '224', '--resume',
              '--sam3_revision', args.sam3_revision or (SAM31_REVISION if args.baseline=='sam3.1' else SAM3_REVISION),
              '--sam3_microbatch_size', str(args.sam3_microbatch_size),
              '--marineinst_microbatch_size', str(args.marineinst_microbatch_size)]
    if args.sam3_model_path:common += ['--sam3_model_path',str(Path(args.sam3_model_path).resolve())]
    if args.marineinst_checkpoint:common += ['--marineinst_checkpoint',str(Path(args.marineinst_checkpoint).resolve())]
    if args.biota_csv:common += ['--biota_csv',str(Path(args.biota_csv).resolve())]
    selected = EXPLICIT+LEGACY if 'all' in args.datasets else args.datasets
    jobs = []
    for name in selected:
        if name in EXPLICIT:
            csv_path = output/'explicit_test.csv'
            command = common + ['--split_protocol','explicit-v1','--eval_split','test','--datasets',name,
                '--substrate_epochs','10','--german_bank_epochs','10','--fathomnet_epochs','20',
                '--brackish_epochs','5','--coralscapes_epochs','10','--coralscapes_batch_size','16']
            protocol = 'explicit-v1'
        else:
            csv_path = output/f'{name}_legacy.csv'
            command = common + ['--split_protocol','legacy','--datasets',name]
            if name == 'deepfish':command += ['--deepfish_dir',args.deepfish_dir,'--deepfish_epochs','10']
            if name == 'coralmask':command += ['--coralmask_epochs','6']
            protocol = 'legacy'
        command += ['--output_csv',str(csv_path)]
        jobs.append(dict(dataset=name,protocol=protocol,command=command,csv=str(csv_path),
                         log=str(output/f'{name}.log')))
    return output,jobs


def result_present(job, baseline):
    path=Path(job['csv'])
    if not path.is_file():return False
    with path.open(newline='') as stream:rows=list(csv.DictReader(stream))
    rows=[row for row in rows if row.get('Model / Label') == baseline+'-encoder (baseline)']
    if job['protocol']=='explicit-v1':
        return any(row.get('Dataset')==job['dataset'] and row.get('Evaluation Split')=='test' for row in rows)
    key={'deepfish':'DeepFish Lin Acc (%)','biota':'Biota Lin mAP (%)','coralmask':'CoralMask mIoU (%)'}[job['dataset']]
    return any(row.get(key) for row in rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',choices=['sam3','sam3.1','marineinst'],required=True)
    parser.add_argument('--datasets',nargs='+',choices=['all',*EXPLICIT,*LEGACY],default=['all'])
    parser.add_argument('--output_dir')
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--sam3_revision',default=None)
    parser.add_argument('--sam3_model_path')
    parser.add_argument('--sam3_microbatch_size',type=int,default=2)
    parser.add_argument('--marineinst_checkpoint')
    parser.add_argument('--marineinst_microbatch_size',type=int,default=4)
    parser.add_argument('--biota_csv',help='Optional audited Biota manifest variant; recorded in the run plan.')
    parser.add_argument('--deepfish_dir',default=str(ROOT/'data/benchmarks/deepfish/DeepFish'))
    parser.add_argument('--dry_run',action='store_true')
    args=parser.parse_args()
    output,jobs=build_jobs(args)
    for job in jobs:print(shlex.join(job['command']),flush=True)
    if args.dry_run:return
    output.mkdir(parents=True,exist_ok=True)
    # Protect legacy label-only resume: changes in source/options require a new
    # output directory instead of silently accepting another configuration.
    code=hashlib.sha256()
    for path in sorted((ROOT/'eval_suite').rglob('*.py')):
        code.update(str(path.relative_to(ROOT)).encode())
        code.update(path.read_bytes())
    plan=dict(baseline=args.baseline,code_sha256=code.hexdigest(),jobs=jobs,
              comparison='frozen image encoder; matched benchmark resolutions; no prompt decoder',
              overlap_note='MarineInst training sources include Brackish, DeepFish and FathomNet; exact overlap unresolved')
    local_files=[]
    if args.baseline=='marineinst':
        from eval_suite.marineinst_backbone import MARINEINST_FILENAME
        local_files=[Path(args.marineinst_checkpoint).expanduser() if args.marineinst_checkpoint else
                     Path.home()/'.cache/seadino/checkpoints'/MARINEINST_FILENAME]
    elif args.sam3_model_path:
        local_files=[Path(args.sam3_model_path)/name for name in ('config.json', 'sam3.1_multiplex.pt' if args.baseline=='sam3.1' else 'model.safetensors')]
    if args.biota_csv:local_files.append(Path(args.biota_csv))
    plan['local_model_sha256']={}
    for path in local_files:
        sha=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):sha.update(chunk)
        plan['local_model_sha256'][str(path.resolve())]=sha.hexdigest()
    plan_path=output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise SystemExit(f'Configuration changed. Choose a new --output_dir; existing plan: {plan_path}')
    plan_path.write_text(json.dumps(plan,indent=2))
    status={}
    for job in jobs:
        print(f"Running {job['dataset']} ({job['protocol']}); log: {job['log']}",flush=True)
        status[job['dataset']]={'state':'running','log':job['log']}
        (output/'status.json').write_text(json.dumps(status,indent=2))
        with Path(job['log']).open('a') as log:
            result=subprocess.run(job['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        complete=result.returncode==0 and result_present(job,args.baseline)
        status[job['dataset']]={'state':'complete' if complete else 'failed',
                               'returncode':result.returncode,'log':job['log'],'csv':job['csv']}
        (output/'status.json').write_text(json.dumps(status,indent=2))
        reporter=ROOT/'scripts/summarize_encoder_results.py'
        if reporter.is_file():
            subprocess.run([sys.executable,str(reporter)],cwd=ROOT,check=False)
        if not complete:
            raise SystemExit(f"{job['dataset']} failed or produced no result. See {job['log']}")
    print('Every requested benchmark produced a result. Explicit and legacy CSVs are separate.',flush=True)


if __name__=='__main__':main()
