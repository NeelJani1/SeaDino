"""Finish queued encoder evaluations while preserving the active MarineInst run."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
QUEUE_FILE=RESULTS/'encoder_benchmark_queue.json'
BIOTA=ROOT/'data/benchmarks/biota/biota_spatial_split_available_v1.csv'
MARINE=RESULTS/'marineinst_encoder'
SAM31=RESULTS/'sam31_encoder'
BIOTA_OUTPUT=RESULTS/'marineinst_biota_available_v1'


def write(path,data):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,indent=2))
    temp.replace(path)


def alive(pid,token):
    path=Path(f'/proc/{pid}/cmdline')
    try:return token.encode() in path.read_bytes()
    except OSError:return False


def stage(name,**details):
    record=json.loads(QUEUE_FILE.read_text()) if QUEUE_FILE.exists() else {}
    record.update(pid=os.getpid(),stage=name,**details)
    write(QUEUE_FILE,record)
    print(name,json.dumps(details),flush=True)


def marine_status(name,state,log=None,returncode=None):
    path=MARINE/'status.json'
    record=json.loads(path.read_text())
    record[name]={'state':state}
    if log:record[name]['log']=str(log)
    if returncode is not None:record[name]['returncode']=returncode
    write(path,record)


def main():
    if not BIOTA.is_file():raise FileNotFoundError(BIOTA)
    pid=json.loads((MARINE/'background_run.json').read_text())['pid']
    stage('waiting_for_active_marineinst',marineinst_pid=pid)
    while alive(pid,'evaluate_encoder_baseline.py'):
        time.sleep(15)
    # German Bank SAM3.1 was launched separately for an immediate end-to-end
    # check. Wait for its recorded completion before sharing its result file.
    stage('waiting_for_sam31_german_bank')
    while True:
        status=json.loads((SAM31/'status.json').read_text())
        state=status.get('german_bank',{}).get('state')
        if state in ('complete','failed'):break
        time.sleep(5)
    stage('running_marineinst_biota_available_v1')
    BIOTA_OUTPUT.mkdir(parents=True,exist_ok=True)
    marine_status('biota_available_v1','running',BIOTA_OUTPUT/'biota.log')
    command=[sys.executable,'-u','scripts/evaluate_encoder_baseline.py',
             '--baseline','marineinst','--datasets','biota','--biota_csv',str(BIOTA),
             '--output_dir',str(BIOTA_OUTPUT)]
    with (BIOTA_OUTPUT/'sweep.log').open('a') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    marine_status('biota_available_v1','complete' if result.returncode==0 else 'failed',
                  BIOTA_OUTPUT/'biota.log',result.returncode)
    marine_returncode=result.returncode
    stage('running_sam31_remaining_benchmarks',marineinst_biota_returncode=marine_returncode)
    command=[sys.executable,'-u','scripts/evaluate_encoder_baseline.py',
             '--baseline','sam3.1','--datasets','substrate','fathomnet','brackish',
             'deepfish','biota','coralscapes','coralmask','--biota_csv',str(BIOTA),
             '--output_dir',str(SAM31)]
    with (SAM31/'remaining_sweep.log').open('a') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    reporting=subprocess.run([sys.executable,str(ROOT/'scripts/summarize_encoder_results.py'),
                              '--require_complete'],cwd=ROOT)
    success=result.returncode==0 and marine_returncode==0 and reporting.returncode==0
    stage('complete' if success else 'needs_attention',sam31_returncode=result.returncode,
          marineinst_biota_returncode=marine_returncode,report_returncode=reporting.returncode,
          report=str(RESULTS/'encoder_baseline_report.md'))
    if not success:raise SystemExit(1)


if __name__=='__main__':main()
