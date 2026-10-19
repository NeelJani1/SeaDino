"""Read-only live benchmark monitor; Ctrl+C returns to the shell."""
import json
from pathlib import Path
import subprocess
import time

root=Path(__file__).resolve().parents[1]
results=root/'results'
folders=[results/'marineinst_encoder',results/'sam31_encoder',results/'sam3_encoder',results/'coralmask_missing_models_20261015',results/'deepfish_refresh_20261016',results/'coralmask_refresh_20261016']
try:
    while True:
        print('\033[2J\033[HSeaDino live benchmark monitor — updates every 2 seconds',flush=True)
        print('Ctrl+C stops this monitor and returns to your WSL shell.\n',flush=True)
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
        print(gpu.stdout.strip(),flush=True)
        for folder in folders:
            status=folder/'status.json'
            if not status.exists():continue
            try:data=json.loads(status.read_text())
            except (OSError,ValueError):continue
            print('\n'+folder.name,flush=True)
            for name,info in data.items():
                print('  '+name+': '+info.get('state','unknown'),flush=True)
                if info.get('state')=='running' and info.get('log'):
                    log=Path(info['log'])
                    if not log.resolve().is_relative_to(results.resolve()):continue
                    try:
                        with log.open('rb') as stream:
                            stream.seek(max(0,log.stat().st_size-4096))
                            tail=stream.read().decode('utf-8',errors='replace').replace('\r','\n').splitlines()
                        for line in [line for line in tail if line.strip()][-5:]:print('    '+line,flush=True)
                    except OSError:pass
        time.sleep(2)
except KeyboardInterrupt:
    print('\nMonitor stopped. Benchmark processes continue; shell is ready.',flush=True)
