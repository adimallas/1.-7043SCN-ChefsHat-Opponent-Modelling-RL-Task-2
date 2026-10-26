#!/usr/bin/env python3
"""Record real demo process output as an asciicast v2 terminal recording.

This is a capture of actual program output, not a generated student viva or a
GUI gameplay video. Play with `asciinema play viva/gameplay/gameplay.cast`.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'viva' / 'gameplay' / 'gameplay.cast'
OUT.parent.mkdir(parents=True, exist_ok=True)
command = [sys.executable, '-u', 'scripts/demo.py', '--seed', '777888', '--seat', '0', '--delay', '0.12']
start = time.monotonic()
with OUT.open('w') as recording:
    recording.write(json.dumps({'version':2,'width':150,'height':40,'timestamp':int(time.time()),'title':'Actual official Chef’s Hat trained-agent terminal demonstration','command':' '.join(command),'env':{'TERM':'xterm-256color'}})+'\n')
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end='', flush=True)
        recording.write(json.dumps([time.monotonic()-start,'o',line.rstrip('\n')+'\r\n'])+'\n')
        recording.flush()
    returncode = process.wait()
    if returncode:
        raise SystemExit(returncode)
print(f'Actual terminal-output recording saved: {OUT}')
