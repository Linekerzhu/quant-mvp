#!/usr/bin/env python3
import subprocess
import sys
import signal

def run_with_timeout(cmd, timeout=120):
    """Run command with timeout."""
    proc = subprocess.Popen(
        cmd, 
        shell=True,
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=lambda: signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    )
    
    output = []
    import time
    start = time.time()
    while proc.poll() is None:
        if time.time() - start > timeout:
            proc.kill()
            return "TIMEOUT", "".join(output)
        time.sleep(1)
    
    out = proc.stdout.read() if proc.stdout else ""
    return proc.returncode, out

# Run daily_job.py
code, out = run_with_timeout("cd /Users/zjz/quant-mvp && PYTHONPATH=/Users/zjz/quant-mvp python3 src/ops/daily_job.py", timeout=120)

print("=== RETURN CODE:", code, "===")
print(out[-3000:] if len(out) > 3000 else out)
