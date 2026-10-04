#!/usr/bin/env python3
"""Launch the qualified single-5090 native cache profile."""
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--cache-directory', required=True)
parser.add_argument('--request-log', default='native-runtime-requests.jsonl')
parser.add_argument('--host', default='127.0.0.1')
parser.add_argument('--port', default='18080')
args = parser.parse_args()
command = [args.binary, args.model, '--host', args.host, '--port', args.port,
           '--model-id', 'qwen3.8-27b', '--max-context', '712704',
           '--kv-capacity', '712704', '--max-concurrency', '1',
           '--kv-dtype', 'nvfp4', '--rope-scaling', 'yarn:3', '--vision',
           '--spec', 'mtp', '--draft-tokens', '7', '--device-state-slots', '0',
           '--prefill-chunk', '1024', '--host-kv-mib', '32768',
           '--host-state-slots', '32', '--max-private-continuations', '8',
           '--max-pending-requests', '16', '--pending-timeout-ms', '1800000',
           '--max-shared-prefixes', '0', '--max-long-anchors-per-continuation', '0',
           '--request-log-jsonl', args.request_log]
if (Path(args.cache_directory) / 'CURRENT').is_file():
    command += ['--cache-restore-directory', args.cache_directory]
os.execv(args.binary, command)
