"""Run one persistent M3Diff model in the isolated Python interpreter."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import traceback

PREFIX = 'M3DIFF_RPC:'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args()
    output = sys.stdout

    def reply(payload):
        output.write(PREFIX + json.dumps(payload, ensure_ascii=True) + '\n')
        output.flush()

    def info():
        return {
            'model_path': str(runner.model_path), 'protocol': runner.protocol,
            'model_instance_id': runner.model_instance_id,
            'model_load_count': runner.model_load_count,
            'successful_requests': runner.successful_requests,
            'load_seconds': runner.load_seconds, 'protocol_id': runner.protocol_id,
            'runtime_manifest': runner.runtime_manifest,
            'worker_python': sys.executable, 'worker_version': sys.version,
        }

    try:
        with redirect_stdout(sys.stderr):
            from PIL import Image
            protocol = json.loads(Path(args.protocol).read_text(encoding='utf-8'))
            if protocol.get('model_variant') == 'B_SCST_128':
                from scst_demo_runner import SCSTRunner
                runner = SCSTRunner(args.model_path, protocol)
            elif 'model_variant' not in protocol:
                from model_runner import M3DiffRunner
                runner = M3DiffRunner(args.model_path, protocol)
            else:
                raise ValueError('Unknown model variant; refusing to fall back to G')
        reply({'ok': True, 'result': info()})
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        reply({'ok': False, 'error': f'{type(exc).__name__}: {exc}'})
        return 1

    for line in sys.stdin:
        try:
            request = json.loads(line)
            action = request['action']
            if action == 'shutdown':
                reply({'ok': True, 'result': {'closed': True}})
                break
            with redirect_stdout(sys.stderr):
                if action == 'info':
                    result = info()
                elif action == 'compare':
                    with Image.open(request['before']) as before, Image.open(request['after']) as after:
                        result = runner.compare(before.convert('RGB'), after.convert('RGB'))
                else:
                    raise ValueError(f'Unknown action: {action}')
            reply({'ok': True, 'result': result})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            reply({'ok': False, 'error': f'{type(exc).__name__}: {exc}'})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
