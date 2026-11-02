"""Notebook bridge to a model process; imports no Torch or Transformers."""

import atexit
import json
from pathlib import Path
import queue
import subprocess
from tempfile import TemporaryDirectory
import threading

PREFIX = 'M3DIFF_RPC:'


class WorkerClient:
    def __init__(self, python, worker, model_path, protocol, work, *, source=None, timeout=1800):
        self.timeout = timeout
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.log_path = self.work / 'model_worker.log'
        self._messages = queue.Queue()
        self._lock = threading.Lock()
        self._log = self.log_path.open('a', encoding='utf-8')
        self._log.write('\n--- Starting isolated model worker ---\n')
        self._log.flush()
        try:
            self.process = subprocess.Popen(
                [str(python), '-u', '-B', str(worker), '--model-path', str(model_path), '--protocol', str(protocol)],
                cwd=str(source or self.work), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._log, text=True, encoding='utf-8', bufsize=1,
            )
        except Exception:
            self._log.close()
            raise
        threading.Thread(target=self._read_responses, daemon=True).start()
        try:
            self.initial_info = self._receive()
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def _read_responses(self):
        try:
            for line in self.process.stdout:
                if line.startswith(PREFIX):
                    self._messages.put(json.loads(line[len(PREFIX):]))
        except Exception as exc:
            self._messages.put({'ok': False, 'error': f'Invalid worker response: {exc}'})
        finally:
            self._messages.put(None)

    def _receive(self):
        try:
            message = self._messages.get(timeout=self.timeout)
        except queue.Empty:
            self.close()  # A late response must never be consumed as the next request.
            raise RuntimeError(f'Model worker timed out; stopped it. See {self.log_path}') from None
        if message is None:
            raise RuntimeError(f'Model worker stopped. See {self.log_path}')
        if not message['ok']:
            raise RuntimeError(f"{message['error']}\nSee {self.log_path}")
        return message['result']

    def _request(self, payload):
        with self._lock:
            if self.process.poll() is not None:
                raise RuntimeError(f'Model worker is no longer running. See {self.log_path}')
            try:
                self.process.stdin.write(json.dumps(payload) + '\n')
                self.process.stdin.flush()
            except BaseException:
                self.close()
                raise
            try:
                return self._receive()
            except (KeyboardInterrupt, SystemExit):
                # A late reply must never be read as the next image pair's result.
                self.close()
                raise

    def info(self):
        return self._request({'action': 'info'})

    def compare(self, before, after):
        with TemporaryDirectory(prefix='m3diff_request_', dir=self.work) as directory:
            before_path = Path(directory) / 'before.png'
            after_path = Path(directory) / 'after.png'
            before.convert('RGB').save(before_path)
            after.convert('RGB').save(after_path)
            return self._request({'action': 'compare', 'before': str(before_path), 'after': str(after_path)})

    def close(self):
        process = self.process
        if process.poll() is None:
            try:
                process.stdin.write('{"action":"shutdown"}\n')
                process.stdin.flush()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.stdout and not process.stdout.closed:
            process.stdout.close()
        if not self._log.closed:
            self._log.close()
