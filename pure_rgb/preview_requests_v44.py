"""Read preview supersession controls while the CPU is processing a frame."""

import hashlib
import json
from pathlib import Path
import queue
import threading


class PreviewSuperseded(Exception):
    pass


def preview_metering_path(current_record):
    return Path(current_record["path"]).with_suffix(".metering.json")


def read_preview_metering(event):
    expected = event.get("metering_sidecar")
    if expected is None:
        return None
    with preview_metering_path(event["records"][0]).open("rb") as stream:
        payload = stream.read(128 * 1024**2 + 1)
    if len(payload) > 128 * 1024**2 or hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("预览分区测量回执不完整")
    return json.loads(payload)


class IncomingRequests:
    def __init__(self, stream):
        self.stream = stream
        self.pending = queue.Queue()
        self.latest_token = self.active_token = None
        self.lock = threading.Lock()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.stream:
                try:
                    document = json.loads(line)
                except (ValueError, TypeError):
                    document = None
                if isinstance(document, dict) and type(document.get("token")) is int:
                    with self.lock:
                        if self.latest_token is None or document["token"] > self.latest_token:
                            self.latest_token = document["token"]
                    if document.get("action") == "supersede":
                        continue
                self.pending.put(line)
        finally:
            self.pending.put(None)

    def __iter__(self):
        while True:
            line = self.pending.get()
            if line is None:
                return
            yield line

    def begin(self, request):
        with self.lock:
            self.active_token = request.get("token")
        self.check()

    def check(self):
        with self.lock:
            active = self.active_token
        self.check_token(active)

    def check_token(self, token):
        with self.lock:
            latest = self.latest_token
        if type(token) is int and type(latest) is int and token < latest:
            raise PreviewSuperseded("预览请求已被更新的调整替代")

    def checkpoint(self):
        with self.lock:
            token = self.active_token
        return lambda: self.check_token(token)

    def finish(self):
        with self.lock:
            self.active_token = None
