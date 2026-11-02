"""Laptop API client; temporary image files are removed on success and failure."""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse


class ConnectionFailure(RuntimeError):
    pass


class DemoClient:
    def __init__(self, url, username, password):
        parsed = urlparse(url.strip())
        if (parsed.scheme not in ('https', 'http') or not parsed.hostname
                or parsed.username or parsed.password
                or (parsed.scheme == 'http' and parsed.hostname not in ('localhost', '127.0.0.1'))):
            raise ValueError('Dán link HTTPS do Colab cung cấp.')
        self.url = url.strip().rstrip('/')
        self.auth = (username, password)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from gradio_client import Client
            self._client = Client(self.url, auth=self.auth, verbose=False,
                                  httpx_kwargs={'timeout': 60})
        return self._client

    def status(self):
        try:
            result = self._get_client().predict(api_name='/status')
            if not isinstance(result, dict) or not result.get('model_instance_id'):
                raise ValueError('invalid status response')
            return result
        except Exception as exc:
            self._client = None
            raise ConnectionFailure('Không kết nối được GPU. Kiểm tra link, tài khoản, '
                                    'mật khẩu và phiên Colab rồi thử lại.') from exc

    def compare(self, before_bytes: bytes, after_bytes: bytes):
        try:
            from gradio_client import handle_file
            with TemporaryDirectory(prefix='m3diff_send_') as temporary:
                paths = [Path(temporary) / name for name in ('before.png', 'after.png')]
                for path, data in zip(paths, (before_bytes, after_bytes)):
                    path.write_bytes(data)
                result = self._get_client().predict(
                    handle_file(str(paths[0])), handle_file(str(paths[1])), api_name='/compare')
            if not isinstance(result, dict) or not isinstance(result.get('caption'), str) or not result['caption'].strip():
                raise ValueError('invalid caption response')
            for side, data in [('before', before_bytes), ('after', after_bytes)]:
                if result.get(side + '_sha256') != hashlib.sha256(data).hexdigest():
                    raise ValueError('image pair mismatch')
            for name in ('elapsed_seconds', 'model_instance_id', 'model_load_count', 'protocol_id'):
                if name not in result:
                    raise ValueError('missing response metadata')
            return result
        except Exception as exc:
            self._client = None
            raise ConnectionFailure('Lượt so sánh chưa hoàn tất. Ảnh và vùng chọn vẫn được giữ. '
                                    'Kiểm tra Colab rồi bấm thử lại; kết quả cũ không được dùng thay.') from exc
