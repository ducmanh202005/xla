"""Gradio bridge to the existing notebook runner. Never loads a model."""

import hashlib
from pathlib import Path
import threading

try:
    from .images import normalize_image
except ImportError:
    from images import normalize_image


class ModelAPI:
    def __init__(self, runner):
        self.runner = runner
        self.lock = threading.Lock()

    def status(self):
        with self.lock:
            info = self.runner.info()
        return {name: info[name] for name in (
            'model_instance_id', 'model_load_count', 'successful_requests',
            'load_seconds', 'protocol_id', 'protocol')}

    def compare(self, before, after):
        if not before or not after:
            raise ValueError('Cần chọn đủ ảnh trước và ảnh sau.')
        data = []
        for path in (before, after):
            path = Path(path)
            if not path.is_file() or path.stat().st_size > 20 * 1024 * 1024:
                raise ValueError('Ảnh không hợp lệ hoặc vượt 20 MB.')
            data.append(path.read_bytes())
        images = [normalize_image(content) for content in data]
        with self.lock:
            result = self.runner.compare(*images)
        return {**result, 'before_sha256': hashlib.sha256(data[0]).hexdigest(),
                'after_sha256': hashlib.sha256(data[1]).hexdigest()}


def build_demo(runner):
    import gradio as gr
    api = ModelAPI(runner)
    with gr.Blocks(title='M3Diff · Kết nối GPU',
                   analytics_enabled=False, delete_cache=(3600, 3600)) as app:
        gr.Markdown('## M³Diff · GPU đang sẵn sàng\n'
                    'Mở giao diện trên laptop để so sánh toàn ảnh hoặc vùng chọn. '
                    'Giữ phiên Colab hoạt động trong lúc sử dụng.')
        with gr.Row():
            before = gr.File(label='Ảnh trước', file_types=['.jpg', '.jpeg', '.png'], type='filepath')
            after = gr.File(label='Ảnh sau', file_types=['.jpg', '.jpeg', '.png'], type='filepath')
        result = gr.JSON(label='Kết quả')
        compare = gr.Button('So sánh', variant='primary')
        compare.click(api.compare, [before, after], result,
                      api_name='compare', concurrency_id='m3diff', concurrency_limit=1)
        status_button = gr.Button('Kiểm tra kết nối')
        status_button.click(api.status, None, result, api_name='status',
                            concurrency_id='m3diff', concurrency_limit=1)
    return app.queue(default_concurrency_limit=1, max_size=8)
