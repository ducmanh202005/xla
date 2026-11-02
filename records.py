"""Offline history and portable reports. Credentials are never recorded."""

import base64
from datetime import datetime, timezone
from html import escape
from io import BytesIO
import json
from pathlib import Path
import uuid
import zipfile

IMAGE_NAMES = ('before', 'after', 'crop_before', 'crop_after')


def save_run(root: Path, record: dict, images: dict[str, bytes]) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    saved = {k: v for k, v in record.items()
             if k in ('mode', 'result', 'inputs', 'rois', 'protocol', 'pair_id', 'scope')}
    saved['run_id'] = uuid.uuid4().hex
    saved['saved_at_utc'] = datetime.now(timezone.utc).isoformat()
    saved['scope'] = 'interactive_demo_not_benchmark'
    directory = root / saved['run_id']
    directory.mkdir(exist_ok=False)
    try:
        for name, content in images.items():
            if name not in IMAGE_NAMES:
                raise ValueError('Tên ảnh báo cáo không hợp lệ.')
            (directory / (name + '.png')).write_bytes(content)
        (directory / 'record.json').write_text(
            json.dumps(saved, ensure_ascii=False, indent=2), encoding='utf-8')
    except BaseException:
        # Only files created in this unique directory are removed.
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()
        raise
    return directory


def list_runs(root: Path, limit=50) -> list[dict]:
    records = []
    for path in Path(root).glob('*/record.json'):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            if record.get('run_id') == path.parent.name:
                records.append(record)
        except (OSError, ValueError):
            continue
    return sorted(records, key=lambda item: item.get('saved_at_utc', ''), reverse=True)[:limit]


def load_run(root: Path, run_id: str):
    if len(run_id) != 32 or any(c not in '0123456789abcdef' for c in run_id):
        raise ValueError('Mã lần lưu không hợp lệ.')
    directory = Path(root) / run_id
    record = json.loads((directory / 'record.json').read_text(encoding='utf-8'))
    images = {name: (directory / (name + '.png')).read_bytes()
              for name in IMAGE_NAMES if (directory / (name + '.png')).is_file()}
    return record, images


def export_html(record: dict, images: dict[str, bytes]) -> str:
    text = lambda value: escape(str(value), quote=True)
    mode = 'Toàn ảnh' if record['mode'] == 'full' else 'Vùng do người dùng chọn'
    labels = {'before': 'Ảnh trước', 'after': 'Ảnh sau',
              'crop_before': 'Vùng trước', 'crop_after': 'Vùng sau'}
    figures = ''.join(
        '<figure><img alt="' + labels[name] + '" src="data:image/png;base64,'
        + base64.b64encode(content).decode('ascii') + '"><figcaption>'
        + labels[name] + '</figcaption></figure>'
        for name, content in images.items() if name in labels)
    result = record['result']
    roi_note = ('<p>Vùng được chọn thủ công. Vị trí trong mô tả được hiểu theo ảnh cắt.</p>'
                if record['mode'] == 'roi' else '')
    return '''<!doctype html><html lang="vi"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Báo cáo M³Diff</title><style>body{font:17px/1.6 system-ui;color:#172e46;
background:#f4f7fa;max-width:1100px;margin:40px auto;padding:24px}h1{font-size:36px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}
figure{margin:0}img{width:100%;max-height:520px;object-fit:contain;background:white}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#fff;padding:20px}
@media(max-width:640px){.grid{grid-template-columns:1fr}}</style><h1>M³Diff / Báo cáo so sánh</h1>''' + (
        '<p>' + text(mode) + ' · ' + text(record.get('saved_at_utc', '')) + '</p>'
        + '<p>Đây là kết quả demo đã lưu, không phải kết quả benchmark.</p>'
        + roi_note + '<div class="grid">' + figures + '</div><h2>Mô tả của model</h2><pre>'
        + text(result['caption']) + '</pre><p>Thời gian suy luận: '
        + text(result['elapsed_seconds']) + ' giây.</p><details><summary>Dữ liệu lần chạy</summary><pre>'
        + text(json.dumps(record, ensure_ascii=False, indent=2)) + '</pre></details></html>')


def export_zip(record: dict, images: dict[str, bytes]) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('report.html', export_html(record, images))
        archive.writestr('record.json', json.dumps(record, ensure_ascii=False, indent=2))
        for name, content in images.items():
            if name in IMAGE_NAMES:
                archive.writestr(name + '.png', content)
    return output.getvalue()
