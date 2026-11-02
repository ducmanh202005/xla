"""Local demo UI. Inference runs only from the two explicit compare buttons."""

import hashlib
import json
import os
from pathlib import Path
import time

import streamlit as st

from client import DemoClient, ConnectionFailure
from images import normalize_image, image_bytes, crop_roi
from records import save_run, list_runs, load_run, export_html, export_zip
from roi_selector import select_roi
from state import pair_changed, roi_changed

HERE = Path(__file__).resolve().parent
HISTORY = Path(os.environ.get('M3DIFF_HISTORY_DIR', str(HERE / 'history')))
st.set_page_config(page_title='M³Diff Studio', page_icon='🔎', layout='wide')
st.markdown('''<style>
.stApp{background:#f6f8fb;color:#17324d}h1,h2,h3{color:#142d46}
[data-testid="stSidebar"]{background:#eaf0f6}
[data-testid="stMetric"]{background:white;border:1px solid #dbe3ec;border-radius:12px;padding:14px}
button[kind="primary"]{background:#0f766e;border-color:#0f766e}
.hero{color:#0f766e;font-size:13px;font-weight:700;letter-spacing:.12em;margin-top:8px}
.stCaption{color:#475569}
</style>''', unsafe_allow_html=True)
ss = st.session_state
if 'workspace' not in ss:
    ss.workspace = {}
ws = ss.workspace


def connect_panel():
    with st.sidebar:
        st.markdown('### Kết nối GPU')
        st.caption('Web ở laptop · model ở Colab')
        configuration = st.file_uploader('Mở file kết nối từ Colab', type=['json'], key='connection_file')
        if configuration is not None:
            digest = hashlib.sha256(configuration.getvalue()).hexdigest()
            if ss.get('connection_file_id') != digest:
                try:
                    values = json.loads(configuration.getvalue())
                    for name in ('url', 'username', 'password'):
                        if not isinstance(values.get(name), str):
                            raise ValueError('missing fields')
                    ss.gpu_url, ss.gpu_user, ss.gpu_password = (values[k] for k in ('url', 'username', 'password'))
                    ss.connection_file_id = digest
                except (ValueError, TypeError):
                    st.error('File kết nối không đúng. Bạn vẫn có thể nhập thông tin bên dưới.')
        url = st.text_input('Link Colab', key='gpu_url', placeholder='https://…gradio.live')
        ss.setdefault('gpu_user', 'm3diff')
        username = st.text_input('Tài khoản', key='gpu_user')
        password = st.text_input('Mật khẩu', type='password', key='gpu_password')
        if st.button('Kết nối / kiểm tra', type='primary', width='stretch'):
            ss.pop('gpu_client', None)
            ss.pop('gpu_info', None)
            ss.pop('connected_signature', None)
            try:
                if not username or not password:
                    raise ValueError('Nhập đủ tài khoản và mật khẩu từ Colab.')
                with st.spinner('Đang kiểm tra kết nối…'):
                    candidate = DemoClient(url, username, password)
                    info = candidate.status()
                ss.gpu_client, ss.gpu_info = candidate, info
                ss.connected_signature = (url.strip().rstrip('/'), username, password)
                # A different worker cannot inherit results presented as its own.
                if ws.get('connected_model') != info['model_instance_id']:
                    ws.update(full_result=None, roi_result=None, connected_model=info['model_instance_id'])
                st.success('Đã kết nối GPU')
            except (ValueError, ConnectionFailure) as exc:
                st.error(str(exc))
        ready = ss.get('connected_signature') == (url.strip().rstrip('/'), username, password) and 'gpu_client' in ss
        if ready:
            info = ss.gpu_info
            st.success('GPU đã kết nối')
            st.caption('Chỉ chạy model khi bạn bấm nút so sánh.')
            with st.expander('Thông tin phiên'):
                st.write('Lần nạp model trong runner:', info['model_load_count'])
                st.code(info['model_instance_id'], language=None)
        else:
            st.info('Có thể chọn ảnh và xem lịch sử trước khi kết nối.')
        st.caption('Giữ Colab hoạt động. Không chia sẻ file kết nối hoặc mật khẩu.')
        return ready


def run_comparison(mode, images, names, ready):
    if not ready:
        st.info('Kết nối GPU ở bên trái trước khi so sánh.')
        return
    if ws.get('busy'):
        st.info('Đang chờ lượt so sánh hiện tại hoàn tất.')
        return
    request_pair = ws['pair_id']
    request_rois = dict(ws.get('rois', {}))
    request_client, info = ss.gpu_client, ss.gpu_info
    ws['busy'] = True
    ws[mode + '_result'] = None
    ws['last_error'] = None
    try:
        sent = images if mode == 'full' else [crop_roi(images[i], request_rois.get(side))
                                             for i, side in enumerate(('before', 'after'))]
        payloads = [image_bytes(image) for image in sent]
        if any(len(data) > 20 * 1024 * 1024 for data in payloads):
            raise ValueError('Ảnh sau khi giải mã vượt 20 MB. Hãy giảm kích thước ảnh rồi thử lại.')
        start = time.perf_counter()
        with st.spinner('Đang gửi hai ảnh và chờ model mô tả…'):
            result = request_client.compare(*payloads)
        if (ws.get('pair_id') != request_pair or ss.get('gpu_client') is not request_client
                or (mode == 'roi' and ws.get('rois', {}) != request_rois)):
            return  # A late response must not be attached to a newly selected pair/ROI.
        if (result['model_instance_id'] != info['model_instance_id']
                or result['protocol_id'] != info['protocol_id']):
            ss.pop('gpu_client', None)
            raise ValueError('Phiên model đã đổi. Bấm kết nối lại trước khi so sánh.')
        result['total_seconds'] = round(time.perf_counter() - start, 3)
        record = {'mode': mode, 'result': result, 'protocol': info['protocol'],
                  'pair_id': request_pair,
                  'inputs': [{'name': name, 'width': image.width, 'height': image.height}
                             for name, image in zip(names, images)],
                  'rois': request_rois if mode == 'roi' else {}}
        evidence = {'before': image_bytes(images[0]), 'after': image_bytes(images[1])}
        if mode == 'roi':
            evidence.update(crop_before=payloads[0], crop_after=payloads[1])
        ws[mode + '_result'] = {'record': record, 'images': evidence}
    except (ValueError, ConnectionFailure) as exc:
        if ws.get('pair_id') == request_pair:
            ws['last_error'] = {'mode': mode, 'message': str(exc)}
    finally:
        ws['busy'] = False


def result_panel(mode):
    error = ws.get('last_error')
    if error and error['mode'] == mode:
        st.error(error['message'])
    saved = ws.get(mode + '_result')
    if not saved:
        return
    result = saved['record']['result']
    st.markdown('#### Mô tả của model')
    st.write(result['caption'])
    a, b = st.columns(2)
    a.metric('Model xử lý', f"{result['elapsed_seconds']:.2f} giây")
    b.metric('Gửi ảnh + chờ + nhận', f"{result['total_seconds']:.2f} giây")
    if mode == 'roi':
        st.caption('Trái/phải trong mô tả được hiểu theo ảnh cắt. Vùng này do bạn chọn thủ công.')
    with st.expander('Chi tiết lần chạy'):
        st.json({k: result[k] for k in ('model_instance_id', 'model_load_count', 'protocol_id')})
    if st.button('Lưu kết quả ' + ('toàn ảnh' if mode == 'full' else 'vùng chọn'), key='save_' + mode):
        try:
            location = save_run(HISTORY, saved['record'], saved['images'])
            st.success('Đã lưu. Mở tab Lịch sử & báo cáo để tải xuống.')
            ws['last_saved'] = location.name
        except OSError:
            st.error('Chưa lưu được trên máy. Kiểm tra dung lượng hoặc quyền ghi thư mục demo.')


def select_panel(image, side, title):
    st.markdown('##### ' + title)
    revision = ws.get('selection_revision', 0)
    value = ws.get('rois', {}).get(side)
    event = select_roi(image, key=f"roi_{ws['pair_id']}_{side}", value=value, revision=revision)
    if isinstance(event, dict) and event.get('revision') == revision:
        if value != event.get('roi'):
            roi_changed(ws, side, event.get('roi'))
            ws['selection_revision'] = revision + 1
            st.rerun()
    with st.expander('Nhập vùng bằng bàn phím'):
        st.caption('Bốn số phần trăm: trái, trên, phải, dưới. Ví dụ: 10, 10, 70, 80.')
        with st.form(f'coords_{side}_{ws["pair_id"]}_{revision}'):
            text = st.text_input('Tọa độ (%)', value=', '.join(str(round(v * 100, 2)) for v in (value or [.1, .1, .9, .9])))
            if st.form_submit_button('Áp dụng vùng'):
                try:
                    proposed = [float(v.strip()) / 100 for v in text.split(',')]
                    crop_roi(image, proposed)
                    roi_changed(ws, side, proposed)
                    ws['selection_revision'] = revision + 1
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))


def comparison_tab(ready):
    st.markdown('### 1. Chọn cặp ảnh')
    source = st.radio('Nguồn ảnh', ['Tải ảnh của bạn', 'Ví dụ có sẵn'], horizontal=True)
    blobs, names = [], []
    if source == 'Tải ảnh của bạn':
        for col, label, key in zip(st.columns(2), ('Ảnh trước', 'Ảnh sau'), ('upload_before', 'upload_after')):
            with col:
                uploaded = st.file_uploader(label, type=['jpg', 'jpeg', 'png'], key=key)
                blobs.append(uploaded.getvalue() if uploaded is not None else None)
                names.append(uploaded.name if uploaded is not None else '')
    else:
        selected = st.selectbox('Cặp ví dụ', ['Cặp 1 · ảnh 000001 → 000002', 'Cặp 2 · ảnh 000001 → 000003', 'Đối chiếu · cùng một ảnh'])
        after_id = '000003' if selected.startswith('Cặp 2') else ('000001' if selected.startswith('Đối') else '000002')
        folder = HERE.parent / 'drive-download-20260916T101111Z-1-001'
        paths = [folder / f'real_test_{i}.jpg' for i in ('000001', after_id)]
        if all(path.is_file() for path in paths):
            blobs, names = [p.read_bytes() for p in paths], [p.name for p in paths]
        else:
            st.info('Máy này chưa có ảnh ví dụ. Chọn “Tải ảnh của bạn”.')
            pair_changed(ws, None)
            return
        st.caption('Ví dụ chỉ điền ảnh; kết quả sẽ được model chạy khi bạn bấm so sánh.')
    if not all(blobs):
        pair_changed(ws, None)
        st.info('Chọn đủ hai ảnh JPG/PNG. Mỗi ảnh tối đa 20 MB và 25 triệu pixel.')
        return
    pair_id = hashlib.sha256(b'\0'.join(hashlib.sha256(b).digest() for b in blobs)).hexdigest()
    pair_changed(ws, pair_id)
    try:
        if ws.get('decoded_pair') != pair_id:
            ws['images'] = [normalize_image(b) for b in blobs]
            ws['decoded_pair'] = pair_id
        images = ws['images']
    except ValueError as exc:
        st.error(str(exc))
        return
    for col, image, name, label in zip(st.columns(2), images, names, ('TRƯỚC', 'SAU')):
        with col:
            st.image(image, caption=f'{label} · {name} · {image.width} × {image.height}', width='stretch')
    st.markdown('### 2. So sánh toàn ảnh')
    st.caption('Gửi cặp ảnh đầy đủ tới M³Diff để mô tả thay đổi.')
    if st.button('So sánh toàn ảnh', type='primary', disabled=not ready or ws.get('busy', False)):
        run_comparison('full', images, names, ready)
    result_panel('full')
    st.divider()
    st.markdown('### 3. Tập trung vào một vùng')
    st.caption('Kéo chuột trên mỗi ảnh để chọn vùng tương ứng. Bạn có thể chọn hai vùng khác nhau.')
    a, b = st.columns(2)
    if a.button('Chép vùng trước → sau', disabled=not ws.get('rois', {}).get('before')):
        roi_changed(ws, 'after', ws['rois']['before'])
        ws['selection_revision'] += 1
        st.rerun()
    if b.button('Xóa cả hai vùng'):
        roi_changed(ws, 'before', None)
        roi_changed(ws, 'after', None)
        ws['selection_revision'] += 1
        st.rerun()
    st.caption('Chép vùng dùng cùng tỷ lệ vị trí trên hai ảnh, không tự căn chỉnh vật thể.')
    for col, image, side, title in zip(st.columns(2), images, ('before', 'after'), ('Vùng ở ảnh trước', 'Vùng ở ảnh sau')):
        with col:
            select_panel(image, side, title)
    crops, problem = [], None
    for image, side in zip(images, ('before', 'after')):
        try:
            crops.append(crop_roi(image, ws.get('rois', {}).get(side)))
        except ValueError as exc:
            problem = str(exc)
    if len(crops) == 2:
        for col, crop, title in zip(st.columns(2), crops, ('Ảnh cắt thực tế · trước', 'Ảnh cắt thực tế · sau')):
            col.image(crop, caption=f'{title} · {crop.width} × {crop.height}', width='stretch')
    else:
        st.info('Chọn vùng ở cả hai ảnh, mỗi vùng ít nhất 32 × 32 pixel. ' + (problem or ''))
    if st.button('So sánh hai vùng đã chọn', type='primary', disabled=not ready or len(crops) != 2 or ws.get('busy', False)):
        run_comparison('roi', images, names, ready)
    result_panel('roi')


def history_tab():
    st.markdown('### Lịch sử & báo cáo')
    st.caption('Kết quả bạn đã bấm lưu trên máy này. Xem lại và tải báo cáo không cần GPU.')
    records = list_runs(HISTORY)
    if not records:
        st.info('Chưa có kết quả được lưu. So sánh ảnh rồi bấm “Lưu kết quả”.')
        return
    def label(index):
        r = records[index]
        mode = 'Toàn ảnh' if r['mode'] == 'full' else 'Vùng chọn'
        return f"{r['saved_at_utc'][:19].replace('T', ' ')} UTC · {mode} · {r['run_id'][:6]}"
    index = st.selectbox('Lần đã lưu (tối đa 50 lần gần nhất)', range(len(records)), format_func=label)
    try:
        record, images = load_run(HISTORY, records[index]['run_id'])
        st.info('Đang xem kết quả đã lưu: ' + ('toàn ảnh' if record['mode'] == 'full' else 'vùng chọn thủ công'))
        for col, name, title in zip(st.columns(2), ('before', 'after'), ('Ảnh trước', 'Ảnh sau')):
            if name in images:
                col.image(images[name], caption=title, width='stretch')
        if record['mode'] == 'roi':
            for col, name, title in zip(st.columns(2), ('crop_before', 'crop_after'), ('Vùng trước', 'Vùng sau')):
                if name in images:
                    col.image(images[name], caption=title, width='stretch')
        st.write(record['result']['caption'])
        st.caption('Thời gian model: ' + str(record['result']['elapsed_seconds']) + ' giây')
        prefix = 'm3diff_' + record['run_id'][:8]
        a, b, c = st.columns(3)
        a.download_button('Tải báo cáo HTML', export_html(record, images), prefix + '.html', 'text/html')
        b.download_button('Tải dữ liệu JSON', json.dumps(record, ensure_ascii=False, indent=2), prefix + '.json', 'application/json')
        c.download_button('Tải gói ZIP', export_zip(record, images), prefix + '.zip', 'application/zip')
    except (OSError, ValueError, KeyError):
        st.error('Lần lưu này không đọc được. Chọn lần khác hoặc kiểm tra thư mục demo/history.')


ready = connect_panel()
st.markdown('<div class="hero">KHÁM PHÁ THAY ĐỔI GIỮA HAI ẢNH</div>', unsafe_allow_html=True)
st.title('M³Diff Studio')
st.write('Xem toàn cảnh. Chọn vùng cần chú ý. Lưu bằng chứng để trình bày.')
compare, history, guide = st.tabs(['So sánh ảnh', 'Lịch sử & báo cáo', 'Hướng dẫn'])
with compare:
    comparison_tab(ready)
with history:
    history_tab()
with guide:
    st.markdown('''### Bắt đầu trong ba bước
1. Upload **M3Diff_Demo_Web_7_O.ipynb** lên Colab, bật GPU rồi chạy **1 → 2 → 3 → 6 → 7**. Ô 4–5 tùy chọn.
2. Tải file kết nối, mở nó ở thanh bên trái rồi bấm **Kết nối / kiểm tra**.
3. Chọn hai ảnh, so sánh toàn ảnh hoặc khoanh hai vùng và so sánh riêng.

### Phần nhóm bổ sung
Khoanh vùng thủ công, xem ảnh cắt trước khi chạy, lưu lịch sử và xuất báo cáo có ảnh.
Notebook hiện tại dùng M³Diff cùng adapter B128 (LoRA + SCST).
Phương pháp và kết quả đánh giá được ghi trong CAI_TIEN_SCST_VA_KET_QUA.md của thư mục demo.

### Đọc kết quả đúng cách
Model mô tả bằng tiếng Anh và có thể mô tả sai hoặc bỏ sót. Vùng chọn do người dùng
quyết định; hệ thống chưa tự xác định vị trí thay đổi. Khi dùng vùng cắt, các từ
trái/phải chỉ vị trí trong vùng cắt. Thời gian tổng có cả gửi ảnh và chờ mạng.

Đổi cặp ảnh sẽ xóa kết quả hiện tại; đổi vùng sẽ xóa kết quả vùng. Lịch sử chỉ có
những lần bạn chủ động lưu và vẫn mở được khi Colab đã ngắt.
''')
