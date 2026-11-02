"""Export and verify the exact B128 adapter; CPU-only until preparation imports Torch.

The training checkpoint is our own pinned, SHA-verified artifact. Never load an
arbitrary .pt received from a third party with weights_only=False.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import zipfile


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def validate_commit(run, release):
    run = Path(run)
    if identity(read(run / 'protocol.json')) != release['parent_protocol']:
        raise ValueError('Sai protocol huấn luyện B128; không tự dùng checkpoint khác.')
    pointer = read(run / 'checkpoints/B/commit.json')
    if (pointer.get('arm') != 'B' or pointer.get('step') != release['step']
            or pointer.get('contract') != release['parent_protocol']
            or type(pointer.get('slot')) is not int or pointer['slot'] not in (0, 1)
            or pointer.get('sha256') != release['checkpoint_sha256']):
        raise ValueError('Cần đúng checkpoint B bước128 đã đánh giá, theo commit.json.')
    expected = {'protocol': release['parent_protocol'], 'weights': release['checkpoint_sha256'], 'step': release['step']}
    if read(run / 'weights_B.json') != expected:
        raise ValueError('weights_B.json không khớp kết quả full1570.')
    path = run / 'checkpoints/B' / f"slot{pointer['slot']}.pt"
    if sha256(path) != release['checkpoint_sha256']:
        raise ValueError('Checkpoint B hỏng hoặc sai SHA256; chưa nạp trọng số.')
    return path


def verify_adapter(folder, release):
    folder = Path(folder)
    manifest = read(folder / 'adapter_manifest.json')
    if manifest.get('release') != release or manifest.get('format') != 'm3diff_lora_tensors_v1':
        raise ValueError('Gói adapter không thuộc bản B128 đã chọn.')
    if sha256(folder / 'adapter_model.pt') != manifest.get('adapter_sha256'):
        raise ValueError('Adapter bị hỏng hoặc thiếu dữ liệu.')
    return manifest


def export_adapter(run, destination, release):
    destination = Path(destination)
    if (destination / 'adapter_manifest.json').exists():
        verify_adapter(destination, release)
        return destination
    path = validate_commit(run, release)  # Verify before trusted pickle deserialization.
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (state.get('arm'), state.get('contract'), state.get('step')) != ('B', release['parent_protocol'], release['step']):
        raise ValueError('Nội dung checkpoint không khớp commit.')
    weights = state.get('adapter')
    if not isinstance(weights, dict) or not weights:
        raise ValueError('Checkpoint không có adapter.')
    for key, value in weights.items():
        if ('lora_' not in key or not torch.is_tensor(value) or value.ndim != 2
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError('Trọng số LoRA không hợp lệ: ' + key)
    tensors = {key: value.detach().cpu().contiguous().clone() for key, value in weights.items()}
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / 'adapter_model.pt'
    temporary = target.with_suffix('.tmp')
    torch.save(tensors, temporary)
    loaded = torch.load(temporary, map_location='cpu', weights_only=True)
    if set(loaded) != set(tensors) or any(not torch.equal(loaded[k], v) for k, v in tensors.items()):
        raise ValueError('Xuất/nạp lại adapter không khớp.')
    os.replace(temporary, target)
    atomic_json(destination / 'adapter_manifest.json', {
        'format': 'm3diff_lora_tensors_v1', 'release': release,
        'adapter_sha256': sha256(target), 'size_bytes': target.stat().st_size,
        'tensor_shapes': {k: list(v.shape) for k, v in tensors.items()},
        'tensor_dtypes': {k: str(v.dtype) for k, v in tensors.items()},
        'note': 'Inference-only adapter; requires the original M3Diff checkpoint. No optimizer or RNG.',
    })
    verify_adapter(destination, release)
    return destination


def copy_checked(source, target, digest):
    source, target = Path(source), Path(target)
    if target.is_file() and sha256(target) == digest:
        return
    if not source.is_file():
        raise FileNotFoundError('Thiếu model gốc: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.incomplete')
    with source.open('rb') as src, temporary.open('wb') as dst:
        shutil.copyfileobj(src, dst, 8 * 1024**2)
    if sha256(temporary) != digest:
        raise ValueError('Sai checksum: ' + str(source))
    os.replace(temporary, target)


def prepare(storage, release_path, protocol_path):
    from huggingface_hub import snapshot_download
    storage = Path(storage)
    release = read(release_path)
    run = storage / 'MDP_Projector_v1/experiments/scst_lora_20261031'
    package = storage / 'demo_scst_B128'
    if not (package / 'adapter_manifest.json').exists():
        print('Xuất adapter B128 từ checkpoint đã đánh giá (CPU).', flush=True)
        export_adapter(run, package, release)
    verify_adapter(package, release)
    model = Path('/content/m3diff_train_checkpoint')
    for name, digest in release['base_files'].items():
        print('Kiểm tra model gốc:', name, flush=True)
        copy_checked(storage / 'M3Diff_checkpoint' / name, model / name, digest)
    v = release['vision']
    vision = Path(snapshot_download(v['repo'], revision=v['revision'],
        allow_patterns=list(release['vision_hashes']), cache_dir='/content/hf_cache'))
    for name, digest in release['vision_hashes'].items():
        if sha256(vision / name) != digest:
            raise ValueError('Sai SigLIP: ' + name)
    atomic_json(protocol_path, {
        'model_variant': 'B_SCST_128', 'adapter_path': str(package),
        'release_path': str(Path(release_path).resolve()), 'vision_path': str(vision),
        'prompt': release['question'], 'torch_dtype': 'bfloat16',
        'generation': release['generation'], 'mdp_projector_dtype': 'float32',
        'parent_protocol': release['parent_protocol'],
    })
    print('B128 READY:', package, '| Model:', model, flush=True)


def write_export_zip(folder, target, release):
    folder, target = Path(folder), Path(target)
    verify_adapter(folder, release)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name in ('adapter_model.pt', 'adapter_manifest.json'):
            z.write(folder / name, 'demo_scst_B128/' + name)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'export'])
    parser.add_argument('--storage', default='/content/drive/MyDrive/M3Diff_Storage')
    parser.add_argument('--release', required=True)
    parser.add_argument('--protocol')
    parser.add_argument('--zip')
    args = parser.parse_args()
    if args.command == 'prepare':
        if not args.protocol:
            parser.error('--protocol required')
        prepare(args.storage, args.release, args.protocol)
    else:
        if not args.zip:
            parser.error('--zip required')
        storage = Path(args.storage)
        release = read(args.release)
        folder = export_adapter(storage / 'MDP_Projector_v1/experiments/scst_lora_20261031', storage / 'demo_scst_B128', release)
        print(write_export_zip(folder, args.zip, release))
