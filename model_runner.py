"""Persistent inference using the author's OmniDiff-Real preprocessing."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import threading
from time import perf_counter
import uuid

from PIL import Image


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _validate_protocol(protocol: dict) -> dict:
    result = deepcopy(protocol)
    required = {
        'prompt', 'conv_mode', 'torch_dtype', 'flash_attention', 'temperature',
        'top_p', 'num_beams', 'max_new_tokens', 'use_cache',
    }
    if set(result) != required:
        raise ValueError(f'Protocol keys must be exactly: {sorted(required)}')
    if not isinstance(result['prompt'], str) or result['prompt'].count('<image>') != 2:
        raise ValueError('Prompt must contain exactly two <image> tokens')
    if result['torch_dtype'] not in {'float16', 'bfloat16', 'float32'}:
        raise ValueError('Unsupported torch dtype')
    for name in ('max_new_tokens', 'num_beams'):
        if type(result[name]) is not int or result[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not isinstance(result['temperature'], (int, float)) or result['temperature'] < 0:
        raise ValueError('temperature must be nonnegative')
    if result['top_p'] is not None and not 0 < result['top_p'] <= 1:
        raise ValueError('top_p must be None or in (0, 1]')
    if type(result['use_cache']) is not bool or type(result['flash_attention']) is not bool:
        raise ValueError('use_cache and flash_attention must be booleans')
    return result


class M3DiffRunner:
    """One loaded checkpoint, reused for each sequential compare request."""

    def __init__(self, model_path: str, protocol: dict):
        self._protocol = _validate_protocol(protocol)
        self.model_path = Path(model_path).expanduser().absolute()
        if not (self.model_path / 'config.json').is_file():
            raise FileNotFoundError(f'Checkpoint config missing: {self.model_path}')
        self._torch = importlib.import_module('torch')
        if not self._torch.cuda.is_available():
            raise RuntimeError('M3Diff requires the prepared Colab CUDA runtime')
        self._evaluation = importlib.import_module('llava.eval.model_vqa_qwen_OmniDiff_Real')
        if self._protocol['conv_mode'] not in self._evaluation.conv_templates:
            raise ValueError('Unknown conversation template')
        self._dtype = getattr(self._torch, self._protocol['torch_dtype'])
        self._lock = threading.Lock()
        self.model_instance_id = uuid.uuid4().hex
        self.successful_requests = 0
        self.model_load_count = 0

        self._evaluation.disable_torch_init()
        started = perf_counter()
        self.tokenizer, self.model, self.image_processor, _ = self._evaluation.load_pretrained_model(
            str(self.model_path), None,
            self._evaluation.get_model_name_from_path(str(self.model_path)),
            use_flash_attn=self._protocol['flash_attention'], torch_dtype=self._dtype,
        )
        self.model.eval()
        self._torch.cuda.synchronize()
        self.load_seconds = perf_counter() - started
        self.model_load_count += 1
        self.runtime_manifest = self._make_manifest()
        self.protocol_id = _canonical_hash({
            'protocol': self._protocol,
            'source_sha256': self.runtime_manifest['source_sha256'],
            'checkpoint_config_sha256': self.runtime_manifest['checkpoint_config_sha256'],
            'package_versions': self.runtime_manifest['package_versions'],
        })

    @property
    def protocol(self) -> dict:
        return deepcopy(self._protocol)

    def _make_manifest(self) -> dict:
        source_root = Path(self._evaluation.__file__).resolve().parents[2]
        names = (
            'llava/eval/model_vqa_qwen_OmniDiff_Real.py',
            'llava/model/builder.py', 'llava/model/llava_arch.py',
            'llava/model/multimodal_resampler/mdp.py',
            'llava/model/multimodal_resampler/builder.py',
        )
        hashes = {name: _sha256(source_root / name) for name in names if (source_root / name).is_file()}
        try:
            revision = subprocess.check_output(
                ['git', '-C', str(source_root), 'rev-parse', 'HEAD'],
                text=True, stderr=subprocess.DEVNULL, timeout=10,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            revision = None
        versions = {}
        for name in ('torch', 'torchvision', 'transformers', 'tokenizers', 'huggingface-hub', 'numpy', 'pillow'):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
        return {
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'python': platform.python_version(), 'source_root': str(source_root),
            'source_revision': revision, 'source_sha256': hashes,
            'package_versions': versions, 'gpu': self._torch.cuda.get_device_name(0),
            'checkpoint_path': str(self.model_path),
            'checkpoint_config_sha256': _sha256(self.model_path / 'config.json'),
            'checkpoint_shards': {
                path.name: path.stat().st_size for path in sorted(self.model_path.glob('*.safetensors'))
            },
            'checkpoint_identity_note': 'Config hash and shard sizes only; weights were not fully hashed.',
        }

    def compare(self, before: Image.Image, after: Image.Image) -> dict:
        if not isinstance(before, Image.Image) or not isinstance(after, Image.Image):
            raise TypeError('compare expects two PIL images, in before/after order')
        with self._lock:
            self._torch.cuda.synchronize()
            started = perf_counter()
            evaluation = self._evaluation
            settings = self._protocol
            conv = evaluation.conv_templates[settings['conv_mode']].copy()
            input_ids = evaluation.preprocess_qwen(
                [{'from': 'human', 'value': settings['prompt']}, {'from': 'gpt', 'value': None}],
                self.tokenizer, has_image=True,
            ).cuda()
            image_tensors = [
                evaluation.process_images([image.convert('RGB')], self.image_processor, self.model.config).to(self._dtype)
                for image in (before, after)
            ]
            image_tensors = [self._torch.cat(image_tensors, dim=0)]
            stop_str = conv.sep if conv.sep_style != evaluation.SeparatorStyle.TWO else conv.sep2
            stopping = evaluation.KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
            with self._torch.inference_mode():
                output_ids = self.model.generate(
                    inputs=input_ids, images=image_tensors,
                    do_sample=settings['temperature'] > 0, temperature=settings['temperature'],
                    top_p=settings['top_p'], num_beams=settings['num_beams'],
                    max_new_tokens=settings['max_new_tokens'], use_cache=settings['use_cache'],
                    stopping_criteria=[stopping],
                )
            caption = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if stop_str and caption.endswith(stop_str):
                caption = caption[:-len(stop_str)].strip()
            if not caption:
                raise RuntimeError('Model returned an empty caption')
            self._torch.cuda.synchronize()
            self.successful_requests += 1
            return {
                'caption': caption, 'elapsed_seconds': round(perf_counter() - started, 4),
                'protocol_id': self.protocol_id, 'model_instance_id': self.model_instance_id,
                'model_load_count': self.model_load_count,
                'successful_requests': self.successful_requests,
            }
