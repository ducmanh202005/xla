"""Persistent B128 inference matching the full1570 evaluation path."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib
from importlib import metadata
from pathlib import Path
import random
import threading
from time import perf_counter
import uuid

from PIL import Image
from scst_artifacts import read, sha256, identity, verify_adapter
from inference_tokens import token_layout


def validate_protocol(protocol, release):
    if (protocol.get('model_variant') != 'B_SCST_128'
            or protocol.get('prompt') != release['question']
            or protocol.get('generation') != release['generation']
            or protocol.get('torch_dtype') != 'bfloat16'
            or protocol.get('mdp_projector_dtype') != 'float32'
            or protocol.get('parent_protocol') != release['parent_protocol']):
        raise ValueError('Cấu hình demo khác bản B đã đánh giá full1570.')


def restore_adapter(model, state, get_state, set_state, torch):
    expected = get_state(model)
    if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
        raise ValueError('Sai keys hoặc kích thước adapter; không chạy với LoRA thiếu trọng số.')
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Adapter chứa NaN/Inf.')
    set_state(model, state)
    actual = get_state(model)
    if any(not torch.equal(actual[k].detach().cpu(), v.cpu()) for k, v in state.items()):
        raise ValueError('Nạp adapter không khớp từng tensor.')


class SCSTRunner:
    def __init__(self, model_path, protocol):
        self._protocol = deepcopy(protocol)
        release = read(protocol['release_path'])
        validate_protocol(protocol, release)
        self.model_path = Path(model_path).expanduser().absolute()
        manifest = verify_adapter(protocol['adapter_path'], release)
        # Fail on a wrong base/vision before allocating a model on GPU.
        for name, digest in release['base_files'].items():
            if sha256(self.model_path / name) != digest:
                raise ValueError('Model gốc không khớp: ' + name)
        vision_path = Path(protocol['vision_path'])
        for name, digest in release['vision_hashes'].items():
            if sha256(vision_path / name) != digest:
                raise ValueError('SigLIP không khớp: ' + name)
        import llava
        source_root = Path(llava.__file__).resolve().parent.parent
        for name, digest in release['source_files'].items():
            if sha256(source_root / name) != digest:
                raise ValueError('Source demo khác bản đã đánh giá: ' + name)
        torch = importlib.import_module('torch')
        self._torch = torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Demo B cần GPU hỗ trợ BF16; chọn A100 như lượt đánh giá.')
        from transformers import AutoTokenizer, GenerationConfig
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM, LlavaQwenConfig
        from llava.mm_utils import process_images
        self._process_images = process_images
        random.seed(release['seed']); torch.manual_seed(release['seed']); torch.cuda.manual_seed_all(release['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        config = LlavaQwenConfig.from_pretrained(str(self.model_path), local_files_only=True)
        if (config.mm_resampler_type != 'mdp' or config.image_aspect_ratio != 'pad'
                or getattr(config, 'attention_dropout', 0.) != 0.):
            raise ValueError('Kiến trúc model không khớp.')
        config.unfreeze_mm_vision_tower = True  # Match evaluator loading, then freeze all parameters.
        config.mm_vision_tower = str(vision_path)
        config.use_cache = False
        config.tokenizer_model_max_length = 4096
        started = perf_counter()
        base, loading_info = LlavaQwenForCausalLM.from_pretrained(str(self.model_path), config=config,
            local_files_only=True, torch_dtype=torch.bfloat16, device_map={'': 0},
            attn_implementation='sdpa', low_cpu_mem_usage=True, output_loading_info=True)
        if any(loading_info.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
            raise ValueError('Model loading mismatch: ' + str(loading_info))
        vision = base.get_vision_tower()
        if not vision.is_loaded:
            raise ValueError('Vision tower chưa được nạp.')
        vision.to(device='cuda', dtype=torch.bfloat16)
        inner = base.get_model()
        inner.vision_resampler.float(); inner.mm_projector.float(); base.requires_grad_(False)
        targets = [n for n, m in base.named_modules() if n.startswith('model.layers.')
            and n.endswith(tuple(release['lora']['target_suffixes'])) and isinstance(m, torch.nn.Linear)]
        if len(targets) != 2 * config.num_hidden_layers:
            raise ValueError('Sai số lớp q_proj/v_proj cho LoRA.')
        lc = {k: v for k, v in release['lora'].items() if k != 'target_suffixes'}
        self.model = get_peft_model(base, LoraConfig(target_modules=targets, **lc))
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                if 'lora_' not in name:
                    raise ValueError('Có tham số ngoài LoRA chưa được đóng băng.')
                p.data = p.data.float()
        state = torch.load(Path(protocol['adapter_path']) / 'adapter_model.pt', map_location='cpu', weights_only=True)
        restore_adapter(self.model, state, get_peft_model_state_dict, set_peft_model_state_dict, torch)
        del state
        self.model.requires_grad_(False); self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), use_fast=False, local_files_only=True)
        if self.tokenizer.convert_tokens_to_ids('<|im_end|>') != 151645 or self.tokenizer.bos_token_id is not None:
            raise ValueError('Tokenizer không khớp bản đánh giá.')
        self.generation_config = GenerationConfig(**release['generation'])
        self.image_processor = vision.image_processor
        self.config = config
        torch.cuda.synchronize()
        self.load_seconds = perf_counter() - started
        self.model_load_count = 1
        self.successful_requests = 0
        self.model_instance_id = uuid.uuid4().hex
        self._lock = threading.Lock()
        self.runtime_manifest = {
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'model_variant': 'B_SCST_128',
            'adapter_manifest': manifest, 'loading_info': loading_info,
            'source_sha256': release['source_sha256'], 'base_hashes_verified': True,
            'vision_hashes_verified': True, 'gpu': torch.cuda.get_device_name(0),
            'precision': 'BF16 autocast; MDP/projector FP32; LoRA FP32',
            'package_versions': {n: metadata.version(n) for n in ['torch', 'transformers', 'peft']},
            'generation': self.generation_config.to_dict(),
        }
        self.protocol_id = identity({'protocol': self._protocol, 'adapter_sha256': manifest['adapter_sha256'],
            'source_sha256': release['source_sha256']})

    @property
    def protocol(self):
        return deepcopy(self._protocol)

    def compare(self, before, after):
        if not isinstance(before, Image.Image) or not isinstance(after, Image.Image):
            raise TypeError('Cần hai ảnh PIL, theo thứ tự trước/sau.')
        torch = self._torch
        with self._lock:
            torch.cuda.synchronize(); started = perf_counter()
            row = {'conversations': [{'from': 'human', 'value': self._protocol['prompt']}]}
            prompt, _ = token_layout(self.tokenizer, row, False)
            pixels = [self._process_images([im.convert('RGB')], self.image_processor, self.config)
                .to(device='cuda', dtype=torch.bfloat16) for im in (before, after)]
            pixels = [torch.cat(pixels, dim=0)]
            self.model.eval()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                output = self.model.generate(inputs=torch.tensor([prompt], device='cuda'),
                    attention_mask=torch.ones((1, len(prompt)), dtype=torch.long, device='cuda'),
                    images=pixels, generation_config=self.generation_config)
            ids = output[0].detach().cpu().tolist()
            if not ids or len(ids) > 256:
                raise ValueError('Unexpected generated token layout')
            caption = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            if not caption:
                raise RuntimeError('Model trả caption rỗng.')
            torch.cuda.synchronize(); self.successful_requests += 1
            return {'caption': caption, 'elapsed_seconds': round(perf_counter() - started, 4),
                'protocol_id': self.protocol_id, 'model_instance_id': self.model_instance_id,
                'model_load_count': self.model_load_count, 'successful_requests': self.successful_requests,
                'model_variant': 'B_SCST_128'}
