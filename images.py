"""Image validation and crops in coordinates of the EXIF-normalized original."""

import base64
from io import BytesIO
import math
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 25_000_000


def normalize_image(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_BYTES:
        raise ValueError('Mỗi ảnh cần nhỏ hơn 20 MB và có nội dung.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in ('JPEG', 'PNG'):
                    raise ValueError('Chọn ảnh JPG hoặc PNG.')
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError('Ảnh vượt giới hạn 25 triệu pixel.')
                return ImageOps.exif_transpose(image).convert('RGB').copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError('Không đọc được ảnh hoặc ảnh quá lớn. Chọn JPG/PNG khác.') from exc


def crop_roi(image: Image.Image, roi) -> Image.Image:
    try:
        x0, y0, x1, y1 = [float(v) for v in roi]
    except (TypeError, ValueError) as exc:
        raise ValueError('Cần chọn đủ bốn tọa độ vùng ảnh.') from exc
    if not (all(math.isfinite(v) for v in (x0, y0, x1, y1))
            and 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError('Vùng chọn phải nằm trong ảnh và có chiều rộng, chiều cao dương.')
    box = (math.floor(x0 * image.width), math.floor(y0 * image.height),
           min(image.width, math.ceil(x1 * image.width)),
           min(image.height, math.ceil(y1 * image.height)))
    if box[2] - box[0] < 32 or box[3] - box[1] < 32:
        raise ValueError('Chọn vùng lớn hơn: ảnh cắt cần ít nhất 32 × 32 pixel.')
    return image.crop(box)


def image_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.convert('RGB').save(output, format='PNG')
    return output.getvalue()


def preview_url(image: Image.Image) -> str:
    preview = image.copy()
    preview.thumbnail((1100, 650))
    output = BytesIO()
    preview.save(output, format='JPEG', quality=88)
    return 'data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode('ascii')
