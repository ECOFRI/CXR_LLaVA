"""Reuse the pinned BLIP-2 processor for full runs; dry-runs use shapes only."""
from pathlib import Path


def load_image(path):
    from PIL import Image
    with Image.open(Path(path)) as image:
        return image.convert('RGB')


def prepare(processor, path, text, device):
    image = load_image(path)
    batch = processor(images=image, text=text, return_tensors='pt', truncation=True, max_length=128)
    return {key: value.to(device) for key, value in batch.items()}
