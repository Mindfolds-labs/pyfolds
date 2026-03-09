"""Reusable TensorFlow data pipelines for Noetic PAWP."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple


def _require_tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow necessário para noetic_pawp.data.tf_pipeline.") from exc
    return tf


def _default_preprocess(image, label, *, image_size: Tuple[int, int]):
    tf = _require_tf()
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def build_tf_dataset(
    samples: Iterable,
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    cache: bool = True,
    prefetch: bool = True,
    preprocess_fn: Optional[Callable] = None,
    image_size: Tuple[int, int] = (224, 224),
):
    """Build a generic tf.data pipeline from tensor-like samples."""
    tf = _require_tf()
    ds = tf.data.Dataset.from_tensor_slices(samples)
    if shuffle:
        ds = ds.shuffle(buffer_size=max(batch_size * 4, 32))
    pp = preprocess_fn or (lambda x, y: _default_preprocess(x, y, image_size=image_size))
    ds = ds.map(pp, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size)
    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _build_vision_pipeline(
    root_dir: str,
    *,
    batch_size: int,
    image_size: Tuple[int, int],
):
    tf = _require_tf()
    root = Path(root_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset root inválido: {root}")

    pattern = str(root / "images" / "*.jpg")
    image_files = tf.io.gfile.glob(pattern)
    if not image_files:
        raise ValueError(f"Nenhuma imagem encontrada em: {pattern}")

    ds = tf.data.Dataset.from_tensor_slices(image_files)

    def _load(path):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3)
        label = tf.constant(0, dtype=tf.int32)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_nyuv2_pipeline(root_dir: str, *, batch_size: int = 8, image_size: Tuple[int, int] = (224, 224)):
    """Build a starter NYUv2 tf.data pipeline from `images/*.jpg`."""
    return _build_vision_pipeline(root_dir, batch_size=batch_size, image_size=image_size)


def build_kitti_pipeline(root_dir: str, *, batch_size: int = 8, image_size: Tuple[int, int] = (224, 224)):
    """Build a starter KITTI tf.data pipeline from `images/*.jpg`."""
    return _build_vision_pipeline(root_dir, batch_size=batch_size, image_size=image_size)
