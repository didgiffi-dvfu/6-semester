#!/usr/bin/env python
"""
End-to-end Neural Tangent Kernel (NTK) analysis for TensorFlow / Keras models.

This script is intentionally practical:
1. It can reconstruct the model used in `start.ipynb` from this workspace.
2. It can load weights from several common formats.
3. It computes the empirical NTK on a batch of inputs.
4. It saves the main plots and metrics to disk.

Default reconstructed architecture (from `start.ipynb`):
    raw input (t, x)
        -> Fourier features for t
        -> Fourier features for x
        -> concat
        -> 4 x Dense(64, tanh)
        -> Dense(1, linear)

Important note about ambiguity:
If you only have dense-layer weights (for example `.npz` with hidden/output kernels)
but not the weights or definition of an upstream embedding / projection block,
the exact model cannot always be recovered uniquely from weights alone.

This script therefore includes:
    - a ready-to-run recovered model for the notebook found in this workspace;
    - a `UserSuppliedModelTemplate` that you can edit for your own architecture;
    - a PyTorch checkpoint inspection path with a mapping template, because
      automatic PyTorch -> Keras conversion is architecture-specific.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib

# Use a non-interactive backend by default so the script also works in headless runs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

try:
    import seaborn as sns

    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

try:
    from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
    from scipy.spatial.distance import pdist

    HAS_SCIPY_CLUSTER = True
except Exception:
    HAS_SCIPY_CLUSTER = False


DEFAULT_DTYPE = "float64"
DEFAULT_BATCH_SIZE = 36
DEFAULT_DOMAIN = {
    "t": (0.0, 4.0),
    "x": (-30.0, 30.0),
}


@dataclass
class LoadedBatch:
    array: np.ndarray
    source: str
    description: str


@dataclass
class LoadedWeightsInfo:
    source: str
    description: str
    extra: Dict[str, object]


def set_global_determinism(seed: int, dtype: str) -> None:
    """Set seeds and TensorFlow default dtype once at startup."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.backend.set_floatx(dtype)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class FallbackFourierFeatureProjection(tf.keras.layers.Layer):
    """
    Self-contained Fourier projection layer.

    We use this layer instead of depending on external packages so the script
    stays runnable in a plain TensorFlow environment.
    """

    def __init__(
        self,
        gaussian_projection: int = 16,
        gaussian_scale: float = 1.0,
        seed: int = 42,
        trainable: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.gaussian_projection = int(gaussian_projection)
        self.gaussian_scale = float(gaussian_scale)
        self.seed = int(seed)
        self.trainable_projection = bool(trainable)

    def build(self, input_shape) -> None:
        self.B = self.add_weight(
            name="B",
            shape=(1, self.gaussian_projection),
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=self.gaussian_scale,
                seed=self.seed,
            ),
            trainable=self.trainable_projection,
            dtype=self.dtype,
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.cast(x, self.compute_dtype)
        projection = 2.0 * math.pi * tf.matmul(x, self.B)
        return tf.concat([tf.sin(projection), tf.cos(projection)], axis=-1)

    def get_config(self) -> Dict[str, object]:
        config = super().get_config()
        config.update(
            {
                "gaussian_projection": self.gaussian_projection,
                "gaussian_scale": self.gaussian_scale,
                "seed": self.seed,
                "trainable": self.trainable_projection,
            }
        )
        return config


def make_fourier_projection(
    gaussian_projection: int,
    gaussian_scale: float,
    seed: int,
    name: str,
    dtype: str,
) -> FallbackFourierFeatureProjection:
    """
    Create a deterministic Fourier projection layer.

    If your original training code used a different projection implementation,
    replace this function with the exact layer definition from your project.
    """
    return FallbackFourierFeatureProjection(
        gaussian_projection=gaussian_projection,
        gaussian_scale=gaussian_scale,
        seed=seed,
        name=name,
        dtype=dtype,
    )


class RecoveredNotebookModel(tf.keras.Model):
    """
    Exact architecture recovered from `start.ipynb` in this workspace.

    Input format:
        inputs[:, 0] = t
        inputs[:, 1] = x
    """

    def __init__(
        self,
        num_hidden_layers: int = 4,
        hidden_dim: int = 64,
        num_fourier: int = 16,
        output_dim: int = 1,
        act: str = "linear",
        dtype: str = DEFAULT_DTYPE,
    ) -> None:
        super().__init__(name="RecoveredNotebookModel")

        self.t_proj = make_fourier_projection(
            gaussian_projection=num_fourier,
            gaussian_scale=1.0 / 4.0,
            seed=101,
            name="t_proj",
            dtype=dtype,
        )
        self.x_proj = make_fourier_projection(
            gaussian_projection=num_fourier,
            gaussian_scale=1.0 / 30.0,
            seed=202,
            name="x_proj",
            dtype=dtype,
        )

        self.hidden_layers = [
            tf.keras.layers.Dense(
                hidden_dim,
                activation="tanh",
                name=f"hidden_{idx + 1}",
                dtype=dtype,
            )
            for idx in range(num_hidden_layers)
        ]
        self.output_layer = tf.keras.layers.Dense(
            output_dim,
            activation=act,
            name="output_layer",
            dtype=dtype,
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        del training
        inputs = tf.cast(inputs, self.compute_dtype)
        t = inputs[:, 0:1]
        x = inputs[:, 1:2]

        t_features = tf.cast(self.t_proj(t), self.compute_dtype)
        x_features = tf.cast(self.x_proj(x), self.compute_dtype)
        features = tf.concat([t_features, x_features], axis=-1)

        h = features
        for layer in self.hidden_layers:
            h = layer(h)

        return self.output_layer(h)


class UserSuppliedModelTemplate(tf.keras.Model):
    """
    Template for the case where architecture cannot be recovered uniquely.

    Edit the marked places if your checkpoint is not the same as the recovered
    notebook model.

    What you usually need to replace:
        1. Raw input dimension and input semantics.
        2. Any preprocessing / embedding / positional / Fourier blocks.
        3. Hidden layer widths, count, and activations.
        4. Output dimension.
        5. Weight-name mapping in `map_pytorch_state_dict_to_keras`.
    """

    def __init__(self, input_dim: int = 2, output_dim: int = 1, dtype: str = DEFAULT_DTYPE) -> None:
        super().__init__(name="UserSuppliedModelTemplate")
        self.input_dim = int(input_dim)

        # Replace this block with your own preprocessing if needed.
        self.preprocess = tf.keras.layers.Dense(
            64,
            activation="linear",
            name="template_preprocess",
            dtype=dtype,
        )

        # Replace the widths / activations below to match your real model.
        self.hidden_layers = [
            tf.keras.layers.Dense(64, activation="tanh", name="template_hidden_1", dtype=dtype),
            tf.keras.layers.Dense(64, activation="tanh", name="template_hidden_2", dtype=dtype),
        ]
        self.output_layer = tf.keras.layers.Dense(
            output_dim,
            activation="linear",
            name="template_output",
            dtype=dtype,
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        del training
        x = tf.cast(inputs, self.compute_dtype)
        h = self.preprocess(x)
        for layer in self.hidden_layers:
            h = layer(h)
        return self.output_layer(h)


def build_model(builder: str, dtype: str, input_dim: int) -> tf.keras.Model:
    """Build and initialize a model instance."""
    if builder == "recovered_pinn":
        model = RecoveredNotebookModel(dtype=dtype)
        dummy = tf.zeros((1, 2), dtype=tf.as_dtype(dtype))
    elif builder == "user_template":
        model = UserSuppliedModelTemplate(input_dim=input_dim, dtype=dtype)
        dummy = tf.zeros((1, input_dim), dtype=tf.as_dtype(dtype))
    else:
        raise ValueError(f"Unknown builder: {builder}")

    _ = model(dummy, training=False)
    return model


def load_full_keras_model(path: Path) -> tf.keras.Model:
    """Load a serialized Keras model with custom objects registered."""
    return tf.keras.models.load_model(
        path,
        compile=False,
        custom_objects={
            "FallbackFourierFeatureProjection": FallbackFourierFeatureProjection,
            "RecoveredNotebookModel": RecoveredNotebookModel,
            "UserSuppliedModelTemplate": UserSuppliedModelTemplate,
        },
    )


def capture_model_summary(model: tf.keras.Model) -> str:
    """Capture `model.summary()` into a string."""
    lines: List[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


def find_first_2d_array(npz_dict: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray]:
    """Pick the first 2D array from an NPZ archive."""
    for key, value in npz_dict.items():
        value = np.asarray(value)
        if value.ndim == 2:
            return key, value
    raise ValueError("No 2D array found in the provided .npz file.")


def maybe_transpose_batch(arr: np.ndarray) -> np.ndarray:
    """
    Heuristic for common batch-storage layouts.

    If data is stored as [features, batch] with a small number of features,
    transpose it into [batch, features].
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D batch array, got shape {arr.shape}.")

    if arr.shape[1] <= 8:
        return arr
    if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
        return arr.T
    return arr


def load_input_batch(path: Optional[str], input_key: Optional[str], batch_size: int, input_dim: int) -> LoadedBatch:
    """
    Load a real input batch if provided, otherwise create a synthetic one.

    Synthetic data is clearly labeled in the output so it is easy to replace.
    """
    if path is None:
        batch = make_synthetic_batch(batch_size=batch_size, input_dim=input_dim)
        return LoadedBatch(
            array=batch,
            source="synthetic",
            description=(
                "Synthetic batch generated by the script. Replace `--input-batch` "
                "with a real batch file when you want NTK on real data."
            ),
        )

    data_path = Path(path)
    suffix = data_path.suffix.lower()

    if suffix == ".npy":
        batch = np.load(data_path)
    elif suffix == ".npz":
        archive = {k: np.asarray(v) for k, v in np.load(data_path, allow_pickle=True).items()}
        if input_key is None:
            input_key, batch = find_first_2d_array(archive)
        else:
            batch = np.asarray(archive[input_key])
    elif suffix in {".csv", ".txt"}:
        delimiter = "," if suffix == ".csv" else None
        batch = np.loadtxt(data_path, delimiter=delimiter)
    else:
        raise ValueError(
            "Unsupported input-batch format. Use one of: .npy, .npz, .csv, .txt"
        )

    batch = np.asarray(batch, dtype=np.float64)
    if batch.ndim == 1:
        batch = batch.reshape(-1, 1)
    batch = maybe_transpose_batch(batch)

    return LoadedBatch(
        array=batch,
        source=str(data_path),
        description=f"Loaded real batch from {data_path}",
    )


def make_synthetic_batch(batch_size: int, input_dim: int) -> np.ndarray:
    """
    Create a synthetic batch.

    For the recovered notebook model we use a structured (t, x) grid over the
    same domain seen in the notebook:
        t in [0, 4], x in [-30, 30]
    """
    if input_dim != 2:
        return np.random.uniform(-1.0, 1.0, size=(batch_size, input_dim)).astype(np.float64)

    # A near-square grid makes NTK heatmaps easier to inspect than a random batch.
    num_t = max(2, int(np.floor(np.sqrt(batch_size))))
    num_x = int(np.ceil(batch_size / num_t))

    t_values = np.linspace(DEFAULT_DOMAIN["t"][0], DEFAULT_DOMAIN["t"][1], num_t)
    x_values = np.linspace(DEFAULT_DOMAIN["x"][0], DEFAULT_DOMAIN["x"][1], num_x)
    t_grid, x_grid = np.meshgrid(t_values, x_values, indexing="ij")

    batch = np.stack([t_grid.reshape(-1), x_grid.reshape(-1)], axis=1)
    batch = batch[:batch_size]
    return batch.astype(np.float64)


def infer_npz_architecture(npz_dict: Dict[str, np.ndarray]) -> Dict[str, object]:
    """
    Infer as much architecture information as possible from NPZ keys / shapes.

    This is only a best-effort description. It cannot recover omitted feature
    extractors or preprocessing layers from dense kernels alone.
    """
    hidden_kernel_pattern = re.compile(r"hidden_(\d+)_kernel$")
    hidden_bias_pattern = re.compile(r"hidden_(\d+)_bias$")

    hidden_kernels: Dict[int, Tuple[int, ...]] = {}
    hidden_biases: Dict[int, Tuple[int, ...]] = {}

    for key, value in npz_dict.items():
        value = np.asarray(value)
        mk = hidden_kernel_pattern.match(key)
        mb = hidden_bias_pattern.match(key)
        if mk:
            hidden_kernels[int(mk.group(1))] = tuple(value.shape)
        if mb:
            hidden_biases[int(mb.group(1))] = tuple(value.shape)

    output_kernel_shape = tuple(np.asarray(npz_dict["output_kernel"]).shape) if "output_kernel" in npz_dict else None
    output_bias_shape = tuple(np.asarray(npz_dict["output_bias"]).shape) if "output_bias" in npz_dict else None

    sorted_hidden = [hidden_kernels[idx] for idx in sorted(hidden_kernels)]
    ambiguous_reason = None

    if sorted_hidden:
        first_kernel = sorted_hidden[0]
        if len(first_kernel) == 2 and first_kernel[0] == first_kernel[1]:
            ambiguous_reason = (
                "The first hidden kernel is square, so an upstream embedding / projection "
                "block may be missing from the checkpoint."
            )

    return {
        "num_hidden_layers_detected": len(sorted_hidden),
        "hidden_kernel_shapes": hidden_kernels,
        "hidden_bias_shapes": hidden_biases,
        "output_kernel_shape": output_kernel_shape,
        "output_bias_shape": output_bias_shape,
        "ambiguous_reason": ambiguous_reason,
    }


def load_npz_dict(path: Path) -> Dict[str, np.ndarray]:
    """Load an NPZ archive into a plain dictionary."""
    with np.load(path, allow_pickle=True) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def assign_npz_to_recovered_model(model: tf.keras.Model, npz_dict: Dict[str, np.ndarray]) -> None:
    """
    Assign dense-layer weights for the recovered notebook model.

    The Fourier projection matrices are not stored in `start_final_parameters.npz`
    from the notebook, so we recreate them from the original deterministic seeds.
    """
    if not hasattr(model, "hidden_layers") or not hasattr(model, "output_layer"):
        raise TypeError("Model does not expose `hidden_layers` / `output_layer` as expected.")

    for idx, layer in enumerate(model.hidden_layers, start=1):
        kernel_key = f"hidden_{idx}_kernel"
        bias_key = f"hidden_{idx}_bias"
        if kernel_key not in npz_dict:
            raise KeyError(
                f"Missing `{kernel_key}` in NPZ archive. "
                "For the recovered notebook model hidden kernels are required."
            )
        kernel = np.asarray(npz_dict[kernel_key], dtype=np.float64)
        if bias_key in npz_dict:
            bias = np.asarray(npz_dict[bias_key], dtype=np.float64)
        else:
            warnings.warn(
                f"`{bias_key}` not found. Falling back to zeros. "
                "Use a full parameter archive when possible.",
                stacklevel=2,
            )
            bias = np.zeros((kernel.shape[1],), dtype=np.float64)
        layer.set_weights([kernel, bias])

    if "output_kernel" not in npz_dict:
        raise KeyError("Missing `output_kernel` in NPZ archive.")
    if "output_bias" in npz_dict:
        output_bias = np.asarray(npz_dict["output_bias"], dtype=np.float64)
    else:
        warnings.warn(
            "`output_bias` not found. Falling back to zeros. "
            "Use a full parameter archive when possible.",
            stacklevel=2,
        )
        output_bias = np.zeros((np.asarray(npz_dict["output_kernel"]).shape[1],), dtype=np.float64)
    model.output_layer.set_weights(
        [
            np.asarray(npz_dict["output_kernel"], dtype=np.float64),
            output_bias,
        ]
    )


def assign_npz_to_template_model(model: tf.keras.Model, npz_dict: Dict[str, np.ndarray]) -> None:
    """
    Placeholder loader for the editable template model.

    This is intentionally conservative. Users should adapt it after they edit
    the template architecture.
    """
    raise NotImplementedError(
        "You selected `user_template`. After you edit the template architecture, "
        "also edit `assign_npz_to_template_model()` so the NPZ keys are mapped "
        "to the correct Keras layers."
    )


def map_pytorch_state_dict_to_keras(model: tf.keras.Model, state_dict: Dict[str, np.ndarray]) -> None:
    """
    Template for PyTorch -> Keras weight transfer.

    Automatic conversion is architecture-specific because names and tensor
    layouts differ. For example:
        - PyTorch nn.Linear stores weights as [out_features, in_features]
        - Keras Dense expects [in_features, out_features]

    Example sketch:
        model.hidden_layers[0].set_weights([
            state_dict["net.0.weight"].T,
            state_dict["net.0.bias"],
        ])
    """
    del model, state_dict
    raise NotImplementedError(
        "Automatic PyTorch -> Keras mapping is not universal. "
        "Edit `map_pytorch_state_dict_to_keras()` for your checkpoint."
    )


def inspect_pytorch_checkpoint(path: Path) -> Dict[str, np.ndarray]:
    """
    Load a PyTorch checkpoint for inspection.

    We only inspect / extract the state dict here. The actual assignment into a
    Keras model must be implemented manually for the specific architecture.
    """
    try:
        import torch
    except Exception as exc:
        raise ImportError(
            "Reading .pt / .pth checkpoints requires PyTorch. "
            "Install torch or convert the checkpoint to .npz / .h5 first."
        ) from exc

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        state_dict = obj.get("state_dict") or obj.get("model_state_dict") or obj
    else:
        state_dict = obj

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Expected a PyTorch state dict in {path}, got {type(state_dict).__name__}."
        )

    numpy_state_dict: Dict[str, np.ndarray] = {}
    for key, value in state_dict.items():
        if hasattr(value, "detach"):
            numpy_state_dict[key] = value.detach().cpu().numpy()
        else:
            numpy_state_dict[key] = np.asarray(value)
    return numpy_state_dict


def looks_like_tf_checkpoint(path: Path) -> bool:
    """Heuristic check for TensorFlow checkpoint paths."""
    if path.suffix in {".ckpt", ".index"}:
        return True
    if path.with_suffix(path.suffix + ".index").exists():
        return True
    if path.with_suffix(".index").exists():
        return True
    return False


def load_weights(
    model: tf.keras.Model,
    weights_path: Path,
    builder: str,
) -> LoadedWeightsInfo:
    """
    Universal-ish weight loading entrypoint.

    Supported:
        - `.npz`
        - `.keras`, `.h5`, `.hdf5`
        - TensorFlow checkpoints (`.ckpt`, `.index`, or prefix)
        - PyTorch `.pt`, `.pth`, `.bin` (inspection + user mapping template)
    """
    suffix = weights_path.suffix.lower()

    if suffix == ".npz":
        npz_dict = load_npz_dict(weights_path)
        architecture_guess = infer_npz_architecture(npz_dict)
        if builder == "recovered_pinn":
            assign_npz_to_recovered_model(model, npz_dict)
        elif builder == "user_template":
            assign_npz_to_template_model(model, npz_dict)
        else:
            raise ValueError(f"Unsupported builder for NPZ loading: {builder}")
        return LoadedWeightsInfo(
            source=str(weights_path),
            description="Loaded NPZ arrays into a Keras model.",
            extra={"npz_keys": list(npz_dict.keys()), "architecture_guess": architecture_guess},
        )

    if suffix in {".keras", ".h5", ".hdf5"}:
        model.load_weights(str(weights_path))
        return LoadedWeightsInfo(
            source=str(weights_path),
            description="Loaded Keras weights into the explicitly built model.",
            extra={},
        )

    if looks_like_tf_checkpoint(weights_path):
        model.load_weights(str(weights_path))
        return LoadedWeightsInfo(
            source=str(weights_path),
            description="Loaded TensorFlow checkpoint weights into the model.",
            extra={},
        )

    if suffix in {".pt", ".pth", ".bin"}:
        state_dict = inspect_pytorch_checkpoint(weights_path)
        map_pytorch_state_dict_to_keras(model, state_dict)
        return LoadedWeightsInfo(
            source=str(weights_path),
            description="Loaded PyTorch checkpoint via a user-defined mapping.",
            extra={"state_dict_keys": list(state_dict.keys())[:50]},
        )

    raise ValueError(
        f"Unsupported weight format: {weights_path}. "
        "Use .npz, .keras, .h5, .hdf5, TensorFlow checkpoint, .pt, .pth, or .bin"
    )


def flatten_jacobian_block(jacobian: tf.Tensor) -> tf.Tensor:
    """
    Flatten a per-variable Jacobian block to shape [batch, output_dim, num_params].
    """
    output_shape = tf.shape(jacobian)
    batch_size = output_shape[0]
    output_dim = output_shape[1]
    return tf.reshape(jacobian, (batch_size, output_dim, -1))


def compute_empirical_ntk(model: tf.keras.Model, batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the empirical NTK on a batch of inputs.

    Definition used here:
        K_ij = sum_o < d f_o(x_i) / d theta , d f_o(x_j) / d theta >

    In other words, we:
        1. compute the Jacobian of model outputs wrt trainable parameters;
        2. flatten parameter axes;
        3. contract over output channels and parameters.
    """
    x = tf.convert_to_tensor(batch, dtype=tf.keras.backend.floatx())
    with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
        outputs = model(x, training=False)
        outputs = tf.reshape(outputs, (tf.shape(outputs)[0], -1))

    jacobian_blocks = []
    for variable in model.trainable_variables:
        # The non-pfor mode is slower, but it is usually quieter and more stable
        # for one-off analysis scripts.
        jac = tape.jacobian(outputs, variable, experimental_use_pfor=False)
        jacobian_blocks.append(flatten_jacobian_block(jac))
    del tape

    full_jacobian = tf.concat(jacobian_blocks, axis=-1)
    ntk = tf.einsum("bop,cop->bc", full_jacobian, full_jacobian)
    ntk = 0.5 * (ntk + tf.transpose(ntk))
    return ntk.numpy(), full_jacobian.numpy()


def entropy_effective_rank(eigenvalues: np.ndarray, eps: float = 1e-12) -> float:
    """
    Entropy-based effective rank.

    For positive eigenvalues lambda_i:
        p_i = lambda_i / sum_j lambda_j
        r_eff = exp( - sum_i p_i log(p_i) )
    """
    positive = eigenvalues[eigenvalues > eps]
    if positive.size == 0:
        return 0.0
    probabilities = positive / np.sum(positive)
    return float(np.exp(-np.sum(probabilities * np.log(probabilities + eps))))


def analyze_ntk(ntk: np.ndarray, eigen_tol: float = 1e-12) -> Dict[str, object]:
    """Compute matrix diagnostics and spectral statistics."""
    ntk = 0.5 * (ntk + ntk.T)
    eigenvalues = np.linalg.eigvalsh(ntk)
    eigenvalues_desc = np.sort(eigenvalues)[::-1]

    max_abs = max(1.0, float(np.max(np.abs(eigenvalues_desc))))
    positive = eigenvalues_desc[eigenvalues_desc > eigen_tol * max_abs]

    if positive.size > 0:
        condition_number = float(positive[0] / positive[-1])
    else:
        condition_number = float("inf")

    metrics = {
        "trace": float(np.trace(ntk)),
        "rank": int(np.linalg.matrix_rank(ntk)),
        "effective_rank": entropy_effective_rank(positive if positive.size else eigenvalues_desc, eps=eigen_tol),
        "condition_number": condition_number,
        "min_eigenvalue": float(np.min(eigenvalues_desc)),
        "max_eigenvalue": float(np.max(eigenvalues_desc)),
        "num_positive_eigenvalues": int(positive.size),
        "symmetry_error_max_abs": float(np.max(np.abs(ntk - ntk.T))),
        "diagonal_min": float(np.min(np.diag(ntk))),
        "diagonal_max": float(np.max(np.diag(ntk))),
        "diagonal_mean": float(np.mean(np.diag(ntk))),
        "eigenvalues_desc": eigenvalues_desc.tolist(),
    }
    return metrics


def pca_project_rows(matrix: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Project NTK rows to 2D with plain NumPy PCA."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    coordinates = u[:, :n_components] * s[:n_components]
    explained_variance = (s ** 2) / max(np.sum(s ** 2), 1e-12)
    return coordinates, explained_variance[:n_components]


def cluster_row_order(matrix: np.ndarray) -> Optional[np.ndarray]:
    """Optional hierarchical clustering order for rows / columns."""
    if not HAS_SCIPY_CLUSTER or matrix.shape[0] < 3:
        return None

    distances = pdist(matrix, metric="euclidean")
    linkage_matrix = linkage(distances, method="average")
    return leaves_list(linkage_matrix)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    """Write JSON with UTF-8 and indentation."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def save_array(path: Path, array: np.ndarray) -> None:
    """Save NumPy array to .npy."""
    np.save(path, array)


def plot_heatmap(
    matrix: np.ndarray,
    path: Path,
    title: str,
    cmap: str = "viridis",
    x_label: str = "Sample index",
    y_label: str = "Sample index",
) -> None:
    """Save a heatmap figure using seaborn if available, otherwise matplotlib."""
    fig, ax = plt.subplots(figsize=(8.0, 6.5), dpi=160)

    if HAS_SEABORN:
        sns.heatmap(matrix, ax=ax, cmap=cmap, square=True, cbar_kws={"label": "NTK value"})
    else:
        image = ax.imshow(matrix, cmap=cmap, aspect="auto")
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("NTK value")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_eigenvalues(eigenvalues_desc: Sequence[float], path: Path) -> None:
    """Plot the NTK spectrum on linear and log scales."""
    eigenvalues_desc = np.asarray(eigenvalues_desc, dtype=np.float64)
    indices = np.arange(1, len(eigenvalues_desc) + 1)

    # Clip only for the log-scale panel; the linear panel shows the raw values.
    log_safe = np.clip(eigenvalues_desc, np.finfo(np.float64).tiny, None)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), dpi=160)

    axes[0].plot(indices, eigenvalues_desc, marker="o", linewidth=1.8, markersize=4.0)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[0].set_title("NTK eigenvalues (linear scale)")
    axes[0].set_xlabel("Eigenvalue index (descending)")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].grid(True, alpha=0.25)

    axes[1].semilogy(indices, log_safe, marker="o", linewidth=1.8, markersize=4.0)
    axes[1].set_title("NTK eigenvalues (log scale)")
    axes[1].set_xlabel("Eigenvalue index (descending)")
    axes[1].set_ylabel("Eigenvalue")
    axes[1].grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_row_projection(
    row_coordinates: np.ndarray,
    batch: np.ndarray,
    explained_variance: np.ndarray,
    path: Path,
) -> None:
    """Plot a 2D PCA projection of NTK rows."""
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=160)

    if batch.shape[1] >= 1:
        color_values = batch[:, 0]
        color_label = "Input dim 0"
    else:
        color_values = np.arange(batch.shape[0])
        color_label = "Sample index"

    scatter = ax.scatter(
        row_coordinates[:, 0],
        row_coordinates[:, 1],
        c=color_values,
        cmap="viridis",
        s=70,
        edgecolors="black",
        linewidths=0.4,
    )
    for idx, (x_coord, y_coord) in enumerate(row_coordinates[:, :2]):
        ax.text(x_coord, y_coord, str(idx), fontsize=8, alpha=0.75)

    ax.set_title(
        "2D PCA projection of NTK rows "
        f"(var: {explained_variance[0]:.2%}, {explained_variance[1]:.2%})"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(color_label)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_clustered_heatmap_if_possible(matrix: np.ndarray, path: Path) -> Optional[np.ndarray]:
    """
    Save a clustered heatmap if scipy clustering is available.

    Returns the permutation order if clustering was performed.
    """
    order = cluster_row_order(matrix)
    if order is None:
        return None

    reordered = matrix[np.ix_(order, order)]
    plot_heatmap(
        reordered,
        path=path,
        title="Clustered NTK heatmap (rows/cols reordered by hierarchical clustering)",
        cmap="magma",
        x_label="Reordered sample index",
        y_label="Reordered sample index",
    )
    return order


def print_report(report: Dict[str, object]) -> None:
    """Pretty-print the main scalar results to stdout."""
    print("\n=== NTK ANALYSIS REPORT ===")
    for key in [
        "trace",
        "rank",
        "effective_rank",
        "condition_number",
        "min_eigenvalue",
        "max_eigenvalue",
        "num_positive_eigenvalues",
        "diagonal_min",
        "diagonal_max",
        "diagonal_mean",
        "symmetry_error_max_abs",
    ]:
        value = report.get(key)
        print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze TensorFlow / Keras model weights through the empirical NTK."
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to weights/checkpoint (.npz, .keras, .h5, TF checkpoint, .pt, .pth, .bin).",
    )
    parser.add_argument(
        "--builder",
        type=str,
        default="recovered_pinn",
        choices=["recovered_pinn", "user_template", "auto_loaded_keras"],
        help=(
            "Which TensorFlow model definition to build before loading weights. "
            "`recovered_pinn` matches the `start.ipynb` model from this workspace. "
            "`auto_loaded_keras` directly opens a serialized `.keras` / `.h5` model."
        ),
    )
    parser.add_argument(
        "--input-batch",
        type=str,
        default=None,
        help="Optional path to a real batch (.npy, .npz, .csv, .txt). If omitted, synthetic data is used.",
    )
    parser.add_argument(
        "--input-key",
        type=str,
        default=None,
        help="If `--input-batch` is an NPZ archive, this selects the array key to use.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=2,
        help="Input dimension for the editable `user_template` builder and synthetic data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Synthetic batch size when `--input-batch` is not provided.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=DEFAULT_DTYPE,
        choices=["float32", "float64"],
        help="Computation dtype.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ntk_analysis_output",
        help="Directory where arrays, metrics, and plots will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_determinism(seed=args.seed, dtype=args.dtype)

    weights_path = Path(args.weights).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())

    if args.builder == "auto_loaded_keras":
        model = load_full_keras_model(weights_path)
        weights_info = LoadedWeightsInfo(
            source=str(weights_path),
            description="Loaded a full serialized Keras model directly from disk.",
            extra={},
        )
    else:
        model = build_model(builder=args.builder, dtype=args.dtype, input_dim=args.input_dim)
        weights_info = load_weights(model=model, weights_path=weights_path, builder=args.builder)

    summary_text = capture_model_summary(model)
    (output_dir / "model_summary.txt").write_text(summary_text, encoding="utf-8")

    batch_info = load_input_batch(
        path=args.input_batch,
        input_key=args.input_key,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
    )

    batch = np.asarray(batch_info.array, dtype=np.float64)
    batch = maybe_transpose_batch(batch)

    predictions = model(tf.convert_to_tensor(batch, dtype=args.dtype), training=False).numpy()
    ntk_matrix, full_jacobian = compute_empirical_ntk(model=model, batch=batch)
    metrics = analyze_ntk(ntk_matrix)
    row_projection, explained_variance = pca_project_rows(ntk_matrix)
    cluster_order = plot_clustered_heatmap_if_possible(
        ntk_matrix,
        path=output_dir / "ntk_clustered_heatmap.png",
    )

    plot_heatmap(
        ntk_matrix,
        path=output_dir / "ntk_heatmap.png",
        title="Empirical NTK heatmap",
        cmap="viridis",
    )
    plot_eigenvalues(metrics["eigenvalues_desc"], path=output_dir / "ntk_eigenvalues.png")
    plot_row_projection(
        row_projection,
        batch=batch,
        explained_variance=explained_variance,
        path=output_dir / "ntk_rows_pca.png",
    )

    save_array(output_dir / "input_batch_used.npy", batch)
    save_array(output_dir / "model_predictions.npy", predictions)
    save_array(output_dir / "ntk_matrix.npy", ntk_matrix)
    save_array(output_dir / "jacobian_tensor.npy", full_jacobian)

    report = {
        "weights_info": {
            "source": weights_info.source,
            "description": weights_info.description,
            "extra": weights_info.extra,
        },
        "batch_info": {
            "source": batch_info.source,
            "description": batch_info.description,
            "shape": list(batch.shape),
        },
        "model_builder": args.builder,
        "dtype": args.dtype,
        "seed": args.seed,
        "output_dir": str(output_dir),
        "cluster_order": cluster_order.tolist() if cluster_order is not None else None,
        "metrics": metrics,
    }
    save_json(output_dir / "ntk_report.json", report)

    print(summary_text)
    print(f"\nWeights: {weights_info.source}")
    print(weights_info.description)
    if weights_info.extra:
        print(json.dumps(weights_info.extra, indent=2, ensure_ascii=True))

    print(f"\nBatch source: {batch_info.source}")
    print(batch_info.description)
    print(f"Batch shape: {batch.shape}")

    print_report(metrics)
    print(f"\nSaved outputs to: {output_dir}")
    print("Main files:")
    print(f"  - {output_dir / 'ntk_heatmap.png'}")
    print(f"  - {output_dir / 'ntk_eigenvalues.png'}")
    print(f"  - {output_dir / 'ntk_rows_pca.png'}")
    if cluster_order is not None:
        print(f"  - {output_dir / 'ntk_clustered_heatmap.png'}")
    print(f"  - {output_dir / 'ntk_report.json'}")
    print(f"  - {output_dir / 'ntk_matrix.npy'}")


if __name__ == "__main__":
    main()
