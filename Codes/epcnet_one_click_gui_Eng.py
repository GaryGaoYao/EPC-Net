# -*- coding: utf-8 -*-
"""
EPC-Net one-click orbital bone STL generator
=============================================

Simple Windows desktop GUI:
1. Select one CT/CBCT NIfTI volume.
2. Select the Stage-1 eye model and Stage-2 bone model.
3. Choose the final STL save path.
4. Click one button.

The application performs internally:
- Stage-1 eye segmentation;
- Stage-2 eye-guided orbital bone segmentation;
- mild 3D Gaussian smoothing of the binary bone mask;
- Marching Cubes surface reconstruction;
- Windowed-Sinc mesh smoothing and STL export.

Only the final STL is kept. All intermediate NIfTI files are created in a
system temporary directory and deleted automatically.

Research use only. This software is not a certified medical device.
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt

from monai.data.meta_tensor import MetaTensor
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    MapTransform,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)
from monai.utils import set_determinism

from PyQt5.QtCore import QThread, Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QLayout,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import (
    vtkCleanPolyData,
    vtkMarchingCubes,
    vtkPolyDataNormals,
    vtkTriangleFilter,
    vtkWindowedSincPolyDataFilter,
)
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkIOGeometry import vtkSTLWriter

# Load the VTK OpenGL backend when bundled on Windows.
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401


# -----------------------------------------------------------------------------
# Fixed inference settings matching the supplied training scripts
# -----------------------------------------------------------------------------
STAGE1_ROI = (96, 96, 96)
STAGE1_SPACING = (1.0, 1.0, 1.0)

STAGE2_ROI = (128, 128, 128)
STAGE2_SPACING = (0.5, 0.5, 1.0)
STAGE2_OVERLAP = 0.125
EYE_SIGMA_MM = 5.0
EYE_DILATE_MM = 1.5

DEFAULT_SW_BATCH_SIZE = 4

# Automatic reconstruction settings.
# A mild physical-space Gaussian removes voxel stair-stepping while preserving
# thin orbital structures better than aggressive smoothing.
GAUSSIAN_SIGMA_MM = 0.60
GAUSSIAN_ISO_VALUE = 0.50
MESH_SMOOTHING_ITERATIONS = 25
MESH_PASS_BAND = 0.05


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------
def application_dir() -> Path:
    """Return the script directory or the PyInstaller executable directory."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def case_id_from_path(path: str) -> str:
    name = Path(path).name
    if name.lower().endswith(".nii.gz"):
        name = name[:-7]
    elif name.lower().endswith(".nii"):
        name = name[:-4]
    if name.endswith("_0000"):
        name = name[:-5]
    return name or "case"


def check_nifti(path: str, label: str) -> None:
    if not path:
        raise ValueError(f"Please select {label}.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} was not found:\n{path}")
    if not path.lower().endswith((".nii", ".nii.gz")):
        raise ValueError(f"{label} must be a .nii or .nii.gz file.")


def check_weight(path: str, label: str) -> None:
    if not path:
        raise ValueError(f"Please select {label}.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} was not found:\n{path}")


def normalize_stl_path(path: str) -> str:
    if not path:
        raise ValueError("Please choose a save path for the final STL file.")
    if not path.lower().endswith(".stl"):
        path += ".stl"
    parent = os.path.dirname(os.path.abspath(path))
    if not parent:
        raise ValueError("The STL save path is invalid.")
    os.makedirs(parent, exist_ok=True)
    return os.path.abspath(path)


def load_torch_object(path: str, device: torch.device) -> Any:
    """Load raw weights or a full checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def normalize_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("model", "model_state_dict", "state_dict", "network"):
            nested = obj.get(key)
            if isinstance(nested, dict):
                obj = nested
                break

    if not isinstance(obj, dict):
        raise ValueError("The model file is not a valid PyTorch state_dict or checkpoint.")

    state: Dict[str, torch.Tensor] = {}
    for key, value in obj.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean_key = str(key)
        for prefix in ("module.", "_orig_mod."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        state[clean_key] = value

    if not state:
        raise ValueError("No loadable network weights were found in the model file.")
    return state


def load_model_weights(
    model: torch.nn.Module,
    path: str,
    device: torch.device,
) -> None:
    model.load_state_dict(
        normalize_state_dict(load_torch_object(path, device)),
        strict=True,
    )


def run_sliding_window(
    image: torch.Tensor,
    model: torch.nn.Module,
    roi_size: Tuple[int, int, int],
    overlap: float,
    device: torch.device,
) -> torch.Tensor:
    """Run AMP inference and retry with batch size 1 after CUDA OOM."""

    def _run(sw_batch_size: int) -> torch.Tensor:
        with torch.inference_mode():
            with torch.amp.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
            ):
                return sliding_window_inference(
                    inputs=image,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    sw_device=device,
                    device=device,
                )

    try:
        return _run(DEFAULT_SW_BATCH_SIZE)
    except RuntimeError as exc:
        if device.type == "cuda" and "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            return _run(1)
        raise


def save_meta_tensor_as_nifti(
    data: MetaTensor | torch.Tensor,
    path: str,
) -> None:
    array = data.detach().cpu().numpy()
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"The predicted mask has an invalid shape: {array.shape}")

    affine = np.eye(4, dtype=np.float64)
    if isinstance(data, MetaTensor):
        affine = np.asarray(data.affine.detach().cpu(), dtype=np.float64)
    nib.save(nib.Nifti1Image(array.astype(np.uint8), affine), path)


def resample_mask_to_reference(
    mask_path: str,
    reference_path: str,
    output_path: str,
) -> str:
    mask = sitk.ReadImage(mask_path)
    reference = sitk.ReadImage(reference_path)
    result = sitk.Resample(
        mask,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    sitk.WriteImage(result, output_path, useCompression=True)
    return output_path


def mask_has_foreground(path: str) -> bool:
    image = sitk.ReadImage(path)
    return bool(np.any(sitk.GetArrayViewFromImage(image) > 0))


# -----------------------------------------------------------------------------
# Stage 1: eye segmentation
# -----------------------------------------------------------------------------
def build_stage1_model(device: torch.device) -> SegResNet:
    return SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=32,
        dropout_prob=0.2,
    ).to(device)


def infer_eye(
    ct_path: str,
    weight_path: str,
    output_path: str,
    device: torch.device,
) -> str:
    preprocessing = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            EnsureTyped(keys=["image"], track_meta=True),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=STAGE1_SPACING,
                mode=("bilinear",),
            ),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.5,
                upper=99.5,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ]
    )

    data = preprocessing({"image": ct_path})
    image = data["image"].unsqueeze(0).to(device, non_blocking=True)

    model = build_stage1_model(device)
    load_model_weights(model, weight_path, device)
    model.eval()

    logits = run_sliding_window(
        image=image,
        model=model,
        roi_size=STAGE1_ROI,
        overlap=0.25,
        device=device,
    )
    prediction = torch.argmax(logits, dim=1, keepdim=True)[0].cpu()

    data["pred"] = prediction
    inverse = Compose(
        [
            EnsureTyped(keys=["pred"], track_meta=True),
            Invertd(
                keys=["pred"],
                transform=preprocessing,
                orig_keys=["image"],
                nearest_interp=True,
                to_tensor=True,
            ),
        ]
    )
    result = inverse(data)["pred"]
    save_meta_tensor_as_nifti(result, output_path)

    del logits, prediction, image, model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not mask_has_foreground(output_path):
        raise RuntimeError("Stage 1 did not detect the eye region. Please check the CT volume and Stage 1 model.")
    return output_path


# -----------------------------------------------------------------------------
# Stage 2: eye-guided orbital bone segmentation
# -----------------------------------------------------------------------------
def affine_spacing_for_array(affine: np.ndarray) -> Tuple[float, float, float]:
    matrix = np.asarray(affine, dtype=np.float64)[:3, :3]
    spacing = np.linalg.norm(matrix, axis=0)
    return tuple(float(value) for value in spacing)  # type: ignore[return-value]


class MakeEyeGuidanced(MapTransform):
    """Create the Stage-2 eye-distance guidance channel."""

    def __init__(self) -> None:
        super().__init__(keys=["eye"])

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(data)
        reference = result["image"]
        eye = result["eye"]

        eye_tensor = eye if isinstance(eye, torch.Tensor) else torch.as_tensor(eye)
        eye_array = eye_tensor.detach().cpu().numpy()
        if eye_array.ndim == 4:
            eye_array = eye_array[0]
        eye_binary = eye_array > 0.5

        if isinstance(reference, MetaTensor):
            spacing = affine_spacing_for_array(
                np.asarray(reference.affine.detach().cpu())
            )
        else:
            spacing = STAGE2_SPACING

        if EYE_DILATE_MM > 0:
            iterations = max(
                1,
                int(round(EYE_DILATE_MM / max(min(spacing), 1e-6))),
            )
            eye_binary = binary_dilation(eye_binary, iterations=iterations)

        if np.any(eye_binary):
            distance_mm = distance_transform_edt(
                ~eye_binary,
                sampling=spacing,
            ).astype(np.float32)
            guidance_array = np.exp(-distance_mm / EYE_SIGMA_MM).astype(
                np.float32
            )
        else:
            guidance_array = np.zeros(eye_binary.shape, dtype=np.float32)

        guidance = torch.from_numpy(guidance_array[None, ...])
        if isinstance(reference, MetaTensor):
            guidance = MetaTensor(
                guidance,
                affine=reference.affine,
                meta=dict(reference.meta),
            )

        result["guidance"] = guidance
        return result


def build_stage2_model(device: torch.device) -> SegResNet:
    return SegResNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=2,
        init_filters=32,
        dropout_prob=0.2,
    ).to(device)


def infer_bone(
    ct_path: str,
    eye_path: str,
    weight_path: str,
    output_path: str,
    device: torch.device,
    temporary_directory: str,
) -> str:
    aligned_eye_path = os.path.join(temporary_directory, "eye_aligned.nii.gz")
    resample_mask_to_reference(eye_path, ct_path, aligned_eye_path)

    keys = ["image", "eye"]
    transforms = Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=STAGE2_SPACING,
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.5,
                upper=99.5,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=keys, track_meta=True),
            MakeEyeGuidanced(),
            ConcatItemsd(keys=["image", "guidance"], name="image2", dim=0),
            EnsureTyped(keys=["image2"], track_meta=False),
        ]
    )

    data = transforms({"image": ct_path, "eye": aligned_eye_path})
    input_tensor = data["image2"].unsqueeze(0).to(device, non_blocking=True)

    model = build_stage2_model(device)
    load_model_weights(model, weight_path, device)
    model.eval()

    logits = run_sliding_window(
        image=input_tensor,
        model=model,
        roi_size=STAGE2_ROI,
        overlap=STAGE2_OVERLAP,
        device=device,
    )
    prediction = (
        torch.argmax(logits, dim=1)[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    affine = np.eye(4, dtype=np.float64)
    image_meta = data.get("image")
    if isinstance(image_meta, MetaTensor):
        affine = np.asarray(image_meta.affine.detach().cpu(), dtype=np.float64)

    stage2_space_path = os.path.join(temporary_directory, "bone_stage2.nii.gz")
    nib.save(nib.Nifti1Image(prediction, affine), stage2_space_path)
    resample_mask_to_reference(stage2_space_path, ct_path, output_path)

    del logits, prediction, input_tensor, model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not mask_has_foreground(output_path):
        raise RuntimeError("Stage 2 did not detect the orbital bone region. Please check the CT volume and Stage 2 model.")
    return output_path


# -----------------------------------------------------------------------------
# Stage 3: automatic Gaussian smoothing + surface reconstruction
# -----------------------------------------------------------------------------
def gaussian_smooth_mask(mask: sitk.Image) -> sitk.Image:
    """Return a physical-space Gaussian-smoothed probability-like mask."""
    binary = sitk.Cast(mask > 0, sitk.sitkFloat32)
    # SmoothingRecursiveGaussian interprets sigma in physical units (mm).
    gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian.SetSigma(GAUSSIAN_SIGMA_MM)
    smoothed = gaussian.Execute(binary)
    smoothed.CopyInformation(mask)
    return smoothed


def sitk_scalar_to_vtk_image(image: sitk.Image) -> vtkImageData:
    array_zyx = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    if not np.any(array_zyx > 0):
        raise RuntimeError("The orbital bone segmentation is empty, so an STL cannot be generated.")

    z_size, y_size, x_size = array_zyx.shape
    vtk_image = vtkImageData()
    vtk_image.SetDimensions(x_size, y_size, z_size)
    vtk_image.SetSpacing(*image.GetSpacing())
    vtk_image.SetOrigin(*image.GetOrigin())

    # [z, y, x] C-order already stores x as the fastest axis.
    vtk_array = numpy_to_vtk(
        array_zyx.ravel(order="C"),
        deep=True,
    )
    vtk_array.SetName("smoothed_mask")
    vtk_image.GetPointData().SetScalars(vtk_array)
    return vtk_image


def direction_transform(image: sitk.Image) -> vtkTransform:
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)

    transform = vtkTransform()
    transform.PostMultiply()
    if np.allclose(direction, np.eye(3), atol=1e-6):
        transform.Identity()
        return transform

    matrix = vtkMatrix4x4()
    matrix.Identity()
    for row in range(3):
        for column in range(3):
            matrix.SetElement(row, column, float(direction[row, column]))

    transform.Translate(-origin[0], -origin[1], -origin[2])
    transform.Concatenate(matrix)
    transform.Translate(origin[0], origin[1], origin[2])
    return transform


def reconstruct_stl(mask_path: str, output_path: str) -> str:
    original_mask = sitk.ReadImage(mask_path)
    smoothed_mask = gaussian_smooth_mask(original_mask)
    vtk_image = sitk_scalar_to_vtk_image(smoothed_mask)

    marching_cubes = vtkMarchingCubes()
    marching_cubes.SetInputData(vtk_image)
    marching_cubes.SetValue(0, GAUSSIAN_ISO_VALUE)
    marching_cubes.ComputeNormalsOff()
    marching_cubes.ComputeGradientsOff()

    triangles = vtkTriangleFilter()
    triangles.SetInputConnection(marching_cubes.GetOutputPort())

    clean = vtkCleanPolyData()
    clean.SetInputConnection(triangles.GetOutputPort())

    # Surface fairing based on the supplied reconstruction example.
    smoothing = vtkWindowedSincPolyDataFilter()
    smoothing.SetInputConnection(clean.GetOutputPort())
    smoothing.SetNumberOfIterations(MESH_SMOOTHING_ITERATIONS)
    smoothing.SetPassBand(MESH_PASS_BAND)
    smoothing.BoundarySmoothingOff()
    smoothing.FeatureEdgeSmoothingOff()
    smoothing.NonManifoldSmoothingOn()
    smoothing.NormalizeCoordinatesOn()

    transform_filter = vtkTransformPolyDataFilter()
    transform_filter.SetTransform(direction_transform(original_mask))
    transform_filter.SetInputConnection(smoothing.GetOutputPort())

    normals = vtkPolyDataNormals()
    normals.SetInputConnection(transform_filter.GetOutputPort())
    normals.SetAutoOrientNormals(True)
    normals.SetConsistency(True)
    normals.SplittingOff()
    normals.Update()

    mesh = normals.GetOutput()
    if mesh is None or mesh.GetNumberOfPoints() == 0:
        raise RuntimeError("No valid mesh could be generated after Gaussian smoothing.")

    output_path = normalize_stl_path(output_path)
    writer = vtkSTLWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(mesh)
    writer.SetFileTypeToBinary()
    if writer.Write() != 1:
        raise RuntimeError("Failed to write the STL file.")
    return output_path


# -----------------------------------------------------------------------------
# Background worker
# -----------------------------------------------------------------------------
class PipelineWorker(QThread):
    progress_changed = pyqtSignal(int, str)
    completed = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        ct_path: str,
        stage1_weight: str,
        stage2_weight: str,
        stl_path: str,
    ) -> None:
        super().__init__()
        self.ct_path = ct_path
        self.stage1_weight = stage1_weight
        self.stage2_weight = stage2_weight
        self.stl_path = stl_path

    def run(self) -> None:
        try:
            warnings.filterwarnings("ignore", category=FutureWarning)
            set_determinism(seed=0)

            check_nifti(self.ct_path, "the input CT volume")
            check_weight(self.stage1_weight, "the Stage 1 eye model")
            check_weight(self.stage2_weight, "the Stage 2 orbital bone model")
            final_stl = normalize_stl_path(self.stl_path)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device_name = (
                torch.cuda.get_device_name(0)
                if device.type == "cuda"
                else "CPU"
            )

            with tempfile.TemporaryDirectory(prefix="epcnet_") as temp_dir:
                eye_path = os.path.join(temp_dir, "eye.nii.gz")
                bone_path = os.path.join(temp_dir, "bone.nii.gz")

                self.progress_changed.emit(5, f"Preparing ({device_name})...")
                self.progress_changed.emit(12, "Segmenting the eye region...")
                infer_eye(
                    ct_path=self.ct_path,
                    weight_path=self.stage1_weight,
                    output_path=eye_path,
                    device=device,
                )

                self.progress_changed.emit(45, "Segmenting the orbital bone...")
                infer_bone(
                    ct_path=self.ct_path,
                    eye_path=eye_path,
                    weight_path=self.stage2_weight,
                    output_path=bone_path,
                    device=device,
                    temporary_directory=temp_dir,
                )

                self.progress_changed.emit(82, "Smoothing the surface and generating the STL...")
                reconstruct_stl(bone_path, final_stl)

            self.progress_changed.emit(100, "Completed")
            self.completed.emit(final_stl)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc) or exc.__class__.__name__)


# -----------------------------------------------------------------------------
# Minimal GUI
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Compact GUI
# -----------------------------------------------------------------------------
class FileRow:
    """One fixed-height row: label, path, and action button."""

    def __init__(self, label_text: str, button_text: str) -> None:
        self.label = QLabel(label_text)
        self.label.setObjectName("fieldLabel")
        self.label.setMinimumWidth(150)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.edit = QLineEdit()
        self.edit.setReadOnly(True)
        self.edit.setPlaceholderText("Not selected")
        self.edit.setFixedHeight(38)
        self.edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.button = QPushButton(button_text)
        self.button.setObjectName("browseButton")
        self.button.setFixedSize(88, 38)

    def add_to_grid(self, grid: QGridLayout, row: int) -> None:
        grid.addWidget(self.label, row, 0)
        grid.addWidget(self.edit, row, 1)
        grid.addWidget(self.button, row, 2)
        grid.setRowMinimumHeight(row, 38)

    def text(self) -> str:
        return self.edit.text().strip()

    def set_text(self, value: str) -> None:
        self.edit.setText(value)
        self.edit.setToolTip(value)

    def set_enabled(self, enabled: bool) -> None:
        self.edit.setEnabled(enabled)
        self.button.setEnabled(enabled)


class ModelSettingsDialog(QDialog):
    """Small dialog for changing the two model files."""

    def __init__(
        self,
        parent: QWidget,
        stage1_path: str,
        stage2_path: str,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Model Settings")
        self.setModal(True)
        self.setMinimumWidth(780)

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(16)
        root.setSizeConstraint(QLayout.SetMinimumSize)

        title = QLabel("Model Settings")
        title.setObjectName("dialogTitle")
        root.addWidget(title)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        grid.setColumnStretch(1, 1)

        self.stage1_field = FileRow("Stage 1 eye model", "Browse")
        self.stage2_field = FileRow("Stage 2 bone model", "Browse")
        self.stage1_field.set_text(stage1_path)
        self.stage2_field.set_text(stage2_path)
        self.stage1_field.add_to_grid(grid, 0)
        self.stage2_field.add_to_grid(grid, 1)
        root.addLayout(grid)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        save_button = buttons.button(QDialogButtonBox.Save)
        cancel_button = buttons.button(QDialogButtonBox.Cancel)
        save_button.setText("Save")
        cancel_button.setText("Cancel")
        save_button.setObjectName("primarySmallButton")
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.stage1_field.button.clicked.connect(
            lambda: self._select_weight(self.stage1_field, "Select Stage 1 weights")
        )
        self.stage2_field.button.clicked.connect(
            lambda: self._select_weight(self.stage2_field, "Select Stage 2 weights")
        )

    def _select_weight(self, field: FileRow, title: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            field.text(),
            "PyTorch weights (*.pth *.pt *.ckpt);;All files (*)",
        )
        if path:
            field.set_text(path)

    def _validate_and_accept(self) -> None:
        try:
            check_weight(self.stage1_field.text(), "the Stage 1 eye model")
            check_weight(self.stage2_field.text(), "the Stage 2 orbital bone model")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Invalid Model Path", str(exc))
            return
        self.accept()

    def paths(self) -> Tuple[str, str]:
        return self.stage1_field.text(), self.stage2_field.text()


class EPCNetSTLWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.worker: PipelineWorker | None = None
        self.stage1_weight_path = ""
        self.stage2_weight_path = ""

        self.setWindowTitle("EPC-Net Orbital Bone STL Generator")
        self.setMinimumWidth(820)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)
        root.setSizeConstraint(QLayout.SetMinimumSize)

        title = QLabel("EPC-Net Orbital Bone STL Generator")
        title.setObjectName("title")
        root.addWidget(title)

        file_grid = QGridLayout()
        file_grid.setContentsMargins(0, 0, 0, 0)
        file_grid.setHorizontalSpacing(12)
        file_grid.setVerticalSpacing(12)
        file_grid.setColumnStretch(1, 1)

        self.ct_field = FileRow("CT / CBCT NIfTI", "Browse")
        self.stl_field = FileRow("STL output", "Save As")
        self.ct_field.add_to_grid(file_grid, 0)
        self.stl_field.add_to_grid(file_grid, 1)
        root.addLayout(file_grid)

        model_row = QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.setSpacing(12)

        self.model_status = QLabel("Models not configured")
        self.model_status.setObjectName("modelStatus")
        self.model_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.model_status.setFixedHeight(36)
        self.model_status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.model_button = QPushButton("Model Settings")
        self.model_button.setObjectName("secondaryButton")
        self.model_button.setFixedSize(128, 36)

        model_row.addWidget(self.model_status, 1)
        model_row.addWidget(self.model_button, 0)
        root.addLayout(model_row)

        self.run_button = QPushButton("Generate STL")
        self.run_button.setObjectName("runButton")
        self.run_button.setFixedHeight(48)
        root.addWidget(self.run_button)

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(10)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("status")
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.status_label.setFixedHeight(24)
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.progress_value = QLabel("0%")
        self.progress_value.setObjectName("progressValue")
        self.progress_value.setFixedSize(42, 24)
        self.progress_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.progress_value, 0)
        root.addLayout(status_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        root.addWidget(self.progress)

        self.ct_field.button.clicked.connect(self._select_ct)
        self.stl_field.button.clicked.connect(self._select_stl_path)
        self.model_button.clicked.connect(self._open_model_settings)
        self.run_button.clicked.connect(self._start)

        self._apply_style()
        self._autofill_weights()
        self._update_model_status()
        self.adjustSize()

    def _apply_style(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        app.setStyleSheet(
            """
            QMainWindow, QDialog, QWidget {
                background: #f5f7f9;
                color: #17212b;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 13px;
            }
            QLabel#title {
                color: #111827;
                font-size: 20px;
                font-weight: 700;
                min-height: 28px;
            }
            QLabel#dialogTitle {
                color: #111827;
                font-size: 18px;
                font-weight: 700;
                min-height: 26px;
            }
            QLabel#fieldLabel {
                color: #374151;
                font-weight: 600;
            }
            QLabel#modelStatus {
                color: #667381;
            }
            QLabel#modelStatus[ready="true"] {
                color: #176b42;
                font-weight: 600;
            }
            QLabel#status {
                color: #455463;
            }
            QLabel#progressValue {
                color: #6b7785;
                font-size: 12px;
            }
            QLineEdit {
                background: #ffffff;
                border: 1px solid #ccd6df;
                border-radius: 7px;
                padding: 0 10px;
                color: #25313d;
                selection-background-color: #2173ad;
            }
            QLineEdit:focus {
                border-color: #2173ad;
            }
            QLineEdit:disabled {
                background: #eef1f4;
                color: #8b96a0;
            }
            QPushButton {
                background: #ffffff;
                border: 1px solid #c9d3dc;
                border-radius: 7px;
                color: #243241;
                font-weight: 600;
                padding: 0 12px;
            }
            QPushButton:hover {
                background: #edf3f7;
            }
            QPushButton:pressed {
                background: #e3ebf1;
            }
            QPushButton:disabled {
                background: #edf0f2;
                color: #9aa4ad;
                border-color: #dde3e7;
            }
            QPushButton#runButton {
                background: #176fa8;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 700;
            }
            QPushButton#runButton:hover {
                background: #145f90;
            }
            QPushButton#runButton:pressed {
                background: #104f79;
            }
            QPushButton#runButton:disabled {
                background: #9fb8c9;
            }
            QPushButton#primarySmallButton {
                background: #176fa8;
                color: #ffffff;
                border: none;
                min-height: 34px;
            }
            QProgressBar {
                background: #e2e7eb;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background: #2173ad;
                border-radius: 4px;
            }
            """
        )

    def _autofill_weights(self) -> None:
        base = application_dir()
        stage1_candidates = (
            base / "weights" / "stage1_eye.pth",
            base / "weights" / "stage1.pth",
            base / "stage1_eye.pth",
        )
        stage2_candidates = (
            base / "weights" / "stage2_bone.pth",
            base / "weights" / "stage2.pth",
            base / "stage2_bone.pth",
        )

        for candidate in stage1_candidates:
            if candidate.is_file():
                self.stage1_weight_path = str(candidate)
                break
        for candidate in stage2_candidates:
            if candidate.is_file():
                self.stage2_weight_path = str(candidate)
                break

    def _update_model_status(self) -> None:
        ready = bool(
            self.stage1_weight_path
            and self.stage2_weight_path
            and os.path.isfile(self.stage1_weight_path)
            and os.path.isfile(self.stage2_weight_path)
        )
        self.model_status.setProperty("ready", ready)
        self.model_status.setText("Models ready" if ready else "Configure models")
        self.model_status.style().unpolish(self.model_status)
        self.model_status.style().polish(self.model_status)

    def _select_ct(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CT / CBCT NIfTI",
            self.ct_field.text(),
            "NIfTI files (*.nii *.nii.gz)",
        )
        if not path:
            return
        self.ct_field.set_text(path)
        default_name = f"{case_id_from_path(path)}_orbital_bone.stl"
        self.stl_field.set_text(os.path.join(os.path.dirname(path), default_name))

    def _select_stl_path(self) -> None:
        initial = self.stl_field.text() or "orbital_bone.stl"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Final STL",
            initial,
            "STL mesh (*.stl)",
        )
        if path:
            self.stl_field.set_text(normalize_stl_path(path))

    def _open_model_settings(self) -> None:
        dialog = ModelSettingsDialog(
            self,
            self.stage1_weight_path,
            self.stage2_weight_path,
        )
        if dialog.exec_() == QDialog.Accepted:
            self.stage1_weight_path, self.stage2_weight_path = dialog.paths()
            self._update_model_status()

    def _set_busy(self, busy: bool) -> None:
        self.ct_field.set_enabled(not busy)
        self.stl_field.set_enabled(not busy)
        self.model_button.setEnabled(not busy)
        self.run_button.setEnabled(not busy)
        self.run_button.setText("Generating..." if busy else "Generate STL")

    def _start(self) -> None:
        try:
            check_nifti(self.ct_field.text(), "the input CT volume")
            check_weight(self.stage1_weight_path, "the Stage 1 eye model")
            check_weight(self.stage2_weight_path, "the Stage 2 orbital bone model")
            final_stl = normalize_stl_path(self.stl_field.text())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Unable to Start", str(exc))
            return

        if os.path.exists(final_stl):
            answer = QMessageBox.question(
                self,
                "Overwrite File",
                f"The file already exists:\n{final_stl}\n\nDo you want to overwrite it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        self.stl_field.set_text(final_stl)
        self.progress.setValue(0)
        self.progress_value.setText("0%")
        self.status_label.setText("Starting...")
        self._set_busy(True)

        self.worker = PipelineWorker(
            ct_path=self.ct_field.text(),
            stage1_weight=self.stage1_weight_path,
            stage2_weight=self.stage2_weight_path,
            stl_path=final_stl,
        )
        self.worker.progress_changed.connect(self._update_progress)
        self.worker.completed.connect(self._completed)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def _update_progress(self, value: int, text: str) -> None:
        self.progress.setValue(value)
        self.progress_value.setText(f"{value}%")
        self.status_label.setText(text)

    def _completed(self, stl_path: str) -> None:
        self.progress.setValue(100)
        self.progress_value.setText("100%")
        self.status_label.setText("Completed")
        QMessageBox.information(self, "Completed", f"The STL file has been saved to:\n\n{stl_path}")

    def _failed(self, message: str) -> None:
        self.progress.setValue(0)
        self.progress_value.setText("0%")
        self.status_label.setText("Failed")
        QMessageBox.critical(self, "Generation Failed", message)

    def _worker_finished(self) -> None:
        self._set_busy(False)
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Processing", "Please wait for the current process to finish before closing the application.")
            event.ignore()
            return
        event.accept()


def main() -> int:
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    QApplication.setAttribute(Qt.AA_UseDesktopOpenGL, True)

    app = QApplication(sys.argv)
    app.setApplicationName("EPC-Net One-Click STL")
    window = EPCNetSTLWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())