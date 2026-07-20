#!/usr/bin/env python3
"""Stage 3: reconstruct and export an STL mesh from a NIfTI label map.

The script supports reproducible command-line execution and an optional PyQt5
GUI. A selected integer label is first converted to a binary mask, then processed
with Marching Cubes, triangle cleaning, optional windowed-sinc smoothing, physical
orientation restoration, and normal generation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import (
    vtkCleanPolyData,
    vtkMarchingCubes,
    vtkPolyDataNormals,
    vtkTriangleFilter,
)
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersModeling import vtkWindowedSincPolyDataFilter
from vtkmodules.vtkIOGeometry import vtkSTLWriter


class ReconstructionError(RuntimeError):
    """Raised when a valid surface cannot be reconstructed."""


def read_label_mask(path: Path, label: int) -> tuple[sitk.Image, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Input NIfTI file not found: {path}")
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3:
        raise ReconstructionError(f"Expected a 3D image, got dimension {image.GetDimension()}.")

    array_zyx = sitk.GetArrayFromImage(image)
    binary_zyx = np.asarray(array_zyx == label, dtype=np.uint8)
    voxel_count = int(binary_zyx.sum())
    if voxel_count == 0:
        available = np.unique(array_zyx)
        preview = ", ".join(str(value) for value in available[:20])
        raise ReconstructionError(
            f"Label {label} is absent. Available values include: {preview}"
        )
    return image, binary_zyx


def numpy_mask_to_vtk(binary_zyx: np.ndarray, image: sitk.Image) -> vtkImageData:
    """Create axis-aligned vtkImageData while preserving x-fastest voxel ordering."""
    if binary_zyx.ndim != 3:
        raise ValueError(f"Expected a 3D NumPy array, got shape {binary_zyx.shape}")

    depth, height, width = binary_zyx.shape
    spacing = image.GetSpacing()
    origin = image.GetOrigin()

    vtk_image = vtkImageData()
    vtk_image.SetDimensions(width, height, depth)
    vtk_image.SetSpacing(*spacing)
    vtk_image.SetOrigin(*origin)

    # SimpleITK arrays have shape [z, y, x]. C-order flattening therefore keeps
    # x as the fastest-changing index, matching vtkImageData point ordering.
    flat = np.ascontiguousarray(binary_zyx).ravel(order="C")
    vtk_array = numpy_to_vtk(flat, deep=True)
    vtk_array.SetName("label")
    vtk_image.GetPointData().SetScalars(vtk_array)
    return vtk_image


def build_direction_transform(image: sitk.Image) -> vtkTransform:
    """Map axis-aligned coordinates to the image's physical direction matrix."""
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    origin = np.asarray(image.GetOrigin(), dtype=float)

    matrix = vtkMatrix4x4()
    matrix.Identity()
    translation = origin - direction @ origin
    for row in range(3):
        for column in range(3):
            matrix.SetElement(row, column, float(direction[row, column]))
        matrix.SetElement(row, 3, float(translation[row]))

    transform = vtkTransform()
    transform.SetMatrix(matrix)
    return transform


def reconstruct_mesh(
    input_path: Path,
    label: int = 1,
    smoothing_iterations: int = 30,
    passband: float = 0.02,
) -> tuple[vtkPolyData, dict[str, Any]]:
    if label < 1:
        raise ValueError("Label must be a positive integer; 0 is reserved for background.")
    if smoothing_iterations < 0:
        raise ValueError("Smoothing iterations must be non-negative.")
    if not 0.0 < passband <= 2.0:
        raise ValueError("Passband must be in (0, 2].")

    image, binary_zyx = read_label_mask(input_path, label)
    vtk_image = numpy_mask_to_vtk(binary_zyx, image)

    marching_cubes = vtkMarchingCubes()
    marching_cubes.SetInputData(vtk_image)
    marching_cubes.SetValue(0, 0.5)
    marching_cubes.ComputeNormalsOff()
    marching_cubes.ComputeGradientsOff()

    triangle = vtkTriangleFilter()
    triangle.SetInputConnection(marching_cubes.GetOutputPort())

    clean = vtkCleanPolyData()
    clean.SetInputConnection(triangle.GetOutputPort())
    clean.PointMergingOn()

    current_port = clean.GetOutputPort()
    smoother = None
    if smoothing_iterations > 0:
        smoother = vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(current_port)
        smoother.SetNumberOfIterations(smoothing_iterations)
        smoother.SetPassBand(passband)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        current_port = smoother.GetOutputPort()

    transform_filter = vtkTransformPolyDataFilter()
    transform_filter.SetTransform(build_direction_transform(image))
    transform_filter.SetInputConnection(current_port)

    normals = vtkPolyDataNormals()
    normals.SetInputConnection(transform_filter.GetOutputPort())
    normals.AutoOrientNormalsOn()
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.Update()

    output = vtkPolyData()
    output.DeepCopy(normals.GetOutput())
    if output.GetNumberOfPoints() == 0 or output.GetNumberOfCells() == 0:
        raise ReconstructionError("Marching Cubes produced an empty surface.")

    stats = {
        "label": label,
        "foreground_voxels": int(binary_zyx.sum()),
        "points": int(output.GetNumberOfPoints()),
        "triangles": int(output.GetNumberOfCells()),
        "spacing_xyz": tuple(float(value) for value in image.GetSpacing()),
        "origin_xyz": tuple(float(value) for value in image.GetOrigin()),
        "smoothing_iterations": smoothing_iterations,
        "passband": passband,
    }
    return output, stats


def write_stl(mesh: vtkPolyData, output_path: Path, binary: bool = True) -> None:
    output_path = output_path.with_suffix(".stl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = vtkSTLWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(mesh)
    if binary:
        writer.SetFileTypeToBinary()
    else:
        writer.SetFileTypeToASCII()
    if writer.Write() != 1:
        raise ReconstructionError(f"VTK failed to write STL: {output_path}")


def run_cli(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    mesh, stats = reconstruct_mesh(
        input_path=input_path,
        label=args.label,
        smoothing_iterations=args.smoothing_iterations,
        passband=args.passband,
    )
    write_stl(mesh, output_path, binary=not args.ascii)
    print(f"Saved STL: {output_path.with_suffix('.stl')}")
    print(
        f"Label={stats['label']} | voxels={stats['foreground_voxels']} | "
        f"points={stats['points']} | triangles={stats['triangles']}"
    )


def launch_gui() -> None:
    try:
        from PyQt5.QtCore import Qt, QTimer
        from PyQt5.QtWidgets import (
            QApplication,
            QDoubleSpinBox,
            QFileDialog,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSlider,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera  # noqa: F401
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer
        from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLRenderer  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "GUI dependencies are unavailable. Install PyQt5 and a VTK build with Qt support, "
            "or run the command-line interface with --input and --output."
        ) from exc

    class ReconstructionWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Stage 3 — NIfTI label to STL")
            self.resize(1200, 760)
            self.input_path: Path | None = None
            self.current_mesh: vtkPolyData | None = None

            self.renderer = vtkRenderer()
            self.renderer.SetBackground(0.12, 0.12, 0.12)
            self.mapper = vtkPolyDataMapper()
            self.mapper.ScalarVisibilityOff()
            self.actor = vtkActor()
            self.actor.SetMapper(self.mapper)
            self.renderer.AddActor(self.actor)

            central = QWidget(self)
            self.setCentralWidget(central)
            root = QHBoxLayout(central)
            controls = QVBoxLayout()
            root.addLayout(controls, 0)

            self.vtk_widget = QVTKRenderWindowInteractor(central)
            self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
            root.addWidget(self.vtk_widget, 1)

            file_box = QGroupBox("Input / output")
            file_layout = QVBoxLayout(file_box)
            open_button = QPushButton("Open NIfTI label map")
            open_button.clicked.connect(self.open_file)
            file_layout.addWidget(open_button)
            self.file_label = QLabel("No file selected")
            self.file_label.setWordWrap(True)
            file_layout.addWidget(self.file_label)
            export_button = QPushButton("Export STL")
            export_button.clicked.connect(self.export_stl)
            file_layout.addWidget(export_button)
            controls.addWidget(file_box)

            parameter_box = QGroupBox("Reconstruction parameters")
            parameter_layout = QVBoxLayout(parameter_box)

            label_row = QHBoxLayout()
            label_row.addWidget(QLabel("Integer label"))
            self.label_spin = QSpinBox()
            self.label_spin.setRange(1, 65535)
            self.label_spin.setValue(1)
            self.label_spin.valueChanged.connect(self.schedule_update)
            label_row.addWidget(self.label_spin)
            parameter_layout.addLayout(label_row)

            iteration_row = QHBoxLayout()
            iteration_row.addWidget(QLabel("Smoothing iterations"))
            self.iteration_slider = QSlider(Qt.Horizontal)
            self.iteration_slider.setRange(0, 200)
            self.iteration_slider.setValue(30)
            self.iteration_slider.valueChanged.connect(self.iteration_changed)
            iteration_row.addWidget(self.iteration_slider)
            self.iteration_label = QLabel("30")
            self.iteration_label.setFixedWidth(40)
            iteration_row.addWidget(self.iteration_label)
            parameter_layout.addLayout(iteration_row)

            passband_row = QHBoxLayout()
            passband_row.addWidget(QLabel("Smoothing passband"))
            self.passband_spin = QDoubleSpinBox()
            self.passband_spin.setDecimals(4)
            self.passband_spin.setRange(0.0005, 0.5)
            self.passband_spin.setSingleStep(0.005)
            self.passband_spin.setValue(0.02)
            self.passband_spin.valueChanged.connect(self.schedule_update)
            passband_row.addWidget(self.passband_spin)
            parameter_layout.addLayout(passband_row)

            controls.addWidget(parameter_box)
            controls.addStretch(1)

            self.timer = QTimer(self)
            self.timer.setSingleShot(True)
            self.timer.timeout.connect(self.update_mesh)
            self.vtk_widget.GetRenderWindow().GetInteractor().Initialize()

        def iteration_changed(self, value: int) -> None:
            self.iteration_label.setText(str(value))
            self.schedule_update()

        def schedule_update(self) -> None:
            if self.input_path is not None:
                self.timer.start(150)

        def open_file(self) -> None:
            selected, _ = QFileDialog.getOpenFileName(
                self, "Open NIfTI label map", "", "NIfTI (*.nii *.nii.gz)"
            )
            if not selected:
                return
            self.input_path = Path(selected)
            self.file_label.setText(selected)
            self.update_mesh(reset_camera=True)

        def update_mesh(self, reset_camera: bool = False) -> None:
            if self.input_path is None:
                return
            try:
                mesh, stats = reconstruct_mesh(
                    self.input_path,
                    label=int(self.label_spin.value()),
                    smoothing_iterations=int(self.iteration_slider.value()),
                    passband=float(self.passband_spin.value()),
                )
                self.current_mesh = mesh
                self.mapper.SetInputData(mesh)
                if reset_camera:
                    self.renderer.ResetCamera()
                self.vtk_widget.GetRenderWindow().Render()
                self.statusBar().showMessage(
                    f"Label={stats['label']} | voxels={stats['foreground_voxels']} | "
                    f"points={stats['points']} | triangles={stats['triangles']}"
                )
            except Exception as exc:  # GUI boundary: show a readable error dialog.
                self.current_mesh = None
                QMessageBox.critical(self, "Reconstruction failed", str(exc))

        def export_stl(self) -> None:
            if self.current_mesh is None:
                QMessageBox.warning(self, "Nothing to export", "Load and reconstruct a label map first.")
                return
            selected, _ = QFileDialog.getSaveFileName(
                self, "Export STL", "reconstruction.stl", "STL (*.stl)"
            )
            if not selected:
                return
            try:
                output_path = Path(selected).with_suffix(".stl")
                write_stl(self.current_mesh, output_path, binary=True)
                QMessageBox.information(self, "Export complete", f"Saved:\n{output_path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export failed", str(exc))

    application = QApplication(sys.argv)
    window = ReconstructionWindow()
    window.show()
    raise SystemExit(application.exec_())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Input NIfTI label map.")
    parser.add_argument("--output", help="Output STL path.")
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--smoothing-iterations", type=int, default=30)
    parser.add_argument("--passband", type=float, default=0.02)
    parser.add_argument("--ascii", action="store_true", help="Write ASCII instead of binary STL.")
    parser.add_argument("--gui", action="store_true", help="Launch the graphical interface.")
    args = parser.parse_args()
    if not args.gui and bool(args.input) != bool(args.output):
        parser.error("Provide both --input and --output, or use --gui.")
    return args


def main() -> None:
    args = parse_args()
    if args.gui or not args.input:
        launch_gui()
    else:
        run_cli(args)


if __name__ == "__main__":
    main()
