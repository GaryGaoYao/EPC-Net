"""3D Slicer module: EPC-Net automatic orbital bone segmentation.

Segments the orbital bone (and orbital content) of a head CT/CBCT volume with the two-stage
EPC-Net cascade and creates a segmentation node that can be reviewed, edited and exported
(e.g. as STL for patient-specific implant design).

Model weights are NOT distributed with the extension: users select their own
``stage1_eye.pth`` and ``stage2_bone.pth`` files (see the EPC-Net repository for access).
"""

import logging
import os
import sys
import threading
import time
from pathlib import Path

import ctk
import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleLogic,
                                           ScriptedLoadableModuleTest, ScriptedLoadableModuleWidget)

EPCNET_REPO_URL = "https://github.com/GaryGaoYao/EPC-Net"
# Python package installed into Slicer's Python environment on first use (pinned to a release tag)
EPCNET_PACKAGE_URL = "https://github.com/GaryGaoYao/EPC-Net/archive/refs/tags/v1.0.0.zip"
EPCNET_MIN_VERSION = "1.0.0"
MINIMUM_TORCH_VERSION = "2.1"
WEIGHTS_INFO_URL = EPCNET_REPO_URL + "/tree/main/weights"
PAPER_URL = "https://doi.org/10.1016/j.bspc.2026.111465"
SETTINGS_PREFIX = "EPCNet/"

DISCLAIMER = ("Research use only - not a medical device. All segmentations and surfaces must be reviewed and, "
              "if necessary, corrected by qualified users before any clinical or manufacturing use.")


class InstallError(Exception):
    pass


# -----------------------------------------------------------------------------------------------
class EPCNet(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "EPC-Net Orbital Bone"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Yao Gao (KU Leuven)"]
        self.parent.helpText = f"""
Automatic segmentation of the orbital bone (and orbital content) in head CT / CBCT with EPC-Net,
an anatomy-guided two-stage cascade. Stage I localises the orbital content; its distance-based priors
guide Stage II, which segments the thin orbital walls at high resolution.
<br><br>
1. Select the input CT volume.<br>
2. Select the model weights <i>stage1_eye.pth</i> and <i>stage2_bone.pth</i>
(<a href="{WEIGHTS_INFO_URL}">how to obtain them</a>).<br>
3. Click <b>Apply</b>. Review / edit the result in Segment Editor, then export the bone surface as STL.
<br><br>
<b>{DISCLAIMER}</b>
<br><br>See <a href="{EPCNET_REPO_URL}">{EPCNET_REPO_URL}</a>.
"""
        self.parent.acknowledgementText = f"""
If you use this module, please cite: Gao Y. et al., Fast and anatomy-guided orbital bone segmentation to support
Patient-Specific Implant design. Biomedical Signal Processing and Control 130 (2027) 111465.
<a href="{PAPER_URL}">{PAPER_URL}</a>
"""


# -----------------------------------------------------------------------------------------------
class EPCNetWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = EPCNetLogic()
        self._busy = False

        # ---- inputs
        box = ctk.ctkCollapsibleButton()
        box.text = "Inputs"
        self.layout.addWidget(box)
        form = qt.QFormLayout(box)

        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = False
        self.inputSelector.showHidden = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.setToolTip("Head CT or CBCT volume (Hounsfield units).")
        form.addRow("Input volume:", self.inputSelector)

        self.outputSelector = slicer.qMRMLNodeComboBox()
        self.outputSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.outputSelector.addEnabled = True
        self.outputSelector.renameEnabled = True
        self.outputSelector.removeEnabled = True
        self.outputSelector.noneEnabled = True
        self.outputSelector.noneDisplay = "(Create new segmentation)"
        self.outputSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("Output segmentation:", self.outputSelector)

        # ---- model weights (user supplied)
        box = ctk.ctkCollapsibleButton()
        box.text = "Model weights"
        self.layout.addWidget(box)
        form = qt.QFormLayout(box)
        self.stage1Path = self._pathEdit("Stage1Weights", "Stage I (orbital content) weights: stage1_eye.pth")
        self.stage2Path = self._pathEdit("Stage2Weights", "Stage II (orbital bone) weights: stage2_bone.pth")
        form.addRow("Stage I weights:", self.stage1Path)
        form.addRow("Stage II weights:", self.stage2Path)
        info = qt.QLabel(f'Weights are not included. They are released on request after approval by the '
                         f'corresponding authors: <a href="{WEIGHTS_INFO_URL}">how to request access</a>.')
        info.wordWrap = True
        info.setOpenExternalLinks(True)
        form.addRow(info)

        # ---- options
        box = ctk.ctkCollapsibleButton()
        box.text = "Options"
        box.collapsed = True
        self.layout.addWidget(box)
        form = qt.QFormLayout(box)

        self.regionCombo = qt.QComboBox()
        self.regionCombo.addItem("Orbital region (fast)", "roi")
        self.regionCombo.addItem("Whole volume (as in the paper)", "full")
        self.regionCombo.setToolTip("Orbital region: Stage II runs only around the orbits found by Stage I "
                                    "(several times faster, recommended without GPU).")
        form.addRow("Region:", self.regionCombo)

        self.marginSpin = qt.QDoubleSpinBox()
        self.marginSpin.setRange(10.0, 100.0)
        self.marginSpin.setSingleStep(5.0)
        self.marginSpin.setDecimals(0)
        self.marginSpin.setSuffix(" mm")
        self.marginSpin.value = 40.0
        self.marginSpin.setToolTip("Margin around the detected orbital content for the orbital-region mode.")
        form.addRow("Region margin:", self.marginSpin)

        self.deviceCombo = qt.QComboBox()
        self.deviceCombo.addItem("Auto (GPU if available)", "auto")
        self.deviceCombo.addItem("GPU (CUDA)", "cuda")
        self.deviceCombo.addItem("CPU", "cpu")
        form.addRow("Device:", self.deviceCombo)

        self.overlapSpin = qt.QDoubleSpinBox()
        self.overlapSpin.setRange(0.0, 0.75)
        self.overlapSpin.setSingleStep(0.25)
        self.overlapSpin.setDecimals(2)
        self.overlapSpin.value = 0.5
        self.overlapSpin.setToolTip("Sliding-window overlap (0.5 as in the paper; 0.25 is about 2x faster).")
        form.addRow("Window overlap:", self.overlapSpin)

        self.orbitCheck = qt.QCheckBox("Also create the orbital content segment")
        self.orbitCheck.checked = True
        form.addRow(self.orbitCheck)

        # ---- run
        self.applyButton = qt.QPushButton("Apply")
        self.applyButton.toolTip = "Run EPC-Net segmentation."
        self.layout.addWidget(self.applyButton)
        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.visible = False
        self.layout.addWidget(self.progressBar)
        self.statusLabel = qt.QLabel("")
        self.statusLabel.wordWrap = True
        self.layout.addWidget(self.statusLabel)

        # ---- export
        self.exportButton = qt.QPushButton("Export orbital bone surface (STL)...")
        self.exportButton.toolTip = ("Surface of the current 'Orbital bone' segment (including manual edits), "
                                     "reconstructed as in the paper (Sec. 3.9), LPS coordinates.")
        self.layout.addWidget(self.exportButton)

        disclaimer = qt.QLabel(DISCLAIMER)
        disclaimer.wordWrap = True
        disclaimer.setStyleSheet("color: #a05a00;")
        self.layout.addWidget(disclaimer)
        self.layout.addStretch(1)

        self._loadSettings()
        self.applyButton.connect("clicked(bool)", self.onApply)
        self.exportButton.connect("clicked(bool)", self.onExport)
        self.regionCombo.connect("currentIndexChanged(int)", self.updateGUI)
        self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateGUI)
        self.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateGUI)
        self.stage1Path.connect("currentPathChanged(QString)", self.updateGUI)
        self.stage2Path.connect("currentPathChanged(QString)", self.updateGUI)
        self.updateGUI()

    def _pathEdit(self, key, tooltip):
        edit = ctk.ctkPathLineEdit()
        edit.filters = ctk.ctkPathLineEdit.Files
        edit.nameFilters = ["PyTorch weights (*.pth *.pt)", "All files (*)"]
        edit.settingKey = SETTINGS_PREFIX + key + "History"
        edit.setToolTip(tooltip)
        return edit

    # ---- settings
    def _loadSettings(self):
        s = qt.QSettings()
        self.stage1Path.currentPath = s.value(SETTINGS_PREFIX + "Stage1Weights", "")
        self.stage2Path.currentPath = s.value(SETTINGS_PREFIX + "Stage2Weights", "")
        for combo, key in ((self.regionCombo, "Region"), (self.deviceCombo, "Device")):
            idx = combo.findData(s.value(SETTINGS_PREFIX + key, ""))
            if idx >= 0:
                combo.currentIndex = idx

    def _saveSettings(self):
        s = qt.QSettings()
        s.setValue(SETTINGS_PREFIX + "Stage1Weights", self.stage1Path.currentPath)
        s.setValue(SETTINGS_PREFIX + "Stage2Weights", self.stage2Path.currentPath)
        s.setValue(SETTINGS_PREFIX + "Region", self.regionCombo.currentData)
        s.setValue(SETTINGS_PREFIX + "Device", self.deviceCombo.currentData)

    def updateGUI(self, *args):
        ready = (self.inputSelector.currentNode() is not None and os.path.isfile(self.stage1Path.currentPath)
                 and os.path.isfile(self.stage2Path.currentPath))
        self.applyButton.enabled = ready and not self._busy
        self.applyButton.toolTip = "Run EPC-Net segmentation." if ready else "Select the input volume and both weight files."
        self.marginSpin.enabled = self.regionCombo.currentData == "roi"
        self.exportButton.enabled = self.outputSelector.currentNode() is not None and not self._busy

    def onProgress(self, fraction, message):
        self.progressBar.value = int(round(100 * fraction))
        self.statusLabel.text = message

    # ---- actions
    def onApply(self):
        self._saveSettings()
        self._busy = True
        self.updateGUI()
        self.progressBar.visible = True
        try:
            with slicer.util.tryWithErrorDisplay("EPC-Net segmentation failed.", waitCursor=True):
                self.onProgress(0.0, "Checking Python packages...")
                self.logic.setupPythonRequirements()
                segNode = self.outputSelector.currentNode()
                if segNode is None:
                    segNode = slicer.mrmlScene.AddNewNodeByClass(
                        "vtkMRMLSegmentationNode", self.inputSelector.currentNode().GetName() + "_EPCNet")
                    self.outputSelector.setCurrentNode(segNode)
                result = self.logic.process(
                    self.inputSelector.currentNode(), segNode,
                    self.stage1Path.currentPath, self.stage2Path.currentPath,
                    device=self.deviceCombo.currentData,
                    roiMarginMm=self.marginSpin.value if self.regionCombo.currentData == "roi" else None,
                    overlap=self.overlapSpin.value,
                    includeOrbit=self.orbitCheck.checked,
                    progressCallback=self.onProgress,
                )
                t = result["timings"]
                warn = "" if result["orbit_found"] else "  WARNING: no orbital content found - check the input."
                self.onProgress(1.0, f"Done in {t['total']:.0f} s on {self.logic.deviceName}. "
                                     f"Please review the result.{warn}")
        finally:
            self._busy = False
            self.updateGUI()

    def onExport(self):
        segNode = self.outputSelector.currentNode()
        volume = self.inputSelector.currentNode()
        default = os.path.join(qt.QStandardPaths.writableLocation(qt.QStandardPaths.DocumentsLocation),
                               (volume.GetName() if volume else "orbital") + "_orbital_bone.stl")
        path = qt.QFileDialog.getSaveFileName(slicer.util.mainWindow(), "Export orbital bone surface", default,
                                              "STL files (*.stl)")
        if not path:
            return
        with slicer.util.tryWithErrorDisplay("STL export failed.", waitCursor=True):
            n = self.logic.exportBoneStl(segNode, volume, path)
            self.statusLabel.text = f"Saved {n} triangles to {path}"


# -----------------------------------------------------------------------------------------------
class EPCNetLogic(ScriptedLoadableModuleLogic):
    BONE_ID, BONE_NAME, BONE_COLOR = "orbital_bone", "Orbital bone", (0.95, 0.88, 0.70)
    ORBIT_ID, ORBIT_NAME, ORBIT_COLOR = "orbital_content", "Orbital content", (0.85, 0.35, 0.30)

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self._predictor = None
        self._predictorKey = None
        self.deviceName = ""

    # ---- dependencies
    @staticmethod
    def _useLocalSource() -> bool:
        """Developer mode: module used directly from an EPC-Net checkout -> import epcnet from the repository."""
        root = Path(__file__).resolve().parents[3]
        if (root / "epcnet" / "__init__.py").is_file():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return True
        return False

    def setupPythonRequirements(self):
        import importlib.metadata
        from packaging import version

        def installed(dist, minimum=None):
            try:
                v = importlib.metadata.version(dist)
            except importlib.metadata.PackageNotFoundError:
                return False
            return minimum is None or version.parse(v) >= version.parse(minimum)

        try:
            import PyTorchUtils
        except ModuleNotFoundError:
            raise InstallError("EPC-Net requires the 'PyTorch' extension. Install it in the Extensions Manager "
                               "and restart Slicer.")
        torchLogic = PyTorchUtils.PyTorchUtilsLogic()
        localSource = self._useLocalSource()

        todo = []
        if not torchLogic.torchInstalled():
            todo.append("PyTorch")
        if not installed("monai", "1.3"):
            todo.append("MONAI")
        if not installed("pyyaml"):
            todo.append("PyYAML")
        if not localSource and not installed("epcnet", EPCNET_MIN_VERSION):
            todo.append("epcnet")
        if todo:
            if not slicer.util.confirmOkCancelDisplay(
                    "EPC-Net needs to install additional Python packages. This requires a network connection "
                    "and may take several minutes (PyTorch with GPU support is ~2-3 GB). Click OK to proceed.",
                    "Install Python packages", detailedText="Packages: " + ", ".join(todo)):
                raise InstallError("Installation cancelled by the user.")

        if "PyTorch" in todo:
            slicer.util.showStatusMessage("Installing PyTorch (may take several minutes)...")
            torch = torchLogic.installTorch(askConfirmation=False, torchVersionRequirement=f">={MINIMUM_TORCH_VERSION}")
            if torch is None:
                raise InstallError("PyTorch installation failed. Use the 'PyTorch Util' module to install it.")
        else:
            if version.parse(torchLogic.torch.__version__.split("+")[0]) < version.parse(MINIMUM_TORCH_VERSION):
                raise InstallError(f"PyTorch {torchLogic.torch.__version__} is too old; >= {MINIMUM_TORCH_VERSION} "
                                   "is required (use the 'PyTorch Util' module to update it).")
        # --no-deps: never let pip replace Slicer's PyTorch / numpy
        if "MONAI" in todo:
            slicer.util.pip_install(["monai>=1.3,<2", "--no-deps"])
        if "PyYAML" in todo:
            slicer.util.pip_install("pyyaml")
        if "epcnet" in todo:
            slicer.util.pip_install([EPCNET_PACKAGE_URL, "--no-deps"])
        slicer.util.showStatusMessage("")

    # ---- inference
    def getPredictor(self, stage1Path, stage2Path, device="auto"):
        from epcnet.inference import EPCNetPredictor, resolve_device

        dev = resolve_device(device)
        if dev.type == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU requested but not available - select 'Auto' or 'CPU'.")
        key = (stage1Path, os.path.getmtime(stage1Path), stage2Path, os.path.getmtime(stage2Path), str(dev))
        if self._predictor is None or key != self._predictorKey:
            self._predictor = None  # release the previous models first
            self._predictor = EPCNetPredictor(stage1Path, stage2Path, device=dev)
            self._predictorKey = key
        if dev.type == "cuda":
            import torch
            self.deviceName = torch.cuda.get_device_name(dev)
        else:
            self.deviceName = "CPU"
        return self._predictor

    def process(self, volumeNode, segmentationNode, stage1Path, stage2Path, device="auto", roiMarginMm=40.0,
                overlap=0.5, includeOrbit=True, progressCallback=None):
        """Run EPC-Net on ``volumeNode`` and write the result into ``segmentationNode``. Returns the result dict."""
        from epcnet.inference import metatensor_from_ijk

        if volumeNode is None or segmentationNode is None:
            raise ValueError("Input volume and output segmentation are required.")
        report = progressCallback or (lambda f, m: None)
        report(0.01, "Loading models...")
        predictor = self.getPredictor(stage1Path, stage2Path, device)
        predictor.overlap = float(overlap)
        predictor.roi_margin_mm = None if roiMarginMm is None else float(roiMarginMm)

        # volume -> model input (array in i, j, k order + IJK->RAS matrix); copied on the main thread
        ijkToRas = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRas)
        image = metatensor_from_ijk(slicer.util.arrayFromVolume(volumeNode).transpose(2, 1, 0),
                                    slicer.util.arrayFromVTKMatrix(ijkToRas))

        # heavy computation in a worker thread so that the application stays responsive
        state = {"fraction": 0.0, "message": "Starting", "result": None, "error": None, "traceback": ""}

        def work():
            try:
                state["result"] = predictor.predict(image, progress=lambda f, m: state.update(fraction=f, message=m))
            except Exception as exc:  # noqa: BLE001
                import traceback
                state["error"], state["traceback"] = exc, traceback.format_exc()

        start = time.time()
        worker = threading.Thread(target=work, name="EPCNetInference", daemon=True)
        worker.start()
        while worker.is_alive():
            report(state["fraction"], f"{state['message']}... ({time.time() - start:.0f} s)")
            slicer.app.processEvents()
            time.sleep(0.1)
        if state["error"] is not None:
            logging.error(state["traceback"])
            raise state["error"]
        result = state["result"]
        report(0.95, "Creating segments...")
        self.writeSegmentation(volumeNode, segmentationNode, result, includeOrbit)
        return result

    def writeSegmentation(self, volumeNode, segmentationNode, result, includeOrbit=True):
        import numpy as np

        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        segmentationNode.SetAndObserveTransformNodeID(volumeNode.GetTransformNodeID())
        segmentation = segmentationNode.GetSegmentation()
        specs = [(self.BONE_ID, self.BONE_NAME, self.BONE_COLOR, result["bone"])]
        if includeOrbit:
            specs.append((self.ORBIT_ID, self.ORBIT_NAME, self.ORBIT_COLOR, result["orbit"]))
        for segmentId, name, color, mask in specs:
            if segmentation.GetSegment(segmentId):
                segmentation.RemoveSegment(segmentId)
            labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "__EPCNet_tmp")
            try:
                labelmap.CopyOrientation(volumeNode)
                labelmap.SetAndObserveTransformNodeID(volumeNode.GetTransformNodeID())
                slicer.util.updateVolumeFromArray(labelmap, np.ascontiguousarray(
                    np.asarray(mask, dtype=np.uint8).transpose(2, 1, 0)))  # (i,j,k) -> (k,j,i)
                segmentation.AddEmptySegment(segmentId, name, color)
                ids = vtk.vtkStringArray()
                ids.InsertNextValue(segmentId)
                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmap, segmentationNode, ids)
            finally:
                slicer.mrmlScene.RemoveNode(labelmap)
        segmentationNode.CreateDefaultDisplayNodes()
        segmentationNode.CreateClosedSurfaceRepresentation()

    def exportBoneStl(self, segmentationNode, volumeNode, path):
        """Export the current orbital-bone segment (including manual edits) as STL (Sec. 3.9 of the paper)."""
        import numpy as np

        if segmentationNode is None or volumeNode is None:
            raise ValueError("Select the input volume and the segmentation first.")
        segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(self.BONE_NAME) or self.BONE_ID
        if not segmentationNode.GetSegmentation().GetSegment(segmentId):
            raise ValueError(f"The segmentation has no '{self.BONE_NAME}' segment.")
        try:
            import skimage  # noqa: F401
        except ModuleNotFoundError:
            slicer.util.pip_install("scikit-image")
        if not self._useLocalSource():
            import epcnet  # noqa: F401  (installed by setupPythonRequirements)
        from epcnet.config import load_config
        from epcnet.surface import mask_to_stl

        mask = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId, volumeNode)  # (k, j, i)
        ijkToRas = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRas)
        affine = slicer.util.arrayFromVTKMatrix(ijkToRas)
        if volumeNode.GetParentTransformNode():  # apply a linear parent transform, if any
            if not volumeNode.GetParentTransformNode().IsLinear():
                raise ValueError("Harden the non-linear transform of the volume before exporting.")
            toWorld = vtk.vtkMatrix4x4()
            volumeNode.GetParentTransformNode().GetMatrixTransformToWorld(toWorld)
            affine = slicer.util.arrayFromVTKMatrix(toWorld) @ affine
        if not path.lower().endswith(".stl"):
            path += ".stl"
        return mask_to_stl(np.asarray(mask).transpose(2, 1, 0) > 0, affine, path, load_config()["surface"])


# -----------------------------------------------------------------------------------------------
class EPCNetTest(ScriptedLoadableModuleTest):
    """Model weights are not public, so the automatic test only checks that the module and its logic load
    and that segments are written correctly from a synthetic result."""

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_WriteSegmentation()

    def test_WriteSegmentation(self):
        import numpy as np

        self.delayDisplay("Starting the test")
        volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "TestCT")
        arr = np.zeros((20, 30, 40), dtype=np.int16)  # (k, j, i)
        volume.SetSpacing(0.5, 0.5, 1.0)
        slicer.util.updateVolumeFromArray(volume, arr)
        bone = np.zeros((40, 30, 20), np.uint8)  # (i, j, k)
        bone[10:20, 5:15, 5:10] = 1
        orbit = np.zeros_like(bone)
        orbit[25:30, 10:20, 8:12] = 1
        seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        EPCNetLogic().writeSegmentation(volume, seg, {"bone": bone, "orbit": orbit})
        out = slicer.util.arrayFromSegmentBinaryLabelmap(seg, EPCNetLogic.BONE_ID, volume)
        self.assertEqual(int(out.sum()), int(bone.sum()))
        self.assertTrue(np.array_equal(out.transpose(2, 1, 0) > 0, bone > 0))
        self.delayDisplay("Test passed")
