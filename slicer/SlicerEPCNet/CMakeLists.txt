cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(EPCNet)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/GaryGaoYao/EPC-Net/tree/main/slicer")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Yao Gao (KU Leuven)")
set(EXTENSION_DESCRIPTION "Automatic, anatomy-guided segmentation of the orbital bone (and orbital content) in head CT/CBCT with EPC-Net, to support patient-specific orbital implant design. Model weights are not included and must be obtained separately.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/GaryGaoYao/EPC-Net/main/slicer/SlicerEPCNet/EPCNet/Resources/Icons/EPCNet.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/GaryGaoYao/EPC-Net/main/slicer/Screenshot01.png")
set(EXTENSION_DEPENDS "PyTorch")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(EPCNet)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
