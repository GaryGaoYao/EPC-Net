# Full EPC-Net training pipeline on Windows (PowerShell).
#
#   Paper setting (Table 2):
#     .\scripts\train_epcnet.ps1
#   Optional 0.5 mm isotropic Stage II, re-using the Stage-I model trained above:
#     .\scripts\train_epcnet.ps1 -Config configs/local_dataset777.yaml,configs/highres_isotropic.yaml `
#         -Stage1Checkpoint D:\Codes\ECP-Net-BSPC-Github\epcnet_work\runs\epcnet_default\stage1\best_model.pt
param(
    [string[]]$Config = @("configs/local_dataset777.yaml"),
    [string]$Python = "D:\Codes\Orbital-Segmentation-nnUNet\.venv\Scripts\python.exe",
    [string]$Stage1Checkpoint = "",   # skip Stage-I training and use this model for the priors
    [int]$NVal = 5,
    [int]$NTest = 5
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$cfgArgs = @(); foreach ($c in $Config) { $cfgArgs += @("-c", $c) }
$s1Args = @(); if ($Stage1Checkpoint) { $s1Args = @("--stage1", $Stage1Checkpoint) }

function Step($title, [string[]]$cmd) {
    Write-Host "`n==== $title ====" -ForegroundColor Cyan
    & $Python -m epcnet @cmd @cfgArgs
    if ($LASTEXITCODE -ne 0) { throw "$title failed (exit $LASTEXITCODE)" }
}

# 1) resample nnU-Net raw data into the EPC-Net cache
Step "1/7 preprocess" @("preprocess")

# 2) patient-level split (created once; delete the json to re-split)
$pyCfg = ($Config | ForEach-Object { "r'$_'" }) -join ","
$split = (& $Python -c "from epcnet.config import load_config; print(load_config([$pyCfg])['data']['split_file'])").Trim()
if (-not (Test-Path $split)) { Step "2/7 split" @("split", "--n-val", "$NVal", "--n-test", "$NTest") }
else { Write-Host "`n==== 2/7 split: using existing $split ====" -ForegroundColor Cyan }

# 3) Stage I - orbital content (anatomical anchor)
if ($Stage1Checkpoint) { Write-Host "`n==== 3/7 stage 1: using $Stage1Checkpoint ====" -ForegroundColor Cyan }
else { Step "3/7 train stage 1" @("train", "--stage", "1") }

# 4) Stage-I predictions -> distance priors on the Stage-II grid
Step "4/7 priors" (@("priors") + $s1Args)

# 5) Stage II - prior-guided orbital bone
Step "5/7 train stage 2" @("train", "--stage", "2")

# 6) end-to-end evaluation on the held-out test cases (global + PO-ROI, native CT resolution)
Step "6/7 evaluate" (@("evaluate", "--split", "test", "--save-predictions") + $s1Args)

# 7) weights-only release package
Step "7/7 export" (@("export") + $s1Args)

Write-Host "`nDone." -ForegroundColor Green
