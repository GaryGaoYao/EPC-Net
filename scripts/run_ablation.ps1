# 2x2 ablation of Stage II (Sec. 3.17, Table 7): Baseline / +Loss / +Feat / full EPC-Net.
# Requires steps 1-4 of train_epcnet.ps1 (cache, split, Stage I, priors).  The full model is
# the "epcnet_default" experiment; the three reduced variants are trained here.
param(
    [string[]]$Config = @("configs/local_dataset777.yaml"),
    [string]$Python = "D:\Codes\Orbital-Segmentation-nnUNet\.venv\Scripts\python.exe",
    [string]$Stage1Experiment = "epcnet_default"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$cfgArgs = @(); foreach ($c in $Config) { $cfgArgs += @("-c", $c) }
$pyCfg = ($Config | ForEach-Object { "r'$_'" }) -join ","
$runs = (& $Python -c "from epcnet.config import load_config; print(load_config([$pyCfg])['runs_dir'])").Trim()
$stage1 = Join-Path $runs "$Stage1Experiment/stage1/best_model.pt"

$variants = @(
    @{ name = "ablation_baseline"; g = "false"; w = "false" },
    @{ name = "ablation_loss_only"; g = "false"; w = "true" },
    @{ name = "ablation_feat_only"; g = "true"; w = "false" }
)
foreach ($v in $variants) {
    $o = @("-o", "experiment=$($v.name)", "-o", "stage2.use_guidance=$($v.g)", "-o", "stage2.use_weighted_loss=$($v.w)")
    Write-Host "`n==== $($v.name): train ====" -ForegroundColor Cyan
    & $Python -m epcnet train --stage 2 @cfgArgs @o
    if ($LASTEXITCODE -ne 0) { throw "training $($v.name) failed" }
    Write-Host "==== $($v.name): evaluate ====" -ForegroundColor Cyan
    & $Python -m epcnet evaluate --split test --stage1 $stage1 @cfgArgs @o
    if ($LASTEXITCODE -ne 0) { throw "evaluation $($v.name) failed" }
}
