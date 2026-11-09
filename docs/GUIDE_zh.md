# EPC-Net 训练与发布指南（中文）

## 1. 代码与论文的对应关系

| 论文 | 代码 |
|---|---|
| 3.6 SegResNet 主干（GroupNorm、dropout 0.2、32 filters） | `epcnet/model.py` |
| 3.7 Stage-I 眶内容物分割（1 mm，128³，0.5/99.5 百分位归一化） | `stage1` 配置 + `epcnet/trainer.py` |
| 3.8.1 先验编码：膨胀 δ → EDT → G = exp(−d/σ)，W = 1+α·exp(−d²/2σ²) | `epcnet/priors.py` |
| 3.8.2 加权 Dice + 加权 CE（Eq. 3–7） | `epcnet/losses.py` |
| Algorithm 1：CT+G 双通道输入、眶周带（0 ≤ d ≤ d_max）优先采样 | `epcnet/data.py: PatchDataset` |
| 3.5 数据增强（翻转 p=0.5，旋转 ±10° p=0.3，等比缩放 0.9–1.1 p=0.3，强度平移/对比度/高斯噪声） | `epcnet/data.py: _build_augment` |
| 3.10 AdamW、余弦退火、早停 50、按验证 DSC 选最优（HD95 打破平局） | `epcnet/trainer.py` |
| 3.9 表面重建（高斯 σ=1 voxel，MC 0.5，Laplacian 20 次 / 0.01） | `epcnet/surface.py` |
| 3.11 Global / PO-ROI（1.1–1.9×）DSC、HD95、ASD；3.12 质心误差 | `epcnet/metrics.py`、`epcnet/pipeline.py: evaluate` |
| 3.17 2×2 消融 | `scripts/run_ablation.ps1`（`stage2.use_guidance` / `stage2.use_weighted_loss`） |
| 3.15 OAT 敏感性分析 | `-o prior.sigma=3.5 -o experiment=sigma3.5` 等 |

## 2. 直接用 nnU-Net 的数据

Stage I 和 Stage II 的标签都从 `Dataset777_OrbitBoneEye/labelsTr` 里取：`eyes(2)` 用于 Stage I，`bone(1)` 用于 Stage II。原数据不需要做任何改动。
`configs/local_dataset777.yaml` 里已经填好了本机路径。

缓存目录 `epcnet_work/cache` 每个病例约占 0.25 GB，43 例一共约 10 GB。

## 3. 训练（PowerShell）

```powershell
cd D:\Codes\ECP-Net-BSPC-Github\EPC-Net
D:\Codes\Orbital-Segmentation-nnUNet\.venv\Scripts\python.exe tests\smoke_test.py   # 先跑合成数据自检（CPU 约 3 分钟）
.\scripts\train_epcnet.ps1                                                       # 预处理→划分→Stage I→先验→Stage II→评估→导出
```

如果 PowerShell 禁止运行脚本，先执行 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。

也可以逐步执行：

```powershell
$py = "D:\Codes\Orbital-Segmentation-nnUNet\.venv\Scripts\python.exe"
$cfg = "configs/local_dataset777.yaml"
& $py -m epcnet preprocess -c $cfg
& $py -m epcnet split -c $cfg --n-val 5 --n-test 5     # 43 例 → 33 / 5 / 5
& $py -m epcnet train -c $cfg --stage 1
& $py -m epcnet priors -c $cfg
& $py -m epcnet train -c $cfg --stage 2
& $py -m epcnet evaluate -c $cfg --split test --save-predictions
& $py -m epcnet export -c $cfg
```

* 训练中断后，加 `--resume` 即可继续。
* 想和 nnU-Net 做公平对比：把 `epcnet_work/splits/splits_final_for_nnunet.json` 覆盖到 `nnUNet_preprocessed/Dataset777_OrbitBoneEye/splits_final.json`（先备份原文件），然后运行 `nnUNetv2_train 777 3d_fullres 0`。
* 如果换到小于 24 GB 显存的卡：加 `-o train.sw_batch_size=2 -o train.val_device=cpu`。

## 3b. GPU 利用（RTX A6000 48 GB，与论文同卡）

* **数据增强在 GPU 上做**：DataLoader 只从缓存里切比 patch 略大的块（约 0.03 s/块），翻转、旋转、缩放、最终裁剪和强度增强都在 GPU 上一次 `grid_sample` 完成。旋转和缩放在物理空间（mm）中进行，在 0.5×0.5×1.0 这种各向异性网格上也是刚性的。CPU 不再拖慢 GPU。
* **加速选项**：AMP（fp16）、cudnn benchmark、TF32、pin_memory + 预取。验证时滑窗一次推 8 个窗口（`train.sw_batch_size: 8`）。
* **每个 epoch 的定义**：250 次迭代 × batch 2 = 500 个新采样的 patch，和 nnU-Net 基线每个 epoch 的采样量相同。论文只说“每个 epoch 重新随机采样 patch”，没有给出数量，这个值是按论文 Table 9 的训练时长（8.2 ± 0.6 h）估算的。
* **batch size 保持论文的 2**：48 GB 显存放得下更大的 batch，但论文 Table 2 写的是 2，这里不改。

## 3c. 分辨率与层厚

* **网络网格（默认，论文 Table 2）**：Stage I 用 1.0 mm，Stage II 用 0.5×0.5×1.0 mm。
* **最终输出是原始 CT 分辨率**：Stage II 输出的是概率图，先线性插值回原始网格，再按 0.5 阈值二值化。比如 0.29 mm 层内、0.3 mm 层厚的扫描，得到的 mask 就是 0.29×0.29×0.3 mm。STL 也是从这个原始分辨率的 mask 生成的（论文 3.9 节）。
* **训练标签**：重采样时，每个类别先做线性插值，再按 0.5 阈值二值化（nnU-Net 的做法），不用最近邻。这样 1–2 个体素厚的眶底和内侧壁在网格上更连续。
* **可选的高分辨率方案**：`configs/highres_isotropic.yaml`，把 Stage II 改成 0.5 mm 各向同性，patch 160×160×128，大约用 30 GB 显存。
  * 原因：Dataset777 的 43 例里，38 例层厚小于 1.0 mm，19 例 ≤ 0.5 mm。论文的 1.0 mm 层厚会丢掉 z 方向的信息。
  * 注意：这**不是论文 Table 2 的设置**，属于扩展实验，发表时要说明。其余设置都不变。
  * Stage I 和论文默认一样，可以直接复用：

```powershell
.\scripts\train_epcnet.ps1 -Config configs/local_dataset777.yaml,configs/highres_isotropic.yaml `
    -Stage1Checkpoint D:\Codes\ECP-Net-BSPC-Github\epcnet_work\runs\epcnet_default\stage1\best_model.pt
```

  训练完后，用 `evaluate` 和论文默认设置在同样的 5 个测试病例上比较。推理和 GUI 会自动从权重里读取 0.5 mm 的网格设置。

## 4. 发布

1. `export` 会生成 `release/EPC-Net-weights-v1.0.0.zip`，里面包含：
   * `weights/stage1_eye.pth` 和 `weights/stage2_bone.pth`：只有权重和元数据，已经去掉病例名和优化器状态；
   * `epcnet_config.yaml`；
   * `manifest.json`（含 SHA256）。

   这两个文件名和现有 GUI 的自动识别规则一致。

   **权重的对外说明**（完整内容见 `weights/README.md`）：
   * **v1.0.0 公开数据集权重**：只用公开数据训练。用户需要填写并签署申请表，发邮件给通讯作者 Van Dessel、Willaert、Sun（抄送 yao.gao@kuleuven.be），作者审批同意后才发下载链接。
   * **论文原始权重**：用 UZ Leuven 临床数据训练（伦理批号 S68944），能否共享以医院和伦理委员会的决定为准，不走上面的申请流程。
   * 导出的权重包里自带 `ACCESS_TERMS.md`，`manifest.json` 中也记录了训练数据类型和获取条件。
2. 代码部分：把本目录推送到 GitHub `GaryGaoYao/EPC-Net`，打 tag `v1.0.0` 并发布 GitHub Release。`.gitignore` 已经排除数据、缓存、权重和 MS.pdf。
3. 推理命令：`python -m epcnet predict -i CT.nii.gz -d out --stl`，也支持 DICOM 文件夹。

   `gui/` 里的 GUI 现在直接调用同一套 `epcnet` 推理代码：网格和 patch 设置从权重里读，结果输出到原始分辨率，STL 按论文 3.9 节生成，和 `evaluate` 评估的流程完全一致。

## 5. 与论文的已知差异 / 需要你确认的地方

* **病例数**：论文是 50 例（40/5/5），本地 nnU-Net 数据集是 43 例，默认划分为 33/5/5。
* **训练病例的先验来源**：论文 Algorithm 1 用的是 Stage-I 的预测 mask。这里默认用训练好的 Stage-I 模型对所有病例做预测（`prior.source: pred`）；也可以改成 `gt`，直接用真值 mask。
* **论文没有给出、这里自定的参数**：
  * 强度增强概率 0.3；
  * 眶周采样比例 `focus_prob` = 0.9；
  * 余弦退火最小学习率 1e-6；
  * 每个 epoch 250 次迭代；
  * 验证滑窗重叠 0.25，推理滑窗重叠 0.5。
* **PO-ROI**：论文写的是“把真值眶内容物按 1.1–1.9 倍扩展”。这里的实现是把每个眼眶连通域绕自身质心等比放大。

## 6. 3D Slicer 扩展（`slicer/SlicerEPCNet`）

**用户看到的界面**：*Segmentation → EPC-Net Orbital Bone*
1. 选择输入 CT 体数据。
2. 选择两个权重文件。权重需要用户自己申请，路径会被记住。
3. 点 **Apply**。
4. 生成“眶骨 / 眶内容物”两个分割（segment），可以在 Segment Editor 里人工修改。
5. 点 **Export STL**。导出的是当前的眶骨分割，包括手工修改的部分，表面按论文 3.9 节重建。

**选项**：
- **区域**：
  - *Orbital region (fast)*：默认选项，Stage II 只算眼眶周围。在这个区域内结果和全图模式完全一致，速度快好几倍。
  - *Whole volume*：全图计算，和论文设置一致。
- **设备**：Auto / GPU / CPU。
- **滑窗重叠**：0.5 和论文一致，0.25 大约快 2 倍。

**自己先试用（不用编译）**：
1. 在 Slicer 菜单 *Edit → Application Settings → Modules → Additional module paths* 里，添加 `D:\Codes\ECP-Net-BSPC-Github\EPC-Net\slicer\SlicerEPCNet\EPCNet`，然后重启 Slicer。
2. 在 Extensions Manager 里先装好 **PyTorch** 扩展。
3. 第一次点 Apply 时，模块会提示安装 PyTorch 和 MONAI。这种开发模式下，epcnet 直接用仓库里的代码。

**公开发布的清单**：
1. 仓库根目录加 `LICENSE.txt`。官方推荐 MIT 或 Apache-2.0，具体用哪个由你决定。
2. 在 Slicer 里截一张效果图，保存为 `slicer/Screenshot01.png`。
3. 推送到 GitHub，打 tag `v1.0.0`。模块从这个 tag 安装 epcnet，地址在 `EPCNet.py` 的 `EPCNET_PACKAGE_URL`。
4. 在 GitHub 仓库的 Topics 里加上 `3d-slicer-extension`。
5. 把 `slicer/SlicerEPCNet/EPCNet.json` 作为 Pull Request 提交到 https://github.com/Slicer/ExtensionsIndex，并按 PR 模板勾选 Tier 1 的清单。
