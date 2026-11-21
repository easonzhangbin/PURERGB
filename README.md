# PURE RGB

**上游项目：[marcinz606/NegPy](https://github.com/marcinz606/NegPy)。** PURE RGB 是基于 NegPy 的派生开发版本，感谢原作者 marcinz606 及 NegPy 贡献者；保留原项目署名与 GPL-3.0 许可。当前采用的 NegPy 基线为 0.68.0（[`d8123a9`](https://github.com/marcinz606/NegPy/tree/d8123a9b2a18fdaab5941347a91e402fa936d3dc)）。

**PURE RGB 0.2.0** 是一款 macOS 胶片数字化工作台。它将同一帧底片在红、绿、蓝光下拍摄的三张 RAW 合成为正像，支持分区密度测量、调色、除尘审核和独立观片，最终输出带 ICC 的 16 位 TIFF。

[下载 macOS 版](https://github.com/easonzhangbin/PURERGB/releases/tag/v0.2.0) · [中文使用指南](docs/OPERATING_MANUAL.zh-CN.md) · [0.2.0 发行说明](docs/releases/0.2.0.md) · [捐赠 PURE Studio 项目](SUPPORT.md)

![PURE RGB 原生观片台，展示建筑胶片样片与当前屏幕的观片提示](docs/images/light-table.png)

## 0.2.0 新功能与修复

- **观片台从第五步打开。** 检查当前照片、挑选整卷，或直接打开 TIFF、JPEG、PNG、HEIF。单张模式可点亮灯板、拖动放大镜，整卷选片同步至导出列表。
- **按显示器打开观片台。** 从内建 XDR 屏打开会进入全屏，Esc 可退出。外接 SDR 屏默认使用窗口，跨屏移动时会更新 HDR 提示。完整 HDR 观片体验需要 XDR、合适的显示模式及 macOS 26 或更新版本。
- **工作台放大镜和滑块。** 镜片默认直径 288 点、倍率 3×，蓝点调大小，绿点或滚轮调倍率，最高 8×。参数轨道可以点选、拖动或用方向键微调。选中步骤和主要操作用明亮橙红色标示。
- **启动与除尘。** 修正旧插件缓存导致的启动闪退，以及选择新建任务后窗口未显示的问题。确认除尘后可继续调色，保存与导出会在新成片上应用已审核的修补蒙版。
- **帮助与支持。** 直接从 macOS 的 PURE RGB 菜单打开「关于 PURE RGB」「使用指南」「捐赠 PURE Studio 项目」或「检查更新」。0.1.1 用户可通过原有的启动更新提示或设置中的手动检查取得 0.2.0。[完整发行说明](docs/releases/0.2.0.md)

## 六步工作流

| 页面 | 用途 |
| --- | --- |
| 1 导入归组 | 选择输入方式，核对 RGB 通道，导入时自动定位片框。 |
| 2 画幅与测量 | 默认取有效画面中心约 80% 面积测量密度，可手绘选区、排除或恢复网格中的区域。 |
| 3 转换调色 | 比较引擎，调整正像与高光。局部高光作用区支持亮度筛选与羽化。 |
| 4 修整审核 | 用独立安装的 PURE Neg 后端检测与修复尘点，确认后返回工作台。 |
| 5 观片台 | 在原生灯板上检查细节，挑选要导出的照片。 |
| 6 导出 | 生成当前或勾选照片的全尺寸 RGB16 TIFF，并嵌入实际使用的 ICC。 |

## 色彩引擎与输入

| 引擎 | 实现范围 |
| --- | --- |
| NegPy · 原生 | 使用 NegPy 的密度分析、反相和数码印相处理。 |
| Nikon CMS · 仅 CMS | 将中性 RGB16 正像送入 CML4 研究链，可选固定曲线。需要本机自备且有权使用的匹配研究资源。 |
| FlexColor · 演算 | 默认以共享密度窗口与公开重建的曲线算法做中性演算。经过验证的本机资源配方可启用原曲线与六色校正。 |

三种引擎的彩色工作流使用同一底片的三次分光拍摄，输入可选择「拜耳 RGB · 三张 RAW」或「单色 RGB · 三张 RAW」，并核对 R、G、B 通道顺序。单色流程直接读取传感器样本，使用文件记录的黑白位归一化。单张彩色翻拍的 Bayer RAW 使用不同的采集域，本版没有对应转换入口。单张黑白负片可走 NegPy 原生流程。Nikon 与 FlexColor 路线用于比较逆向研究所得的处理阶段，色彩输出以相机采集与当前的工作域假设为基础。

## 安装与启动

**Apple 芯片 Mac 下载，要求 macOS 14 或更新版本**。打开 PKG 后按 macOS 安装助手指引安装到「应用程序」。也可用访达解压 ZIP，双击同目录的「打开 PURE RGB.command」校验应用，然后选择在访达中安装或直接打开。两种方式都包含运行环境。PKG 没有 Developer ID Installer 签名，应用使用本机临时签名，尚未通过 Apple 公证。若首次打开时 macOS 提示无法验证开发者，先核对发布页上的 SHA-256。如确定来自本项目，可在「系统设置 → 隐私与安全性」中选择「仍要打开」，按当时的系统提示确认。 [下载 PKG 安装助手](https://github.com/easonzhangbin/PURERGB/releases/download/v0.2.0/PURE-RGB-0.2.0-macos-arm64.pkg) · [下载 ZIP](https://github.com/easonzhangbin/PURERGB/releases/download/v0.2.0/PURE-RGB-0.2.0-macos-arm64.zip) · [Apple 官方说明](https://support.apple.com/zh-cn/102445)

### 从源码运行

需要 Git、[uv](https://docs.astral.sh/uv/getting-started/installation/)、稳定版 [Rust 工具链](https://rust-lang.org/tools/install/) 和 Apple 命令行开发者工具。可用 `xcode-select --install` 安装 Apple 工具。准备好这些工具后，再执行以下命令。安装脚本使用 Python 3.13 和固定依赖，首次编译可能需要数分钟。

```bash
git clone --branch v0.2.0 https://github.com/easonzhangbin/PURERGB.git
cd PURERGB
python3 bootstrap.py
./PURE\ RGB.command
```

要在 Finder 里启动，可在本机完成上述安装后构建应用。该应用仍使用构建时的源码目录和虚拟环境。

```bash
vendor/negpy/.venv/bin/python tools/build_local_test_app.py "$HOME/Applications/PURE RGB 0.2.0.app"
```

应用在启动后及持续运行期间每 24 小时查询 GitHub 正式发布通道。点击「查看更新」阅读说明并下载安装，也可以在 macOS 的 PURE RGB 菜单中主动检查。

### 可选 Nikon 资源

仅在拥有可使用的匹配资源时，把 CML4 资源、ICC、固定曲线和验证文件放入本机的 `vendor/cool-colors/assets/nikon_cml4/`。安装脚本只取得参考源码。运行以下探针可核对所需名称和哈希。

```bash
vendor/negpy/.venv/bin/python -m pure_rgb.nikon_resource_probe
```

缺少资源时可使用 NegPy 和 FlexColor 路线。Nikon 的预扫描原始特性数据、扫描仪厂商配置与实验捕获数据保留在其各自的权利范围内。

## 后续计划

目录管理和图片库管理待开发。本版的整卷挑选使用当前项目的照片列表。

## 捐赠 PURE Studio 项目

感谢捐助者 **nothing老庄、J3RRYK33PF1T、熊小默、基岩、一笑而过**。欢迎使用支付宝扫描项目收款码，自愿支持后续开发。

<img src="docs/images/SupportAlipay.jpg" alt="PURE Studio 项目支付宝捐赠收款码" width="240">

应用内也可在「PURE RGB → 捐赠 PURE Studio 项目」查看同一张二维码。[捐赠说明](SUPPORT.md)

## 许可与致谢

软件源码采用 [GPL-3.0](LICENSE)。项目由 NegPy 的开源工作发展而来，并参考 cool-colors、flexcolor-tool 与 nkscan。品牌图标和字标保留各自权利。详见 [第三方来源与许可](THIRD_PARTY_NOTICES.md)。

宣传截图使用 Milad Fakurian 的桌面壁纸。 [Unsplash 原作](https://unsplash.com/photos/red-and-black-horizontal-striped-abstract-background-WlFOb5WgWvY) · [许可](https://unsplash.com/license)
