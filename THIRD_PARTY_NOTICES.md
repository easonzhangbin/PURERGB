# Third-party sources

- NegPy 0.68.0, commit d8123a9b2a18fdaab5941347a91e402fa936d3dc, GPL-3.0. The unmodified checkout retains its NOTICE.md and other dependency notices. See licenses/NegPy-GPL-3.0.txt.
- cool-colors, Rohan Pandula, commit 74ffceb79e6e976858d9bffddfeefddba626378f. The density adapter derives from invert_c41.py::invert_auto, with frozen analysis, floating-point output and separate output transfer encoding. See licenses/cool-colors-LICENSE.txt, including the Nikon data exclusion. No Nikon table/profile is loaded by this adapter.
- flexcolor-tool contributors, commit 04e745a2fde3559acb2d4f194b7a192150b26f5c, MIT. The adapter calls build_lut, contrast_brightness and neg_gamma from the unmodified source checkout. LUT resolution/interpolation and input normalization are PURE RGB adapter choices. The representative sensor LUT stage is excluded. See licenses/flexcolor-tool-LICENSE.txt.
- activexray/nkscan, commit f971cfd8efe89f6e02507c80b0945206fdaf0bcd, MIT OR Apache-2.0; the MIT option is used for the independent prescan exposure/thumbnail metering port in pure_rgb/nikon_metering.py and shared analyzer arithmetic in pure_rgb/nikon_parameters.py. Source paths and provenance are recorded in those modules, outputs/nikon-metering-v1 and outputs/nikon-parameter-study-v1. See licenses/nkscan-MIT.txt. Matching the public implementation is distinct from complete Nikon Scan reproduction; original resource/distribution constraints remain separate.

Reference checkouts remain separate from the PURE RGB source repository. Binary releases include the pinned source portions used by the application and their original notices; proprietary Nikon tables and personal calibration profiles are excluded.

- The Qt scrubber controls adapt the interaction and timing of [SmoothUI Scrubber](https://smoothui.dev/docs/components/scrubber). SmoothUI is MIT licensed. See [its license](licenses/SmoothUI-MIT.txt).
- The native Spring dropdown, Jelly slider, Rubber dock and Glass lens use the interaction designs and spring parameters published by [DesignBookmark Motion](https://designbookmark.com/motion).

## PURE Neg 观片台

原生灯板和胶片放大镜改编自 PURE Neg 的 BatchLightTable.swift 与 FilmLoupe.swift，版权归 Film Dust for PhotoCraft contributors 所有，采用 MIT 许可。[原项目](https://github.com/easonzhangbin/PURENeg) · [许可全文](licenses/PURENeg-MIT.txt)

## Qt 6.11.2 runtime fixes

The macOS binary includes Qt 6.11.2 with a Cocoa accessibility ownership backport and a local QtWidgets stale-accessibility-cache guard. Original copyright and license notices, exact modified files, patches and build instructions are in [licenses/qt-cocoa-6.11.2](licenses/qt-cocoa-6.11.2/README.txt). Complete corresponding QtBase source and these modifications are provided as `Qt-6.11.2-Corresponding-Source.tar.gz` alongside the [0.2.2 release downloads](https://github.com/easonzhangbin/PURERGB/releases/tag/v0.2.2). The application retains accessibility support.
