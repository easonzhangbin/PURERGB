# RGB 灯光配平与柯达资料 / RGB light balancing and Kodak references

[中文九图指南](RGB_TRICHROME_GUIDE.zh-CN.md) · [English illustrated guide](RGB_TRICHROME_GUIDE.en.md)

## 柯达 H-61A 的 D-min 与 LAD / Kodak H-61A D-min and LAD

H-61A 用未受光的场景背景说明 **D-min**。对应的负片区域几乎未曝光，冲洗后呈最低密度。**LAD** 是实验室用于核对印片过程的参考。负片 LAD 标准色块取自反射率 16% 的中性目标，名义 Status M 密度为 R 0.80、G 1.20、B 1.60。

H-61A illustrates **D-min** with an unlit scene background that leaves the negative virtually unexposed. That area has the lowest density after processing. **LAD** is a laboratory reference used to check the printing process. The negative LAD patch comes from a neutral target with 16% reflectance. Its nominal Status M densities are R 0.80, G 1.20 and B 1.60. [柯达 H-61A 原文 / Kodak H-61A PDF](https://www.kodak.com/content/products-brochures/Film/LAD-for-KODAK-Color-Negative-Film-H-61a.pdf)

## 扩展动态范围扫描 / Scanning extended dynamic range

柯达描述的传统 10 位 DPX 按每码值 0.002 密度单位编码。最高码值为 1023，D-min 传统放在 **CV 95**，所以 D-min 以上可记录约 **1.856** 密度单位。高密度的 VISION3 负片可能超出这个范围，导致原场景的高光细节丢失。扫描器本身能测量足够的密度范围时，柯达优先建议 **16 位编码**。保留 10 位时，也可把步长提高到 0.0025 密度单位，扩大可记录范围。

In Kodak’s traditional 10-bit DPX example, each code step represents 0.002 density units. The top code is 1023 and **CV 95** is the historical D-min placement, leaving about **1.856** density units above D-min. Dense VISION3 negatives can exceed that encoding range, losing detail from the original scene highlights. Kodak prefers **16-bit encoding** when the scanner can measure the full density range. A wider step of 0.0025 density units is an alternative for 10-bit encoding. [柯达扩展动态范围扫描建议，第 1–4 页 / Kodak scanning recommendations, pp. 1–4](https://www.kodak.com/content/products-brochures/Film/Scanning-Recommendations-for-Extended-Dynamic-Range-Camera-Films-EN.pdf)

## PURE RGB 的配平与曝光 / Balance and exposure in PURE RGB

同卷空白片基让三路照明通过同一块材料，软件可以用三次实测比较当前装置的相对信号。上面的 CV 数值用于扫描输出的密度编码。相机翻拍时，右侧的 100% 指 RAW 白电平上限。输出位深用于保留已经采集到的数据，RAW 如有削顶，应先调整曝光再复拍。PURE RGB 根据本次片基与 RAW 实测给出建议，并自行定义 ±5% 保持区、10% 复测分界和快门试拍判据。

An unexposed piece from the same roll gives the three exposures a common material reference for comparing signals in your setup. The CV figures above describe density encoding in scanner output. In camera capture, 100% on the right-hand chart marks the RAW white-level limit. Output bit depth preserves data already captured. If the RAW clips, adjust the exposure and recapture. PURE RGB uses your film-base and RAW measurements and defines the ±5% hold band, the 10% retest boundary and the shutter trial criteria.

### 拍摄与复测

1. **三色试拍。** 固定同卷已冲洗、未曝光的片基，保持 ISO、快门和光圈相同。在红、绿、蓝灯光下各拍一张 RAW。
2. **确认橙框和蓝框。** 可在确认蓝框前移动和缩放橙框。按实际素材，把整个橙框放进三张共有的同一块干净、已冲洗、未曝光的片基，位置可在蓝框内或蓝框外。例如左侧为均匀的空白胶片、右侧有画面时，橙框应整个落在左侧胶片内的空白处，避开齿孔、胶片外的黑色区域和过渡边缘。蓝框圈住实际要保留的整张照片，按输出需要纳入齿孔，然后确认保留画幅。

   手边没有单独片基时，三张 RAW 都拍入相邻两格画面和中间的透明窄条。选择「片间空隙」后，软件会尝试在三路共同位置给出橙色建议框。可以拖动或缩放，框内留出与两侧帧线的距离，避开齿孔、漏光、字码、阴影和灰尘。无法辨认同一空隙时，可在预览中手动画框。核对三张中的窄条都未曝光再勾选确认。蓝框继续圈定实际保留画幅。

   如需平场，点「载入平场…」选一张平场 RAW 或 NPZ 增益文件，或用「已有平场…」选择项目或库中的配置，也可跟随项目平场。核对平场与翻拍条件是否匹配。勾选「套用平场」后，左侧使用校正后的片基信号，右侧按原始 RAW 检查曝光余量。

3. **先看曝光。** 有削顶风险时，按右侧提示缩短三路共用的快门时间。估算有余量且右侧建议试加曝光时，可试用较慢的共同快门。保持灯光、ISO 和光圈一致，复拍三路确认。
4. **再调灯光。** 片基已确认且三路偏差在 ±5% 内时保持灯光。最大偏差超过 5%、至多 10% 时先复测。偏差更大时，可按提示方向微调对应通道，再重拍核对。

增加单路灯光前，软件还会检查保留画幅的原始 RAW 峰值；缺少可靠峰值、已有零星近上限或削顶点，或预计峰值达到 RAW 上限的 99.8% 时，暂不提供增加量。当前曝光风险仍按报告中的 P99.9 和近上限点比例判据显示。

### Capture and retest

1. **Capture three test RAWs.** Use an unexposed, processed piece of film from the same roll. Keep ISO, shutter speed and aperture identical across separate red, green and blue exposures.
2. **Confirm the orange and blue regions.** You can move and resize the orange box before confirming the blue box. Put the entire orange box on the same clean, processed, unexposed film base in all three exposures. Its position follows the film base and can be inside or outside the blue box. For example, if the blank film is on the left and the image is on the right, place the whole orange box within the blank film on the left. Avoid sprocket holes, the dark area outside the film, and the transition into the image. Set the blue box to cover the entire intended output, including sprocket holes if needed, and then confirm the retained area.

   If you have no separate film-base sample, include two adjacent frames and the clear strip between them in all three RAWs. Choose **Inter-frame gap** to get a suggested orange box where the three previews agree. Move or resize it to leave room on both sides of the frame lines and avoid sprocket holes, light leaks, edge markings, shadows and dust. If there is no suggestion, draw the box on the preview. Check that the strip is unexposed in all three captures before confirming it. Use the blue box for the full area you intend to keep.

   To use a flat field, open one RAW reference or NPZ gain file with 「载入平场…」. Use 「已有平场…」 to select a saved project or library profile, or follow the project's current flat field. Match the reference to your capture setup. With flat-field correction enabled, the left chart uses the corrected film-base readings. The right chart checks exposure headroom from the original RAW samples.

3. **Check exposure first.** If clipping is likely, try the faster shutter speed shown on the right for all three exposures. If the meter suggests a slower shutter speed, you can test it with the same lamp settings, ISO and aperture. Capture the trio again and recheck.
4. **Then balance the lamps.** With a confirmed film-base reference, keep the lamp settings when all three deviations are within ±5%. Retest when the maximum deviation exceeds 5% and is at most 10%. For larger deviations, make small changes in the indicated direction and verify with a new set.

Before suggesting a lamp increase, the meter also checks the original RAW peak in the retained area. It withholds the increase if a reliable peak is unavailable, any near-limit or clipped samples already exist, or the predicted peak reaches 99.8% of the RAW limit. The current exposure-risk indicator continues to use the reported P99.9 and near-limit sample-fraction criteria.

## 灯具参考的保存与复用 / Save and reuse a lamp reference

按设备原刻度记录 R、G、B，可以在分析前保存灯光参考，也可以在分析完成后保存完整报告。同一灯具用相同刻度时，载入 TXT 参考并输入实际读数，查看「按参考调整灯具」。示例中，参考为 24、42、69，当前为 20、45、69，旋钮应 R 增 4、G 减 3、B 保持。百分比刻度的差值以百分点计。调整后复拍核对。更换设备或胶片，或改动拍摄参数后，重新测量。

Enter the R, G and B readings in the lamp’s own units. You can save a lamp reference before analysis or save a full report afterward. For the same lamp and scale, load the TXT reference and enter the actual current readings to see the knob changes. If the reference is 24, 42, 69 and the current readings are 20, 45, 69, increase R by 4, decrease G by 3 and leave B at 69. On a percentage scale, those changes are percentage points. Verify with another capture. Remeasure after changing equipment, film or camera settings.

## 完整报告与火眼联动 / Full reports and Flameeye

完成三张 RAW 分析，核对并确认同卷空白片基和实际保留画幅后，点「保存报告…」选择 **TXT 文本报告**或 **JSON 联动报告**。选择「整幅 RAW」时，曝光检查范围是完整 RAW 画幅。完整 TXT 同时包含可读说明、灯具参考和完整测量数据；JSON 包含相同测量数据，也保留已填写的灯具参考。两种完整报告均可导入支持 PURE RGB 报告的火眼版本，用于相机片基配平流程。未确认片基、未确认保留画幅、信号无效或保留画幅存在 RAW 削顶风险时，应先处理并复测，不能把报告当作已验收的采集结果。

After analyzing all three RAWs, verify and confirm the unexposed film base from the same roll and the actual retained area. Use **Save report…** to choose a **TXT report** or a **JSON companion report**. Selecting the full-RAW option checks the complete RAW canvas. The full TXT contains the readable explanation, lamp reference and complete measurements. JSON contains the same measurements and any entered lamp reference. Both full formats can be imported into a version of Flameeye that supports PURE RGB reports for the camera film-base balance workflow. Resolve missing confirmations, invalid signals or RAW clipping risk in the retained area and repeat the measurement before treating the capture as accepted.

火眼与 PURE RGB 使用同一套中位数目标：三路片基中位信号的中间值作为共同目标，三路相对偏差都在 **±5%** 内时通过配平；最大偏差超过 5%、至多 10% 时复拍确认。改变灯光后，保持约定的相机设置、片基和量区，重新拍摄 R、G、B 三张 RAW，再分析并导出新报告。报告记录源文件内容身份和分析时间；重复导出、重命名或重新分析同一组三张 RAW，均不能代替新的相机复拍验收。原文件被覆盖或发生变化时，保存完整报告前会要求重新分析并确认。

Flameeye and PURE RGB use the same median target: the middle of the three film-base median signals is the shared target. Balance passes when every channel is within **±5%** of that target. A maximum deviation above 5% and at most 10% calls for another capture. After adjusting the lamps, retain the agreed camera settings, film base and measurement region, capture a new R/G/B set, analyze it and export a new report. Reports record source-content identities and the analysis time. Re-exporting, renaming or reanalyzing the same three RAWs cannot substitute for a new camera verification capture. If an original file is overwritten or changes, saving the full report requires a fresh analysis and confirmation.

「保存灯光参考…」生成的独立灯具参考只用于保存旋钮读数和可用的拍摄参数，**不包含完整片基测量，不能用于火眼的测量联动或复拍验收**。它仍可通过 PURE RGB 的「载入参考…」复用；完整 TXT 和带灯具参考的 JSON 也可以在此载入。两种参考都需要核对同一灯具、同一种刻度和实际拍摄条件。

The separate **Save lamp reference…** action stores control readings and any available capture settings. It **does not contain full film-base measurements and cannot be used for Flameeye measurement transfer or camera verification**. It remains loadable through PURE RGB’s reference-loading action, which also accepts the full TXT and JSON containing a lamp reference. Check that the same lamp, control scale and capture conditions apply.
