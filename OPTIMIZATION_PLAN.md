# 两项优化执行计划

> 基于上游 Gourieff/ComfyUI-ReActor 0.7.0（fork 起点 0.5.2-a2）的调研结论制定。
> 两项优化相互独立，各自单独测试、单独提交，可独立 revert。

## 背景与关键调研结论

- 上游 0.7.0 的"新核心"是用纯 ONNX 复刻 insightface 流水线，**画质无实质提升**，不整体跟进。
- 真正值得吸收的两点：① HyperSwap/ReSwapper 换脸模型支持；② 基于 3D landmark 的真实姿态角（替代本 fork 的启发式朝向估计）。
- 上游 MaskHelper 落后于本 fork（整批只检测第 0 帧的 bug 未修），不合并上游 nodes.py。

## 现状盘点（实施前必读）

1. **HyperSwap/ReSwapper 已有半成品**（此前已部分移植，勿从零重写）：
   - `scripts/reactor_swapper.py` L115-125 `_get_model_type` 按文件名分流
   - `scripts/reactor_swapper.py` L127-145 `getFaceSwapModel`：hyperswap → 原生 `ort.InferenceSession`；reswapper/inswapper → `insightface.model_zoo.get_model`
   - `scripts/reactor_swapper.py` L149-203 `get_landmarks_5` / `create_gradient_mask`；L205+ `paste_back`；L257 `run_hyperswap`
   - `scripts/reactor_swapper.py` L870-875（swap_face）、L1104-1109（swap_face_many）已有 hyperswap 分支；**Face Boost 对 hyperswap 跳过**
   - 模型目录常量 L54-55：`models/reswapper`、`models/hyperswap`
2. **朝向角当前是启发式**，不是 3D 姿态：`scripts/reactor_swapper.py` 约 L440-660（宽高比 + 左右可见度距离比 + 鼻子位置 → 角度）。
3. **1k3d68 landmark 已在逐脸运行**：`analyze_faces` → `FaceAnalysis.get()` 会加载并执行 buffalo_l 全部模型，`face.landmark_3d_68` 现成可用 → 姿态解算**零新增推理开销**。
4. 模型目录注册助手已有：`reactor_utils.py` L249 `add_folder_path_and_extensions`。

上游参考实现（对照基准）：
- https://raw.githubusercontent.com/Gourieff/ComfyUI-ReActor/main/reactor_core/hyperswap.py
- https://raw.githubusercontent.com/Gourieff/ComfyUI-ReActor/main/reactor_core/inswap.py
- https://raw.githubusercontent.com/Gourieff/ComfyUI-ReActor/main/reactor_core/face_objects.py

---

## 优化一：HyperSwap / ReSwapper 换脸模型支持（审计补完）

### 目标

在保持 insightface 检测/分析层不变、inswapper_128 行为完全不变的前提下，正确支持：
- `hyperswap_1a_256 / 1b_256 / 1c_256.onnx`（FaceFusion Labs，256px）
- `reswapper_128 / 256 / 512.onnx`（somanchiu，inswapper 同架构重训）

### 实施步骤

- [ ] 1. 对照上游 `hyperswap.py` 审计 `run_hyperswap`：
  - 输入名是否硬编码 `'source'/'target'`（上游用 session 动态获取 `input_names`）
  - 归一化链路：RGB 转换、`(x/255 - 0.5)/0.5`、CHW
  - 输出范围自适应（[-1,1] → [0,255] 判定）、NaN/Inf 防护（上游 L156-170）
  - paste_back 椭圆渐变 mask 参数与上游一致性
- [ ] 2. 验证 reswapper 走 `insightface.model_zoo.get_model` 对 256/512 是否正确：
  - insightface 自带 INSwapper 的对齐点是 128 标准点；上游 reactor_core 的 `INSwapper` 用 `ratio = input_size/128` 缩放标准点后对齐。
  - 若 insightface 版本对非 128 输入处理不正确，移植上游 `INSwapper.get` 的对齐逻辑（需新增 `onnx>=1.14.0` 依赖用于提取 emap，纯 Python 包，无编译问题）。
- [ ] 3. 确认模型目录注册：hyperswap/reswapper 目录是否通过 `add_folder_path_and_extensions` 注册，UI 下拉能否列出模型；不能则补注册。
- [ ] 4. 模型获取（手动下载，不进 install 自动流程）：
  - hyperswap_1a/1b/1c_256.onnx ← https://huggingface.co/facefusion/models-3.3.0/tree/main → `ComfyUI/models/hyperswap/`
  - reswapper_128/256/512.onnx ← https://huggingface.co/datasets/Gourieff/ReActor/tree/main/models → `ComfyUI/models/reswapper/`
- [ ] 5. 决策 Face Boost（GFPGAN/CodeFormer）对 hyperswap 的策略：当前跳过（保持）；256px 裸输出质量足够时 restore 可选开放。先保持现状，测试后再定。
- [ ] 6. 清理 hyperswap/reswapper 路径上的 DEBUG 打印。
- [ ] 7. README/文档补充模型下载说明（可选）。

### 已知风险与注意事项

- **许可证**：HyperSwap 为 ResearchRAIL-MS，商用需评估。
- **paste_back 固定椭圆 mask**：上游用固定椭圆渐变，遮挡/发际线贴合弱于 inswapper 的差值自适应 mask。本管线若接 MaskHelper（SAM mask 二次合成），边界由 mask 节点决定，影响被抵消；直接用换脸节点输出时需留意。
- **时序一致性**：换模型后伪影特性改变，`angle_threshold` 及 `smooth_blend_values`（L663-724）的 75.0/transition 参数可能需要重调。
- **性能**：256px 换脸段约为 128px 的 4 倍开销；可通过减弱/关闭 face restore 拉平总耗时。
- **结果不可比**：与 inswapper 输出"略有不同"，已产出的旧视频无法逐帧比对。

### 测试清单

- [ ] 单图 A/B：inswapper_128+GFPGAN vs hyperswap_1a 裸输出 vs hyperswap_1a+GFPGAN
- [ ] 三变体（1a/1b/1c）各出一张图对比
- [ ] reswapper_256 出图正常（若提供支持）
- [ ] 难例：侧脸、大角度、手/麦克风遮挡帧
- [ ] 视频全管线回归（含 MaskHelper、瞳孔保持、平滑混合），检查时序闪烁
- [ ] 性能记录：换脸段 / restore 段耗时对比
- [ ] 异常路径：模型文件缺失时的报错信息清晰
- [ ] 回归确认：inswapper_128 路径输出与改动前一致

### 验收标准

三类模型均可从 UI 选择并正常出图；inswapper 路径零行为变化；视频管线无回归；异常提示明确。

### 提交建议

```
feat: 补完 HyperSwap/ReSwapper 换脸模型支持（对照上游 0.6.2 参考）
```

---

## 优化二：真 3D 姿态角替换启发式朝向估计

### 目标

用 1k3d68 三维 landmark + 均形拟合解算欧拉角（pitch/yaw/roll），替换现有基于宽高比/可见度的启发式估计，提高侧脸判定鲁棒性。**不改变 `smooth_blend_values` 的输入输出接口**（per-frame 标量角度）。

### 实施步骤

- [ ] 1. 验证 `face.landmark_3d_68` 在现有管线可用（`analyze_faces` 返回的 Face 对象上直接检查，一次性验证即可）。
- [ ] 2. 新增姿态解算：移植上游 `inswap.py` 的 `estimate_affine_matrix_3d23d` / `P2sRt` / `matrix2angle`（约 30 行）+ `meanshape_68.py` 的 `MEANSHAPE_68` 常量表，从 `landmark_3d_68` 得到 (pitch, yaw, roll)，弧度转角度。
- [ ] 3. 接入角度产出（`swap_face` / `swap_face_many` 中现调用启发式的位置）：
  - 主角度取 `|yaw|`（与现语义"水平偏转"一致），保持 per-frame 标量输出；
  - 是否引入 pitch 参与阈值判断：先不引入，保持行为最小变化，标记为后续可选项。
- [ ] 4. 保留启发式函数作为回退：`landmark_3d_68` 缺失（模型被删/异常）时退回旧路径并 log 一次。
- [ ] 5. 清理启发式路径的 `DEBUG_MODE` 打印。
- [ ] 6. 保留/新增角度序列 dump（逐帧 CSV：pose 角度 vs 旧启发式角度），供标定与回归对比。

### 已知风险与注意事项

- **数值分布不同**：欧拉角 yaw 与启发式角度不是同一分布，`angle_threshold`（默认 60°）与 `smooth_blend_values` 内的 75.0/±5° 过渡带常量需用 dump 数据重新标定。
- **极端姿态**：仰头/低头过大时 1k3d68 的 yaw 解算稳定性需在难例上确认。
- **行为变化**：侧脸混合的触发时机可能提前/滞后，属预期变化，需在视频上确认观感。

### 测试清单

- [ ] 同一测试视频：旧启发式角度 vs 新 pose 角度逐帧 CSV 对比
- [ ] 平滑混合权重（Angles & weights dump）前后对比
- [ ] 视频全管线回归：侧脸段混合是否合理，无闪烁/突变
- [ ] 性能确认：无新增模型推理（对比改前改后换脸段耗时）
- [ ] 回退路径：人为使 landmark 不可用，确认回退启发式且仅告警一次

### 验收标准

角度全部来自 3D landmark 解算；按 dump 数据重标定后侧脸混合行为合理；inswapper/hyperswap 两种模型下均正常；无性能回退。

### 提交建议

```
feat: 用 1k3d68 三维姿态角替换启发式朝向估计
```

---

## 执行顺序建议

**先优化二，后优化一**：优化二改动小、零新增推理开销、当天可完成闭环；优化一涉及模型下载与 A/B 调参，周期更长。两项都完成后，优化一测试换模型时可直接受益于优化二更稳的朝向判定。

## 回滚策略

两项各自独立成 commit（不混提交），任一出问题单独 `git revert` 即可，互不影响。
