# PhytoDA 用户使用手册（中文版）

> **PhytoDA**（Phytolith Digital Analyzer）是一套面向农业与植物科学的高通量图像分析软件，集成了深度学习大模型（SAM、Depth Anything V2）与自然语言 AI 助手，支持全自动表型分析、手动图像测量和 AI 辅助语义标注三大工作模式。

---

## 目录

1. [环境要求](#1-环境要求)
2. [项目获取](#2-项目获取)
3. [环境安装](#3-环境安装)
4. [模型权重下载](#4-模型权重下载)
5. [配置 AI Copilot（可选）](#5-配置-ai-copilot可选)
6. [启动软件](#6-启动软件)
7. [界面总览](#7-界面总览)
8. [模式一：Auto Phenotyping（全自动表型分析）](#8-模式一auto-phenotyping全自动表型分析)
9. [模式二：Image Workbench（图像工作台）](#9-模式二image-workbench图像工作台)
10. [模式三：Annotation Studio（标注工作室）](#10-模式三annotation-studio标注工作室)
11. [AI Copilot 智能助手](#11-ai-copilot-智能助手)
12. [结果导出](#12-结果导出)
13. [常见问题](#13-常见问题)

---

## 1. 环境要求

| 项目 | 最低配置 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Windows 10 / Linux (Ubuntu 20.04+) | Windows 11 / Ubuntu 22.04 |
| GPU | NVIDIA GTX 1060 (6GB VRAM) | NVIDIA RTX 3060+ (8GB+ VRAM) |
| 内存 | 16 GB | 32 GB |
| Python | 3.10 | 3.10 或 3.11 |
| CUDA | 11.8+ | 12.1 |

> **注意**：软件可在无 GPU 的 CPU 环境下运行，但推理速度会非常慢，不建议用于批量分析。

---

## 2. 项目获取

### 方式一：直接下载

将项目文件夹 `PhytoDA` 完整拷贝至本地任意路径（建议路径中不包含中文和空格），例如：

```
C:\Projects\PhytoDA
```

### 方式二：Git 克隆（如已配置版本控制）

```bash
git clone <your-repo-url> PhytoDA
cd PhytoDA
```

### 确认目录结构

项目根目录下应包含以下关键文件夹和文件：

```
PhytoDA/
├── main.py                    # 程序入口
├── requirement.txt            # Python 依赖列表
├── checkpoints/               # 模型权重存放目录
│   ├── sam_vit_b_01ec64.pth
│   ├── sam_vit_l_0b3195.pth
│   ├── sam_vit_h_4b8939.pth
│   └── depth_anything_v2_vitl.pth
├── core/                      # 核心引擎
├── ui/                        # 用户界面
├── plugins/                   # 分析插件（种子/叶片/玉米/番茄/麦穗/冠层）
├── segment_anything/          # SAM 模型代码
└── depth_anything_v2/         # Depth Anything V2 模型代码
```

---

## 3. 环境安装

### 3.1 创建虚拟环境（强烈推荐）

使用 Conda：

```bash
conda create -n PhytoDA python=3.10 -y
conda activate PhytoDA
```

或使用 venv：

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3.2 安装 PyTorch（GPU 版本）

```bash
# CUDA 12.1 版本（推荐）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

如使用 CPU 或不同 CUDA 版本，请参照 [PyTorch 官网](https://pytorch.org/get-started/locally/) 调整命令。

### 3.3 安装其余依赖

```bash
pip install -r requirement.txt
```

### 3.4 验证安装

在终端中执行以下命令，确认所有关键库导入无误：

```bash
python -c "import torch; import cv2; import numpy; from PySide6.QtWidgets import QApplication; print('All dependencies OK')"
```

---

## 4. 模型权重下载

本软件依赖两个 AI 模型，需将其权重文件下载并放入 `checkpoints/` 文件夹中。

### 4.1 SAM（Segment Anything Model）

从以下链接下载对应权重文件：

| 模型 | 文件名 | 大小 | 说明 |
|------|--------|------|------|
| ViT-B | `sam_vit_b_01ec64.pth` | ~358 MB | 快速模式，精度较低 |
| ViT-L | `sam_vit_l_0b3195.pth` | ~1.2 GB | 均衡模式 |
| ViT-H | `sam_vit_h_4b8939.pth` | ~2.4 GB | 高精度模式，显存占用大 |

官方仓库：[facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

> 至少需要下载 ViT-B 版本。将下载的文件放入 `PhytoDA/checkpoints/` 目录。

### 4.2 Depth Anything V2

| 文件 | 大小 | 说明 |
|------|------|------|
| `depth_anything_v2_vitl.pth` | ~1.3 GB | 深度估计模型（必选） |

官方仓库：[DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)

### 4.3 确认 checkpoints 目录

```
checkpoints/
├── sam_vit_b_01ec64.pth    ✅
├── sam_vit_l_0b3195.pth    （可选）
├── sam_vit_h_4b8939.pth    （可选）
└── depth_anything_v2_vitl.pth  ✅
```

---

## 5. 配置 AI Copilot（可选）

PhytoDA 内置了一个 AI 助手，可以理解自然语言指令来自动操控软件。它支持任何兼容 OpenAI API 格式的大模型服务商（如 DeepSeek、通义千问、OpenAI 等）。

配置方式全部在软件界面上完成——**无需编辑代码，无需手动设置环境变量**。

### 5.1 获取 API Key

前往任意提供 OpenAI 兼容 Chat Completions API 的服务商平台注册并生成 API Key。常见示例：

| 服务商 | 模型示例 | API URL 示例 |
|--------|----------|-------------|
| DeepSeek | `deepseek-chat` | `https://api.deepseek.com/chat/completions` |
| 阿里云百炼 (Qwen) | `qwen-plus` | `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions` |
| OpenAI | `gpt-4o` | `https://api.openai.com/v1/chat/completions` |

记下三个关键信息：**API 地址**、**模型名称** 和 **API Key**。

### 5.2 通过 API Settings 对话框配置

1. 启动 PhytoDA
2. 点击顶部工具栏中的 **API Settings** 按钮
3. 在弹出的对话框中填写：
   - **Full API URL** — Chat Completions 接口地址（如 `https://api.deepseek.com/chat/completions`）
   - **Model Name** — 模型标识符（如 `deepseek-chat`）
   - **API Key** — 你的密钥
4. 点击 **Save** — 配置将保存至 `api_config.json`，并立即生效

如需更换服务商，只需再次点击 **API Settings** 修改内容即可，无需重启软件。

### 5.3 验证配置

在 Copilot 输入框中输入 `Hello` 并回车。如果收到回复，说明配置成功。

如果出现缺少 API Key 的提示，请点击 **API Settings** 确认密钥填写正确。

### 5.4 备选方式：环境变量配置（高级用户）

如果习惯使用环境变量方式，PhytoDA 会读取以下变量作为备选：

| 变量名 | 用途 |
|--------|------|
| `PhytoDA_API_KEY` | API 密钥 |
| `PhytoDA_API_BASE_URL` | Chat Completions 接口地址 |
| `PhytoDA_MODEL_NAME` | 模型名称 |

> **注意**：API Settings 对话框生成的 `api_config.json` 优先级高于环境变量。如果 `api_config.json` 已存在，其中的值会覆盖环境变量。

> **注意**：AI Copilot 是可选功能。即使不配置 API，除 Copilot 外的所有图像分析功能均可正常使用，仅对话面板会提示配置 API 设置。

---

## 6. 启动软件

### 6.1 命令行启动

```bash
python main.py
```

### 6.2 预期效果

启动后，界面将包含以下区域：
- **左上角**：工作模式切换下拉框
- **主区域**：当前工作模式的全部功能面板
- **右侧**：AI Copilot 对话面板（可通过中间的小三角按钮折叠/展开）

首次启动时，软件会自动加载 AI 模型，加载进度可在 Copilot 面板中查看。

---

## 7. 界面总览

PhytoDA 提供三种工作模式，通过顶部下拉框切换：

```
🤖 Auto Phenotyping   — 全自动批量表型分析
✏️ Image Workbench     — 手动测量与绘图工具
🔬 Annotation Studio  — AI 辅助语义标注
```

| 模式 | 适用场景 | 输入 | 输出 |
|------|----------|------|------|
| Auto Phenotyping | 大批量作物表型测量 | 单张或多张照片 | 图片 + CSV 数据表 |
| Image Workbench | 单图精确测量/标记 | 单张照片 | 图片 + CSV |
| Annotation Studio | 数据集制作（训练 AI） | 图片文件夹 | JSON / YOLO 标注文件 |

---

## 8. 模式一：Auto Phenotyping（全自动表型分析）

这是 PhytoDA 的核心功能，用于对植物图像进行全自动分析，提取形态学指标。

### 8.1 支持的分析任务

| 任务名称 | 英文标识 | 适用对象 | 主要输出指标 |
|----------|----------|----------|-------------|
| 种子分析 | Seed Analysis | 谷物、豆类种子 | 粒数、粒长、粒宽、面积、长宽比 |
| 叶片表型 | Leaf Phenotyping | 植物叶片 | 叶长、叶宽、面积、周长 |
| 玉米分析 | Corn Analysis | 玉米果穗 | 穗长、穗宽、行数、粒数、粒色 |
| 番茄表型 | Tomato Phenotyping | 番茄等近球形果实 | 果径、面积、体积、颜色 |
| 小麦叶夹角 | Wheat Leaf Angle | 小麦旗叶 | 茎叶夹角角度 |
| 麦穗分析 | Wheat Ear Analysis | 小麦麦穗 | 穗长、小穗数 |
| 冠层分析 | Canopy Analysis | 作物冠层 | 覆盖度、冠层孔隙 |

### 8.2 操作流程

**步骤 1：选择分析任务**

在左侧控制面板的"Task & Model"下拉框中选择对应的分析任务。

**步骤 2：选择 SAM 模型精度**

| 选项 | VRAM 占用 | 速度 | 精度 | 适用场景 |
|------|-----------|------|------|----------|
| Fast (ViT-B) | ~2 GB | 快 | 标准 | 大批量快速筛选 |
| Balanced (ViT-L) | ~4 GB | 中等 | 较高 | 常规研究分析 |
| High Precision (ViT-H) | ~6 GB | 慢 | 最高 | 高精度出版级数据 |

点击右侧的 **Load** 按钮可手动重新加载/切换模型。

**步骤 3：导入图片**

点击 **📂 Import Images** 按钮，在弹出的对话框中：
- 选择**多张图片**：按住 Ctrl 点选多个文件
- 选择**整个文件夹**：在文件对话框中进入目标文件夹后确认

导入后，左侧面板会显示已加载的图片数量，右侧显示第一张图片的预览。

**步骤 4：开始分析**

点击 **⚡ START ANALYSIS** 按钮即可开始批量分析。分析过程如下：
1. 系统对每张图片依次执行深度估计 + SAM 分割
2. 根据所选任务提取对应的形态学指标
3. 在 Details 表格中实时显示每一条检测结果的详细数据
4. 进度条实时更新

**步骤 5：查看结果**

- **Summary 标签页**：显示当前图片的汇总统计数据（总数、平均值等）
- **Details 标签页**：显示每个检测目标的详细指标（ID、长度、宽度、面积、颜色等）
- **👁️ Show Process Details**：打开后以 6 宫格形式展示每一步中间处理结果（原始图→深度图→深度掩码→深度裁剪→SAM 分割→最终结果）

**步骤 6：数据清理**

- 右键点击 Details 表格中的某一行，可**编辑 ID**、**删除脏数据**或**一键重排所有 ID**
- 右键点击图片上的标注文字，可直接从图像上删除该目标
- 删除后系统自动重新计算汇总统计量

**步骤 7：浏览切换**

使用 **◀ Prev** / **Next ▶** 按钮在不同图片间切换，或使用表格下方的导航控件。

### 8.3 自定义可视化样式

在图像显示区顶部工具栏中，可实时调整：
- **Text size**（0 – 5.0）：标注文字大小
- **Position**（Center / Top / Bottom / Left / Right）：文字相对于目标的位置
- **Thickness**（0.5 – 10.0）：标注线条的粗细

---

## 9. 模式二：Image Workbench（图像工作台）

这是一个自由的手动测量与绘图工具，适用于需要对单张图片进行精确测量的场景。

### 9.1 界面布局

- **左侧面板**：图片导入、导出按钮 + 测量数据表格
- **右侧**：双排工具栏 + 图像画布 + 状态栏

### 9.2 工具栏说明

**上排工具栏（通用工具）**：

| 工具 | 图标 | 功能 | 快捷键提示 |
|------|------|------|-----------|
| Select | ↖ | 选择/移动图形 | 点击选中，拖拽移动，框选多选 |
| Zoom | 🔍 | 缩放画布 | 左键放大，Alt+左键缩小 |
| Hand | ✋ | 平移画布 | 左键拖拽平移 |
| Arrow | ⬇️ | 绘制箭头 | 拖拽绘制，Shift 锁定方向 |
| Text | T | 添加文字标签 | 点击输入文字 |
| Pencil | ✏️ | 自由手绘 | 拖拽绘制 |
| Fill | 💧 | 颜色填充 | 点击填充封闭区域 |
| Picker | 💉 | 取色器 | 点击从图像上取色 |
| Eraser | 🧽 | 橡皮擦 | 点击/拖拽擦除图形 |

**下排工具栏（测量工具）**：

| 工具 | 图标 | 功能 |
|------|------|------|
| Line | 📏 | 绘制测量线段 |
| Poly | 📉 | 多点折线 |
| Free | 〰 | 自由曲线 |
| Angle | 📐 | 三点测量角度 |
| Point | 🎯 | 放置计数标记 |
| Rect | ⬜ | 矩形 |
| R-Rect | ⬜ | 圆角矩形 |
| Oval | ⚪ | 椭圆/圆形 |
| PolySel | ⬠ | 多边形选区 |
| FreeSel | 〰 | 自由选区 |
| Brush | 🖌️ | 画笔选区 |
| Magic | 🪄 | 智能磁力套索 |

### 9.3 设置比例尺（标定）

在测量前设置正确的比例尺，可将像素测量值转换为物理单位（如 cm、mm）。

1. 选择 **Line** 工具
2. 在图像上沿已知长度的参照物绘制一条线段
3. 右键点击该线段，选择 **📏 Set Scale**
4. 在弹出的对话框中输入：
   - **Known distance**：该线段在现实中对应的真实长度（例如 5.0）
   - **Unit of length**：单位名称（例如 cm）
5. 点击 OK，此后所有测量值将自动换算为物理单位

> 点击 **🔍 1:1 Reset** 按钮可重置比例尺为像素模式。

### 9.4 数据导出

点击 **💾 Export CSV** 按钮：
1. 选择保存路径和文件名
2. 系统会同时生成一张带标注的 PNG 图片和一个 CSV 数据表
3. CSV 中包含每条测量的 ID、类型、工具、长度、宽度、面积、周长、RGB 颜色等信息

---

## 10. 模式三：Annotation Studio（标注工作室）

这是一个 AI 辅助的语义分割标注工具，可以将普通图片制作为深度学习训练数据集。

### 10.1 界面布局

- **左侧工具栏**：10 种工具按钮
- **顶部栏**：深度切片控制 + SAM 模型选择 + 热键设置
- **中央画布**：图像显示与交互标注区域
- **右侧面板**：文件列表 + 类别管理 + 实例列表

### 10.2 工具说明

| 工具 | 快捷键 | 功能 |
|------|--------|------|
| Select | V | 选择/移动已标注的对象 |
| Hand | H | 平移画布 |
| SAM Point | Q | 点击正样本（绿点）+ 右键负样本（红点）进行 SAM 提示分割 |
| SAM Box | W | 框选目标区域进行 SAM 分割 |
| Auto 3D | T | 基于深度 + SAM 的一键全自动分割 |
| Lasso | L | 智能磁力套索，吸附边缘 |
| Poly | C | 点击绘制多边形 |
| Free Poly | F | 拖拽自由绘制闭合多边形 |
| Edit | E | 编辑已有多边形的顶点（拖拽移动/右键删除/双击加顶点） |
| Delete | DEL | 删除选中的标注对象 |

### 10.3 操作流程

**步骤 1：打开图片文件夹**

点击右侧面板的 **📁 Open Directory** 按钮，选择一个包含图片的文件夹。

系统会自动对每张图片运行深度估计，并在文件列表中显示三种状态标记：
- 📄 无标注的原始图片
- ✏️ 已有修改但尚未保存
- ✅ 已有已保存的 JSON 标注文件

**步骤 2：管理类别**

在右侧 **Categories** 面板中：
- **+ Add Class**：添加新类别（如 Leaf、Tomato、Stem）
- **双击类别名**：修改类别名称和颜色
- **右键类别**：编辑或删除类别

**步骤 3：使用 SAM 工具进行 AI 标注**

1. 首先在顶部栏中选择 SAM 模型精度，点击 **Load** 按钮加载 SAM 模型
2. 选择一个类别（如 Leaf），确保它在类别列表中高亮选中
3. 选择以下任一 AI 工具：

   - **SAM Point**：在目标上点击左键（绿色标记）添加正样本提示，在背景上点击右键（红色标记）添加负样本提示。软件实时显示分割预览，双击确认保存
   - **SAM Box**：在目标周围拖拽一个矩形框，SAI 将自动分割框内的主要目标
   - **Auto 3D**：利用深度图自动定位前景区域并生成分割结果，无需手动点击

**步骤 4：使用手动工具精细调整**

- **Edit 工具**（E）：选中已有标注的多边形，拖拽顶点调整形状，双击边缘添加新顶点，右键删除顶点
- 如果 AI 标注效果不理想，可使用 **Poly** 或 **Free Poly** 工具手动绘制

**步骤 5：使用深度切片辅助标注**

顶部栏的**深度切片滑块**可以帮助分离不同距离的物体：
- 拖动两个滑块设置深度范围（0-255）
- 只有深度值在范围内的像素会被显示，其余变暗
- 松开滑块后 SAM 会自动重新编码当前深度切片
- 使用 +/- 按钮或 `[` / `]` 快捷键微调

**步骤 6：保存标注**

- 点击 **💾 Save JSON** 按钮将当前所有修改过的标注统一保存
- 每张图片的标注存储为同名的 `.json` 文件（LabelMe 格式）
- 可使用 **🔄 YOLO Exporter** 将所有 JSON 转为 YOLO 分割格式（生成 `.txt` 文件）

### 10.4 快捷键总览

| 快捷键 | 功能 |
|--------|------|
| A / D | 上一张 / 下一张图片 |
| V | Select 工具 |
| H | Hand 工具 |
| Q | SAM Point 工具 |
| W | SAM Box 工具 |
| C | Polygon 工具 |
| F | Freehand Polygon 工具 |
| L | Magnetic Lasso 工具 |
| E | Edit 工具 |
| T | Auto 3D 一键分割 |
| ESC | 取消当前操作 |
| DEL | 删除选中的标注对象 |
| S | 保存 JSON |
| [ / ] | 深度滑块微调 |

> 所有快捷键均可通过顶部 **⌨️ Hotkeys** 按钮自定义。

---

## 11. AI Copilot 智能助手

PhytoDA 内置了一个 AI 助手，可以通过自然语言理解用户意图并自动操控软件。

### 11.1 可用指令示例

**全局指令（任何模式下可用）**：

| 你说 | Copilot 的行为 |
|------|---------------|
| "切换到标注模式" | 自动切换到 Annotation Studio |
| "帮我加载 C:\images 文件夹" | 自动导入该文件夹中的所有图片 |
| "打开 D:\data\wheat.jpg 这张图" | 加载指定的单张图片 |

**Auto Phenotyping 模式下的指令**：

| 你说 | Copilot 的行为 |
|------|---------------|
| "运行种子分析" | 自动切换到 Seed Analysis 并开始分析 |
| "用高精度模型分析" | 切换到 ViT-H 并加载 |
| "看下一张" | 导航到下一张图片 |
| "把字体调大一点，放到上方" | 调整标注文字大小和位置 |
| "删除 ID 5" | 从数据中删除编号为 5 的目标 |
| "重新排列所有 ID" | 按 1, 2, 3... 重新编号 |
| "隐藏颜色相关的列" | 隐藏表格中与颜色相关的列 |
| "高亮面积最大的那个" | 在表格中高亮面积最大的行 |
| "导出结果" | 自动导出图片和 CSV |
| "清空工作区" | 清除所有加载的图片和结果 |

### 11.2 使用技巧

1. **具体明确**：指令越精确，AI 执行效果越好。例如用"删除 ID 5"比"把那个错的删掉"更好
2. **一次一个操作**：Copilot 一次处理一条指令，如需连续操作请分步发出
3. **查看内部过程**：点击 AI 回复气泡下方的 "Show Internal Process" 可展开查看每个步骤的执行日志

---

## 12. 结果导出

### 12.1 Auto Phenotyping 模式

**方式一：手动导出**

点击 **💾 Export Results** 按钮，选择保存位置：
- **单张图片**：弹出保存文件对话框，可自定义文件名
- **多张图片**：弹出选择文件夹对话框，所有结果保存到该目录

每个分析结果包含：
- `xxx_result.jpg`：带标注的可视化结果图片
- `xxx_result.csv`：包含所有目标详细数据的表格文件

**方式二：通过 Copilot 自动导出**

在 Copilot 对话框中输入"导出结果"，系统会自动将文件保存到原始图片所在目录。

### 12.2 Image Workbench 模式

点击 **💾 Export CSV** 按钮，系统会同时保存：
- 一张包含所有绘图的 PNG 图片
- 一个包含所有测量数据的 CSV 文件

### 12.3 Annotation Studio 模式

- **💾 Save JSON**：保存为 LabelMe 格式的 JSON 文件，每张图片生成一个对应的 `.json`
- **🔄 YOLO Exporter**：将所有 JSON 转换为 YOLO 分割格式，生成 `.txt` 文件和 `classes.txt`

---

## 13. 常见问题

### Q1：启动时提示 "Core Error: 缺少依赖库"

**原因**：Python 环境未正确安装依赖。

**解决**：
```bash
pip install -r requirement.txt
```

如果 `segment_anything` 或 `depth_anything_v2` 导入失败，请确认这两个文件夹位于项目根目录下且完整。

### Q2：启动后报 "AI Engines 加载失败" 或长时间停在加载

**原因**：模型权重文件缺失或路径不正确。

**解决**：
1. 检查 `checkpoints/` 文件夹是否包含必需的 `.pth` 文件
2. 确认文件名与 [第 4 节](#4-模型权重下载) 中列出的完全一致
3. 如果 VRAM 不足（< 4 GB），请使用 Fast (ViT-B) 模式

### Q3：Copilot 提示 "API Key not detected"

**原因**：环境变量 `PhytoDA_API_KEY` 未设置或拼写错误。

**解决**：
1. 确认环境变量名称完全为 `PhytoDA_API_KEY`（区分大小写）
2. 在终端中执行 `echo $env:PhytoDA_API_KEY`（Windows PowerShell）或 `echo $PhytoDA_API_KEY`（Linux）验证
3. 如果是 IDE 中启动，可能需要重启 IDE 以读取新的环境变量

### Q4：分析结果中某些目标是多余的/错误的

**解决**：
1. 右键点击 Details 表格中对应的行，选择 "Delete This Object"
2. 或右键点击图片上对应的标注文字，选择 "Delete Object ID: X"
3. 删除后系统会自动重新计算统计值

### Q5：标注文字太小看不清

**解决**：在 Auto Phenotyping 模式下的图像显示区顶部，调整 **Text size**（建议 1.5 – 2.5）和 **Thickness**（建议 1.5 – 2.5）。

### Q6：SAM 分割效果不理想

**解决**：
1. 尝试切换到更高精度的 SAM 模型（ViT-L 或 ViT-H）
2. 在 Annotation Studio 中使用 SAM Point 工具，手动添加正/负样本提示点
3. 调整深度切片范围，帮助模型聚焦正确的前景区域

### Q7：界面显示异常（字体过大/过小、模糊）

**原因**：高分屏（4K）缩放问题。

**解决**：程序已在 `main.py` 中设置了 `QT_AUTO_SCREEN_SCALE_FACTOR=1`。如果仍有问题，可在系统显示设置中调整缩放比例。

---

> **技术支持**：如遇到本手册未涵盖的问题，请通过项目 Issue 或讨论区反馈，并附上错误信息和运行环境描述。
