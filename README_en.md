# PhytoDA User Manual

> **PhytoDA** (Phytolith Digital Analyzer) is a high-throughput image analysis platform for agricultural and plant science research. It integrates large vision models (SAM, Depth Anything V2) with a natural-language AI assistant, supporting three working modes: fully automated phenotyping, manual image measurement, and AI-assisted semantic annotation.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Obtaining the Project](#2-obtaining-the-project)
3. [Environment Setup](#3-environment-setup)
4. [Downloading Model Weights](#4-downloading-model-weights)
5. [Configuring AI Copilot (Optional)](#5-configuring-ai-copilot-optional)
6. [Launching the Software](#6-launching-the-software)
7. [UI Overview](#7-ui-overview)
8. [Mode 1: Auto Phenotyping](#8-mode-1-auto-phenotyping)
9. [Mode 2: Image Workbench](#9-mode-2-image-workbench)
10. [Mode 3: Annotation Studio](#10-mode-3-annotation-studio)
11. [AI Copilot](#11-ai-copilot)
12. [Exporting Results](#12-exporting-results)
13. [FAQ](#13-faq)

---

## 1. System Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| OS | Windows 10 / Linux (Ubuntu 20.04+) | Windows 11 / Ubuntu 22.04 |
| GPU | NVIDIA GTX 1060 (6 GB VRAM) | NVIDIA RTX 3060+ (8 GB+ VRAM) |
| RAM | 16 GB | 32 GB |
| Python | 3.10 | 3.10 or 3.11 |
| CUDA | 11.8+ | 12.1 |

> **Note**: The software can run on CPU-only systems, but inference will be significantly slower and is not recommended for batch analysis.

---

## 2. Obtaining the Project

### Option A: Direct Download

Copy the entire `PhytoDA` project folder to any local path (avoid paths containing spaces or non-ASCII characters), for example:

```
C:\Projects\PhytoDA
```

### Option B: Git Clone (if version control is configured)

```bash
git clone <your-repo-url> PhytoDA
cd PhytoDA
```

### Verify Directory Structure

The project root should contain the following key folders and files:

```
PhytoDA/
├── main.py                    # Application entry point
├── requirement.txt            # Python dependency list
├── checkpoints/               # Model weight storage
│   ├── sam_vit_b_01ec64.pth
│   ├── sam_vit_l_0b3195.pth
│   ├── sam_vit_h_4b8939.pth
│   └── depth_anything_v2_vitl.pth
├── core/                      # Core engine
├── ui/                        # User interface
├── plugins/                   # Analysis plugins (seed/leaf/corn/tomato/wheat ear/canopy)
├── segment_anything/          # SAM model code
└── depth_anything_v2/         # Depth Anything V2 model code
```

---

## 3. Environment Setup

### 3.1 Create Virtual Environment (Strongly Recommended)

Using Conda:

```bash
conda create -n PhytoDA python=3.10 -y
conda activate PhytoDA
```

Or using venv:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3.2 Install PyTorch (GPU Version)

```bash
# CUDA 12.1 (recommended)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

If using CPU or a different CUDA version, refer to the [PyTorch website](https://pytorch.org/get-started/locally/) to adjust the command accordingly.

### 3.3 Install Remaining Dependencies

```bash
pip install -r requirement.txt
```

### 3.4 Verify Installation

Run the following command in your terminal to confirm all critical libraries import correctly:

```bash
python -c "import torch; import cv2; import numpy; from PySide6.QtWidgets import QApplication; print('All dependencies OK')"
```

---

## 4. Downloading Model Weights

This software depends on two AI models. Their weight files must be downloaded and placed into the `checkpoints/` folder.

### 4.1 SAM (Segment Anything Model)

Download the weight files from the official repository:

| Model | Filename | Size | Description |
|-------|----------|------|-------------|
| ViT-B | `sam_vit_b_01ec64.pth` | ~358 MB | Fast mode, standard accuracy |
| ViT-L | `sam_vit_l_0b3195.pth` | ~1.2 GB | Balanced mode |
| ViT-H | `sam_vit_h_4b8939.pth` | ~2.4 GB | High precision, larger VRAM footprint |

Official repository: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

> At minimum, the ViT-B version is required. Place the downloaded files into `PhytoDA/checkpoints/`.

### 4.2 Depth Anything V2

| File | Size | Description |
|------|------|-------------|
| `depth_anything_v2_vitl.pth` | ~1.3 GB | Depth estimation model (required) |

Official repository: [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)

### 4.3 Confirm checkpoints Directory

```
checkpoints/
├── sam_vit_b_01ec64.pth          ✅
├── sam_vit_l_0b3195.pth          (optional)
├── sam_vit_h_4b8939.pth          (optional)
└── depth_anything_v2_vitl.pth    ✅
```

---

## 5. Configuring AI Copilot (Optional)

PhytoDA includes an AI assistant that understands natural-language commands to control the software automatically. It supports any OpenAI-compatible API provider (DeepSeek, Qwen, OpenAI, etc.).

Configuration is done entirely through the UI — **no code editing or environment variables required**.

### 5.1 Obtain an API Key

Sign up with any provider that offers an OpenAI-compatible chat completions API, and generate an API key. Examples:

| Provider | Model Example | API URL Example |
|----------|---------------|-----------------|
| DeepSeek | `deepseek-chat` | `https://api.deepseek.com/chat/completions` |
| Alibaba Qwen (Bailian) | `qwen-plus` | `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions` |
| OpenAI | `gpt-4o` | `https://api.openai.com/v1/chat/completions` |

Take note of three pieces of information: the **API URL**, the **model name**, and your **API key**.

### 5.2 Configure via the API Settings Dialog

1. Launch PhytoDA
2. In the top bar, click the **API Settings** button
3. In the dialog that appears, fill in:
   - **Full API URL** — the chat completions endpoint (e.g., `https://api.deepseek.com/chat/completions`)
   - **Model Name** — the model identifier (e.g., `deepseek-chat`)
   - **API Key** — your secret key
4. Click **Save** — the configuration is stored in `api_config.json` and takes effect immediately

You can change providers at any time by clicking **API Settings** again and updating the values. No restart required.

### 5.3 Verify Configuration

Type `Hello` into the Copilot input box and press Enter. If you receive a reply, the configuration is successful.

If a "missing key" error appears, click **API Settings** to verify the API Key field is filled in correctly.

### 5.4 Alternative: Environment Variables (for advanced users)

If you prefer configuration via environment variables, PhytoDA reads the following as fallbacks:

| Variable | Purpose |
|----------|---------|
| `PhytoDA_API_KEY` | API key |
| `PhytoDA_API_BASE_URL` | Chat completions endpoint URL |
| `PhytoDA_MODEL_NAME` | Model name |

> **Note**: The API Settings dialog (stored in `api_config.json`) takes priority over environment variables. If `api_config.json` exists, its values override the environment.

> **Note**: The AI Copilot is an optional feature. All image analysis functions work normally without it — only the chat panel will show a reminder to configure API settings.

---

## 6. Launching the Software

### 6.1 Start from Terminal

```bash
cd PhytoDA
python main.py
```

### 6.2 Expected result

Upon launch, the interface consists of:
- **Top-left**: Working mode selector dropdown
- **Main area**: All functional panels for the current working mode
- **Right side**: AI Copilot chat panel (collapsible via the small triangular button in the middle)

On first launch, the software automatically loads AI models; loading progress can be monitored in the Copilot panel.

---

## 7. UI Overview

PhytoDA provides three working modes, switchable via the dropdown at the top:

```
🤖 Auto Phenotyping   — Fully automated batch phenotyping analysis
✏️ Image Workbench     — Manual measurement and drawing tools
🔬 Annotation Studio  — AI-assisted semantic annotation
```

| Mode | Use Case | Input | Output |
|------|----------|-------|--------|
| Auto Phenotyping | High-throughput crop trait measurement | One or multiple photos | Image + CSV data table |
| Image Workbench | Precision single-image measurement/marking | Single photo | Image + CSV |
| Annotation Studio | Dataset creation (for training AI) | Image folder | JSON / YOLO annotation files |

---

## 8. Mode 1: Auto Phenotyping

This is the core capability of PhytoDA, designed for fully automated analysis of plant images to extract morphological traits.

### 8.1 Supported Analysis Tasks

| Task | Target | Key Output Metrics |
|------|--------|-------------------|
| Seed Analysis | Grain, bean seeds | Count, length, width, area, L/W ratio |
| Leaf Phenotyping | Plant leaves | Leaf length, width, area, perimeter |
| Corn Analysis | Maize ears | Ear length, width, row count, kernel count, kernel color |
| Tomato Phenotyping | Tomato and similar round fruits | Diameter, area, volume, color |
| Wheat Leaf Angle | Wheat flag leaf | Stem-leaf angle |
| Wheat Ear Analysis | Wheat ears | Ear length, spikelet count |
| Canopy Analysis | Crop canopy | Coverage, canopy porosity |

### 8.2 Workflow

**Step 1: Select the analysis task**

Choose the corresponding analysis task from the "Task & Model" dropdown in the left control panel.

**Step 2: Select SAM model precision**

| Option | VRAM Usage | Speed | Accuracy | Best For |
|--------|-----------|------|----------|----------|
| Fast (ViT-B) | ~2 GB | Fast | Standard | Rapid batch screening |
| Balanced (ViT-L) | ~4 GB | Medium | High | Routine research |
| High Precision (ViT-H) | ~6 GB | Slow | Highest | Publication-grade data |

Click the **Load** button to manually reload or switch models.

**Step 3: Import images**

Click **📂 Import Images**, then in the dialog:
- Select **multiple images**: Ctrl+click individual files
- Select **an entire folder**: navigate into the folder and confirm

After import, the left panel shows the image count and the right side shows a preview of the first image.

**Step 4: Start analysis**

Click **⚡ START ANALYSIS** to begin batch analysis. The pipeline:
1. Depth estimation + SAM segmentation for each image
2. Task-specific morphological trait extraction
3. Real-time display of detailed per-object data in the Details table
4. Live progress bar updates

**Step 5: View results**

- **Summary tab**: Aggregate statistics for the current image (total count, averages, etc.)
- **Details tab**: Per-object metrics (ID, length, width, area, color, etc.)
- **👁️ Show Process Details**: Toggle a 6-panel grid showing intermediate processing steps (Original → Depth Map → Depth Mask → Depth Cutout → SAM Seg → Final Result)

**Step 6: Data cleaning**

- Right-click a row in the Details table to **edit ID**, **delete spurious objects**, or **auto-reorder all IDs**
- Right-click an annotation label on the image to remove that object directly from the canvas
- Summary statistics are automatically recalculated after each deletion

**Step 7: Navigation**

Use **◀ Prev** / **Next ▶** buttons to switch between images in the loaded queue.

### 8.3 Customizing Visualization

In the top toolbar of the image display area, adjust:

- **Text size** (0 – 5.0): Annotation label font size
- **Position** (Center / Top / Bottom / Left / Right): Label placement relative to the object
- **Thickness** (0.5 – 10.0): Contour line width

---

## 9. Mode 2: Image Workbench

A flexible manual measurement and drawing tool for precision single-image analysis.

### 9.1 Layout

- **Left panel**: Load/export buttons + measurement data table
- **Right side**: Dual toolbars + image canvas + status bar

### 9.2 Toolbars

**Upper toolbar (general tools)**:

| Tool | Function |
|------|----------|
| Select | Select/move/box-select graphics |
| Zoom | Left-click to zoom in, Alt+click to zoom out |
| Hand | Pan the canvas |
| Arrow | Draw arrows |
| Text | Place text labels |
| Pencil | Freehand drawing |
| Fill | Flood-fill enclosed areas |
| Picker | Pick a color from the image |
| Eraser | Eraser drawings |

**Lower toolbar (measurement tools)**:

| Tool | Function |
|------|----------|
| Line | Draw a measurement line segment |
| Poly | Multi-point polyline |
| Free | Freehand curve |
| Angle | Three-point angle measurement |
| Point | Place auto-numbered point markers |
| Rect | Rectangle |
| R-Rect | Rounded rectangle |
| Oval | Ellipse/circle |
| PolySel | Polygon selection |
| FreeSel | Freehand selection |
| Brush | Brush selection |
| Magic | Intelligent magnetic lasso |

### 9.3 Setting the Scale (Calibration)

Set the correct scale before measuring to convert pixel values into physical units (cm, mm, etc.).

1. Select the **Line** tool
2. Draw a line segment along a reference object of known length on the image
3. Right-click the line and choose **📏 Set Scale**
4. In the dialog, enter:
   - **Known distance**: the real-world length of the reference (e.g., 5.0)
   - **Unit of length**: the unit name (e.g., cm)
5. Click OK — all subsequent measurements will be converted to physical units

> Click **🔍 1:1 Reset** to revert the scale to pixel mode.

### 9.4 Data Export

Click **💾 Export CSV** to:
1. Choose a save location and filename
2. The system saves both an annotated PNG image and a CSV data table
3. The CSV contains ID, type, tool, length, width, area, perimeter, RGB color, and more for each measurement

---

## 10. Mode 3: Annotation Studio

An AI-assisted semantic segmentation annotation tool for creating deep-learning training datasets.

### 10.1 Layout

- **Left toolbar**: 10 tool buttons
- **Top bar**: Depth slice controls + SAM model selector + hotkey settings
- **Center canvas**: Image display and interactive annotation area
- **Right panel**: File list + category management + instance list

### 10.2 Tools

| Tool | Shortcut | Function |
|------|----------|----------|
| Select | V | Select/move annotated objects |
| Hand | H | Pan the canvas |
| SAM Point | Q | Click positive (green) / right-click negative (red) prompts for SAM segmentation |
| SAM Box | W | Draw a bounding box for SAM segmentation |
| Auto 3D | T | Depth-guided one-click zero-shot segmentation |
| Lasso | L | Intelligent magnetic lasso, snaps to edges |
| Poly | C | Click-to-place polygon vertices |
| Free Poly | F | Drag to draw a freehand closed polygon |
| Edit | E | Edit polygon vertices (drag/right-click delete/double-click add) |
| Delete | DEL | Delete selected annotation objects |

### 10.3 Workflow

**Step 1: Open an image folder**

Click **📁 Open Directory** in the right panel and select a folder containing images.

The system automatically runs depth estimation for each image and displays three status markers in the file list:
- 📄 Raw image (no annotations)
- ✏️ Modified but not yet saved
- ✅ Has a saved JSON annotation file

**Step 2: Manage categories**

In the **Categories** panel:
- **+ Add Class**: Create a new category (e.g., Leaf, Tomato, Stem)
- **Double-click a category**: Edit its name and color
- **Right-click a category**: Edit or delete

**Step 3: AI-assisted annotation with SAM**

1. First select the SAM model precision in the top bar and click **Load** to load the model
2. Select a category (e.g., Leaf) so it is highlighted in the category list
3. Choose one of the AI tools:

   - **SAM Point**: Left-click on the target (green marker) to add a positive prompt; right-click on the background (red marker) to add a negative prompt. A live segmentation preview is shown; double-click to confirm
   - **SAM Box**: Drag a rectangle around the target — SAM will automatically segment the primary object within the box
   - **Auto 3D**: Automatically locates foreground regions using the depth map and generates a segmentation result — no manual clicking required

**Step 4: Manual fine-tuning**

- **Edit tool** (E): Select an existing polygon, drag vertices to reshape, double-click an edge to insert a new vertex, right-click a vertex to delete it
- If AI results are unsatisfactory, use **Poly** or **Free Poly** to manually create polygons

**Step 5: Use depth slicing to assist annotation**

The **depth slice slider** in the top bar helps separate objects at different distances:
- Drag the two handles to set the depth range (0–255)
- Only pixels within the range are displayed; everything else is dimmed
- SAM automatically re-encodes the current depth slice when you release the slider
- Use the +/- buttons or `[` / `]` hotkeys for fine adjustments

**Step 6: Save annotations**

- Click **💾 Save JSON** to batch-save all modified annotations
- Each image's annotations are stored as a `.json` file (LabelMe format) with the same base name
- Use **🔄 YOLO Exporter** to convert all JSON files to YOLO segmentation format (generates `.txt` files)

### 10.4 Shortcut Reference

| Shortcut | Action |
|----------|--------|
| A / D | Previous / Next image |
| V | Select tool |
| H | Hand tool |
| Q | SAM Point tool |
| W | SAM Box tool |
| C | Polygon tool |
| F | Freehand Polygon tool |
| L | Magnetic Lasso tool |
| E | Edit tool |
| T | Auto 3D one-click segmentation |
| ESC | Cancel current action |
| DEL | Delete selected annotation object |
| S | Save JSON |
| [ / ] | Fine-tune depth slider |

> All shortcuts can be customized via the **⌨️ Hotkeys** button in the top bar.

---

## 11. AI Copilot

PhytoDA includes an AI assistant that understands natural-language intent and controls the software automatically.

### 11.1 Example Commands

**Global commands (work in any mode)**:

| You say | Copilot's action |
|---------|-----------------|
| "Switch to annotation mode" | Switches to Annotation Studio |
| "Load C:\images folder" | Imports all images from that folder |
| "Open D:\data\wheat.jpg" | Loads the specified single image |

**Auto Phenotyping commands**:

| You say | Copilot's action |
|---------|-----------------|
| "Run seed analysis" | Switches to Seed Analysis and starts |
| "Use the high precision model" | Switches to ViT-H and loads it |
| "Next image" | Navigates to the next image |
| "Make the text bigger and move it to the top" | Adjusts text size and position |
| "Delete ID 5" | Removes object with ID 5 from the data |
| "Reorder all IDs" | Renumbers all objects as 1, 2, 3... |
| "Hide color-related columns" | Hides color columns in the table |
| "Highlight the one with the largest area" | Highlights the largest-area row |
| "Export results" | Auto-exports images and CSV |
| "Clear workspace" | Removes all loaded images and results |

### 11.2 Tips

1. **Be specific**: The more precise your command, the better the AI performs. For example, "Delete ID 5" is better than "remove that wrong one"
2. **One action at a time**: Copilot processes one command per round; issue sequential commands for multi-step operations
3. **Inspect internal process**: Click "Show Internal Process" below an AI response bubble to expand the execution log for each step

---

## 12. Exporting Results

### 12.1 Auto Phenotyping Mode

**Manual export**:

Click **💾 Export Results** and choose a save location:
- **Single image**: A save-file dialog appears; you can customize the filename
- **Multiple images**: A folder selection dialog appears; all results are saved into that directory

Each result includes:
- `xxx_result.jpg`: Annotated visualization image
- `xxx_result.csv`: Table file with detailed per-object data

**Auto export via Copilot**:

Type "export results" in the Copilot dialog — the system automatically saves files to the source image directory.

### 12.2 Image Workbench Mode

Click **💾 Export CSV** to save:
- A PNG image containing all drawings
- A CSV file containing all measurement data

### 12.3 Annotation Studio Mode

- **💾 Save JSON**: Saves annotations in LabelMe JSON format, one `.json` per image
- **🔄 YOLO Exporter**: Converts all JSON files to YOLO segmentation format, generating `.txt` files and `classes.txt`

---

## 13. FAQ

### Q1: "Core Error: missing dependency" on startup

**Cause**: Python environment dependencies not installed correctly.

**Solution**:
```bash
pip install -r requirement.txt
```

If `segment_anything` or `depth_anything_v2` fails to import, verify that both folders exist under the project root and are complete.

### Q2: AI engine fails to load or gets stuck on loading

**Cause**: Model weight files are missing or at incorrect paths.

**Solution**:
1. Check that the `checkpoints/` folder contains the required `.pth` files
2. Verify the filenames match those listed in [Section 4](#4-downloading-model-weights) exactly
3. If VRAM is insufficient (< 4 GB), use Fast (ViT-B) mode

### Q3: Copilot reports "API Key not detected"

**Cause**: The `PhytoDA_API_KEY` environment variable is not set or is misspelled.

**Solution**:
1. Verify the variable name is exactly `PhytoDA_API_KEY` (case-sensitive)
2. Run `echo $env:PhytoDA_API_KEY` (Windows PowerShell) or `echo $PhytoDA_API_KEY` (Linux) to verify
3. If launching from an IDE, you may need to restart the IDE to pick up the new environment variable

### Q4: Some detected objects in the analysis are spurious

**Solution**:
1. Right-click the corresponding row in the Details table and choose "Delete This Object"
2. Or right-click the annotation label on the image and choose "Delete Object ID: X"
3. Summary statistics are automatically recalculated after deletion

### Q5: Annotation labels are too small to read

**Solution**: In Auto Phenotyping mode, adjust **Text size** (recommended 1.5–2.5) and **Thickness** (recommended 1.5–2.5) in the top toolbar of the image display area.

### Q6: SAM segmentation results are poor

**Solution**:
1. Try switching to a higher-precision SAM model (ViT-L or ViT-H)
2. In Annotation Studio, use the SAM Point tool to manually add positive/negative prompt points
3. Adjust the depth slice range to help the model focus on the correct foreground region

### Q7: UI display issues (fonts too large/small, blurry)

**Cause**: High-DPI (4K) scaling issues.

**Solution**: The application already sets `QT_AUTO_SCREEN_SCALE_FACTOR=1` in `main.py`. If issues persist, adjust the scaling setting in your system display settings.

---

> **Support**: For issues not covered in this manual, please report via the project's Issue tracker or discussion board, including error messages and environment details.
