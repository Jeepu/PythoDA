import sys
import os

# ---------------------------------------------------------
# 1. 全局环境路径配置 (必须在所有自定义包导入之前)
# ---------------------------------------------------------
# 获取当前 main.py 所在的绝对路径
root_dir = os.path.dirname(os.path.abspath(__file__))

# 将项目根目录加入路径 (为了能找到 core, ui, plugins)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 将第三方大模型目录显式加入最高优先级路径
sam_path = os.path.join(root_dir, 'segment_anything')
depth_path = os.path.join(root_dir, 'Depth_anything_v2')

if sam_path not in sys.path:
    sys.path.insert(0, sam_path)
if depth_path not in sys.path:
    sys.path.insert(0, depth_path)


from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

if __name__ == "__main__":
    # 高分屏适配 (防止界面模糊或太小)
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # 使用现代扁平化风格
    
    window = MainWindow()
    window.show()
    
    print(f"--> App running from: {root_dir}")
    sys.exit(app.exec())