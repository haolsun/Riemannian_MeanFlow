import os
os.environ['DISPLAY'] = ':10.0'
import trimesh
from pathlib import Path
# 加载 GLB 文件
source_dir = '/home/zhongzichen/code/RFM-Grasp-main/logs/wandb/offline-run-20260325_141238-ud36dtng/files/media/object3D/test'
glb_s = Path(source_dir)
scene = trimesh.Scene()
for glb in glb_s.glob("*"):
    mesh = trimesh.load(glb)
    mesh.show()
    # scene.add_geometry(mesh)
# 查看基本信息
# print(f"顶点数：{len(mesh.vertices)}")
# print(f"面数：{len(mesh.faces)}")
# print(f"边界框：{mesh.bounds}")

# 可视化（需要 X11 显示）
# scene.show()

# 或保存为其他格式
# mesh.export('output.obj')