import open3d as o3d
import numpy as np


def visualize_point_cloud(pt1, pt2=None):

    # 将 NumPy 数组转换为 Open3D 的 PointCloud 对象
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pt1)
    pcd1.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0] for _ in range(pt1.shape[0])], dtype=np.float64))

    # 创建一个可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 将点云添加到可视化窗口
    vis.add_geometry(pcd1)

    if pt2 is not None:
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pt2)
        pcd2.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1] for _ in range(pt2.shape[0])], dtype=np.float64))
        vis.add_geometry(pcd2)

    # 运行可视化
    vis.run()