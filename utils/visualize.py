import cv2
import numpy as np
import open3d as o3d


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


def visualize_pose_bbox(results, save_dir):
    rgb = results['image']
    pred_rot = results['pred_Rs'].reshape((-1, 3, 3))
    pred_trans = results['pred_Ts'].reshape((-1, 3))
    model_points = results['whole_model_points']
    K = results['camera_k'].reshape((3, 3))

    # draw_detections返回numpy数组格式的图像
    img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
    img = np.uint8(img)  # 确保类型为uint8

    save_path = save_dir / 'vis_pose_bbox.png'
    cv2.imwrite(save_path, img)

    prediction = cv2.imread(save_path, cv2.IMREAD_COLOR)

    rgb = np.uint8(rgb)

    concat = np.concatenate((rgb, prediction), axis=1)
    return concat


def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input:
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


def get_3d_bbox(scale, shift=0):
    """
    Input:
        scale: [3] or scalar
        shift: [3] or scalar
    Return
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                            [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                            [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                            [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                            [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                            [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                            [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                            [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                            [scale / 2, +scale / 2, -scale / 2],
                            [-scale / 2, +scale / 2, scale / 2],
                            [-scale / 2, +scale / 2, -scale / 2],
                            [+scale / 2, -scale / 2, scale / 2],
                            [+scale / 2, -scale / 2, -scale / 2],
                            [-scale / 2, -scale / 2, scale / 2],
                            [-scale / 2, -scale / 2, -scale / 2]]) + shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d


def draw_3d_bbox(img, imgpts, color, size=3):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, size)

    # draw pillars in blue color
    color_pillar = (int(color[0] * 0.6), int(color[1] * 0.6), int(color[2] * 0.6))
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, size)

    # finally, draw top layer in color
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, size)
    return img


def draw_3d_pts(img, imgpts, color, size=1):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    for point in imgpts:
        img = cv2.circle(img, (point[0], point[1]), size, color, -1)
    return img


def draw_detections(image, pred_rots, pred_trans, model_points, intrinsics, color=(255, 0, 0)):
    num_pred_instances = len(pred_rots)
    draw_image_bbox = image.copy()

    for ind in range(num_pred_instances):
        # 3d bbox
        scale = (np.max(model_points[ind], axis=0) - np.min(model_points[ind], axis=0))
        shift = np.mean(model_points[ind], axis=0)
        bbox_3d = get_3d_bbox(scale, shift)

        # 3d point
        choose = np.random.choice(np.arange(len(model_points[ind])), 512)
        pts_3d = model_points[ind][choose].T
        # draw 3d bounding box
        transformed_bbox_3d = pred_rots[ind] @ bbox_3d + pred_trans[ind][:, np.newaxis]
        projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
        draw_image_bbox = draw_3d_bbox(draw_image_bbox, projected_bbox, color)
        # draw point cloud
        transformed_pts_3d = pred_rots[ind] @ pts_3d + pred_trans[ind][:, np.newaxis]
        projected_pts = calculate_2d_projections(transformed_pts_3d, intrinsics)
        draw_image_bbox = draw_3d_pts(draw_image_bbox, projected_pts, color)

    return draw_image_bbox
