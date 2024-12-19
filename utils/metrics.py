from pathlib import Path

import numpy as np

from utils import SimpleClass


def compute_translation_error_np(pred_Ts, gt_Ts):
    """
    计算平移误差。

    Args:
        pred_Ts (np.ndarray): 预测平移矩阵，形状为 (-1, 3)。
        gt_Ts (np.ndarray): 真实平移矩阵，形状为 (-1, 3)。

    Returns:
        np.ndarray: 平移误差，形状为 (-1,)。
    """
    return np.linalg.norm(pred_Ts - gt_Ts, axis=1)


def compute_rotation_error_np(pred_Rs, gt_Rs):
    """
    计算旋转误差。

    Args:
        pred_Rs (np.ndarray): 预测旋转矩阵，形状为 (-1, 9)，每行是展平的旋转矩阵。
        gt_Rs (np.ndarray): 真实旋转矩阵，形状为 (-1, 9)。

    Returns:
        np.ndarray: 旋转误差（弧度），形状为 (-1,)。
    """
    pred_Rs = pred_Rs.reshape(-1, 3, 3)  # 恢复为 (-1, 3, 3)
    gt_Rs = gt_Rs.reshape(-1, 3, 3)

    R_diff = np.matmul(pred_Rs.transpose(0, 2, 1), gt_Rs)  # R_pred^T * R_gt
    trace = np.trace(R_diff, axis1=1, axis2=2)  # 计算迹
    trace = np.clip(trace, -1.0, 3.0)  # 防止数值误差导致 acos 超出定义域
    return np.arccos((trace - 1) / 2)


def compute_ADD_np(pred_Rs, pred_Ts, gt_Rs, gt_Ts, points):
    """
    计算几何一致性 ADD 指标。

    Args:
        pred_Rs (np.ndarray): 预测旋转矩阵，形状为 (b, 9)。
        pred_Ts (np.ndarray): 预测平移矩阵，形状为 (b, 3)。
        gt_Rs (np.ndarray): 真实旋转矩阵，形状为 (b, 9)。
        gt_Ts (np.ndarray): 真实平移矩阵，形状为 (b, 3)。
        points (np.ndarray): 物体点云，形状为 (b, n, 3)。

    Returns:
        np.ndarray: ADD 指标，形状为 (-1,)。
    """
    pred_Rs = pred_Rs.reshape(-1, 3, 3)  # 恢复为 (-1, 3, 3)
    gt_Rs = gt_Rs.reshape(-1, 3, 3)

    # 预测点云位姿变换
    pred_points = (points - pred_Ts[:, np.newaxis, :]) @ pred_Rs
    # 真实点云位姿变换
    gt_points = (points - gt_Ts[:, np.newaxis, :]) @ gt_Rs

    # 计算 L2 距离
    add_error = np.linalg.norm(pred_points - gt_points, axis=2).mean(axis=1)  # (b,)
    return add_error


class Metric(SimpleClass):
    """
    Class for computing evaluation metrics for YOLOv8 model.

    Attributes:
        p (list): Precision for each class. Shape: (nc,).
        r (list): Recall for each class. Shape: (nc,).
        f1 (list): F1 score for each class. Shape: (nc,).
        all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
        ap_class_index (list): Index of class for each AP score. Shape: (nc,).
        nc (int): Number of classes.

    Methods:
        ap50(): AP at IoU threshold of 0.5 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        ap(): AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        mp(): Mean precision of all classes. Returns: Float.
        mr(): Mean recall of all classes. Returns: Float.
        map50(): Mean AP at IoU threshold of 0.5 for all classes. Returns: Float.
        map75(): Mean AP at IoU threshold of 0.75 for all classes. Returns: Float.
        map(): Mean AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: Float.
        mean_results(): Mean of results, returns mp, mr, map50, map.
        class_result(i): Class-aware result, returns p[i], r[i], ap50[i], ap[i].
        maps(): mAP of each class. Returns: Array of mAP scores, shape: (nc,).
        fitness(): Model fitness as a weighted combination of metrics. Returns: Float.
        update(results): Update metric attributes with new evaluation results.
    """

    def __init__(self) -> None:
        """Initializes a Metric instance for computing evaluation metrics for the YOLOv8 model."""
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0

    @property
    def ap50(self):
        """
        Returns the Average Precision (AP) at an IoU threshold of 0.5 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50 values per class, or an empty list if not available.
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """
        Returns the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50-95 values per class, or an empty list if not available.
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """
        Returns the Mean Precision of all classes.

        Returns:
            (float): The mean precision of all classes.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """
        Returns the Mean Recall of all classes.

        Returns:
            (float): The mean recall of all classes.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.5.

        Returns:
            (float): The mAP at an IoU threshold of 0.5.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.75.

        Returns:
            (float): The mAP at an IoU threshold of 0.75.
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """
        Returns the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05.

        Returns:
            (float): The mAP over IoU thresholds of 0.5 - 0.95 in steps of 0.05.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """Mean of results, return mp, mr, map50, map."""
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i):
        """Class-aware result, return p[i], r[i], ap50[i], ap[i]."""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self):
        """MAP of each class."""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self):
        """Model fitness as a weighted combination of metrics."""
        w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (np.array(self.mean_results()) * w).sum()

    def update(self, results):
        """
        Updates the evaluation metrics of the model with a new set of results.

        Args:
            results (tuple): A tuple containing the following evaluation metrics:
                - p (list): Precision for each class. Shape: (nc,).
                - r (list): Recall for each class. Shape: (nc,).
                - f1 (list): F1 score for each class. Shape: (nc,).
                - all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
                - ap_class_index (list): Index of class for each AP score. Shape: (nc,).

        Side Effects:
            Updates the class attributes `self.p`, `self.r`, `self.f1`, `self.all_ap`, and `self.ap_class_index` based
            on the values provided in the `results` tuple.
        """
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            self.p_curve,
            self.r_curve,
            self.f1_curve,
            self.px,
            self.prec_values,
        ) = results

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []

    @property
    def curves_results(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return [
            [self.px, self.prec_values, "Recall", "Precision"],
            [self.px, self.f1_curve, "Confidence", "F1"],
            [self.px, self.p_curve, "Confidence", "Precision"],
            [self.px, self.r_curve, "Confidence", "Recall"],
        ]


class PoseMetric(SimpleClass):
    """
    Class for storing pose evaluation metrics results:
    - re: rotation errors
    - te: translation errors
    - add: ADD errors
    - ps: predicted pose scores
    """

    def __init__(self) -> None:
        """Initializes a Metric instance for computing evaluation metrics for the YOLOv8 model."""
        self.re = []
        self.te = []
        self.add = []
        self.ps = []

    def update(self, results):
        """
        Updates the evaluation metrics of the model with a new set of results.

        Args:
            results (tuple): A tuple containing:
                - re (np.ndarray): rotation errors
                - te (np.ndarray): translation errors
                - add (np.ndarray): ADD errors
                - ps (np.ndarray): predicted pose scores
        """
        self.re, self.te, self.add, self.ps = results

    def fitness(self):
        """
        Model fitness as a weighted combination of metrics.
        Here we assume fitness should reward high pose score and penalize large errors.
        You can adjust weights as needed.
        """
        if len(self.re) == 0:
            return 0.0
        mean_re = np.mean(self.re)
        mean_te = np.mean(self.te)
        mean_add = np.mean(self.add)
        mean_ps = np.mean(self.ps) if len(self.ps) else 0.0
        # Example: fitness = mean_ps - (mean_re + mean_te + mean_add)
        return mean_ps - (mean_re + mean_te + mean_add)


class PoseMetrics(SimpleClass):
    """
    Utility class for computing pose estimation metrics such as rotation error, translation error, and ADD.

    Attributes:
        save_dir (Path): A path to the directory where the output plots will be saved.
        plot (bool): A flag that indicates whether to plot additional metrics or not.
        on_plot (func): An optional callback for plot handling.
        names (dict of str): A dict of strings for class names if needed.
        pose (PoseMetric): An instance of PoseMetric class for storing pose metrics results.

    Methods:
        process(pred_Rs, pred_Ts, gt_Rs, gt_Ts, points, pose_scores):
            Process predicted results for pose estimation and update metrics.
        pred_pose_score:
            Returns the average pose score of the predicted poses.
        keys:
            Returns a list of keys for accessing specific metrics.
        mean_results:
            Returns mean values for translation error, rotation error, ADD, and predicted pose score.
        fitness:
            Returns the fitness of the pose estimation.
        results_dict:
            Returns a dictionary that maps pose metric keys to their computed values.
    """

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=None) -> None:
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "pose"
        self.pose = PoseMetric()

    def process(self, pred_Rs, pred_Ts, gt_Rs, gt_Ts, points, pose_scores):
        """
        Process predicted results for pose estimation and update metrics.

        Args:
            pred_Rs (np.ndarray): Predicted rotations, shape (-1, 9)
            pred_Ts (np.ndarray): Predicted translations, shape (-1, 3)
            gt_Rs (np.ndarray): Ground truth rotations, shape (-1, 9)
            gt_Ts (np.ndarray): Ground truth translations, shape (-1, 3)
            points (np.ndarray): Object 3D model points, shape (n, 3)
            pose_scores (np.ndarray): Pose confidence scores, shape (-1,)

        Updates:
            self.pose: updates pose metrics internal arrays.
        """
        if len(pred_Rs) == 0:
            # No predictions, no updates
            return
        te = compute_translation_error_np(pred_Ts, gt_Ts)
        re = compute_rotation_error_np(pred_Rs, gt_Rs)
        add = compute_ADD_np(pred_Rs, pred_Ts, gt_Rs, gt_Ts, points)
        ps = pose_scores
        self.pose.update((re, te, add, ps))

    @property
    def pred_pose_score(self):
        """Returns the average pose score of the predicted bounding boxes."""
        return np.mean(self.pose.ps) if len(self.pose.ps) else 0.0

    @property
    def keys(self):
        """Returns a list of keys for accessing specific metrics."""
        return ["metrics/trans_err", "metrics/rot_err", "metrics/ADD_err", "metrics/score"]

    @property
    def mean_results(self):
        """Returns a list of mean values for the computed detection metrics."""
        if len(self.pose.te) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [np.mean(self.pose.te), np.mean(self.pose.re), np.mean(self.pose.add), self.pred_pose_score]

    @property
    def fitness(self):
        """Model fitness as a weighted combination of metrics."""
        return self.pose.fitness()

    @property
    def results_dict(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return dict(zip(self.keys + ["metrics/fitness"], self.mean_results + [self.fitness]))
