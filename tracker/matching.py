import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist
from cython_bbox import bbox_overlaps as bbox_ious
from . import kalman_filter


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type btlbrs: list[tlbr] | np.ndarray
    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious


def iou_distance(atracks, btracks, buffer_scale=0.1):
    """
    Compute cost based on Buffered IoU (BIoU), which applies a buffer around
    bounding boxes to increase matching flexibility. This is referred to as C-BIoU
    in the AppMoTrack paper.
    
    :param atracks: list[STrack] or np.ndarray
    :param btracks: list[STrack] or np.ndarray
    :param buffer_scale: float, proportion to expand each bounding box
    :return: cost_matrix np.ndarray
    """

    # Convert tlbr to xywh for buffered expansion
    def tlbr_to_xywh(tlbr):
        x1, y1, x2, y2 = tlbr
        w, h = x2 - x1, y2 - y1
        x, y = x1 + w / 2, y1 + h / 2
        return [x, y, w, h]

    def xywh_to_tlbr(xywh):
        x, y, w, h = xywh
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x1 + w, y1 + h
        return [x1, y1, x2, y2]

    # Add buffer to a bounding box in xywh format
    def add_buffer(xywh, scale):
        x, y, w, h = xywh
        buffered_w = w * (1 + 2 * scale)
        buffered_h = h * (1 + 2 * scale)
        return [x, y, buffered_w, buffered_h]

    # Get tlbr bounding boxes
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    # Apply buffer
    atlbrs_buffered = [xywh_to_tlbr(add_buffer(tlbr_to_xywh(box), buffer_scale)) for box in atlbrs]
    btlbrs_buffered = [xywh_to_tlbr(add_buffer(tlbr_to_xywh(box), buffer_scale)) for box in btlbrs]

    # Compute IoU with buffered boxes
    _ious = ious(atlbrs_buffered, btlbrs_buffered)
    cost_matrix = 1 - _ious

    return cost_matrix


def feature_distance(atracks, btracks):
    """
    Calculates a cost matrix based on appearance feature cosine distances.
    
    :param atracks: list[STrack]
    :param btracks: list[STrack]
    :return: cost_matrix np.ndarray
    """
    
    # Extract features from tracks and detections
    track_feats = np.array([track.feature for track in atracks if track.feature is not None])
    det_feats = np.array([det.feature for det in btracks if det.feature is not None])
    
    # Handle cases with no features
    if len(track_feats) == 0 or len(det_feats) == 0:
        return np.full((len(atracks), len(btracks)), float('inf'), dtype=float)

    # Compute cosine distance matrix
    cost_matrix = cdist(track_feats, det_feats, 'cosine')
    
    return cost_matrix


def fuse_score(cost_matrix, detections, tracks, scale_factor=0.001):
    """
    Applies the uncertainty-aware cost function adjustment described in AppMoTrack.
    It fuses IoU similarity, detection confidence, and Kalman filter covariance.
    
    :param cost_matrix: The initial cost matrix (e.g., from combined appearance/motion).
    :param detections: list[STrack]
    :param tracks: list[STrack]
    :param scale_factor: A scaling factor for the covariance uncertainty.
    :return: fuse_cost np.ndarray
    """
    if cost_matrix.size == 0:
        return cost_matrix

    iou_sim = 1 - cost_matrix
    
    # Detection scores
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)

    # Track covariance uncertainty
    cov_uncertainty = np.array([np.sum(np.diag(track.covariance)[:2]) for track in tracks])
    cov_uncertainty_normalized = np.tanh(scale_factor * cov_uncertainty)
    cov_uncertainty_normalized = np.expand_dims(cov_uncertainty_normalized, axis=1).repeat(cost_matrix.shape[1], axis=1)

    # Fuse similarity with detection scores and covariance uncertainty
    fuse_sim = iou_sim * det_scores * (1 - 0.1 * cov_uncertainty_normalized)
    fuse_cost = 1 - fuse_sim

    return fuse_cost
