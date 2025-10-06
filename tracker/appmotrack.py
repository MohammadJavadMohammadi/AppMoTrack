import cv2
import numpy as np
import time
from collections import deque, OrderedDict
from PIL import Image

from . import matching
from .kalman_filter import KalmanFilter

def hellinger_fast(p, q):
    """Hellinger distance between two discrete distributions."""
    z = np.sqrt(p) - np.sqrt(q)
    return np.sqrt(z @ z / 2)

def image_desc(image_in):
    """Compute HSV histograms and concatenate them into a single descriptor."""
    hsv_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv_image], [0], None, [50], [0, 180])  # Hue
    hist_s = cv2.calcHist([hsv_image], [1], None, [60], [0, 256])  # Saturation

    # Normalize histograms
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()

    # Concatenate histograms
    return np.concatenate([hist_h, hist_s])

def compare_two_images(image1, image2):
    """Compare two images using their HSV histogram descriptors."""
    size1 = image1.shape[:2]
    size2 = image2.shape[:2]
    h = min(size1[0], size2[0])
    w = min(size1[1], size2[1])

    center1 = (size1[1] // 2, size1[0] // 2)
    center2 = (size2[1] // 2, size2[0] // 2)

    x1_start = max(center1[0] - w // 2, 0)
    y1_start = max(center1[1] - h // 2, 0)
    cropped_image1 = image1[y1_start:y1_start + h, x1_start:x1_start + w]

    x2_start = max(center2[0] - w // 2, 0)
    y2_start = max(center2[1] - h // 2, 0)
    cropped_image2 = image2[y2_start:y2_start + h, x2_start:x2_start + w]

    desc1 = image_desc(cropped_image1)
    desc2 = image_desc(cropped_image2)

    dist = (hellinger_fast(desc1, desc2)) / 10
    return dist

def extract_roi(image, box, img_width, img_height):
    """Extract ROI based on a bounding box."""
    x1, y1, x2, y2 = map(int, box[:4])
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width - 1))
    y2 = max(0, min(y2, img_height - 1))

    # Ensure valid box dimensions
    if x2 <= x1: x2 = x1 + 1
    if y2 <= y1: y2 = y1 + 1
    
    roi = image[y1:y2, x1:x2]
    return roi if roi.size > 0 else np.zeros((1, 1, 3), dtype=np.uint8)

def crop_to_central_75_percent(bbox):
    """Crop a bounding box to its central 75% area."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    new_w, new_h = w * 0.866, h * 0.866 # sqrt(0.75)
    new_x1, new_y1 = cx - new_w / 2, cy - new_h / 2
    new_x2, new_y2 = cx + new_w / 2, cy + new_h / 2
    return new_x1, new_y1, new_x2, new_y2

def hsv_distance(tracks, detections):
    """Calculates a cost matrix based on HSV feature distances."""
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix

    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            if track.roi is not None and det.roi is not None:
                cost_matrix[i, j] = compare_two_images(track.roi, det.roi)
            else:
                cost_matrix[i, j] = float('inf')
    return cost_matrix

class GMC:
    """Global Motion Compensation using sparse optical flow."""
    def __init__(self, downscale=4):
        self.downscale = max(1, int(downscale))
        self.feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3)
        self.prevFrame = None
        self.prevKeyPoints = None
        self.initialized = False

    def apply(self, raw_frame):
        height, width = raw_frame.shape[:2]
        frame_gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        
        if self.downscale > 1:
            frame = cv2.resize(frame_gray, (width // self.downscale, height // self.downscale))
        else:
            frame = frame_gray

        H = np.eye(2, 3)
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        if not self.initialized:
            self.prevFrame = frame
            self.prevKeyPoints = keypoints
            self.initialized = True
            return H

        matchedKeypoints, status, _ = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)
        status = status.ravel() > 0
        prevPoints = self.prevKeyPoints[status]
        currPoints = matchedKeypoints[status]

        if len(prevPoints) > 4:
            H, _ = cv2.estimateAffinePartial2D(prevPoints, currPoints, method=cv2.RANSAC)
            if self.downscale > 1:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        
        self.prevFrame = frame
        self.prevKeyPoints = keypoints
        return H

def process_bboxes_optimized(bboxes, img, ipca, lda_model, img_width, img_height):
    """Optimized function to extract features using the IPCA-LDA pipeline."""
    rois = []
    for box in bboxes:
        roi = extract_roi(img, box, img_width, img_height)
        if roi.size > 0:
            roi = cv2.resize(roi, (10, 30))
            rois.append(roi)
    
    if not rois:
        return np.array([])
        
    rois = np.array(rois).transpose(0, 3, 1, 2).reshape(len(rois), -1)
    
    # Apply PCA and LDA transformations
    pca_features = ipca.transform(rois)
    lda_features = lda_model.transform(pca_features)
    return lda_features

class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class STrack:
    _count = 0
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feature, roi):
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.score = score
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        self.tracklet_len = 0
        
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None

        self.feature = feature
        self.roi = roi

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        STrack._count += 1
        return STrack._count

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        self.feature = new_track.feature
        self.roi = new_track.roi
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        self.feature = new_track.feature
        self.roi = new_track.roi
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    def mark_lost(self): self.state = TrackState.Lost
    def mark_removed(self): self.state = TrackState.Removed
    @staticmethod
    def clear_count(): STrack._count = 0

    def predict(self):
        if self.state != TrackState.Tracked:
            self.mean[6:8] = 0
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if not stracks: return
        multi_mean = np.array([st.mean for st in stracks])
        multi_covariance = np.array([st.covariance for st in stracks])
        
        mask = np.array([st.state != TrackState.Tracked for st in stracks])
        multi_mean[mask, 6:8] = 0

        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, st in enumerate(stracks):
            st.mean, st.covariance = multi_mean[i], multi_covariance[i]

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if not stracks: return
        R = H[:2, :2]
        R8x8 = np.kron(np.eye(4, dtype=float), R)
        t = H[:2, 2]
        
        for st in stracks:
            st.mean = R8x8.dot(st.mean)
            st.mean[:2] += t
            st.covariance = R8x8.dot(st.covariance).dot(R8x8.T)

    @property
    def tlwh(self):
        if self.mean is None: return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @staticmethod
    def tlwh_to_xywh(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self): return self.tlwh_to_xywh(self.tlwh)
    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

class AppMoTrack:
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        STrack.clear_count()

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = KalmanFilter()
        self.gmc = GMC()

    def update(self, output_results, img):
        self.frame_id += 1
        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        scores = output_results[:, 4] if output_results.shape[1] >= 5 else np.array([])
        bboxes = output_results[:, :4] if output_results.shape[1] >= 4 else np.empty((0, 4))
        
        # Filter detections by score
        remain_inds = scores > self.args.track_low_thresh
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        
        # Create detection objects
        img_h, img_w = img.shape[:2]
        detections = []
        if len(dets) > 0:
            features = process_bboxes_optimized(dets, img, self.args.loaded_ipca, self.args.lda_model, img_w, img_h)
            roi_list = [extract_roi(img, crop_to_central_75_percent(item), img_w, img_h) for item in dets]
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f, r) for (tlbr, s, f, r) in zip(dets, scores_keep, features, roi_list)]

        # Separate tracks
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked_stracks = [t for t in self.tracked_stracks if t.is_activated]

        # --- First Association (High-confidence) ---
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        
        # GMC
        warp = self.gmc.apply(img)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)
        
        # Filter high-confidence detections
        high_conf_inds = [i for i, d in enumerate(detections) if d.score >= self.args.track_high_thresh]
        detections_high = [detections[i] for i in high_conf_inds]

        if detections_high:
            ious_dists = matching.iou_distance(strack_pool, detections_high, 0.4)
            feat_dists = matching.feature_distance(strack_pool, detections_high)
            total_dists = 0.8 * ious_dists + 0.2 * feat_dists
            total_dists = matching.fuse_score(total_dists, detections_high, strack_pool)

            # Adaptive thresholding with K-means
            th = 0.7
            if total_dists.size > 1:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(total_dists.flatten().reshape(-1, 1))
                centers = np.sort(kmeans.cluster_centers_.flatten())
                th = (centers[0] + centers[1]) / 2
            
            matches, u_track, u_detection = matching.linear_assignment(total_dists, thresh=th)

            # Hierarchical HSV matching for unmatched
            if u_track.size > 0 and u_detection.size > 0:
                unmatched_tracks = [strack_pool[i] for i in u_track]
                unmatched_dets = [detections_high[i] for i in u_detection]
                hsv_dists = hsv_distance(unmatched_tracks, unmatched_dets)
                
                # Apply IoU gating
                iou_gate = matching.iou_distance(unmatched_tracks, unmatched_dets)
                hsv_dists[iou_gate > self.args.proximity_thresh] = 1.0
                
                new_matches, _, _ = matching.linear_assignment(hsv_dists, thresh=0.03)
                
                # Map new matches back to original indices and combine
                new_matches_global = np.array([[u_track[t], u_detection[d]] for t, d in new_matches])
                if new_matches_global.size > 0:
                    matches = np.vstack((matches, new_matches_global))
            
            # Update matched tracks
            for i, (itracked, idet) in enumerate(matches):
                track = strack_pool[itracked]
                det = detections_high[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id)
                    refind_stracks.append(track)
            
            # Update unmatched lists based on combined matches
            matched_track_indices = set(matches[:, 0])
            matched_det_indices = set(matches[:, 1])
            u_track = [i for i in range(len(strack_pool)) if i not in matched_track_indices]
            u_detection = [i for i, d in enumerate(detections) if d.score >= self.args.track_high_thresh and i not in matched_det_indices]
        else:
            u_track = list(range(len(strack_pool)))
            u_detection = []

        # --- Second Association (Low-confidence) ---
        low_conf_inds = [i for i, d in enumerate(detections) if self.args.track_low_thresh < d.score < self.args.track_high_thresh]
        detections_low = [detections[i] for i in low_conf_inds]
        
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
        if r_tracked_stracks and detections_low:
            ious_dists = matching.iou_distance(r_tracked_stracks, detections_low, 0.5)
            matches, u_track_low, _ = matching.linear_assignment(ious_dists, thresh=0.5)
            
            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = detections_low[idet]
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            
            for it in u_track_low:
                track = r_tracked_stracks[it]
                if track.state != TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)

        # --- Handle Unconfirmed Tracks ---
        u_detection_final = [detections[i] for i in u_detection]
        
        if unconfirmed and u_detection_final:
            ious_dists = matching.iou_distance(unconfirmed, u_detection_final, 0.4)
            matches, u_unconfirmed, u_detection_unconfirmed = matching.linear_assignment(ious_dists, thresh=0.7)
            
            for itracked, idet in matches:
                unconfirmed[itracked].update(u_detection_final[idet], self.frame_id)
                activated_stracks.append(unconfirmed[itracked])
            
            for it in u_unconfirmed:
                unconfirmed[it].mark_removed()
                removed_stracks.append(unconfirmed[it])
            
            u_detection_final = [u_detection_final[i] for i in u_detection_unconfirmed]

        # --- Initialize new tracks ---
        for det in u_detection_final:
            if det.score > self.args.new_track_thresh:
                det.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(det)

        # --- Update state and merge ---
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        return [t for t in self.tracked_stracks if t.is_activated]

def joint_stracks(tlista, tlistb):
    exists = {t.track_id for t in tlista}
    res = tlista[:]
    res.extend(t for t in tlistb if t.track_id not in exists)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {t.track_id: t for t in tlista}
    for t in tlistb:
        stracks.pop(t.track_id, None)
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb, 0.0)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = set(), set()
    for p, q in zip(*pairs):
        if stracksa[p].frame_id - stracksa[p].start_frame > stracksb[q].frame_id - stracksb[q].start_frame:
            dupb.add(q)
        else:
            dupa.add(p)
    return [t for i, t in enumerate(stracksa) if i not in dupa], [t for i, t in enumerate(stracksb) if i not in dupb]
