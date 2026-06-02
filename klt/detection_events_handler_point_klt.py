from datetime import datetime
from typing import Dict, List, Tuple

import cv2
import supervision as sv
from ultralytics import YOLO

from src.logger.logger import LocalLogger
from src.bin.config_provider import ContainerConfig
from src.tracker.detection_events_handler import DetectionEventsHandler
from src.tracker.tracklet import Tracklet

from .point_klt_config import PointKLTConfig
from .point_klt_tracker import PointKLTPersonTracker, PointKLTStats


class PointKLTDetectionEventsHandler(DetectionEventsHandler):
    """
    Same handler style as the project DetectionEventsHandler, but person detections
    come from PointKLT instead of YOLO+ByteTrack.

    Kept project behavior:
      - FILTER_FRAMES early return stays;
      - ZIZ/PPE branch stays through legacy _track(frame);
      - Tracklet drawing uses tracklet.confirmed;
      - Zones.draw(frame, tracklets) is used;
      - no Tracklet.extrapolate() in the PointKLT path.
    """

    def __init__(
        self,
        model: YOLO,
        zones: Dict[int, List[sv.PolygonZone]],
        frame_hw: Tuple[int, int],
        logger: LocalLogger,
        container_config: ContainerConfig,
        verbose: bool = False,
        annotate_all_frames: bool = False,
        processing_rate: float = 0.3,  # accepted for caller compatibility; not passed to parent
        point_klt_config: PointKLTConfig | None = None,
        fps: float | None = None,
    ) -> None:
        super().__init__(
            model=model,
            zones=zones,
            frame_hw=frame_hw,
            logger=logger,
            container_config=container_config,
            verbose=verbose,
            annotate_all_frames=annotate_all_frames,
        )

        self.processing_rate = processing_rate

        self.point_klt_cfg = point_klt_config or PointKLTConfig()
        self.person_tracker = PointKLTPersonTracker(
            yolo_model=self.model,
            cfg=self.point_klt_cfg,
            frame_hw=frame_hw,
            logger=logger,
        )

        if fps is None:
            fps = self.container_config.stream_config.FPS
        self.person_tracker.set_fps(float(fps))

        self.point_klt_stats: PointKLTStats = self.person_tracker.stats
        self.motion_saved_frames = 0

    def process_frame(self, frame):
        self.frames_received += 1

        annotated_frame = frame.copy()
        motion_score = 1.0

        # Keep the original motion filter behavior: if no motion, skip processing.
        # Important difference from the old skipped-frame branch: no Tracklet.extrapolate() here.
        if self.container_config.events_config.FILTER_FRAMES:
            motion_score = self.frames_filter.process_frame(frame.copy())

            if motion_score < 0.00001:
                self.motion_saved_frames += 1
                self.frames_skipped += 1

                if self.annotate_all_frames:
                    annotated_frame = self._annotate_frame(
                        annotated_frame,
                        [],
                        self.tracklets,
                        motion_score,
                    )

                return annotated_frame, []

        events_tracklets = []

        # Person branch: PointKLT always runs on non-motion-skipped frames.
        # It decides internally whether this frame needs YOLO or only KLT.
        person_detections, point_stats = self.person_tracker.update(frame, allow_yolo=True)
        self.point_klt_stats = point_stats

        events_tracklets_zone, tracklets = self.detection_zone.process_frame(
            frame=frame,
            detections=person_detections,
            class_ids=[0],
            verbose=self.verbose,
        )
        events_tracklets += events_tracklets_zone
        self.tracklets = tracklets

        # ZIZ / PPE branch: keep the project behavior exactly by using the legacy _track(frame).
        if self.container_config.events_config.PROCESS_ZIZ:
            ppe_detections = self._track(frame)

            events_tracklets_ppe, tracklets_ppe = self.detection_ppe.process_frame(
                frame=frame,
                detections=ppe_detections,
                class_ids=[2],
                verbose=self.verbose,
            )

            events_tracklets += events_tracklets_ppe

        # Legacy counters now reflect the new split:
        # frames_processed = actual YOLO runs inside PointKLT;
        # frames_skipped = motion-skipped + KLT-saved frames.
        self.frames_processed = point_stats.yolo_runs
        self.frames_skipped = self.motion_saved_frames + max(
            0,
            point_stats.frames_received - point_stats.yolo_runs,
        )

        should_annotate = (
            bool(events_tracklets)
            or self.annotate_all_frames
            or (
                self.point_klt_cfg.force_annotate_when_debug
                and (self.point_klt_cfg.draw_debug_points or self.point_klt_cfg.draw_debug_status)
            )
        )

        if should_annotate:
            annotated_frame = self._annotate_frame(
                annotated_frame,
                events_tracklets,
                self.tracklets,
                motion_score,
            )

        return annotated_frame, events_tracklets

    def _saving_counters(self):
        stats = self.point_klt_stats

        total_frames = self.frames_received
        motion_saved = self.motion_saved_frames

        klt_frames = stats.frames_received
        klt_saved = max(0, klt_frames - stats.yolo_runs)
        total_saved = motion_saved + klt_saved

        def pct(value, denom):
            return 0.0 if denom <= 0 else 100.0 * value / denom

        return {
            "total_frames": total_frames,
            "yolo_runs": stats.yolo_runs,
            "student_runs": stats.student_runs,
            "motion_saved": motion_saved,
            "motion_saved_pct_total": pct(motion_saved, total_frames),
            "klt_saved": klt_saved,
            "klt_saved_pct_total": pct(klt_saved, total_frames),
            "klt_saved_pct_klt_frames": pct(klt_saved, klt_frames),
            "total_saved": total_saved,
            "total_saved_pct": pct(total_saved, total_frames),
            "klt_frames": klt_frames,
            "active_tracks": stats.active_tracks,
            "last_yolo_reason": stats.last_yolo_reason,
        }

    def _annotate_frame(
        self,
        frame,
        events_tracklets: List[Tracklet],
        tracklets: dict[int, Tracklet],
        motion_score: float,
    ):
        annotated_frame = frame.copy()

        # Internal PointKLT debug layer: bbox + points + statuses.
        # This is independent from event Tracklet drawing.
        if self.point_klt_cfg.draw_debug_points or self.point_klt_cfg.draw_debug_status:
            annotated_frame = self.person_tracker.draw_debug(annotated_frame)

        for tracklet in tracklets.values():
            event_tracklet = tracklet in events_tracklets

            if tracklet.confirmed:
                annotated_frame = tracklet.draw(
                    annotated_frame,
                    event_tracklet,
                )

        annotated_frame = self.danger_zones.draw(
            annotated_frame,
            tracklets,
        )

        counters = self._saving_counters()

        overlay_lines = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"motion_score: {motion_score:.7f}",
            f"yolo: {counters['yolo_runs']}/{counters['total_frames']}",
            f"save_klt: {counters['klt_saved']} ({counters['klt_saved_pct_total']:.1f}%)",
            f"save_motion: {counters['motion_saved']} ({counters['motion_saved_pct_total']:.1f}%)",
            f"save_total: {counters['total_saved']} ({counters['total_saved_pct']:.1f}%)",
            f"student: {counters['student_runs']}",
            f"tracks: {counters['active_tracks']}",
            f"last_yolo: {counters['last_yolo_reason']}",
            f"proc/skip: {self.frames_processed}/{self.frames_skipped}",
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        color = (0, 255, 0)
        line_spacing = 6
        padding = 5

        max_text_w = 0
        total_text_h = 0
        sizes = []

        for line in overlay_lines:
            (text_w, text_h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
            sizes.append((text_w, text_h, baseline))
            max_text_w = max(max_text_w, text_w)
            total_text_h += text_h + line_spacing

        total_text_h -= line_spacing
        box_w = max_text_w + 2 * padding
        box_h = total_text_h + 2 * padding

        x = annotated_frame.shape[1] - box_w - 10
        y = 30

        cv2.rectangle(
            annotated_frame,
            (x - padding, y - padding),
            (x - padding + box_w, y - padding + box_h),
            (0, 0, 0),
            -1,
        )

        cur_y = y
        for line, (_, text_h, _) in zip(overlay_lines, sizes):
            cv2.putText(
                annotated_frame,
                line,
                (x, cur_y + text_h),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            cur_y += text_h + line_spacing

        return annotated_frame
