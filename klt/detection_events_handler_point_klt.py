from datetime import datetime
from typing import Dict, List, Tuple

import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO

from src.logger.logger import LocalLogger
from src.bin.config_provider import ContainerConfig
from src.tracker.detection_events_handler import DetectionEventsHandler
from src.tracker.tracklet import Tracklet

from .point_klt_config import PointKLTConfig
from .point_klt_tracker import PointKLTPersonTracker, PointKLTStats


class PointKLTDetectionEventsHandler(DetectionEventsHandler):
    """
    Drop-in alternative to DetectionEventsHandler for person events.

    External contract stays the same:
        annotated_frame, events = handler.process_frame(frame)

    Internally:
        persons: YOLO sometimes + KLT/student between YOLO refreshes
        zones/events: existing DangerZone/Tracklet logic
        PPE: optional legacy path, left unchanged if PROCESS_PIZ=True
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
        processing_rate: float = 0.3,
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
            processing_rate=processing_rate,
        )

        self.point_klt_cfg = point_klt_config or PointKLTConfig()
        self.person_tracker = PointKLTPersonTracker(
            yolo_model=self.model,
            cfg=self.point_klt_cfg,
            frame_hw=frame_hw,
            logger=logger,
        )

        if fps is None:
            fps = getattr(self.container_config.stream_config, "FPS", 10)
        self.person_tracker.set_fps(float(fps))

        self.point_klt_stats: PointKLTStats = self.person_tracker.stats

        # Extra savings counters at the integration-handler level.
        # PointKLTStats counts only frames actually submitted to PointKLT.
        # motion_saved_frames counts frames skipped before PointKLT by FramesFilter.
        self.motion_saved_frames = 0

    def process_frame(self, frame):
        self.frames_received += 1

        annotated_frame = frame.copy()
        motion_score = 1.0

        # Keep original FILTER_FRAMES behavior: if the ROI is static, skip all
        # expensive processing for this frame. Important difference from the
        # old handler: we do NOT extrapolate Tracklets here. We simply freeze
        # the current event-layer state for this frame.
        if self.container_config.events_config.FILTER_FRAMES:
            motion_score = self.frames_filter.process_frame(frame.copy())

            if motion_score < 0.00001:
                self.motion_saved_frames += 1
                self.frames_skipped += 1

                # Same spirit as the original handler: on motion-skip we still
                # draw the current known state/zones/overlay, but we do not call
                # detection_zone.process_frame(...) with empty detections.
                annotated_frame = self._annotate_frame(
                    annotated_frame,
                    [],
                    self.tracklets,
                    motion_score,
                )
                return annotated_frame, []

        events_tracklets = []

        # PointKLT is now the person tracker. It is called on frames that pass
        # the motion filter. It decides internally whether YOLO is needed or KLT
        # is enough.
        person_detections, point_stats = self.person_tracker.update(frame=frame, allow_yolo=True)
        self.point_klt_stats = point_stats

        events_tracklets_zone, tracklets = self.detection_zone.process_frame(
            frame=frame,
            detections=person_detections,
            class_ids=[0],
            verbose=self.verbose,
        )
        events_tracklets += events_tracklets_zone
        self.tracklets = tracklets

        # Keep old PPE behavior for now. This may run legacy YOLO/ByteTrack if enabled.
        if self.container_config.events_config.PROCESS_PIZ:
            ppe_detections = self._track(frame=frame)
            events_tracklets_ppe = self.detection_ppe.process_frame(
                frame=frame,
                detections=ppe_detections,
                class_ids=[2],
                verbose=self.verbose,
            )
            events_tracklets += events_tracklets_ppe

        # Legacy counters are kept for compatibility, but the meaningful numbers
        # are printed by _saving_counters().
        self.frames_processed = point_stats.yolo_runs
        self.frames_skipped = self.motion_saved_frames + max(0, point_stats.frames_received - point_stats.yolo_runs)

        should_annotate = (
            bool(events_tracklets)
            or self.annotate_all_frames
            or (self.point_klt_cfg.force_annotate_when_debug and (
                self.point_klt_cfg.draw_debug_points or self.point_klt_cfg.draw_debug_status
            ))
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

        # Frames that passed motion filter and were handled by PointKLT without YOLO.
        klt_frames = stats.frames_received
        klt_saved = max(0, klt_frames - stats.yolo_runs)

        total_saved = motion_saved + klt_saved

        def pct(x, denom):
            if denom <= 0:
                return 0.0
            return 100.0 * x / denom

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

        # First draw the internal point-KLT state. This is debug-level tracker state,
        # not event-level Tracklet state. It makes visible people outside zones and
        # KLT points even if Tracklet.draw() would hide static/unconfirmed objects.
        annotated_frame = self.person_tracker.draw_debug(annotated_frame)

        # Then draw the existing event-level Tracklet overlay exactly like the old handler.
        for tracklet in tracklets.values():
            event_tracklet = tracklet in events_tracklets
            if event_tracklet or not tracklet.static or not tracklet.unconfirmed:
                annotated_frame = tracklet.draw(annotated_frame, event_tracklet)

        annotated_frame = self.danger_zones.draw(annotated_frame)

        c = self._saving_counters()
        current_time = (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            + f"  motion_score: {motion_score:.7f}"
            + f"  yolo: {c['yolo_runs']}/{c['total_frames']}"
            + f"  save_klt: {c['klt_saved']} ({c['klt_saved_pct_total']:.1f}%)"
            + f"  save_motion: {c['motion_saved']} ({c['motion_saved_pct_total']:.1f}%)"
            + f"  save_total: {c['total_saved']} ({c['total_saved_pct']:.1f}%)"
            + f"  student: {c['student_runs']}"
            + f"  tracks: {c['active_tracks']}"
            + f"  last_yolo: {c['last_yolo_reason']}"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        color = (0, 255, 0)

        (text_w, text_h), baseline = cv2.getTextSize(current_time, font, font_scale, thickness)
        x, y = 10, 20

        cv2.rectangle(
            annotated_frame,
            (x - 3, y - text_h - 3),
            (x + text_w + 3, y + baseline + 3),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            annotated_frame,
            current_time,
            (x, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        return annotated_frame
