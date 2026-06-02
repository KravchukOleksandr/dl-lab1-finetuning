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

    def process_frame(self, frame):
        self.frames_received += 1

        annotated_frame = frame.copy()
        motion_score = 1.0

        # Important: do not skip PointKLT update entirely.
        # Motion filter may be used only as overlay/stat signal here.
        if self.container_config.events_config.FILTER_FRAMES:
            motion_score = self.frames_filter.process_frame(frame.copy())

        events_tracklets = []

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
        # If PROCESS_PIZ is False, the new person tracker is the only heavy path.
        if self.container_config.events_config.PROCESS_PIZ:
            ppe_detections = self._track(frame=frame)
            events_tracklets_ppe = self.detection_ppe.process_frame(
                frame=frame,
                detections=ppe_detections,
                class_ids=[2],
                verbose=self.verbose,
            )
            events_tracklets += events_tracklets_ppe

        self.frames_processed = point_stats.yolo_runs
        self.frames_skipped = max(0, self.frames_received - self.frames_processed)

        if events_tracklets or self.annotate_all_frames:
            annotated_frame = self._annotate_frame(
                annotated_frame,
                events_tracklets,
                self.tracklets,
                motion_score,
            )

        return annotated_frame, events_tracklets

    def _annotate_frame(
        self,
        frame,
        events_tracklets: List[Tracklet],
        tracklets: dict[int, Tracklet],
        motion_score: float,
    ):
        annotated_frame = frame.copy()

        for tracklet in tracklets.values():
            event_tracklet = tracklet in events_tracklets
            if event_tracklet or not tracklet.static or not tracklet.unconfirmed:
                annotated_frame = tracklet.draw(annotated_frame, event_tracklet)

        annotated_frame = self.danger_zones.draw(annotated_frame)

        stats = self.point_klt_stats
        current_time = (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            + f"  motion_score: {motion_score:.7f}"
            + f"  yolo: {stats.yolo_runs}/{stats.frames_received}"
            + f"  saved: {stats.yolo_saved_frames} ({stats.yolo_saved_pct:.1f}%)"
            + f"  student: {stats.student_runs}"
            + f"  tracks: {stats.active_tracks}"
            + f"  last_yolo: {stats.last_yolo_reason}"
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
