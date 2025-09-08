# pip install ultralytics supervision==0.20.0 opencv-python ocsort-bytestrack
from ultralytics import YOLO
import supervision as sv
import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List

CLASS_MAP = {0: "male", 1: "female", 2: "staff", 3: "car"}

@dataclass
class LineZone:
    name: str
    A: Tuple[int, int]
    B: Tuple[int, int]   # сторона входа = левая относительно A->B

@dataclass
class Session:
    entry_time: float
    entry_zone: str

@dataclass
class TrackState:
    class_id: int
    confirmed_hits: int = 0
    last_point: Optional[Tuple[float,float]] = None
    last_seen_time: float = 0.0
    dead_count: int = 0
    session: Optional[Session] = None

def cross(z,w): return z[0]*w[1]-z[1]*w[0]
def side(A,B,P): return cross(np.array(B)-A, np.array(P)-A)

def seg_intersection(p, r, q, s):
    p,r,q,s = map(np.array,[p,r,q,s])
    r2, s2 = r-p, s-q
    denom = cross(r2, s2)
    if denom == 0: return False
    t = cross(q-p, s2)/denom
    u = cross(q-p, r2)/denom
    return 0 <= t <= 1 and 0 <= u <= 1

class EventEngine:
    def __init__(self, model_path:str, lines:List[LineZone], camera_id:int,
                 conf=0.4, iou=0.5, min_hits=3, max_age=30, use_ocsort=True):
        self.model = YOLO(model_path)
        self.lines = lines
        self.camera_id = camera_id
        self.conf, self.iou = conf, iou
        self.min_hits, self.max_age = min_hits, max_age
        self.states: Dict[int, TrackState] = {}
        self.tracker = sv.tracker.OCSort(min_hits=min_hits, max_age=max_age) \
            if use_ocsort else sv.tracker.ByteTrack(track_buffer=max_age)

    def _fmt_time(self, ts:float)->str:
        return time.strftime("%d.%m.%Y %H:%M:%S", time.localtime(ts))

    def _close_emit(self, st:TrackState, exit_zone:str, t_now:float)->dict:
        if st.session is None: return None
        ev = {
            "object_type": CLASS_MAP.get(st.class_id, str(st.class_id)),
            "entry_time":  self._fmt_time(st.session.entry_time),
            "exit_time":   self._fmt_time(t_now),
            "zone_of_entry": st.session.entry_zone,
            "zone_of_exit":  exit_zone,
            "camera_id": self.camera_id
        }
        st.session = None
        return ev

    def _open(self, st:TrackState, entry_zone:str, t_now:float):
        st.session = Session(entry_time=t_now, entry_zone=entry_zone)

    def process_frame(self, frame, ts:Optional[float]=None)->List[dict]:
        """На вход кадр и, опционально, timestamp (секунды). Возвращает список событий."""
        t_now = ts if ts is not None else time.time()
        events: List[dict] = []

        # --- детекция
        y = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
        dets = []
        for b, cls, conf in zip(y.boxes.xyxy.cpu().numpy(),
                                y.boxes.cls.cpu().numpy(),
                                y.boxes.conf.cpu().numpy()):
            cls = int(cls)
            if cls == 2:   # staff игнорируем
                continue
            x1,y1,x2,y2 = map(float, b)
            cx, cy = (x1+x2)/2, (y1+y2)/2
            dets.append(sv.Detection(xyxy=b, class_id=cls, confidence=conf,
                                     data={"centroid":(cx,cy)}))

        tracked = self.tracker.update_with_detections(
            detections=sv.Detections.merge(dets)
        )

        seen = set()

        # --- обработка живых треков
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            cls = int(tracked.class_id[i])
            cx, cy = tracked.data["centroid"][i]
            seen.add(tid)

            st = self.states.get(tid) or TrackState(class_id=cls)
            self.states[tid] = st
            st.class_id = cls
            st.last_seen_time = t_now
            st.confirmed_hits = min(self.min_hits, st.confirmed_hits+1)

            # класс 3: только появление/исчезновение
            if cls == 3:
                if st.session is None and st.confirmed_hits >= self.min_hits:
                    self._open(st, "None", t_now)
                st.last_point = (cx, cy); st.dead_count = 0
                continue

            # появление: вход в None
            if st.session is None and st.confirmed_hits >= self.min_hits:
                self._open(st, "None", t_now)

            # пересечения линий
            if st.last_point is not None and st.session is not None:
                p_prev, p_cur = st.last_point, (cx, cy)
                for L in self.lines:
                    if not seg_intersection(p_prev, p_cur, L.A, L.B):
                        continue
                    # выход всегда по линии L
                    ev = self._close_emit(st, L.name, t_now)
                    if ev: events.append(ev)
                    # направление: «входная» = левая сторона до перехода
                    s_prev = np.sign(side(np.array(L.A), np.array(L.B), np.array(p_prev)))
                    self._open(st, L.name if s_prev>0 else "None", t_now)
                    break

            st.last_point = (cx, cy)
            st.dead_count = 0

        # --- умершие (исчезли) → выход в None
        for tid, st in list(self.states.items()):
            if tid in seen: continue
            st.dead_count += 1
            if st.dead_count > self.max_age:
                if st.session is not None:
                    ev = self._close_emit(st, "None", st.last_seen_time)
                    if ev: events.append(ev)
                del self.states[tid]

        return events


lines = [
    LineZone("Entrance 1", (100,200), (540,200)),
    LineZone("Entrance 2", (200,400), (700,400)),
]
engine = EventEngine("yolov8n.pt", lines, camera_id=1)

# В твоём цикле чтения камеры:
events = engine.process_frame(frame, ts=your_timestamp)
for row in events:
    # row уже готов для записи/отправки
    print(row)
