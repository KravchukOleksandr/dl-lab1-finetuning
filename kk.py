# pip install ultralytics opencv-python
from ultralytics import YOLO
import time, numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List

CLASS_MAP = {0: "male", 1: "female", 2: "staff", 3: "car"}

@dataclass
class LineZone:
    name: str
    A: Tuple[int, int]
    B: Tuple[int, int]   # «входная» сторона = левая относительно A->B

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

# ---- геометрия ----
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
    """
    Тот же интерфейс: process_frame(frame, ts=None) -> List[dict]
    Трекинг через model.track(persist=True, tracker='bytetrack.yaml').
    """
    def __init__(self, model_path:str, lines:List[LineZone], camera_id:int,
                 conf=0.4, iou=0.5, min_hits=3, max_age=30,
                 tracker_yaml: str = "bytetrack.yaml"):
        self.model = YOLO(model_path)
        self.lines = lines
        self.camera_id = camera_id
        self.conf, self.iou = conf, iou
        self.min_hits, self.max_age = min_hits, max_age
        self.tracker_yaml = tracker_yaml

        # состояние по track_id
        self.states: Dict[int, TrackState] = {}

    def _fmt_time(self, ts:float)->str:
        return time.strftime("%d.%m.%Y %H:%M:%S", time.localtime(ts))

    def _close_emit(self, st:TrackState, exit_zone:str, t_now:float)->Optional[dict]:
        if st.session is None:
            return None
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
        """
        На вход кадр (numpy array BGR/RGB — как в Ultralytics), опционально timestamp (секунды).
        На выход — список 0..N событий в формате таблицы.
        """
        t_now = ts if ts is not None else time.time()
        events: List[dict] = []

        # --- детекция+трекинг (persist=True сохраняет трекер между вызовами)
        results = self.model.track(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            stream=False,
            persist=True,
            verbose=False,
            tracker=self.tracker_yaml
        )
        # results — список из одного кадра
        if not results:
            return events
        r = results[0]

        seen: set[int] = set()

        # извлекаем боксы
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls  = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            ids  = boxes.id  # может быть None на первых кадрах
            ids  = None if ids is None else ids.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                class_id = int(cls[i])
                if class_id == 2:  # staff — полностью игнорируем
                    continue
                if ids is None:    # пока нет id — пропускаем
                    continue
                tid = int(ids[i])
                seen.add(tid)

                x1,y1,x2,y2 = xyxy[i]
                cx, cy = (x1+x2)/2.0, (y1+y2)/2.0

                st = self.states.get(tid)
                if st is None:
                    st = TrackState(class_id=class_id)
                    self.states[tid] = st

                st.class_id = class_id
                st.last_seen_time = t_now
                st.confirmed_hits = min(self.min_hits, st.confirmed_hits + 1)

                # класс 3: считаем только появление/исчезновение
                if class_id == 3:
                    if st.session is None and st.confirmed_hits >= self.min_hits:
                        self._open(st, "None", t_now)
                    st.last_point = (cx, cy); st.dead_count = 0
                    continue

                # появление → вход в None
                if st.session is None and st.confirmed_hits >= self.min_hits:
                    self._open(st, "None", t_now)

                # пересечения линий
                if st.last_point is not None and st.session is not None:
                    p_prev, p_cur = st.last_point, (cx, cy)
                    for L in self.lines:
                        if not seg_intersection(p_prev, p_cur, L.A, L.B):
                            continue
                        # выход всегда по зоне линии
                        ev = self._close_emit(st, exit_zone=L.name, t_now=t_now)
                        if ev: events.append(ev)
                        # направление: «вход» — если были слева от A->B
                        s_prev = np.sign(side(np.array(L.A), np.array(L.B), np.array(p_prev)))
                        self._open(st, L.name if s_prev>0 else "None", t_now)
                        break

                st.last_point = (cx, cy)
                st.dead_count = 0

        # --- исчезнувшие треки → выход в None
        for tid, st in list(self.states.items()):
            if tid in seen:
                continue
            st.dead_count += 1
            if st.dead_count > self.max_age:
                if st.session is not None:
                    ev = self._close_emit(st, "None", st.last_seen_time)
                    if ev: events.append(ev)
                del self.states[tid]

        return events
