#!/usr/bin/env python3

import os, time
import numpy as np
import cv2
import pyrealsense2 as rs
import gi, hailo
gi.require_version('Gst', '1.0')
from gi.repository import Gst

HEF_PATH     = "/home/zoqtmxhs/hailo-CLIP/hailo_clip_venv/lib/python3.11/site-packages/resources/yolov8m.hef"
YOLO_POST_SO = "/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so"
HAILO_FUNC   = "yolov8m"
W0, H0, FPS = 640, 480, 30
Wi, Hi      = 640, 640

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

def build_pipeline(): #데이터 처리 파이프라인 구축 (gstreamer)
    desc = f"""
      appsrc name=src is-live=true do-timestamp=true format=time !
      video/x-raw,format=RGB,width={Wi},height={Hi},framerate={FPS}/1 !
      hailonet hef-path="{HEF_PATH}" force-writable=true !
      hailofilter function-name={HAILO_FUNC} so-path="{YOLO_POST_SO}" !
      appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true
    """
    return Gst.parse_launch(desc)

def letterbox_params(W0, H0, Wi, Hi): #카메라 이미지를 ai모델용 이미지로..전처리
    s = min(Wi / float(W0), Hi / float(H0))
    w1 = int(round(W0 * s)); h1 = int(round(H0 * s))
    pad_x = (Wi - w1) / 2.0
    pad_y = (Hi - h1) / 2.0
    return s, pad_x, pad_y, w1, h1

def make_letterbox_rgb(color_bgr, Wi, Hi, s, pad_x, pad_y, w1, h1):
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w1, h1), interpolation=cv2.INTER_LINEAR)
    left = int(np.floor(pad_x)); right = int(np.ceil(pad_x))
    top  = int(np.floor(pad_y)); bottom = int(np.ceil(pad_y))
    if left + w1 + right != Wi:   right  += (Wi - (left + w1 + right))
    if top  + h1 + bottom != Hi:  bottom += (Hi - (top  + h1 + bottom))
    return np.pad(resized, ((top,bottom),(left,right),(0,0)), mode="constant")

def _get_num(v):
    if callable(v): v = v()
    return float(v)

def parse_dets(sample): #bb 데이터 사용하기 쉬운 데이터로 변환*****
    out = []
    buf = sample.get_buffer()
    roi = hailo.get_roi_from_buffer(buf) #데이터 추출
    dets = roi.get_objects_typed(hailo.HAILO_DETECTION) #객체 목록 가져오기
    for d in dets:
        b = d.get_bbox() #bb 좌표 추출
        xmin = getattr(b, 'xmin', None); ymin = getattr(b, 'ymin', None)
        xmax = getattr(b, 'xmax', None); ymax = getattr(b, 'ymax', None)
        if xmin is not None and xmax is not None and ymin is not None and ymax is not None:
            x1 = _get_num(xmin); y1 = _get_num(ymin)
            x2 = _get_num(xmax); y2 = _get_num(ymax)
        else:
            width_attr  = getattr(b, 'width',  None)
            height_attr = getattr(b, 'height', None)
            if xmin is None or ymin is None or not callable(width_attr) or not callable(height_attr):
                continue
            x1 = _get_num(xmin); y1 = _get_num(ymin)
            x2 = x1 + _get_num(width_attr); y2 = y1 + _get_num(height_attr)

        mx = max(abs(x1), abs(y1), abs(x2), abs(y2))
        if 0.0 < mx <= 2.0:
            x1 *= Wi; y1 *= Hi; x2 *= Wi; y2 *= Hi

        if x2 <= x1 or y2 <= y1:
            continue

        conf = float(d.get_confidence()) if hasattr(d,'get_confidence') else 0.0
        cls_id = -1
        if hasattr(d,'get_class_id'):
            try: cls_id = int(d.get_class_id())
            except: pass
        out.append((x1, y1, x2, y2, cls_id, conf))
    return out

def main():
    Gst.init(None)

    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_stream(rs.stream.depth, W0, H0, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, W0, H0, rs.format.bgr8, FPS)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()

    pipeline = build_pipeline()
    appsrc   = pipeline.get_by_name("src")
    appsink  = pipeline.get_by_name("sink")
    pipeline.set_state(Gst.State.PLAYING)

    s, pad_x, pad_y, w1, h1 = letterbox_params(W0, H0, Wi, Hi)

    cv2.namedWindow("YOLO + Depth", cv2.WINDOW_NORMAL)

    print("RUN: Press Q to quit.")
    try:
        while True:
            frames = pipe.wait_for_frames()
            frames = align.process(frames)
            df = frames.get_depth_frame(); cf = frames.get_color_frame()
            if not df or not cf: continue
            depth = np.asanyarray(df.get_data())
            color = np.asanyarray(cf.get_data())

            rgb = make_letterbox_rgb(color, Wi, Hi, s, pad_x, pad_y, w1, h1)
            buf = Gst.Buffer.new_allocate(None, rgb.nbytes, None)
            buf.fill(0, rgb.tobytes())
            pts = time.monotonic_ns()
            buf.pts = pts; buf.dts = pts
            appsrc.emit("push-buffer", buf)

            sample = appsink.emit("pull-sample")
            dets = parse_dets(sample) if sample else []

            vis = color.copy() #원본 이미지 복사
            lines = []
            for (x1m,y1m,x2m,y2m,cls,conf) in dets: #원본 카메라 좌표로 돌리기
                x1 = int(round((x1m - pad_x)/s)); y1 = int(round((y1m - pad_y)/s))
                x2 = int(round((x2m - pad_x)/s)); y2 = int(round((y2m - pad_y)/s))
                x1 = max(0,min(W0-1,x1)); y1 = max(0,min(H0-1,y1))
                x2 = max(0,min(W0-1,x2)); y2 = max(0,min(H0-1,y2))
                if x2 <= x1 or y2 <= y1: continue

                #------------------추가
                grid_x = np.linspace(x1, x2, 5, dtype=int)[1:-1]
                grid_y = np.linspace(y1, y2, 5, dtype=int)[1:-1]
                points = [(int(x), int(y)) for y in grid_y for x in grid_x]

                distances = []
                for gy in grid_y:
                    for gx in grid_x:
                        d = float(depth[gy, gx]) * depth_scale 
                        distances.append(d)
                        cv2.circle(vis, (gx, gy), 3, (0, 0, 255), -1)
                
                valid_distances = [d for d in distances if d > 0]

                dist_m = -1.0 # 기본값은 -1 (측정 불가)
                if len(valid_distances) >= 3:
                    # 3개 이상이면 가장 가까운 3개의 평균 계산
                    valid_distances.sort()
                    dist_m = (valid_distances[0] + valid_distances[1] + valid_distances[2]) / 3.0
                elif len(valid_distances) > 0:
                    # 1~2개라도 있으면 그냥 그 값들의 평균을 사용
                    dist_m = np.mean(valid_distances)
                
                label = f"id{cls} {dist_m:.2f} m"
                #------------------추가

                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2) #화면에 그리기
                cv2.putText(vis, label, (x1, max(0,y1-7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("YOLO + Depth", vis)
            if dets:
                print(f"[detections: {len(dets)}] " + ", ".join(lines))
            else:
                print("[detections: 0]")

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q'), ord('Q')): break

    finally:
        try: appsrc.emit("end-of-stream")
        except: pass
        pipeline.set_state(Gst.State.NULL)
        pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()