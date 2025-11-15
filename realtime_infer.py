"""

مثال على الداتا الي رح يستقبلها من jeson
علمنا انه القيم الي رح نحاتاجها في السينسور هي التسارع والجيروسكوب
ax = التسارع     gx = الجيروسكوب
{
  "samples": [
    {"ax": -0.12, "ay": 0.98, "az": 0.05, "gx": 0.01, "gy": -0.02, "gz": 0.00},
    {"ax": -0.13, "ay": 0.97, "az": 0.06, "gx": 0.01, "gy": -0.02, "gz": 0.00}
    // ...
  ]
}

كود التشغيل من التيرمينال
python realtime_json_infer.py ^
  --ckpt .\runs\tcn_hier\exp3\best.pt ^
  --device cpu ^
  --json_file .\windows\window.json

"""



import argparse
import json
from collections import deque

import numpy as np
import torch

from infer_tcn_hier import TCNHier, FINE_CLASS_NAMES, aggregate_with_gate

# إعدادات عامة
N_CHANNELS = 6
WINDOW_SIZE = 100  #عدد العينات الي رح يستقبلها المودل عشان يعطي نتيجه
FEATURE_KEYS = ["ax", "ay", "az", "gx", "gy", "gz"]


# ---------- 1) تحميل المودل من ckpt ----------
def load_model(ckpt_path, device="auto"):
    if device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    ckpt = torch.load(ckpt_path, map_location=device_t)
    cargs = ckpt.get("args", {})

    widths = tuple(int(x) for x in str(cargs.get("widths", "64,64,96,96")).split(","))
    kernel = int(cargs.get("kernel", 5))
    pooling = cargs.get("pooling", "gap")

    print(f"[INFO] Using device: {device_t}")
    print(f"[INFO] From ckpt -> widths={widths}, kernel={kernel}, pooling={pooling}")

    model = TCNHier(
        in_ch=N_CHANNELS,
        fine_classes=len(FINE_CLASS_NAMES),
        widths=widths,
        kernel=kernel,
        drop=0.0,
        causal=True,
        pooling=pooling,
    ).to(device_t)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, device_t


# ---------- 2) تحويل JSON -> عينة واحدة [6] ----------
def json_to_sample(obj):
    """
    obj ممكن يكون:
      - dict فيه المفاتيح ax,ay,az,gx,gy,gz
      - list فيها 6 أرقام
    ترجع list طولها 6
    """
    if isinstance(obj, dict):
        if all(k in obj for k in FEATURE_KEYS):
            vals = [obj[k] for k in FEATURE_KEYS]
        elif "values" in obj:
            vals = obj["values"]
        else:
            raise ValueError(f"JSON dict لا يحتوي على المفاتيح المتوقعة: {obj}")
    elif isinstance(obj, list):
        vals = obj
    else:
        raise ValueError(f"نوع JSON غير مدعوم: {type(obj)}")

    if len(vals) != N_CHANNELS:
        raise ValueError(f"العينة يجب أن تحتوي {N_CHANNELS} قيم، لكن فيها {len(vals)}")

    return [float(v) for v in vals]


# ---------- 3) تجهيز نافذة [T,C] -> [1,C,T] ----------
def prepare_window(window_2d):
    arr = np.asarray(window_2d, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D window, got shape {arr.shape}")

    # arr: [T,C]  (100,6)
    if arr.shape[1] == N_CHANNELS:
        ct = arr.T  # -> [C,T]
    elif arr.shape[0] == N_CHANNELS:
        ct = arr
    else:
        raise ValueError(f"One dimension must be {N_CHANNELS}, got {arr.shape}")

    ct = np.nan_to_num(ct, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.from_numpy(ct[None, ...])  # [1,C,T]
    return x


# ---------- 4) inference لنافذة واحدة ----------
def infer_single_window(window_2d, model, device, gate_thr=0.6):
    x = prepare_window(window_2d).to(device)

    with torch.no_grad():
        fine_logits, coarse_logits = model(x)  # [1,11], [1,2]

    pred_id, p_fall, pred_prob = aggregate_with_gate(
        fine_logits, coarse_logits,
        mode="logits_mean",
        gate_thr=float(gate_thr),
        min_conf=0.0,
    )

    pred_name = FINE_CLASS_NAMES[pred_id]
    return pred_id, pred_name, float(pred_prob), float(p_fall)


# ---------- 5) stream JSON من Serial (رِيال-تايم) ----------
def json_serial_stream(port="COM5", baudrate=115200):
    import serial

    ser = serial.Serial(port, baudrate, timeout=1)
    print(f"[INFO] Listening on {port} @ {baudrate} ...")

    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # لو السطر مش JSON صالح نتجاهله
            continue

        try:
            sample = json_to_sample(obj)  # [6]
        except Exception as e:
            print(f"[WARN] Invalid sample: {e}")
            continue

        yield sample


# ---------- 6) رِيال-تايم: تجميع نافذة من stream ----------
def run_realtime_json(ckpt_path, port="COM5", baudrate=115200,
                      device="auto", gate_thr=0.6, stride=5):
    model, device_t = load_model(ckpt_path, device=device)
    buf = deque(maxlen=WINDOW_SIZE)

    step = 0
    for sample in json_serial_stream(port, baudrate):
        buf.append(sample)

        if len(buf) < WINDOW_SIZE:
            step += 1
            continue

        if step % stride == 0:
            window = np.array(buf, dtype=np.float32)  # [T,C] = [100,6]
            pred_id, pred_name, pred_prob, p_fall = infer_single_window(
                window, model, device_t, gate_thr=gate_thr
            )

            is_fall_class = (pred_id >= 8)   # indices 8..10 = falls
            is_fall_event = is_fall_class and (p_fall >= gate_thr)

            print(
                f"Pred: {pred_name} [{pred_id}] "
                f"(p={pred_prob:.3f})  p(FALL)={p_fall:.3f}  -> FALL_EVENT={is_fall_event}"
            )

        step += 1


# ---------- 7) inference من ملف JSON واحد ----------
def run_from_json_file(ckpt_path, json_path, device="auto", gate_thr=0.6):
    model, device_t = load_model(ckpt_path, device=device)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "samples" in data:
        samples_raw = data["samples"]
    else:
        samples_raw = data

    if len(samples_raw) < WINDOW_SIZE:
        raise ValueError(
            f"الملف يحتوي {len(samples_raw)} عينة فقط؛ نحتاج على الأقل {WINDOW_SIZE}"
        )

    # نأخذ آخر 100 عينة كنافذة
    last_samples = samples_raw[-WINDOW_SIZE:]
    samples = [json_to_sample(obj) for obj in last_samples]  # list[100][6]
    window = np.array(samples, dtype=np.float32)  # [T,C]

    pred_id, pred_name, pred_prob, p_fall = infer_single_window(
        window, model, device_t, gate_thr=gate_thr
    )

    is_fall_class = (pred_id >= 8)
    is_fall_event = is_fall_class and (p_fall >= gate_thr)

    print("---- JSON file inference ----")
    print(f"File: {json_path}")
    print(
        f"Pred: {pred_name} [{pred_id}] "
        f"(p={pred_prob:.3f})  p(FALL)={p_fall:.3f}  -> FALL_EVENT={is_fall_event}"
    )


# ---------- 8) CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to best.pt")
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--gate_thr", type=float, default=0.6)
    ap.add_argument("--stride", type=int, default=5, help="خطوة الإنزلاق بين النوافذ في الرِيال-تايم")
    ap.add_argument("--port", default="COM5", help="Serial port للرِيال-تايم")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--json_file", help="لو حددته: يعمل inference على ملف JSON ويخرج")
    args = ap.parse_args()

    if args.json_file:
        run_from_json_file(
            ckpt_path=args.ckpt,
            json_path=args.json_file,
            device=args.device,
            gate_thr=args.gate_thr,
        )
    else:
        run_realtime_json(
            ckpt_path=args.ckpt,
            port=args.port,
            baudrate=args.baud,
            device=args.device,
            gate_thr=args.gate_thr,
            stride=args.stride,
        )


if __name__ == "__main__":
    main()
