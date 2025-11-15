import os, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FINE_CLASS_NAMES = [
    "walking","jogging","bending","hopping","lyingdownonabed",
    "sittinggettinguponachair","goupstairs","godownstairs",
    "fall_forward","fall_backward","fall_lateral"
]
FINE_NAME_TO_ID = {n:i for i,n in enumerate(FINE_CLASS_NAMES)}
ADL_IDS = set(range(0,8))
FALL_IDS = set(range(8,11))

def infer_label_from_filename(path):
    name = os.path.basename(path).lower().replace("-","_").replace(" ","_")
    for c in FINE_CLASS_NAMES:
        if c in name:
            return FINE_NAME_TO_ID[c]
    return None

# ---------- TCN blocks ----------
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1, drop=0.0, causal=True):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.pad_left = pad if causal else pad // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=0, dilation=dilation, bias=True)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=0, dilation=dilation, bias=True)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        if self.pad_left > 0:
            x1 = F.pad(x, (self.pad_left, 0))
        else:
            x1 = x
        y = self.drop(self.act(self.bn1(self.conv1(x1))))
        if self.pad_left > 0:
            y1 = F.pad(y, (self.pad_left, 0))
        else:
            y1 = y
        y = self.drop(self.act(self.bn2(self.conv2(y1))))
        res = x if self.downsample is None else self.downsample(x)
        return y + res

class TCNBackbone(nn.Module):
    def __init__(self, in_ch=6, widths=(64,64,96,96), kernel=5, drop=0.0, causal=True):
        super().__init__()
        layers, ch_in, dilation = [], in_ch, 1
        for ch_out in widths:
            layers.append(TemporalBlock(ch_in, ch_out, kernel_size=kernel, dilation=dilation, drop=drop, causal=causal))
            ch_in = ch_out
            dilation *= 2
        self.net = nn.Sequential(*layers)
        self.out_ch = ch_in
    def forward(self, x):  # [B,C,T]
        return self.net(x)

class TCNHier(nn.Module):
    def __init__(self, in_ch=6, fine_classes=11, widths=(64,64,96,96), kernel=5, drop=0.0, causal=True, pooling="gap"):
        super().__init__()
        self.backbone = TCNBackbone(in_ch, widths, kernel, drop, causal)
        self.pooling = pooling
        self.head_coarse = nn.Linear(self.backbone.out_ch, 2)
        self.head_fine   = nn.Linear(self.backbone.out_ch, fine_classes)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):  # [B,C,T]
        h = self.backbone(x)
        feat = h.mean(dim=-1) if self.pooling=="gap" else h[..., -1]
        return self.head_fine(feat), self.head_coarse(feat)

def load_windows(path, in_ch_hint=6):
    """ يقرأ ملف .npz/.pt ويعيد جميع النوافذ كشكل [N,C,T] (أو [1,C,T] لو نافذة واحدة) + الليبل إن وجد """
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        X = d.get("X", d.get("x", None))
        y = d.get("y", d.get("y_fine", d.get("labels", None)))
    elif path.endswith(".pt"):
        d = torch.load(path, map_location="cpu")
        X = d.get("X", d.get("x", None))
        y = d.get("y", d.get("y_fine", d.get("labels", None)))
        if isinstance(X, torch.Tensor): X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()
    else:
        raise ValueError("Only .npz/.pt are supported")

    if X is None:
        raise ValueError(f"Missing X/x in {path}")

    X = np.asarray(X)

    def to_nct(arr):
        # arr: [T,C] أو [C,T] أو [N,T,C] / [N,C,T]
        if arr.ndim == 2:
            T, C = arr.shape
            if C == in_ch_hint:
                ct = arr.T  # -> [C,T]
            else:
                # حاول الاستدلال
                ct = arr if arr.shape[0] == in_ch_hint else arr.T
            ct = ct.astype(np.float32)
            ct = np.nan_to_num(ct, nan=0.0, posinf=0.0, neginf=0.0)
            return ct[None, ...]  # [1,C,T]
        elif arr.ndim == 3:
            # [N,T,C] أو [N,C,T]
            if arr.shape[2] <= 16:
                nct = np.transpose(arr, (0, 2, 1))  # -> [N,C,T]
            else:
                nct = arr  # [N,C,T]
            nct = nct.astype(np.float32)
            nct = np.nan_to_num(nct, nan=0.0, posinf=0.0, neginf=0.0)
            return nct
        else:
            raise ValueError(f"Unsupported shape {arr.shape}")

    nct = to_nct(X)  # [N,C,T]
    y_true = None
    if y is None:
        y_true = infer_label_from_filename(path)
    else:
        y = np.asarray(y).reshape(-1)
        y_true = int(y[0])

    return torch.from_numpy(nct), y_true  # [N,C,T], int|None

# --------- التجميع + الـGating عبر النوافذ ---------
def aggregate_with_gate(fine_logits_all, coarse_logits_all, mode="logits_mean", gate_thr=0.6, min_conf=0.0):
    """
    fine_logits_all: [N,11]
    coarse_logits_all: [N,2]
    mode: logits_mean | probs_mean | majority
    gate_thr: -1 لتعطيل الـgating، وإلا [0..1]
    min_conf: يستخدم فقط مع majority لتجاهل النوافذ منخفضة الثقة
    """
    import torch
    import torch.nn.functional as F

    if mode == "logits_mean":
        cl_mean = coarse_logits_all.mean(dim=0)              # [2]
        p_coarse = F.softmax(cl_mean, dim=-1)                # [2]
        p_fall = float(p_coarse[1].item())

        fl_base = fine_logits_all.mean(dim=0)                # [11]
        if gate_thr >= 0:
            fl = fl_base.clone()
            if p_fall < gate_thr: fl[8:] = -1e9
            else:                 fl[:8] = -1e9
        else:
            fl = fl_base
        probs = F.softmax(fl, dim=-1)
        pred_id = int(probs.argmax().item())
        pred_p  = float(probs[pred_id].item())
        return pred_id, p_fall, pred_p

    elif mode == "probs_mean":
        cp = F.softmax(coarse_logits_all, dim=-1).mean(dim=0)  # [2]
        p_fall = float(cp[1].item())

        fp = F.softmax(fine_logits_all, dim=-1).mean(dim=0)    # [11]
        if gate_thr >= 0:
            fp = fp.clone()
            if p_fall < gate_thr:
                fp[8:] = 0.0
                s = fp[:8].sum().clamp_min(1e-12); fp[:8] /= s
            else:
                fp[:8] = 0.0
                s = fp[8:].sum().clamp_min(1e-12); fp[8:] /= s
        pred_id = int(fp.argmax().item())
        pred_p  = float(fp[pred_id].item())
        return pred_id, p_fall, pred_p

    else:  # "majority"
        cp_all = F.softmax(coarse_logits_all, dim=-1)          # [N,2]
        p_fall = float(cp_all[:,1].mean().item())

        fl = fine_logits_all.clone()
        if gate_thr >= 0:
            if p_fall < gate_thr: fl[:, 8:] = -1e9   # حصر ضمن ADL
            else:                fl[:, :8] = -1e9    # حصر ضمن FALL
        fp = F.softmax(fl, dim=-1)                   # [N,11]

        # فلترة حسب الثقة إن لزم
        if min_conf > 0.0:
            conf, pred = fp.max(dim=1)               # [N], [N]
            keep = conf >= min_conf
            if keep.any():
                pred = pred[keep]
            else:
                pred = fp.argmax(dim=1)
        else:
            pred = fp.argmax(dim=1)

        counts = torch.bincount(pred, minlength=11)  # [11]
        pred_id = int(counts.argmax().item())
        pred_p  = float((counts.float() / counts.sum().clamp_min(1.0))[pred_id].item())
        return pred_id, p_fall, pred_p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to best.pt")
    ap.add_argument("--file", help="single window file (.npz/.pt)")
    ap.add_argument("--folder", help="folder with .npz/.pt windows")
    ap.add_argument("--device", default="cuda", choices=["cpu","cuda","auto"])
    ap.add_argument("--limit", type=int, default=0, help="max files from folder (0=all)")
    ap.add_argument("--csv_out", type=str, default="", help="optional save predictions to CSV")
    # الإضافات:
    ap.add_argument("--gate_thr", type=float, default=0.6, help="عتبة قرار السقوط من الرأس الثنائي لقطع أصناف fine (استعمل -1 لتعطيله)")
    ap.add_argument("--infer_bs", type=int, default=2048, help="حجم باتش inference لتجميع النوافذ على دفعات")
    ap.add_argument("--agg", type=str, default="logits_mean",
                    choices=["logits_mean","probs_mean","majority"],
                    help="طريقة التجميع عبر النوافذ")
    ap.add_argument("--min_conf", type=float, default=0.0,
                    help="حد الثقة لتجاهل النوافذ (يُستخدم مع majority فقط). 0 لتعطيله")
    args = ap.parse_args()

    # device
    if args.device=="auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device=="cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA not available, using CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # load ckpt & model params
    ckpt = torch.load(args.ckpt, map_location=device)
    cargs = ckpt.get("args", {})
    widths  = tuple(int(x) for x in str(cargs.get("widths","64,64,96,96")).split(","))
    kernel  = int(cargs.get("kernel",5))
    pooling = cargs.get("pooling","gap")
    print(f"[INFO] From ckpt -> widths={widths}, kernel={kernel}, pooling={pooling}")

    model = TCNHier(in_ch=6, fine_classes=11, widths=widths, kernel=kernel, drop=0.0, causal=True, pooling=pooling).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    def predict_one(p):
        # X_windows: [N,C,T] (كل النوافذ)
        X_windows, y_true = load_windows(p, in_ch_hint=6)
        X_windows = X_windows.to(device)
        N = X_windows.shape[0]

        # مرر على دفعات واجمع اللوجِتس
        bs = max(1, int(args.infer_bs))
        all_fine_logits = []
        all_coarse_logits = []
        with torch.no_grad():
            for i in range(0, N, bs):
                xb = X_windows[i:i+bs]  # [B,C,T]
                lf, lc = model(xb)      # logits: [B,11], [B,2]
                all_fine_logits.append(lf.detach().cpu())
                all_coarse_logits.append(lc.detach().cpu())

        fine_logits_all   = torch.cat(all_fine_logits, dim=0)    # [N,11]
        coarse_logits_all = torch.cat(all_coarse_logits, dim=0)  # [N,2]

        # تجميع + Gating بحسب الإعدادات
        pred_id, p_fall, pred_p = aggregate_with_gate(
            fine_logits_all, coarse_logits_all,
            mode=args.agg, gate_thr=float(args.gate_thr), min_conf=float(args.min_conf)
        )
        pred_name = FINE_CLASS_NAMES[pred_id]
        return y_true, pred_id, pred_name, pred_p, p_fall

    rows = []
    if args.file:
        y_true, pred_id, pred_name, pred_prob, p_fall = predict_one(args.file)
        print("\n=== Single file ===")
        print(f"File: {args.file}")
        if y_true is not None:
            print(f"GT (if known): {FINE_CLASS_NAMES[y_true]} [{y_true}]")
        print(f"Pred fine: {pred_name} [{pred_id}]  (p={pred_prob:.3f})")
        print(f"p(FALL): {p_fall:.3f}  (gate_thr={args.gate_thr}, agg={args.agg})\n")

    if args.folder:
        paths = sorted(glob.glob(os.path.join(args.folder, "*.npz")) + glob.glob(os.path.join(args.folder, "*.pt")))
        if args.limit>0:
            paths = paths[:args.limit]
        print(f"[INFO] Testing on {len(paths)} files from: {args.folder}")
        n_has_gt = 0
        n_correct = 0
        for i,p in enumerate(paths,1):
            y_true, pred_id, pred_name, pred_prob, p_fall = predict_one(p)
            rows.append((p, y_true, pred_id, pred_name, pred_prob, p_fall))
            if y_true is not None:
                n_has_gt += 1
                if y_true == pred_id: n_correct += 1
            if i%5000==0:
                print(f"  ...{i} files")

        if n_has_gt>0:
            acc = n_correct / n_has_gt
            print(f"[RESULT] fine top-1 acc = {acc:.3f}  (on {n_has_gt} files with GT)")
        else:
            print("[RESULT] No ground-truth found in filenames/files; printed predictions only.")

        if args.csv_out:
            import csv
            with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["file","y_true","pred_id","pred_name","pred_prob","p_fall"])
                for r in rows:
                    w.writerow(r)
            print(f"[INFO] CSV saved: {args.csv_out}")

if __name__ == "__main__":
    main()
