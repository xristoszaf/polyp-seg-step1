import subprocess, cv2, kagglehub
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def build_dataset(pairs, out_dir, split="train"):
    for sub in ("images", "masks"): Path(f"{out_dir}/{sub}").mkdir(parents=True, exist_ok=True)
    rows = []
    for img_p, msk_p in pairs:
        img = cv2.imread(str(img_p))
        msk = ((cv2.imread(str(msk_p), 0) > 127).astype(np.uint8)) * 255
        name = f"{int(img_p.stem):04d}.png" if img_p.stem.isdigit() else f"{img_p.stem}.png"
        ip, mp = f"{out_dir}/images/{name}", f"{out_dir}/masks/{name}"
        cv2.imwrite(ip, img); cv2.imwrite(mp, msk)
        rows.append({"filename": name, "image_path": ip, "mask_path": mp, "split": split,
                     "height": img.shape[0], "width": img.shape[1], "polyp_ratio": float((msk > 0).mean())})
    return pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)

for d in ("manifests", "predictions", "results", "figures"): Path(d).mkdir(exist_ok=True)

# 1. Kvasir-SEG
subprocess.run("wget -q --no-check-certificate https://datasets.simula.no/downloads/kvasir-seg.zip -O /tmp/k.zip && unzip -qo /tmp/k.zip -d /tmp/k", shell=True, check=True)
k_imgs = sorted(p for p in Path("/tmp/k").rglob("*") if p.suffix.lower() in EXTS and p.parent.name == "images")
k_msks = {p.stem: p for p in Path("/tmp/k").rglob("*") if p.suffix.lower() in EXTS and p.parent.name == "masks"}
df_k = build_dataset([(p, k_msks[p.stem]) for p in k_imgs if p.stem in k_msks], "data/kvasir")
if not Path("manifests/kvasir_split.csv").exists():
    _, tmp = train_test_split(df_k, test_size=0.30, random_state=42)
    val, test = train_test_split(tmp, test_size=0.50, random_state=42)
    df_k.loc[val.index, "split"], df_k.loc[test.index, "split"] = "val", "test"
    df_k.to_csv("manifests/kvasir_split.csv", index=False)

# 2. CVC-ClinicDB (Kaggle)
cvc_root = Path(kagglehub.dataset_download("balraj98/cvcclinicdb"))
c_all = [p for p in cvc_root.rglob("*") if p.suffix.lower() in EXTS and "TIF" not in str(p)]
c_imgs = sorted(p for p in c_all if "original" in p.parent.name.lower())
c_msks = {p.stem: p for p in c_all if "ground" in p.parent.name.lower() or "mask" in p.parent.name.lower()}
df_c = build_dataset([(p, c_msks[p.stem]) for p in c_imgs if p.stem in c_msks], "data/cvc", "external_test")
if not Path("manifests/cvc_manifest.csv").exists():
    df_c.to_csv("manifests/cvc_manifest.csv", index=False)

print("Kvasir:", pd.read_csv("manifests/kvasir_split.csv")["split"].value_counts().to_dict())
print("CVC:", len(pd.read_csv("manifests/cvc_manifest.csv")), "| Mask values:", np.unique(cv2.imread(df_k.iloc[0]["mask_path"], 0)))
