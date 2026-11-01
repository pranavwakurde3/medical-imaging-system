import io, os, sys, time, sqlite3
import numpy as np, requests, pydicom, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
from matplotlib import colormaps

ORTHANC = "http://localhost:8042"
DB = "data/results.db"
os.makedirs("data/results", exist_ok=True)

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
plain = transforms.Compose([transforms.ToTensor(), norm])

m = models.densenet121(weights=None)
m.classifier = nn.Linear(1024, 1)
m.load_state_dict(torch.load("models/densenet_pneumonia.pt", map_location="cpu"))
m.eval()

con = sqlite3.connect(DB)
con.execute("""CREATE TABLE IF NOT EXISTS results (
    study_id TEXT PRIMARY KEY, patient_id TEXT, score REAL, overlay_path TEXT,
    status TEXT DEFAULT 'pending', decision TEXT, decided_at REAL, created_at REAL)""")
con.commit()


def preprocess(ds):
    a = ds.pixel_array.astype("float32")
    lo, hi = np.percentile(a, [1, 99])
    a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
    return Image.fromarray((a * 255).astype("uint8")).resize((224, 224)).convert("RGB")


def gradcam(img):
    x = plain(img).unsqueeze(0)
    m.zero_grad()
    r = F.relu(m.features(x)); r.retain_grad()
    out = m.classifier(torch.flatten(F.adaptive_avg_pool2d(r, 1), 1))
    out.sum().backward()
    w = r.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * r).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear")[0, 0].detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, torch.sigmoid(out).item()


def process(study_id):
    inst = requests.get(f"{ORTHANC}/studies/{study_id}/instances").json()[0]["ID"]
    raw = requests.get(f"{ORTHANC}/instances/{inst}/file").content
    ds = pydicom.dcmread(io.BytesIO(raw))
    img = preprocess(ds)
    cam, score = gradcam(img)
    heat = (colormaps["jet"](cam)[..., :3] * 255).astype("uint8")
    overlay = Image.blend(img, Image.fromarray(heat), 0.4)
    path = f"data/results/{study_id}.png"
    overlay.save(path)
    con.execute("INSERT OR REPLACE INTO results (study_id, patient_id, score, overlay_path, created_at) VALUES (?,?,?,?,?)",
                (study_id, str(ds.get("PatientID", "")), score, path, time.time()))
    con.commit()
    print(f"scored {study_id[:8]}... score {score:.3f}")


def run_once():
    done = {r[0] for r in con.execute("SELECT study_id FROM results")}
    new = [s for s in requests.get(f"{ORTHANC}/studies").json() if s not in done]
    for s in new:
        process(s)
    return len(new)


if __name__ == "__main__":
    if "--once" in sys.argv:
        print("processed", run_once(), "new studies")
    else:
        while True:
            run_once(); time.sleep(5)
