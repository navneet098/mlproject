from pathlib import Path
import hashlib, json, platform, subprocess, sys
import importlib.metadata as md

ROOT = Path(__file__).resolve().parents[1]
def git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

packages = {}
for name in ["numpy","pandas","scikit-learn","matplotlib","joblib","requests","streamlit","PyYAML"]:
    try: packages[name] = md.version(name)
    except md.PackageNotFoundError: packages[name] = None

artifacts = {}
for folder in ["models","checkpoints","results"]:
    for p in (ROOT/folder).rglob("*"):
        if p.is_file() and p.name != "provenance.json":
            artifacts[str(p.relative_to(ROOT))] = {"bytes": p.stat().st_size, "sha256": sha256(p)}

record = {
    "python": sys.version,
    "platform": platform.platform(),
    "packages": packages,
    "git_commit": git("rev-parse", "HEAD"),
    "git_log": git("log", "-5", "--format=%H|%cI|%s"),
    "git_status": git("status", "--short"),
    "artifact_hashes": artifacts
}
(ROOT/"results").mkdir(exist_ok=True)
(ROOT/"results"/"provenance.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
