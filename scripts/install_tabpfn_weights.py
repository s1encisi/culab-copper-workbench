"""Download the pinned public TabPFN v2 weights into the private local runtime cache."""
from pathlib import Path
import hashlib,json,os,urllib.request
root=Path(__file__).resolve().parents[1]
plan=json.loads((root/"configs/runtime/tabpfn_weights.json").read_text())
runtime=Path(os.environ.get("COPPER_TABPFN_RUNTIME_DIR",root/"runs/dependencies/tabpfn-2.1.3"))
target=runtime.parent/"tabpfn-weights"/plan["filename"]
target.parent.mkdir(parents=True,exist_ok=True)
if not target.exists():
    url=f"https://huggingface.co/{plan['repository']}/resolve/{plan['revision']}/{plan['filename']}"
    partial=target.with_suffix(".ckpt.part")
    with urllib.request.urlopen(url,timeout=60) as response,partial.open("wb") as output:
        for block in iter(lambda:response.read(1024*1024),b""):output.write(block)
    assert partial.stat().st_size==plan["size"]
    assert hashlib.sha256(partial.read_bytes()).hexdigest()==plan["sha256"]
    partial.replace(target)
assert hashlib.sha256(target.read_bytes()).hexdigest()==plan["sha256"]
(target.parent/"LICENSE.txt").write_bytes((root/"third_party/TabPFN_LICENSE.txt").read_bytes())
print(json.dumps({"status":"verified","sha256":plan["sha256"],"attribution":plan["attribution"]}))
