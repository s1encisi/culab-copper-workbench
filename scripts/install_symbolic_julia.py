import hashlib,json,subprocess,urllib.request,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
plan=json.loads((root/"configs/runtime/symbolic_julia.json").read_text())
import os
dependencies=Path(os.environ.get("COPPER_SYMBOLIC_RUNTIME_DIR",root/"runs/dependencies/pysr-1.5.9")).resolve().parent
archive=dependencies/("julia-"+plan["version"]+"-win64.zip")
partial=archive.with_suffix(".zip.part")
destination=dependencies/"julia-runtime"
executable=destination/("julia-"+plan["version"])/"bin/julia.exe"
def digest(path):
 h=hashlib.sha256()
 with path.open("rb") as stream:
  for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
 return h.hexdigest()
if not archive.exists():
 offset=partial.stat().st_size if partial.exists() else 0
 headers={"Range":f"bytes={offset}-"} if offset else {}
 with urllib.request.urlopen(urllib.request.Request(plan["url"],headers=headers),timeout=60) as response:
  if offset and response.status!=206:offset=0
  with partial.open("ab" if offset else "wb") as stream:
   size=offset;last=size
   while True:
    block=response.read(1024*1024)
    if not block:break
    stream.write(block);size+=len(block)
    if size-last>=16*1024*1024:
     print(json.dumps({"downloaded_bytes":size,"total_bytes":plan["size"]}),flush=True);last=size
 assert partial.stat().st_size==plan["size"]
 assert digest(partial)==plan["sha256"],"Julia download hash mismatch"
 partial.replace(archive)
assert digest(archive)==plan["sha256"]
if not executable.exists():
 destination.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(archive) as source:
  for member in source.infolist():
   assert (destination/member.filename).resolve().is_relative_to(destination.resolve()),member.filename
  source.extractall(destination)
result=subprocess.run([str(executable),"--version"],capture_output=True,text=True,check=True,
                      creationflags=subprocess.CREATE_NO_WINDOW)
record={**plan,"executable":str(executable),"verified_sha256":digest(archive),"version_output":result.stdout.strip(),"status":"installed"}
(dependencies/"julia_installation.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
print(json.dumps({"status":"installed","version":record["version_output"],"executable":str(executable)}),flush=True)
