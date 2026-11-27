from pathlib import Path
import shutil,subprocess,re,zipfile,tarfile,json,hashlib,os
base=Path(__file__).resolve().parent;root=base/'rootfs'
if root.exists():shutil.rmtree(root)
root.mkdir()
def cp(src,dest=None):
 p=root/(dest or str(src).lstrip('/'));p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p,follow_symlinks=True);return p
# POSIX shell and basic initialization utilities are required by the evaluator.
for utility in ['dash','env','mkdir','cp','rm','chmod','cat']:
 cp('/usr/bin/'+utility)
(root/'bin').mkdir(exist_ok=True)
(root/'bin/sh').symlink_to('/usr/bin/dash')
(root/'usr/bin/sh').symlink_to('dash')
cp('/usr/bin/python3.13');(root/'usr/bin/python3').symlink_to('python3.13')
z=root/'usr/lib/python313.zip';z.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as out:
 for p in Path('/usr/lib/python3.13').rglob('*.py'):
  if any(x in p.parts for x in ('__pycache__','test','tests','idlelib','tkinter','ensurepip','venv','lib-dynload')):continue
  out.write(p,str(p.relative_to('/usr/lib/python3.13')))
for p in Path('/usr/lib/python3.13/lib-dynload').glob('*.so'):
 if any(x in p.name for x in ('_tkinter','_curses','_sqlite3','readline','_dbm','_gdbm','_ssl','_hashlib')):continue
 cp(p)
np=Path('/usr/lib/python3/dist-packages/numpy')
for p in np.rglob('*'):
 if p.name.startswith('_simd.'):continue
 if not p.is_file() or any(x in p.relative_to(np).parts for x in ('tests','__pycache__','include','_core/include')) or p.suffix in ('.pyc','.a','.h','.pxd','.pyi','.csv','.gz'):continue
 cp(p)
for p in list(root.rglob('*.so'))+[p for p in (root/'usr/bin').iterdir() if p.is_file() and not p.is_symlink()]:
 for line in subprocess.run(['ldd',str(p)],capture_output=True,text=True).stdout.splitlines():
  m=re.search(r'(\/[^\s]+)',line)
  if m and Path(m[1]).exists():cp(m[1])
shutil.copytree(base/'src'/'baselines',root/'app/baselines',ignore=shutil.ignore_patterns('__pycache__','tests','docs'))
shutil.copytree(base/'src'/'sdk'/'contest_harness',root/'app/contest_harness',ignore=shutil.ignore_patterns('__pycache__'))
for n in ['contest_submission.py','harness.py','drift.py','adaptive_train.py']:cp(base/'src'/n,'app/'+n)
for d in ['workspace','tmp','model','data','state','proc','dev']: (root/d).mkdir(exist_ok=True)
(root/'tmp').chmod(0o1777);(root/'workspace').chmod(0o777)
# Upstream copyright and license notices accompany the image.
for n in ['python3.13','python3-numpy','libc6','libstdc++6','libblas3','liblapack3','libssl3t64','zlib1g','dash','coreutils','libselinux1','libpcre2-8-0','libacl1']:
 p=Path('/usr/share/doc')/n/'copyright'
 if p.exists():cp(p,'licenses/'+n+'.txt')
(root/'app/PROJECT.txt').write_text('工业场景自适应模型自进化系统\nDriftGuard-IM 为副标题。\n')
layer=base/'layer.tar'
def norm(t):t.uid=t.gid=0;t.uname=t.gname='root';t.mtime=0;return t
with tarfile.open(layer,'w',format=tarfile.PAX_FORMAT) as t:
 for p in sorted(root.rglob('*')):t.add(p,arcname=str(p.relative_to(root)),recursive=False,filter=norm)
digest=hashlib.sha256(layer.read_bytes()).hexdigest()
config={'created':'2026-11-24T00:00:00Z','architecture':'amd64','os':'linux','config':{'Env':['PATH=/usr/local/bin:/usr/bin:/bin','PYTHONDONTWRITEBYTECODE=1','PYTHONUNBUFFERED=1','PYTHONPATH=/app','BASELINE_CPU_THREADS=1','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1'],'WorkingDir':'/app','Entrypoint':['python3','/app/harness.py'],'Cmd':['describe']},'rootfs':{'type':'layers','diff_ids':['sha256:'+digest]},'history':[{'created':'2026-11-24T00:00:00Z','created_by':'audited minimal Python NumPy runtime assembly'}]}
b=json.dumps(config,sort_keys=True).encode();cid=hashlib.sha256(b).hexdigest();cpath=base/(cid+'.json');cpath.write_bytes(b)
mp=base/'manifest.json';mp.write_text(json.dumps([{'Config':cpath.name,'RepoTags':['industrial-evolution:20261127-v5'],'Layers':['layer.tar']}]))
out=base/'output/02_Harness_Image_v5.tar'
with tarfile.open(out,'w') as t:
 for p in [layer,cpath,mp]:t.add(p,arcname=p.name,filter=norm)
print(json.dumps({'image_bytes':out.stat().st_size,'under_50MB':out.stat().st_size<=50_000_000,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},indent=2))
