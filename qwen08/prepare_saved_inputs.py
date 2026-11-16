"""Expand published frozen traces for fitting/scoring; never calls a model."""
import gzip,pathlib,shutil
root=pathlib.Path(__file__).resolve().parent
for name in ['fit-traces.jsonl','math500-edited.jsonl']:
    target=root/'results'/name
    if target.exists():
        print('Already present:',target.name);continue
    with gzip.open(str(target)+'.gz','rb') as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
    print('Expanded',target.name)
