"""Download pinned public datasets; writes inputs for generation and fitting."""
import hashlib,json,pathlib,urllib.request
import pyarrow.parquet as pq
ROOT=pathlib.Path(__file__).resolve().parent;D=ROOT/'data';D.mkdir(exist_ok=True)
sources={
 'math500':('HuggingFaceH4/MATH-500','6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be','test.jsonl','math500.jsonl'),
 'math_train':('DigitalLearningGmbH/MATH-lighteval','0530c78699ea5e8eb5530600900e1f328b48acad','data/train-00000-of-00001.parquet','math_train.json'),
 'gsm8k':('openai/gsm8k','740312add88f781978c0658806c59bc2815b9866','main/test-00000-of-00001.parquet','gsm8k_test.json'),
 'ifeval':('google/IFEval','966cd89545d6b6acfd7638bc708b98261ca58e84','ifeval_input_data.jsonl','ifeval.json'),
}
report={}
for name,(repo,rev,file,out) in sources.items():
    url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{file}'
    raw=urllib.request.urlopen(url,timeout=90).read()
    if file.endswith('.parquet'):
        import io
        rows=pq.read_table(io.BytesIO(raw)).to_pylist()
        (D/out).write_text(json.dumps(rows,indent=1,default=str)+'\n')
    elif name=='ifeval':
        rows=[json.loads(l) for l in raw.decode().splitlines() if l.strip()]
        (D/out).write_text(json.dumps(rows,indent=1)+'\n')
    else:
        (D/out).write_bytes(raw);rows=[json.loads(l) for l in raw.splitlines() if l.strip()]
    expected={'math500':500,'math_train':7500,'gsm8k':1319,'ifeval':541}[name]
    assert len(rows)==expected
    report[name]=dict(repo=repo,revision=rev,file=file,raw_sha256=hashlib.sha256(raw).hexdigest(),rows=len(rows))
    print(name,len(rows),flush=True)
(D/'dataset-download-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
