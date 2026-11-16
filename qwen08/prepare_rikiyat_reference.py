"""Rebuild the published reference from its two hash-verified changed tensors."""
import hashlib,json,pathlib,struct,urllib.request
import numpy as np
from fit08 import gguf,SRC

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'reference/rikiyat';OUT.mkdir(exist_ok=True)
source=json.loads((OUT/'source.json').read_text())
expected={15:'82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53',
          19:'8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2'}

def fetch_range(start,end):
    url=source['url']+f'?download=true&range_start={start}'
    request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(request,timeout=180) as response:
        assert response.status==206
        assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
        data=response.read(end-start+2)
    assert len(data)==end-start+1
    return data

header=json.loads(fetch_range(8,7+source['header_bytes']))
(OUT/'safetensors-header.json').write_text(json.dumps(header,indent=2)+'\n')
reader=gguf.GGUFReader(SRC);tensors={t.name:t for t in reader.tensors}
changes={};records=[]
for layer,wanted in expected.items():
    key=f'model.language_model.layers.{layer}.mlp.down_proj.weight'
    spec=header[key];assert spec['dtype']=='BF16' and spec['shape']==[1024,3584]
    path=OUT/f'layer{layer}.bf16'
    if not path.exists() or hashlib.file_digest(path.open('rb'),'sha256').hexdigest()!=wanted:
        begin,end=spec['data_offsets'];offset=8+source['header_bytes']
        data=fetch_range(offset+begin,offset+end-1)
        assert hashlib.sha256(data).hexdigest()==wanted
        path.write_bytes(data)
    raw=np.fromfile(path,np.uint8).reshape(1024,3584*2)
    name=f'blk.{layer}.ffn_down.weight';tensor=tensors[name]
    assert tensor.tensor_type==gguf.GGMLQuantizationType.BF16 and tensor.data.shape==raw.shape
    changes[name]=raw
    records.append(dict(tensor=key,gguf_tensor=name,sha256=wanted,bytes=path.stat().st_size))

dst=OUT/'Qwen3.5-0.8B-RikiyaT-BF16.gguf';tmp=str(dst)+'.partial'
writer=gguf.GGUFWriter(tmp,arch=reader.fields['general.architecture'].contents(),endianess=reader.endianess)
for field in reader.fields.values():
    if field.name in (gguf.Keys.General.ARCHITECTURE,) or field.name.startswith('GGUF.'):continue
    typ=field.types[0];sub=field.types[-1] if typ==gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(field.name,field.contents(),typ,sub_type=sub)
writer.add_key_value('general.reference',source['repository']+' @ '+source['revision'],gguf.GGUFValueType.STRING)
for t in reader.tensors:
    a=changes.get(t.name,t.data);writer.add_tensor_info(t.name,a.shape,a.dtype,a.nbytes,t.tensor_type)
writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_ti_data_to_file()
for t in reader.tensors:writer.write_tensor_data(changes.get(t.name,t.data),tensor_endianess=reader.endianess)
writer.close();pathlib.Path(tmp).replace(dst)
check=gguf.GGUFReader(str(dst));changed=[]
for t in check.tensors:
    old=tensors[t.name]
    if not np.array_equal(t.data,old.data):changed.append(t.name)
assert set(changed)==set(changes)
report=dict(**source,model_path=str(dst),model_sha256=hashlib.file_digest(dst.open('rb'),'sha256').hexdigest(),
            changed_tensors=records,unchanged_tensors=len(tensors)-2,
            reconstruction='Pinned original base GGUF plus the two published BF16 tensors; both match model-card SHA256 hashes. Other tensors retained from pinned base as specified by the reference model card.')
(OUT/'reconstruction.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
