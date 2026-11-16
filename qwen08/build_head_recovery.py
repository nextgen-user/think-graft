"""Write an output-head-only repair; all original tensors are copied verbatim."""
import json,pathlib,sys,numpy as np
from fit08 import gguf,SRC
from gguf import quants
root=pathlib.Path(__file__).resolve().parent;out=root/'loop-repair';strength=float(sys.argv[1])
fit=json.load(open(out/'head-recovery-fit.json'));assert len(fit['fit_ids'])==80 and all(u.startswith('train/') for u in fit['fit_ids'])
done=np.load(out/'done-probe.npy');loop=np.load(out/'loop-probe.npy');allloop=np.load(out/'loop-all-token-probe.npy')
deltas={248069:12*done+24*loop}
for token in fit['selected_tokens']:deltas[token['id']]=-strength*allloop
r=gguf.GGUFReader(SRC);te=next(t for t in r.tensors if t.name=='token_embd.weight');qt=te.tensor_type
raw=np.array(te.data,copy=True);effective=[];targets=[]
for token,delta in deltas.items():
    original=quants.dequantize(te.data[token:token+1],qt)[0].astype(np.float32)
    raw[token]=quants.quantize((original+delta).astype(np.float32)[None,:],qt).reshape(-1).view(raw.dtype)
    written=quants.dequantize(raw[token:token+1],qt)[0]-original
    effective.append(written);targets.append(token)
    print('row',token,'relative rounding error',np.linalg.norm(written-delta)/np.linalg.norm(delta),flush=True)
path=out/f'Qwen3.5-0.8B-head-recovery-g{strength:g}.gguf';tmp=str(path)+'.partial'
w=gguf.GGUFWriter(tmp,arch=r.fields['general.architecture'].contents(),endianess=r.endianess)
for f in r.fields.values():
    if f.name==gguf.Keys.General.ARCHITECTURE or f.name.startswith('GGUF.'):continue
    vt=f.types[0];st=f.types[-1] if vt==gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name,f.contents(),vt,sub_type=st)
w.add_key_value('general.graft',f'output-head-only: loop-aware stopping plus learned loop-token suppression {strength:g}; MATH-train fit only',gguf.GGUFValueType.STRING)
for t in r.tensors:w.add_tensor_info(t.name,t.data.shape,t.data.dtype,t.data.nbytes,t.tensor_type)
w.add_tensor_info('output.weight',raw.shape,raw.dtype,raw.nbytes,qt)
w.write_header_to_file();w.write_kv_data_to_file();w.write_ti_data_to_file()
for t in r.tensors:w.write_tensor_data(t.data,tensor_endianess=r.endianess)
w.write_tensor_data(raw,tensor_endianess=r.endianess);w.close();pathlib.Path(tmp).replace(path)
np.savez(path.with_suffix('.deltas.npz'),target_ids=np.array(targets),effective_deltas=np.stack(effective))
path.with_suffix('.recipe.json').write_text(json.dumps(dict(method='output-head-only, three rows',stop_rechecking_strength=12,stop_loop_strength=24,loop_token_penalty_strength=strength,selected_tokens=fit['selected_tokens'],fit_problems=80,calibration_problems=16,backbone_changes=0),indent=2)+'\n')
print('WROTE',path,flush=True)
