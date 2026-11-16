# Qwen3.5-0.8B generation under the Astra/LP evaluation protocol, through a stock llama-server (Nov 16 2026).
# Protocol (from RikiyaT/Qwen3.5-0.8B-weight-space-programming eval/): thinking on, temperature 0.6, top_p 0.95,
# top_k 20, presence_penalty 1.5 (whole generation), min_p 0, max 32,768 new tokens, user prompt + the boxed suffix.
# llama.cpp difference (disclosed): the penalty window also covers the ~100 prompt tokens (vLLM: output tokens only).
# usage: python3 gen08.py PORT ARM OUT.jsonl SET IDS_FILE|all [samples] [n_predict]
#   SET: math500 | math_train | gsm8k | ifeval ; ids: one id per line (or "all")
import json, sys, os, time, hashlib, urllib.request, concurrent.futures as cf, re
port, arm, out, dset, idsf = sys.argv[1:6]
samples = int(sys.argv[6]) if len(sys.argv) > 6 else 1; npred = int(sys.argv[7]) if len(sys.argv) > 7 else 32768
fixed_seed = int(os.environ['FIXED_SEED']) if 'FIXED_SEED' in os.environ else None
model_id = os.environ.get('MODEL_ID')
U = f'http://127.0.0.1:{port}'; D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'
def last_boxed(s):
    i = s.rfind('\\boxed');
    if i < 0: return None
    j = s.find('{', i); d = 0
    for k in range(j, len(s)):
        d += (s[k] == '{') - (s[k] == '}')
        if d == 0: return s[j + 1:k]
    return None
P = {}
if dset == 'math500':
    for l in open(f'{D}/math500.jsonl'):
        r = json.loads(l); P[r['unique_id']] = dict(user=r['problem'] + SUFFIX, expected=r['answer'], level=r['level'])
elif dset == 'math_train':
    for i, r in enumerate(json.load(open(f'{D}/math_train.json'))):
        P[f'train/{i}'] = dict(user=r['problem'] + SUFFIX, expected=last_boxed(r['solution']), level=r['level'], type=r['type'])
elif dset == 'gsm8k':
    for i, r in enumerate(json.load(open(f'{D}/gsm8k_test.json'))):
        P[f'gsm8k/{i}'] = dict(user=r['question'] + SUFFIX, expected=r['answer'].split('####')[-1].strip().replace(',', ''))
elif dset == 'ifeval':
    for r in json.load(open(f'{D}/ifeval.json')):
        P[f"ifeval/{r['key']}"] = dict(user=r['prompt'], expected=None, instruction_id_list=r['instruction_id_list'], kwargs=r['kwargs'])
ids = list(P) if idsf == 'all' else [l.strip() for l in open(idsf) if l.strip()]
done = set()
if os.path.exists(out):
    previous = [json.loads(l) for l in open(out)]
    if fixed_seed is not None:
        assert all(r.get('seed') == fixed_seed for r in previous), 'Existing output uses a different seed; archive it first'
    if model_id is not None:
        assert all(r.get('model_id') == model_id for r in previous), 'Existing output uses another model; archive it first'
    done = {(r['id'], r['sample']) for r in previous}
def post(path, body, timeout=7200):
    req = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as f: return json.load(f)
def one(job):
    uid, s = job; p = P[uid]
    prompt = post('/apply-template', {'messages': [{'role': 'user', 'content': p['user']}], 'chat_template_kwargs': {'enable_thinking': True}})['prompt']
    ptoks = post('/tokenize', {'content': prompt, 'add_special': False, 'parse_special': True})['tokens']
    seed = fixed_seed if fixed_seed is not None else int(hashlib.md5(f'{uid}#{s}'.encode()).hexdigest()[:8], 16)
    t0 = time.time()
    r = post('/completion', {'prompt': ptoks, 'n_predict': npred, 'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0,
                             'presence_penalty': 1.5, 'repeat_last_n': npred + 1024, 'repeat_penalty': 1.0, 'seed': seed,
                             'return_tokens': True, 'cache_prompt': False, 'stream': False,
                             'samplers': ['penalties','top_k','top_p','temperature']})
    return dict(arm=arm, model_id=model_id, set=dset, id=uid, sample=s, expected=p.get('expected'), level=p.get('level'), prompt_ids=ptoks,
                gen_ids=r.get('tokens', []), text=r['content'], n_gen=r.get('tokens_predicted'), stop_type=r.get('stop_type'),
                seconds=round(time.time() - t0, 1), seed=seed, generation_settings=r.get('generation_settings'), **({k: p[k] for k in ('instruction_id_list', 'kwargs') if k in p}))
todo = [(u, s) for u in ids for s in range(samples) if (u, s) not in done]
print(f'{arm}/{dset}: {len(todo)} generations to run ({len(done)} done)', flush=True)
# as_completed so each row is flushed the moment it finishes (with map(), one slow problem blocks every
# row behind it and a killed run loses the whole in-flight batch)
failures=[]
with cf.ThreadPoolExecutor(int(os.environ.get('PAR', '8'))) as ex, open(out, 'a') as fo:
    jobs={ex.submit(one,j):j for j in todo}
    for fut in cf.as_completed(jobs):
        try:
            row=fut.result()
        except Exception as exc:
            failure=dict(job=jobs[fut],error=repr(exc),time=time.time())
            failures.append(failure)
            with open(out+'.errors.jsonl','a') as errors: errors.write(json.dumps(failure)+'\n')
            print('FAILED',failure,flush=True)
            continue
        fo.write(json.dumps(row) + '\n'); fo.flush()
        print(f"{row['id']}#{row['sample']}: {row['n_gen']} tok, {row['seconds']} s, {row['stop_type']}", flush=True)
if failures: raise SystemExit(f'{len(failures)} requests failed; successful rows saved, rerun to resume.')
