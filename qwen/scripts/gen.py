# Generate Qwen3.5-4B maths answers through an upstream llama-server (Nov 15 2026, the edited-on-Qwen scratchpad).
# usage: python3 gen.py PORT ARM OUT.jsonl IDS_FILE [thinking=on|off] [n_predict]
#   PORT: a running llama-server; ARM: a label stored with each row; IDS_FILE: one MATH-500 unique_id per line.
# Prompt: the model's own chat template (server /apply-template) with the MATH-500 problem + a boxed-answer instruction;
# sampling = Qwen's thinking-mode recommendation (temperature 0.6, top_p 0.95, top_k 20), fixed seed per problem.
# Each row keeps the prompt token ids and the generated token ids (needed to capture head inputs for the fit).
import json, sys, time, urllib.request, concurrent.futures as cf, hashlib, os
port, arm, out, idsf = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
think = (sys.argv[5] if len(sys.argv) > 5 else 'on') == 'on'; npred = int(sys.argv[6]) if len(sys.argv) > 6 else 12000
U = f'http://127.0.0.1:{port}'
D = {json.loads(l)['unique_id']: json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'math500.jsonl'))}
D.update({p['unique_id']: p for p in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'ds_problems.json')))})  # (DeepSeek's 9 maths problems)
D.update({f"aime25/{p['problem_idx']}": dict(unique_id=f"aime25/{p['problem_idx']}", problem=p['problem'], answer=str(p['answer']), level='aime', subject=p.get('problem_type', '')) for p in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'aime25.json')))})  # (AIME 2025 I=1-15, II=16-30)
ids = [l.strip() for l in open(idsf) if l.strip()]
done = set()
if os.path.exists(out): done = {json.loads(l)['id'] for l in open(out)}
SUFFIX = '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'
def post(path, body, timeout=3600):
    req = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as f: return json.load(f)
def one(uid):
    p = D[uid]
    prompt = post('/apply-template', {'messages': [{'role': 'user', 'content': p['problem'] + SUFFIX}],
                                      'chat_template_kwargs': {'enable_thinking': think}})['prompt']
    ptoks = post('/tokenize', {'content': prompt, 'add_special': False, 'parse_special': True})['tokens']
    seed = int(hashlib.md5(uid.encode()).hexdigest()[:8], 16)
    t0 = time.time()
    r = post('/completion', {'prompt': ptoks, 'n_predict': npred, 'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'seed': seed,
                             'return_tokens': True, 'cache_prompt': False, 'stream': False})
    return dict(arm=arm, id=uid, level=p['level'], subject=p['subject'], expected=p['answer'], thinking=think, n_predict=npred,
                prompt_ids=ptoks, gen_ids=r.get('tokens', []), text=r['content'], n_gen=r.get('tokens_predicted'),
                stop_type=r.get('stop_type'), seconds=round(time.time() - t0, 1), tps=r.get('timings', {}).get('predicted_per_second'))
todo = [u for u in ids if u not in done]
print(f'{arm}: {len(todo)} problems to run ({len(done)} done already)', flush=True)
with cf.ThreadPoolExecutor(int(os.environ.get('PAR', '4'))) as ex, open(out, 'a') as fo:
    for row in ex.map(one, todo):
        fo.write(json.dumps(row) + '\n'); fo.flush()
        print(f"{row['id']}: {row['n_gen']} tokens, {row['seconds']} s, stop {row['stop_type']}", flush=True)
