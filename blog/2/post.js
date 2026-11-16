'use strict';
const $=id=>document.getElementById(id);
const state={trace:null,playing:false,position:0,last:0,pending:0,request:0};
const nf=new Intl.NumberFormat('en-US');
function typeset(root=document){
  if(!window.katex)return;
  root.querySelectorAll('[data-tex]').forEach(el=>{try{window.katex.render(el.dataset.tex,el,{displayMode:el.classList.contains('equation'),throwOnError:false,strict:false});}catch{/* Text remains readable. */}});
}
function pause(){state.playing=false;$('play').textContent='Play recording';}
function draw(){
  if(!state.trace)return;
  const p=Math.floor(state.position);$('position').value=String(p);$('position-label').textContent=`${nf.format(p)} tokens`;
  for(const key of ['base','reference','ours']){
    const t=state.trace.trajectories[key],n=Math.min(p,t.tokens),end=n?t.ends[n-1]:0;
    const pre=$(key+'-text');pre.textContent=n?t.text.slice(0,end):'Recording paused at the start.';
    if(state.playing||p===0||p>=Number($('position').max))pre.scrollTop=pre.scrollHeight;
    const ended=p>=t.tokens,thought=t.close!==null&&n>t.close;
    $(key+'-meta').textContent=`${t.native_tokens?'':'~'}${nf.format(n)} / ${nf.format(t.tokens)} tokens · ${thought?'answer phase':'thinking'}${ended?' · end of recording':''}`;
    const status=$(key+'-status');status.textContent=ended?(t.correct?'RIGHT':'WRONG'):'replaying';status.className='result '+(ended?(t.correct?'right':'wrong'):'');
    if(ended&&t.extra_close_tags)$(key+'-meta').textContent+=` · ${nf.format(t.extra_close_tags)} extra closing tags`;
  }
}
async function load(index){
  pause();const request=++state.request;state.trace=null;state.position=0;$('play').disabled=true;$('position').disabled=true;
  $('problem').textContent='Loading recorded output…';$('replay-error').hidden=true;
  try{
    const response=await fetch('/2/data/'+index.file);if(!response.ok)throw Error(`Trace unavailable (${response.status})`);
    const trace=await response.json();if(request!==state.request)return;
    state.trace=trace;$('case-id').textContent=trace.id.replace('test/','').replace('.json','');
    $('problem').textContent=trace.problem;$('position').max=String(Math.max(...Object.values(trace.trajectories).map(t=>t.tokens)));
    state.position=Math.min(128,Number($('position').max));$('play').disabled=false;$('position').disabled=false;draw();
  }catch(error){if(request!==state.request)return;$('problem').textContent='Recorded output could not be loaded.';$('replay-error').hidden=false;$('replay-error').textContent=error.message;}
}
function frame(now){
  if(!state.playing)return;
  if(state.playing&&state.trace){
    const delta=Math.min(now-state.last,250);state.pending+=delta;
    state.position=Math.min(Number($('position').max),state.position+delta/1000*Number($('speed').value));
    if(state.pending>=66){draw();state.pending=0;}
    if(state.position>=Number($('position').max)){draw();pause();}
  }
  state.last=now;if(state.playing)requestAnimationFrame(frame);
}
async function init(){
  typeset();window.addEventListener('load',()=>typeset(),{once:true});
  $('play').addEventListener('click',()=>{if(!state.trace)return;if(state.playing){pause();return;}if(state.position>=Number($('position').max))state.position=0;state.playing=true;state.last=performance.now();$('play').textContent='Pause';requestAnimationFrame(frame);});
  $('reset').addEventListener('click',()=>{pause();state.position=0;draw();});
  $('ending').addEventListener('click',()=>{pause();state.position=Number($('position').max);draw();});
  $('position').addEventListener('input',()=>{pause();state.position=Number($('position').value);draw();});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
  try{
    const response=await fetch('/2/data/traces.json');if(!response.ok)throw Error('Trace index unavailable');
    const index=await response.json();$('trace-choice').replaceChildren(...index.map((r,i)=>{const o=document.createElement('option');o.value=String(i);o.textContent=r.label;return o;}));
    $('trace-choice').addEventListener('change',()=>load(index[Number($('trace-choice').value)]));
    await load(index[0]);
  }catch(error){$('replay-error').hidden=false;$('replay-error').textContent=error.message;}
}
init();
