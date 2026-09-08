import {useEffect,useRef,useState} from 'react';
import {Info,LoaderCircle} from 'lucide-react';
import {api,type Explanation,type Run} from './api';

export function ExplanationPanel({run,llmEnabled}:{run:Run|null;llmEnabled:boolean}){
  const [value,setValue]=useState<Explanation|null>(null);
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState('');
  const [live,setLive]=useState(false);
  const pending=useRef<AbortController|null>(null);

  useEffect(()=>{
    setValue(null);setError('');setBusy(false);
    return()=>pending.current?.abort();
  },[run?.run_id]);

  if(!run||run.status!=='completed'||run.task_type==='agent_diagnostic')return null;
  const ask=async(question:string)=>{
    pending.current?.abort();
    const controller=new AbortController();
    pending.current=controller;
    setBusy(true);setError('');
    try{
      const result=await api<Explanation>(`/runs/${run.run_id}/explanation`,{question,use_llm:live},controller.signal);
      if(!controller.signal.aborted)setValue(result);
    }catch(e){
      if(!controller.signal.aborted)setError((e as Error).message);
    }finally{
      if(pending.current===controller&&!controller.signal.aborted)setBusy(false);
    }
  };

  return <section className="panel explanation-panel">
    <div className="panel-heading"><h3>结果解读</h3><label className="inline-check"><input type="checkbox" checked={live} disabled={!llmEnabled} onChange={e=>setLive(e.target.checked)}/>{llmEnabled?'LLM 增强':'本地解释'}</label></div>
    <div className="question-list">{[['summary','解释本次结果'],['mode','工况依据'],['model','模型选择'],['tradeoff','候选权衡'],['constraints','约束说明']].map(([key,label])=><button disabled={busy} onClick={()=>void ask(key)} key={key}>{label}</button>)}</div>
    {busy?<p className="muted"><LoaderCircle className="spin" size={16}/> 正在整理已有计算结果</p>:value?<div className="explanation-text">{value.text.split('\n\n').map((p,i)=><p key={i}>{p}</p>)}<div className="source-note">来源：本次运行 · {value.mode==='local'?'本地计算说明':'LLM 增强'}{value.cached?' · 已复用':''}{value.fallback_reason?' · 已使用本地回退':''}</div></div>:<p className="muted">选择一个问题，查看与本次计算关联的说明。</p>}
    {error&&<div role="alert" className="error-box"><Info size={17}/><span>{error}</span></div>}
  </section>;
}
