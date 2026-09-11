import {useEffect,useRef,useState} from 'react';
import {Send,Plus,Pause,Play,Square,MessageSquare,LoaderCircle,BookOpen} from 'lucide-react';
import {api,date,fmt,type Run} from './api';

type Session={id:string;title:string;context:Record<string,string|null>;messages:{role:string;text:string;task_id:string|null}[]};
type Evidence={evidence_id:string;tool:string;data:{summary:unknown;local_facts:Record<string,{value:number|null;unit:string}>}};
type Task={id:string;status:string;version:number;model_calls:number;tool_calls:number;cache_hits:number;
  usage:{spent_cny:number;unsettled_cny:number};error:{message:string}|null;result:{answer:string}|null;evidence:Evidence[]};
type Info={model:string;live_calls_enabled:boolean;max_cost_cny:number};
type Resource={run_id:string;status:string;created_at:string;request?:Record<string,unknown>};
const active=new Set(['queued','running','pausing','cancelling']);
const labels:Record<string,string>={queued:'等待执行',running:'正在取证',pausing:'等待暂停边界',paused:'已暂停',cancelling:'正在取消',cancelled:'已取消',completed:'已完成',failed:'未完成',needs_attention:'需要处理'};
const tools:Record<string,string>={project_status:'工作台能力',model_metrics:'模型指标',list_model_comparisons:'模型比较目录',list_optimizations:'优化目录',optimization_summary:'优化统计',probe_constraints:'约束探测',probe_response:'模型响应',probe_resolution:'分辨率对照',event_context:'事件工况'};

const factLabels:Record<string,string>={development_events:'开发事件',oof_events:'OOF 事件',front_points:'原前沿点数',probe_points:'探测点数',feasible_points:'可行点数',probe_front_points:'探测前沿点数',feasible_rate:'可行率',search_evaluations:'搜索求值',variable_count:'变量数量',common_events:'共同评价事件',as_allowance_mg_l:'As 允许增量 (mg/L)',status:'状态',decision_at:'决策时间'};
function EvidenceSummary({value}:{value:unknown}){
  if(!value||typeof value!=='object'||Array.isArray(value))return null;
  const data=value as Record<string,unknown>;
  const fields=Object.entries(data).filter(([key,v])=>key in factLabels&&(typeof v==='number'||typeof v==='string')).slice(0,8);
  const models=data.models as {registered_count?:number}|undefined;
  const optimizers=data.optimizers as {items?:unknown[]}|undefined;
  return <div className="research-facts">{models?.registered_count?<span>注册预测方法<strong>{models.registered_count}</strong></span>:null}{optimizers?.items?<span>注册优化方法<strong>{optimizers.items.length}</strong></span>:null}{fields.map(([key,v])=><span key={key}>{factLabels[key]}<strong>{typeof v==='number'?fmt(v,3):String(v)}</strong></span>)}</div>;
}

function EvidenceCard({item}:{item:Evidence}){
  const facts=Object.entries(item.data.local_facts||{});
  return <details className="research-evidence" id={item.evidence_id}>
    <summary><BookOpen size={15}/>{tools[item.tool]||item.tool}<small>{item.evidence_id.slice(-6)}</small></summary>
    {facts.length?<div className="research-facts">{facts.map(([key,fact])=><span key={key}>{key==='current_cu'?'当前 Cu':key==='current_as'?'当前 As':key}<strong>{fmt(fact.value,3)} {fact.unit}</strong></span>)}</div>:null}
    <EvidenceSummary value={item.data.summary}/><details className="research-raw"><summary>查看完整数据</summary><pre>{JSON.stringify(item.data.summary,null,2)}</pre></details>
  </details>;
}
export function ResearchPanel({eventId}:{eventId:string}){
  const [sessions,setSessions]=useState<Session[]>([]);
  const [sessionId,setSessionId]=useState<string|null>(null);
  const [conversation,setConversation]=useState<Session|null>(null);
  const [task,setTask]=useState<Task|null>(null);
  const [question,setQuestion]=useState('');
  const [error,setError]=useState('');
  const [sending,setSending]=useState(false);
  const [info,setInfo]=useState<Info|null>(null);
  const [runs,setRuns]=useState<Run[]>([]);
  const [models,setModels]=useState<Resource[]>([]);
  const [comparisons,setComparisons]=useState<Resource[]>([]);
  const [runId,setRunId]=useState('');
  const [modelId,setModelId]=useState('');
  const [comparisonId,setComparisonId]=useState('');
  const bottom=useRef<HTMLDivElement>(null);
  const justCreated=useRef<string|null>(null);
  const running=task?active.has(task.status):false;
  const blocked=sending||running||task?.status==='paused';
  useEffect(()=>{
    const controller=new AbortController();const signal=controller.signal;
    Promise.all([
      api<{items:Session[]}>('/v2/sessions',undefined,signal),
      api<Info>('/v2/assistant',undefined,signal),
      api<{items:Run[]}>('/runs?task_type=optimize&limit=30',undefined,signal),
      api<{items:Resource[]}>('/v2/model-comparisons',undefined,signal),
      api<{items:Resource[]}>('/v2/optimizer-comparisons',undefined,signal)
    ]).then(([s,i,r,m,c])=>{
      setSessions(s.items);setInfo(i);setRuns(r.items.filter(x=>x.status==='completed'&&x.request.mode==='plant'));
      setModels(m.items.filter(x=>x.status==='completed'));setComparisons(c.items.filter(x=>x.status==='completed'));
    }).catch(e=>{if(!signal.aborted)setError(e.message);});
    return()=>controller.abort();
  },[]);
  useEffect(()=>{
    if(sessionId&&justCreated.current===sessionId){justCreated.current=null;return;}
    setConversation(null);setTask(null);setError('');
    if(!sessionId)return;
    const controller=new AbortController();
    api<Session>('/v2/sessions/'+sessionId,undefined,controller.signal).then(async value=>{
      if(controller.signal.aborted)return;
      setConversation(value);setRunId(value.context.optimization_run_id||'');setModelId(value.context.model_comparison_id||'');setComparisonId(value.context.optimizer_comparison_id||'');
      const last=value.messages.at(-1);
      if(last?.task_id){const current=await api<Task>('/v2/tasks/'+last.task_id,undefined,controller.signal);if(!controller.signal.aborted)setTask(current);}
    }).catch(e=>{if(!controller.signal.aborted)setError(e.message);});
    return()=>controller.abort();
  },[sessionId]);
  useEffect(()=>{
    if(!task?.id||!running)return;
    const controller=new AbortController();let timer:ReturnType<typeof setTimeout>;
    const poll=async()=>{
      try{
        const current=await api<Task>('/v2/tasks/'+task.id,undefined,controller.signal);
        if(controller.signal.aborted)return;
        if(!active.has(current.status)&&sessionId){
          const updated=await api<Session>('/v2/sessions/'+sessionId,undefined,controller.signal);
          if(controller.signal.aborted)return;
          setConversation(updated);
        }
        setTask(current);
        if(active.has(current.status))timer=setTimeout(poll,700);
      }catch(e){if(!controller.signal.aborted)setError(e instanceof Error?e.message:'查询失败');}
    };
    timer=setTimeout(poll,250);
    return()=>{controller.abort();clearTimeout(timer);};
  },[task?.id,running,sessionId]);
  useEffect(()=>{bottom.current?.scrollIntoView({behavior:'smooth',block:'nearest'});},[conversation?.messages.length,task?.status]);
  async function send(){
    if(!question.trim()||blocked)return;
    setSending(true);setError('');
    const text=question.trim();
    const selected=runs.find(r=>r.run_id===runId);
    const context={event_id:(selected?.request.event_id as string)||eventId||null,optimization_run_id:runId||null,
                   model_comparison_id:modelId||null,optimizer_comparison_id:comparisonId||null};
    try{
      let id=sessionId;
      if(!id){const created=await api<Session>('/v2/sessions',{title:text.slice(0,40),context});id=created.id;justCreated.current=id;setConversation(created);setSessions(values=>[created,...values]);setSessionId(id);}
      const current=await api<Task>('/v2/sessions/'+id+'/messages',{question:text,request_key:crypto.randomUUID(),context});
      setQuestion('');setTask(current);setConversation(await api<Session>('/v2/sessions/'+id));
    }catch(e){setError(e instanceof Error?e.message:'发送失败');}
    finally{setSending(false);}
  }
  async function control(action:string){
    if(!task)return;setError('');
    try{const current=await api<Task>('/v2/tasks/'+task.id);setTask(await api<Task>('/v2/tasks/'+task.id+'/'+action,{expected_version:current.version}));}
    catch(e){setError(e instanceof Error?e.message:'操作未完成');setTask(await api<Task>('/v2/tasks/'+task.id));}
  }
  return <div className="research-layout">
    <aside className="research-sessions"><button className="secondary" onClick={()=>{setSessionId(null);setConversation(null);setTask(null);setRunId('');setModelId('');setComparisonId('');}}><Plus size={16}/>新对话</button>
      {sessions.map(s=><button key={s.id} className={s.id===sessionId?'selected':''} onClick={()=>setSessionId(s.id)}><MessageSquare size={15}/><span>{s.title}</span></button>)}
    </aside>
    <section className="research-main">
      <div className="research-context">
        <label className="field">参考优化<select aria-label="参考优化" value={runId} onChange={e=>{setRunId(e.target.value);setComparisonId('');}} disabled={!!blocked}><option value="">未指定</option>{runs.map(r=><option key={r.run_id} value={r.run_id}>{date(r.created_at)} · {r.run_id.slice(0,6)}</option>)}</select></label>
        <label className="field">模型比较<select aria-label="模型比较" value={modelId} onChange={e=>setModelId(e.target.value)} disabled={!!blocked}><option value="">未指定</option>{models.map(r=><option key={r.run_id} value={r.run_id}>{r.request?.request_key as string||r.run_id.slice(0,8)}</option>)}</select></label>
        <label className="field">优化器比较<select aria-label="优化器比较" value={comparisonId} onChange={e=>{setComparisonId(e.target.value);setRunId('');}} disabled={!!blocked}><option value="">未指定</option>{comparisons.map(r=><option key={r.run_id} value={r.run_id}>{r.request?.request_key as string||r.run_id.slice(0,8)}</option>)}</select></label>
      </div>
      <div className="research-messages" aria-live="polite">
        {!conversation?.messages.length?<div className="research-intro"><MessageSquare size={30}/><h2>围绕证据，直接提问</h2><p>比较模型、查看工况，或追问一次优化为什么得到这样的结果。</p>
          <div>{['当前有哪些模型和优化方法？','比较这次模型实验的结果','为什么这次优化的候选很少？'].map(text=><button className="secondary" key={text} onClick={()=>setQuestion(text)}>{text}</button>)}</div></div>:null}
        {conversation?.messages.map((m,i)=><article key={i} className={'research-message '+m.role}><strong>{m.role==='user'?'你':'CuLab'}</strong><p>{m.text}</p></article>)}
        {task?<div className="research-task"><span className={'status '+task.status}>{labels[task.status]||task.status}</span><small>模型 {task.model_calls} 次 · 工具 {task.tool_calls} 次 · 费用 ¥{fmt(task.usage?.spent_cny,4)}{task.usage?.unsettled_cny?' · 待结算 ¥'+fmt(task.usage.unsettled_cny,4):''}</small>
          <div>{running?<button className="secondary" onClick={()=>control('pause')}><Pause size={14}/>暂停</button>:null}{task.status==='paused'?<button className="secondary" onClick={()=>control('resume')}><Play size={14}/>继续</button>:null}{running||task.status==='paused'?<button className="secondary" onClick={()=>control('cancel')}><Square size={14}/>取消</button>:null}</div>
          {task.error?<p role="alert">{task.error.message}</p>:null}
        </div>:null}
        {task?.evidence.map(e=><EvidenceCard key={e.evidence_id} item={e}/>)}
        <div ref={bottom}/>
      </div>
      <form className="research-compose" onSubmit={e=>{e.preventDefault();void send();}}>
        <label className="sr-only" htmlFor="research-question">研究问题</label>
        <textarea id="research-question" value={question} onChange={e=>setQuestion(e.target.value)} placeholder="输入问题，也可以继续追问…" rows={3} maxLength={6000} disabled={!!blocked}/>
        <div><small>{info?.live_calls_enabled?info.model+' · 每任务上限 ¥'+info.max_cost_cny:'真实模型调用尚未启用'}</small><button className="primary" disabled={!!blocked||!question.trim()||!info?.live_calls_enabled}>{sending||running?<LoaderCircle size={16} className="spin"/>:<Send size={16}/>}发送</button></div>
      </form>
      {error?<div className="error-box" role="alert">{error}</div>:null}
    </section>
  </div>;
}
