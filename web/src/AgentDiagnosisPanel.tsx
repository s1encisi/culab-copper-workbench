import {useEffect,useState} from 'react';
import {Bot,Download,LoaderCircle,RefreshCw} from 'lucide-react';
import {api,fmt,statusLabel,type AgentResult,type AgentRun,type Run} from './api';

type Question='why_few_candidates'|'constraint_effect'|'model_response';
type AgentInfo={configured:boolean;model:string;max_calls:number;max_cost_cny:number};
const questions:[Question,string][]=[['why_few_candidates','为什么候选较少？'],['constraint_effect','As 约束影响多大？'],['model_response','模型对电流敏感吗？']];
const toolLabels:Record<string,string>={inspect_run:'读取原运行',probe_constraints:'探测约束影响',probe_response:'检查模型响应',compare_probe_resolution:'比较探测密度',finish_diagnosis:'提交诊断'};

export function AgentDiagnosisResult({result,runId}:{result:AgentResult;runId:string}){
  return <>
    {result.report&&<div className="agent-report">
      <p className="agent-summary">{result.report.summary}</p>
      <div className="agent-findings">{result.report.findings.map((finding,index)=><article key={index}>
        <span className={'agent-finding-status '+finding.status}>{finding.status==='supported'?'模型判断：有证据支持':'模型判断：尚待确认'}</span>
        <p>{finding.claim}</p><div className="agent-evidence-tags">{finding.evidence_ids.map(id=><a key={id} href={`#${runId}-${id}`} onClick={()=>{const node=document.getElementById(`${runId}-${id}`);if(node instanceof HTMLDetailsElement)node.open=true;}}>{id}</a>)}</div>
      </article>)}</div>
      {result.report.next_checks.length>0&&<div className="agent-next"><strong>建议补充的检查</strong><ul>{result.report.next_checks.map((item,index)=><li key={index}>{item}</li>)}</ul></div>}
    </div>}
    {result.steps.length>0&&<div className="agent-investigation"><h4>实际工具调用</h4><ol>{result.steps.map((step,index)=><li key={index}>
      <div><strong>{toolLabels[step.tool]||step.tool}</strong><span>{step.evidence_id}{step.cached?' · 复用探测':''}{step.status==='failed'?' · 未执行成功':''}</span></div>
      <p>{step.purpose||step.error?.message}</p>
    </li>)}</ol></div>}
    {result.evidence.length>0&&<div className="agent-evidence"><h4>查看计算证据</h4>{result.evidence.map(item=><details key={item.evidence_id} id={`${runId}-${item.evidence_id}`}>
      <summary><span>{item.evidence_id}</span>{toolLabels[item.tool]||item.tool}<small>{item.data.evidence_scope==='original_saved_run'?'原运行记录':'新增数值探测'}</small></summary>
      <pre>{JSON.stringify(item.data,null,2)}</pre>
    </details>)}</div>}
    <div className="agent-usage"><span>{result.model}</span><span>API {result.usage.api_calls} 次 · 工具 {result.usage.tool_calls} 次</span><span>输入 {result.usage.input_tokens.toLocaleString()} / 输出 {result.usage.output_tokens.toLocaleString()} tokens</span><span>估算 ¥{fmt(result.usage.estimated_cost_cny,4)}</span>{result.usage.unsettled_reservation_cny>0&&<span>待确认费用预留 ¥{fmt(result.usage.unsettled_reservation_cny,4)}</span>}</div>
  </>;
}

export function AgentDiagnosisPanel({source}:{source:Run|null}){
  const eligible=source?.status==='completed'&&source.result?.kind==='optimization'&&source.result.mode==='plant';
  const sourceId=eligible?source.run_id:null;
  const [info,setInfo]=useState<AgentInfo|null>(null);
  const [question,setQuestion]=useState<Question>('why_few_candidates');
  const [history,setHistory]=useState<AgentRun[]>([]);
  const [job,setJob]=useState<AgentRun|null>(null);
  const [submitting,setSubmitting]=useState(false);
  const [error,setError]=useState('');
  const [reused,setReused]=useState(false);
  useEffect(()=>{
    if(!sourceId)return;
    const controller=new AbortController();
    Promise.all([api<AgentInfo>('/diagnostic-agent',undefined,controller.signal),api<{items:AgentRun[]}>(`/runs/${sourceId}/diagnoses`,undefined,controller.signal)])
      .then(([settings,runs])=>{
        if(controller.signal.aborted)return;
        setInfo(settings);setHistory(runs.items);
        const latest=runs.items[0];
        if(latest){setJob(latest);setQuestion(latest.request.question as Question);}
      }).catch(e=>{if(e.name!=='AbortError')setError(e.message);});
    return()=>controller.abort();
  },[sourceId]);
  useEffect(()=>{
    if(!job||!['queued','running'].includes(job.status))return;
    const controller=new AbortController();
    let timer:ReturnType<typeof setTimeout>;
    const poll=async()=>{
      try{
        const next=await api<AgentRun>(`/runs/${job.run_id}`,undefined,controller.signal);
        if(controller.signal.aborted)return;
        setJob(next);setError('');
        if(['queued','running'].includes(next.status))timer=setTimeout(poll,1000);
        else setHistory(previous=>[next,...previous.filter(item=>item.run_id!==next.run_id)]);
      }catch(e){if(!controller.signal.aborted){setError((e as Error).message);timer=setTimeout(poll,2500);}}
    };
    timer=setTimeout(poll,350);
    return()=>{controller.abort();clearTimeout(timer);};
  },[job?.run_id,job?.status]);
  if(!sourceId)return null;
  const busy=submitting||!!job&&['queued','running'].includes(job.status);
  const start=async(fresh=false)=>{
    setSubmitting(true);setError('');setReused(false);
    try{
      const next=await api<AgentRun>(`/runs/${sourceId}/diagnoses`,{question,...(fresh?{request_key:crypto.randomUUID()}:{})});
      setJob(next);setReused(!!next.reused);setHistory(previous=>[next,...previous.filter(item=>item.run_id!==next.run_id)]);
    }catch(e){setError((e as Error).message);}finally{setSubmitting(false);}
  };
  return <section className="panel agent-panel" aria-label="优化诊断助手">
    <div className="panel-heading"><h3><Bot size={21}/>优化诊断助手</h3><span className="subtle">DeepSeek V4 Flash · 真实 API</span></div>
    <p className="agent-intro">助手根据原运行和工具证据选择检查步骤，分析候选数量、As 约束及模型响应。</p>
    <div className="agent-controls"><label className="field">诊断问题<select aria-label="诊断问题" value={question} disabled={busy} onChange={e=>{const value=e.target.value as Question;setQuestion(value);setJob(history.find(item=>item.request.question===value)||null);setReused(false);setError('');}}>{questions.map(([value,label])=><option key={value} value={value}>{label}</option>)}</select></label>
      <button className="primary" disabled={busy||!info?.configured} onClick={()=>void start()}>{busy?<LoaderCircle size={17} className="spin"/>:<Bot size={17}/>}开始诊断</button>
      {job&&!busy&&<button className="secondary" disabled={!info?.configured} onClick={()=>void start(true)}><RefreshCw size={15}/>重新诊断</button>}
    </div>
    <p className="source-note">{info?.configured?`单次最多 ${info.max_calls} 次模型调用，费用上限 ¥${fmt(info.max_cost_cny,2)}。同问题默认复用已有运行；“重新诊断”会发起新调用。`:info?'未找到本机 DEEPSEEK_API_KEY，历史诊断仍可查看。':'正在检查配置。'}</p>
    {reused&&<p role="status" className="source-note">已复用已有诊断，没有重复调用 API。</p>}
    {busy&&<p role="status" className="agent-progress"><LoaderCircle size={16} className="spin"/>{job?.trace.at(-1)?.node||'诊断已排队'} · 工具结果会逐步显示</p>}
    {(error||job?.error)&&<div role="alert" className="error-box">{error||job?.error?.message}</div>}
    {job&&<div className="agent-run-heading"><span className={'status '+job.status}>{statusLabel[job.status]}</span><span className="subtle">{job.run_id.slice(0,10)}</span><a href={`/api/runs/${job.run_id}/export?format=md`} download><Download size={14}/>导出诊断</a></div>}
    {job?.result&&<AgentDiagnosisResult result={job.result} runId={job.run_id}/>}
  </section>;
}
