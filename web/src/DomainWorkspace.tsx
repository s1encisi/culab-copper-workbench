import {useCallback,useEffect,useState} from 'react';
import {Play,Plus} from 'lucide-react';
import {DataTable,Drawer,EmptyState,Facts,JsonDetails,Loading,Notice,Status,Toolbar,date,fmt,items,object,request,text,useRemote,type RecordValue} from './WorkspaceUI';

export default function DomainWorkspace(){
  const info=useRemote<RecordValue>('/v2/domain-evaluations/defaults'),access=useRemote<RecordValue>('/v2/workspace');
  const history=useRemote<{items:RecordValue[]}>('/v2/domain-evaluations');
  const [title,setTitle]=useState('项目规则对照'),[questions,setQuestions]=useState(''),[document,setDocument]=useState(''),[source,setSource]=useState('合成项目规则，用于工程评测');
  const [budget,setBudget]=useState(2),[selected,setSelected]=useState(''),[busy,setBusy]=useState(false),[error,setError]=useState('');
  const [verdict,setVerdict]=useState('accepted'),[notes,setNotes]=useState('');
  const detail=useRemote<RecordValue>(selected?'/v2/domain-evaluations/'+selected:null);
  const close=useCallback(()=>{setSelected('');setError('');},[]);
  const owner=object(access.data?.user).role==='owner';
  useEffect(()=>{if(info.data){setQuestions((info.data.questions as string[]).join('\n'));setDocument(text(info.data.document_text,''));}},[info.data]);
  useEffect(()=>{if(!['queued','running'].includes(text(detail.data?.status)))return;const timer=setInterval(()=>{detail.reload();history.reload();},1400);return()=>clearInterval(timer);},[detail.data?.status,detail.reload,history.reload]);
  async function action(run:()=>Promise<RecordValue>){
    setBusy(true);setError('');
    try{const value=await run();history.reload();detail.reload();return value;}
    catch(e){setError((e as Error).message);}finally{setBusy(false);}
  }
  async function create(){
    const value=await action(()=>request('/v2/domain-evaluations',{body:{request_key:crypto.randomUUID(),title,questions:questions.split('\n').map(v=>v.trim()).filter(Boolean),document_text:document,source_note:source,max_cost_cny:budget}}));
    if(value)setSelected(text(value.id));
  }
  const study=detail.data,cases=items(study?.cases),summary=items(study?.summary);
  return <section className="workspace-stack">
    <Toolbar title="提示词与资料对照评测" onRefresh={()=>{info.reload();history.reload();}}/>
    <section className="workspace-panel">
      <p>使用同一组问题，对比当前提示词与加入领域资料后的回答，保留引用、耗时、调用量和费用。</p>
      <div className="workspace-form-grid">
        <label className="field">评测名称<input value={title} onChange={e=>setTitle(e.target.value)} maxLength={120}/></label>
        <label className="field">总预算上限 / 元<input type="number" min={0.1} max={2} step={0.1} value={budget} onChange={e=>setBudget(Number(e.target.value))}/></label>
      </div>
      <label className="field">问题（每行一个，最多四题）<textarea rows={4} value={questions} onChange={e=>setQuestions(e.target.value)}/></label>
      <label className="field">对照资料<textarea rows={6} value={document} onChange={e=>setDocument(e.target.value)} maxLength={6000}/></label>
      <label className="field">资料来源与用途<input value={source} onChange={e=>setSource(e.target.value)} maxLength={300}/></label>
      <button className="primary" disabled={busy||!owner||!title.trim()||!questions.trim()||document.trim().length<20} onClick={()=>void create()}><Plus size={16}/>保存评测方案</button>
      <p className="workspace-note">每个问题分别运行两种条件。保存方案使用本地资料库，运行按钮会调用已配置模型。</p>
    </section>
    {error||history.error||info.error?<Notice tone="error">{error||history.error||info.error}</Notice>:null}
    {history.loading&&!history.data?<Loading/>:history.data?.items.length?<DataTable rows={history.data.items} onRow={row=>{setNotes('');setSelected(text(row.id));}} columns={[
      {key:'title',label:'评测'},{key:'status',label:'状态',render:r=><Status value={r.status==='completed'?'evaluation_done':r.status}/>},
      {key:'created_at',label:'创建时间',render:r=>date(r.created_at)},{key:'max_cost_cny',label:'预算 / 元',render:r=>fmt(r.max_cost_cny,2)}
    ]}/>:<EmptyState title="还没有评测记录">保存一组问题与资料后，即可查看数据卡并运行对照。</EmptyState>}
    {selected?<Drawer title="领域评测结果" onClose={close}>{detail.loading&&!study?<Loading/>:null}{detail.error?<Notice tone="error">{detail.error}</Notice>:null}
      {study?<section className="workspace-stack"><Status value={study.status==="completed"?"evaluation_done":study.status}/><Facts values={[
        {label:'资料来源',value:text(study.source_note)},{label:'问题数',value:fmt((study.questions as unknown[]||[]).length,0)},
        {label:'总预算 / 元',value:fmt(study.max_cost_cny,2)},{label:'调用类型',value:study.provider_mode==='test_transport'?'测试替身':study.provider_mode==='live'?'真实模型':'尚未运行'}
      ]}/><a href={'/api/v2/knowledge/documents/'+text(object(study.document).doc_id)+'/versions/1/source'} target="_blank" rel="noreferrer">查看本次资料</a>
      <button className="primary" disabled={busy||!owner||!info.data?.can_run||['queued','running','completed'].includes(text(study.status))} onClick={()=>void action(()=>request('/v2/domain-evaluations/'+selected+'/run',{body:{}}))}><Play size={15}/>{study.status==='planned'?'运行对照':'继续未完成项'}</button>
      {!info.data?.can_run?<Notice>当前模型调用未启用，方案已保存。</Notice>:null}
      {object(study.error).message?<Notice tone="error">{text(object(study.error).message)}</Notice>:null}
      {summary.length?<DataTable rows={summary} columns={[
        {key:'variant',label:'条件',render:r=>r.variant==='prompt'?'当前提示词':'提示词 + 资料'},
        {key:'completed',label:'完成任务'},{key:'answers',label:'生成回答'},{key:'document_citations',label:'文档引用'},
        {key:'p50_ms',label:'p50 / ms',render:r=>fmt(r.p50_ms,0)},{key:'p95_ms',label:'p95 / ms',render:r=>fmt(r.p95_ms,0)},
        {key:'cost_cny',label:'费用 / 元',render:r=>fmt(r.cost_cny,4)}
      ]}/>:null}
      {cases.map((row,i)=><article className="workspace-panel" key={i}><strong>{row.variant==='prompt'?'当前提示词':'提示词 + 资料'} · 问题 {text(row.case_id)}</strong><p>{text(row.question)}</p><Status value={row.task_status??row.status}/>{object(row.result).answer?<p style={{whiteSpace:'pre-wrap'}}>{text(object(row.result).answer)}</p>:null}{object(row.error).message?<Notice tone="error">{text(object(row.error).message)}</Notice>:null}{row.task_id?<small>任务 {text(row.task_id)}</small>:null}<JsonDetails value={row} label="调用与引用记录"/></article>)}
      {study.status==='completed'?<section className="workspace-panel"><h3>结果复核</h3><label className="field">结论<select value={verdict} onChange={e=>setVerdict(e.target.value)}><option value="accepted">接受当前方案</option><option value="needs_revision">需要调整</option></select></label><label className="field">复核说明<textarea value={notes} onChange={e=>setNotes(e.target.value)}/></label><button disabled={busy||!owner||!notes.trim()} onClick={()=>void action(()=>request('/v2/domain-evaluations/'+selected+'/review',{body:{verdict,notes}}))}>保存复核</button>{study.review?<JsonDetails value={study.review} label="已有复核记录"/>:null}</section>:null}
      {error?<Notice tone="error">{error}</Notice>:null}
      </section>:null}
    </Drawer>:null}
  </section>;
}
