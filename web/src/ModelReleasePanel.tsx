import {useCallback,useEffect,useState} from 'react';
import {ShieldCheck,RotateCcw,Archive} from 'lucide-react';
import {DataTable,Drawer,EmptyState,Facts,JsonDetails,Loading,Notice,Status,Toolbar,date,fmt,object,request,text,useRemote,type RecordValue} from './WorkspaceUI';

const activeStates=new Set(['queued','running']);
const targetName=(target:unknown)=>target==='cu'?'Cu · g/L':'As · mg/L';
const candidateName=(id:unknown)=>id==='builtin-persistence'?'Persistence 基线':'工件 '+text(id).slice(0,12);

function ProposalReview({value,canApprove,onChanged}:{value:RecordValue;canApprove:boolean;onChanged:(value:RecordValue)=>void}){
  const [busy,setBusy]=useState(false),[error,setError]=useState(''),[reason,setReason]=useState('');
  const payload=object(value.payload),previous=object(payload.previous);
  const expired=Number(payload.expires_at)*1000<=Date.now();
  async function decide(decision:'approve'|'reject'){
    setBusy(true);setError('');
    try{onChanged(await request<RecordValue>('/v2/releases/'+text(value.id)+'/approve',{body:{payload_hash:value.payload_hash,decision,reason}}));}
    catch(e){setError((e as Error).message);}finally{setBusy(false);}
  }
  return <section className="workspace-stack">
    <Status value={value.status}/>
    <Facts values={[
      {label:'操作',value:payload.operation==='rollback'?'回退研究模型':'发布研究模型'},
      {label:'目标 / 单位',value:targetName(payload.target)},
      {label:'当前版本',value:candidateName(previous.candidate_id)+' · 指针 '+text(payload.expected_version)},
      {label:'拟切换到',value:candidateName(payload.candidate_id)},
      {label:'用途',value:payload.purpose==='forecast'?'历史研究预测':text(payload.purpose)},
      {label:'提案原因',value:text(payload.reason)},
      {label:'有效至',value:date(payload.expires_at)}
    ]}/>
    <JsonDetails value={payload} label="核对完整提案与来源绑定"/>
    <label className="field">审批意见<textarea value={reason} onChange={e=>setReason(e.target.value)} maxLength={500}/></label>
    <div className="workspace-actions">
      <button className="primary" disabled={busy||!canApprove||value.status!=='pending'||expired} title={!canApprove?'需要管理员权限':undefined} onClick={()=>void decide('approve')}>批准此提案</button>
      <button disabled={busy||!canApprove||value.status!=='pending'} title={!canApprove?'需要管理员权限':undefined} onClick={()=>void decide('reject')}>拒绝此提案</button>
    </div>
    {value.status==='pending'&&expired?<Notice>提案已过有效期，请按当前指针重新创建提案。</Notice>:null}
    {value.approval?<JsonDetails value={value.approval} label="查看审批记录"/>:null}
    {error?<Notice tone="error">{error}</Notice>:null}
  </section>;
}

export function ArtifactActions({artifact,canCompute,canApprove,isCurrent,onChanged}:{artifact:RecordValue;canCompute:boolean;canApprove:boolean;isCurrent:boolean;onChanged:()=>void}){
  const id=text(artifact.id),descriptor=object(artifact.descriptor);
  const history=useRemote<{items:RecordValue[]}>('/v2/model-artifacts/'+id+'/shadows');
  const [shadowId,setShadowId]=useState(''),[busy,setBusy]=useState(false),[error,setError]=useState(''),[reason,setReason]=useState('');
  const [proposal,setProposal]=useState<RecordValue|null>(null),[retiring,setRetiring]=useState(false);
  const shadow=useRemote<RecordValue>(shadowId?'/v2/model-shadows/'+shadowId:null);
  const gate=useRemote<RecordValue>(shadow.data?.status==='completed'?'/v2/model-artifacts/'+id+'/qualification?shadow_id='+shadowId:null);
  useEffect(()=>{if(!shadowId&&history.data?.items.length)setShadowId(text(history.data.items[0].id));},[history.data,shadowId]);
  useEffect(()=>{
    if(!activeStates.has(text(shadow.data?.status)))return;
    const timer=setInterval(shadow.reload,1200);return()=>clearInterval(timer);
  },[shadow.data?.status,shadow.reload]);
  useEffect(()=>{if(shadow.data&&!activeStates.has(text(shadow.data.status))){history.reload();onChanged();}},[shadow.data?.status,history.reload,onChanged]);
  async function perform(action:()=>Promise<RecordValue>){
    setBusy(true);setError('');
    try{const result=await action();history.reload();onChanged();return result;}
    catch(e){setError((e as Error).message);}finally{setBusy(false);}
  }
  async function startShadow(){
    const result=await perform(()=>request('/v2/model-artifacts/'+id+'/shadow',{body:{request_key:crypto.randomUUID(),samples_per_fold:8}}));
    if(result)setShadowId(text(result.id));
  }
  async function propose(){
    const result=await perform(async()=>{
      const latest=await request<RecordValue>('/v2/model-pointers');
      return request('/v2/releases',{body:{request_key:crypto.randomUUID(),candidate_id:id,shadow_id:shadowId,target:descriptor.target,
        expected_version:object(latest[text(descriptor.target)]).version,reason}});
    });
    if(result)setProposal(result);
  }
  const running=activeStates.has(text(shadow.data?.status));
  const canShadow=['benchmarked','shadow','approved'].includes(text(artifact.status));
  return <section className="workspace-stack">
    <div className="workspace-actions"><button disabled={busy||running||!canCompute||!canShadow} title={!canCompute?'需要研究员或管理员权限':undefined} onClick={()=>void startShadow()}><ShieldCheck size={15}/>运行影子回放</button>
      <button onClick={()=>{history.reload();shadow.reload();gate.reload();onChanged();}}>刷新验证记录</button></div>
    {history.error?<Notice tone="error">{history.error}</Notice>:null}
    {history.data?.items.length?<label className="field">影子验证记录<select value={shadowId} onChange={e=>setShadowId(e.target.value)}>{history.data.items.map(row=><option value={text(row.id)} key={text(row.id)}>{date(row.created_at)} · {text(row.id).slice(0,8)}</option>)}</select></label>:null}
    {shadow.loading&&!shadow.data?<Loading/>:null}
    {shadow.data?<><div>影子回放：<Status value={shadow.data.status}/></div>{running?<Notice>正在回放已保存工件，结果会自动更新。</Notice>:null}
      <Facts values={[{label:'已回放样本',value:fmt(object(shadow.data.evidence).samples,0)},{label:'预测一致性',value:object(shadow.data.evidence).parity_passed===undefined?'等待结果':object(shadow.data.evidence).parity_passed?'通过':'未通过'}]}/>
      <JsonDetails value={shadow.data.evidence} label="查看影子回放证据"/></>:null}
    {shadow.error||gate.error?<Notice tone="error">{shadow.error||gate.error}</Notice>:null}
    {gate.data?<Notice tone={gate.data.passed?'success':'neutral'}>{gate.data.passed?'当前满足发布提案条件':'尚未满足：'+((gate.data.failures as string[])||[]).join('、')}</Notice>:null}
    <label className="field">发布提案原因<textarea value={reason} onChange={e=>setReason(e.target.value)} maxLength={500}/></label>
    <button className="primary" disabled={busy||running||!gate.data?.passed||!canCompute||!reason.trim()} onClick={()=>void propose()}>创建发布提案</button>
    {proposal?<section className="workspace-review"><h3>审查具体提案</h3><ProposalReview value={proposal} canApprove={canApprove} onChanged={value=>{setProposal(value);onChanged();}}/></section>:null}
    <details className="workspace-danger"><summary>退休此工件</summary><p>退休后保留历史工件与预测记录。正在使用的模型需先通过提案回退。</p>
      <button disabled={busy||!canApprove||isCurrent||artifact.status==='retired'||artifact.status==='quarantined'} onClick={()=>setRetiring(true)}><Archive size={15}/>申请退休</button>
      {retiring?<Notice>确认退休 {text(descriptor.method_id)}（{targetName(descriptor.target)}），工件版本 {text(artifact.version)}？
        <button disabled={busy||!canApprove||isCurrent} onClick={()=>void perform(()=>request('/v2/model-artifacts/'+id+'/retire',{body:{expected_version:artifact.version}})).then(value=>{if(value)setRetiring(false);})}>确认退休</button></Notice>:null}
    </details>
    {error?<Notice tone="error">{error}</Notice>:null}
  </section>;
}

export function ReleaseHistory({canApprove,onChanged}:{canApprove:boolean;onChanged:()=>void}){
  const collection=useRemote<{items:RecordValue[]}>('/v2/releases');
  const pointers=useRemote<RecordValue>('/v2/model-pointers');
  const [target,setTarget]=useState('cu'),[destination,setDestination]=useState(''),[reason,setReason]=useState('');
  const [selected,setSelected]=useState(''),[busy,setBusy]=useState(false),[error,setError]=useState('');
  const detail=useRemote<RecordValue>(selected?'/v2/releases/'+selected:null);
  const close=useCallback(()=>setSelected(''),[]);
  function refresh(){collection.reload();pointers.reload();detail.reload();onChanged();}
  async function rollback(){
    setBusy(true);setError('');
    try{
      const current=await request<RecordValue>('/v2/model-pointers');
      const result=await request<RecordValue>('/v2/release-rollbacks',{body:{request_key:crypto.randomUUID(),target,
        expected_version:object(current[target]).version,destination_release_id:destination||null,reason}});
      setSelected(text(result.id));refresh();
    }catch(e){setError((e as Error).message);}finally{setBusy(false);}
  }
  const choices=(collection.data?.items||[]).filter(row=>row.status==='applied'&&object(row.payload).target===target&&object(row.payload).purpose==='forecast');
  return <section className="workspace-stack">
    <section className="workspace-panel"><Toolbar title="创建模型回退提案"/>
      <div className="workspace-form-grid">
        <label className="field">回退目标<select value={target} onChange={e=>{setTarget(e.target.value);setDestination('');}}><option value="cu">Cu · g/L</option><option value="as">As · mg/L</option></select></label>
        <label className="field">回退版本<select value={destination} onChange={e=>setDestination(e.target.value)}><option value="">上一版本（无历史时为 Persistence）</option>{choices.map(row=><option key={text(row.id)} value={text(row.id)}>{date(row.created_at)} · {candidateName(object(row.payload).candidate_id)}</option>)}</select></label>
      </div>
      <label className="field">回退原因<textarea value={reason} onChange={e=>setReason(e.target.value)} maxLength={500}/></label>
      <button disabled={busy||!canApprove||!reason.trim()||!pointers.data} title={!canApprove?'需要管理员权限':undefined} onClick={()=>void rollback()}><RotateCcw size={15}/>创建回退提案</button>
      <p className="workspace-note">创建提案后需要审查并批准，当前模型指针才会改变。</p>
    </section>
    <Toolbar title="发布与回退记录" onRefresh={refresh}/>
    {error||collection.error||pointers.error?<Notice tone="error">{error||collection.error||pointers.error}</Notice>:null}
    {collection.loading&&!collection.data?<Loading/>:collection.data?.items.length?<DataTable rows={collection.data.items} onRow={row=>setSelected(text(row.id))} columns={[
      {key:'created_at',label:'创建时间',render:row=>date(row.created_at)},
      {key:'operation',label:'操作',render:row=>object(row.payload).operation==='rollback'?'回退':'发布'},
      {key:'target',label:'目标',render:row=>targetName(object(row.payload).target)},
      {key:'candidate',label:'目标版本',render:row=>candidateName(object(row.payload).candidate_id)},
      {key:'status',label:'状态',render:row=><Status value={row.status}/>}
    ]}/>:<EmptyState title="尚无发布提案">工件验证通过后可创建发布提案；回退同样保留审批记录。</EmptyState>}
    {selected?<Drawer title="发布提案与审批" onClose={close}>{detail.loading&&!detail.data?<Loading/>:null}{detail.error?<Notice tone="error">{detail.error}</Notice>:null}
      {detail.data?<ProposalReview key={selected} value={detail.data} canApprove={canApprove} onChanged={refresh}/>:null}</Drawer>:null}
  </section>;
}
