import {useCallback,useEffect,useState} from 'react';
import {Clock3,ExternalLink,Pause,Play,Square} from 'lucide-react';
import {DataTable,Drawer,EmptyState,Facts,JsonDetails,Loading,Notice,Status,Tabs,Toolbar,date,fmt,items,object,request,text,useRemote,type RecordValue} from './WorkspaceUI';

const routes:Record<string,string>={research:'/v2/tasks/',run:'/runs/',model_comparison:'/v2/model-comparisons/',optimizer_comparison:'/v2/optimizer-comparisons/',calibration:'/v2/calibrations/'};
const kinds:Record<string,string>={research:'研究对话',run:'计算运行',model_comparison:'模型比较',optimizer_comparison:'优化比较',calibration:'区间校准'};
const active=new Set(['queued','running','pausing','cancelling']);
export function TasksWorkspace({onOpenRun}:{onOpenRun:(id:string)=>void}){
  const collection=useRemote<{items:RecordValue[]}>('/v2/workspace/tasks');
  const access=useRemote<RecordValue>('/v2/workspace');const canControl=!!access.data&&object(access.data.user).role!=='viewer';
  const [filter,setFilter]=useState('all');const[selected,setSelected]=useState<RecordValue|null>(null);const[error,setError]=useState('');
  const close=useCallback(()=>setSelected(null),[]);
  const detail=useRemote<RecordValue>(selected?(routes[text(selected.kind)]||'/runs/')+text(selected.id):null);
  const trace=useRemote<{items:RecordValue[]}>(selected?.kind==='research'?'/v2/workspace/research-tasks/'+text(selected.id)+'/events':null);
  useEffect(()=>{if(!selected||!active.has(text(detail.data?.status??selected.status)))return;const timer=setInterval(()=>{detail.reload();trace.reload();collection.reload();},1500);return()=>clearInterval(timer);},[selected,detail.data?.status,detail.reload,trace.reload,collection.reload]);
  async function control(action:string){if(!selected)return;setError('');try{const latest=await request<RecordValue>('/v2/tasks/'+text(selected.id));await request('/v2/tasks/'+text(selected.id)+'/'+action,{body:{expected_version:latest.version}});detail.reload();trace.reload();collection.reload();}catch(e){setError((e as Error).message);}}
  const rows=(collection.data?.items||[]).filter(row=>filter==='all'||(filter==='active'?active.has(text(row.status)):row.status===filter));
  return <section className="workspace-stack"><Toolbar title="任务记录" onRefresh={collection.reload}><select aria-label="筛选任务状态" value={filter} onChange={e=>setFilter(e.target.value)}><option value="all">全部状态</option><option value="active">执行中</option><option value="completed">已完成</option><option value="failed">失败</option><option value="needs_attention">需要处理</option></select></Toolbar>
    {collection.error?<Notice tone="error">{collection.error}</Notice>:null}
    {collection.loading&&!collection.data?<Loading/>:rows.length?<DataTable rows={rows} onRow={setSelected} columns={[{key:'title',label:'任务',render:r=><div className="workspace-task-title"><strong>{text(r.title)}</strong><small>{text(r.id)}</small></div>},{key:'kind',label:'类型',render:r=>kinds[text(r.kind)]||text(r.kind)},{key:'status',label:'状态',render:r=><Status value={r.status}/>},{key:'created_at',label:'创建时间',render:r=>date(text(r.created_at))},{key:'elapsed_ms',label:'耗时 / ms',render:r=>fmt(r.elapsed_ms,0)}]}/>:<EmptyState title="还没有任务记录">从数据、模型池或优化实验室开始一次研究。</EmptyState>}
    {selected?<Drawer title={text(selected.title)} onClose={close}>{detail.loading&&!detail.data?<Loading/>:null}{detail.error?<Notice tone="error">{detail.error}</Notice>:null}<Status value={detail.data?.status??selected.status}/><Facts values={[{label:'任务类型',value:kinds[text(selected.kind)]},{label:'记录编号',value:<code>{text(selected.id)}</code>},{label:'创建时间',value:date(text(selected.created_at))}]}/>
      {selected.kind==='research'?<div className="workspace-actions">{active.has(text(detail.data?.status))?<button disabled={!canControl} title={!canControl?'需要研究员或管理员权限':undefined} onClick={()=>void control('pause')}><Pause size={15}/>请求暂停</button>:null}{detail.data?.status==='paused'?<button disabled={!canControl} title={!canControl?'需要研究员或管理员权限':undefined} onClick={()=>void control('resume')}><Play size={15}/>恢复任务</button>:null}{active.has(text(detail.data?.status))||detail.data?.status==='paused'?<button disabled={!canControl} title={!canControl?'需要研究员或管理员权限':undefined} onClick={()=>void control('cancel')}><Square size={15}/>请求取消</button>:null}</div>:null}
      {selected.kind==='run'?<button onClick={()=>onOpenRun(text(selected.id))}><ExternalLink size={15}/>打开原运行</button>:null}
      {error?<Notice tone="error">{error}</Notice>:null}
      <ol className="workspace-timeline">{(trace.data?.items||items(detail.data?.trace)).map((row,i)=><li key={text(row.seq,i.toString())}><Clock3 size={15}/><div><strong>{text(row.kind??row.node)}</strong><small>{date(text(row.created_at??row.started_at))}</small><JsonDetails value={row.payload??row.detail}/></div></li>)}</ol>
      {object(detail.data?.error).message?<Notice tone="error">{text(object(detail.data?.error).message)}</Notice>:null}<JsonDetails value={detail.data}/></Drawer>:null}
  </section>;
}

export function GovernanceWorkspace({diagnostics}:{diagnostics:React.ReactNode}){
  const [tab,setTab]=useState('overview');const summary=useRemote<RecordValue>('/v2/workspace');const audit=useRemote<RecordValue>('/v2/workspace/governance');
  const budget=object(summary.data?.budget),usage=object(audit.data?.usage),counts=object(summary.data?.research_tasks);
  return <section className="workspace-stack"><Tabs value={tab} onChange={setTab} options={[{id:'overview',label:'费用与状态'},{id:'audit',label:'研究审计'},{id:'checks',label:'异常路径检查'}]}/>
    {tab==='checks'?diagnostics:tab==='overview'?<><Toolbar title="评估与治理" onRefresh={()=>{summary.reload();audit.reload();}}/>
      {summary.error?<Notice tone="error">{summary.error}</Notice>:null}
      <div className="workspace-metric-grid">{[{label:'已结算记录 / 元',value:fmt(budget.spent,4)},{label:'待结算预留 / 元',value:fmt(budget.reserved,4)},{label:'当前账号模型调用',value:fmt(usage.calls,0)},{label:'当前账号输入 token',value:fmt(usage.input_tokens,0)}].map(v=><div className="workspace-metric" key={v.label}><span>{v.label}</span><strong>{v.value}</strong></div>)}</div>
      <section className="workspace-panel"><Facts values={[{label:'费用范围',value:budget.scope==='project'?'当前项目记录':'当前账号研究对话'},{label:'账号角色',value:text(object(summary.data?.user).role)},{label:'真实模型调用',value:summary.data?.assistant_live?'已启用':'尚未启用'},{label:'已完成研究任务',value:fmt(counts.completed??0,0)},{label:'失败研究任务',value:fmt(counts.failed??0,0)},{label:'需要处理',value:fmt(counts.needs_attention??0,0)}]}/><Notice>费用来自运行账本。未结算请求保留预留，不能视为没有发生调用。</Notice></section>
    </>:<><Toolbar title="研究审计记录" onRefresh={audit.reload}/>{audit.error?<Notice tone="error">{audit.error}</Notice>:null}{items(audit.data?.events).length?<DataTable rows={items(audit.data?.events)} columns={[{key:'kind',label:'事件'},{key:'task_id',label:'任务',render:r=><code>{text(r.task_id).slice(0,12)}</code>},{key:'detail',label:'记录',render:r=><JsonDetails value={r.detail} label="查看详情"/>}]}/>:<EmptyState title="当前账号还没有审计事件">执行研究任务后会显示调用、工具与恢复记录。</EmptyState>}</>}
  </section>;
}

export {default as AdaptationWorkspace} from './DomainWorkspace';
