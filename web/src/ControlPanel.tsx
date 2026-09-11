import {useEffect,useState} from 'react';
import {Check,Clock3,RefreshCw,ShieldCheck,X} from 'lucide-react';
import {api,fmt} from './api';

type Info={connected:boolean;can_control:boolean;can_test_admin:boolean;notice:string};
type Point={sp:number;pv:number;unit:string;quality:string;sample_time:number;max_delta:number;max_rate_per_second:number};
type Device={device_id:string;mode:string;connected:boolean;running:boolean;estop_latched:boolean;interlocks_ok:boolean;state_version:number;device_epoch:number;observation_seq:number;virtual_time:number;points:Record<string,Point>};
type Target={point_id:string;value:number;unit:string};
type Feedback={target:number;sp?:number;pv?:number;unit?:string;quality?:string;good_samples?:number;status:string};
type Command={id:string;status:string;payload_hash:string;cancel_requested:boolean;created_at:number;payload:{request:{targets:Target[];reason:string;ramp_seconds:number;tolerance:number;settling_deadline_seconds:number};expected_state_version:number;expected_device_epoch:number;expires_at:number};result:{ack_received?:boolean;ack_confirmed_by_status?:boolean;reason?:string;points?:Record<string,Feedback>};approval:{approver_id:string}|null;events:{seq:number;kind:string;occurred_at:number}[];outbox:{attempts?:number}};
const labels:Record<string,string>={WAITING_APPROVAL:'待审批',QUEUED:'待执行',DISPATCHING:'正在发送',VERIFYING:'正在回读',VERIFIED:'已验证达到',PARTIAL:'部分写入',FAILED:'未达到条件',REJECTED:'已拒绝',CANCELLED:'已取消',EXPIRED:'已过期',UNKNOWN_OUTCOME:'结果待协调'};
const terminal=new Set(['VERIFIED','PARTIAL','FAILED','REJECTED','CANCELLED','EXPIRED']);
const pointNames:Record<string,string>={'mock.cell3.current_setpoint':'主电流','mock.cell3.aux_current_setpoint':'辅助电流'};
const when=(seconds:number)=>new Date(seconds*1000).toLocaleTimeString('zh-CN',{hour12:false});

export function ControlPanel(){
  const [info,setInfo]=useState<Info|null>(null);
  const [device,setDevice]=useState<Device|null>(null);
  const [commands,setCommands]=useState<Command[]>([]);
  const [selected,setSelected]=useState('');
  const [command,setCommand]=useState<Command|null>(null);
  const [target,setTarget]=useState('105');
  const [reason,setReason]=useState('合成设备设定调整');
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState('');
  const [revision,setRevision]=useState(0);

  useEffect(()=>{
    const controller=new AbortController();let timer:ReturnType<typeof setTimeout>;
    async function load(){
      try{
        const availability=await api<Info>('/v2/control',undefined,controller.signal);setInfo(availability);
        const [history,state,current]=await Promise.all([
          api<{items:Command[]}>('/v2/commands',undefined,controller.signal),
          availability.connected?api<Device>('/v2/mock/devices/mock.cell3/state',undefined,controller.signal):Promise.resolve(null),
          selected?api<Command>('/v2/commands/'+selected,undefined,controller.signal):Promise.resolve(null),
        ]);
        if(controller.signal.aborted)return;
        setCommands(history.items);setDevice(state);setCommand(current);setError('');
      }catch(e){if(!controller.signal.aborted)setError(e instanceof Error?e.message:'读取失败');}
      if(!controller.signal.aborted)timer=setTimeout(load,800);
    }
    void load();return()=>{controller.abort();clearTimeout(timer);};
  },[selected,revision]);

  async function propose(){
    setBusy(true);setError('');
    try{
      const value=await api<Command>('/v2/commands',{request_key:crypto.randomUUID(),reason:reason.trim(),ttl_seconds:60,
        targets:[{point_id:'mock.cell3.current_setpoint',value:Number(target),unit:'A'}]});
      setCommand(value);setSelected(value.id);setRevision(n=>n+1);
    }catch(e){setError(e instanceof Error?e.message:'提案未创建');}
    finally{setBusy(false);}
  }
  async function action(name:string,body:unknown={}){
    if(!command)return;setBusy(true);setError('');
    try{setCommand(await api<Command>('/v2/commands/'+command.id+'/'+name,body));setRevision(n=>n+1);}
    catch(e){setError(e instanceof Error?e.message:'操作未完成');}
    finally{setBusy(false);}
  }
  const expired=!!command&&Date.now()/1000>=command.payload.expires_at;
  const waiting=command?.status==='WAITING_APPROVAL';
  return <div className="control-page">
    <div className="mock-notice"><ShieldCheck size={18}/><strong>Mock 环境，真实工厂未连接</strong><span>合成点表 · 0–200 A</span></div>
    {error?<div className="error-box" role="alert">{error}</div>:null}
    {!info?<p>正在读取 Mock 配置…</p>:!info.connected?<div className="panel control-card"><h2>Mock 服务未连接</h2><p>请按本地运行说明启动独立模拟器，配置工作台的 Mock 连接后刷新。</p><button className="secondary" onClick={()=>setRevision(n=>n+1)}><RefreshCw size={16}/>重新读取</button></div>:null}
    {device?<div className="panel control-card">
      <div className="control-heading"><div><h2>合成电流设备</h2><p>{device.device_id} · {device.mode==='remote'?'远程模式':device.mode==='manual'?'手动模式':'维护模式'} · {device.connected?'连接正常':'已断线'}</p></div><span className="status">{device.estop_latched?'急停锁存':!device.interlocks_ok?'联锁未满足':'联锁正常'}</span></div>
      <div className="control-points">{Object.entries(device.points).map(([id,p])=><section key={id}><h3>{pointNames[id]||id}</h3><div><span>设定 SP<strong>{fmt(p.sp,2)} <small>{p.unit}</small></strong></span><span>测量 PV<strong>{fmt(p.pv,2)} <small>{p.unit}</small></strong></span></div><p>质量 {p.quality==='good'?'合格':'异常'} · 允许单次变化 ≤{p.max_delta} {p.unit}</p></section>)}</div>
      <small>控制版本 {device.state_version} · 启动代次 {device.device_epoch} · 观测序号 {device.observation_seq} · 模拟时间 {fmt(device.virtual_time,1)} s</small>
    </div>:null}
    <div className="control-columns">
      <div>
        {info?.can_control?<form className="panel control-card" onSubmit={e=>{e.preventDefault();void propose();}}>
          <h2>创建调整提案</h2>
          <label className="field" htmlFor="mock-target">主电流目标值（A）<input id="mock-target" type="number" min="0" max="200" step="0.1" value={target} onChange={e=>setTarget(e.target.value)} required/></label>
          <label className="field" htmlFor="mock-reason">调整说明<input id="mock-reason" value={reason} onChange={e=>setReason(e.target.value)} maxLength={500} required/></label>
          <p className="muted">变化量上限 10 A，1 秒完成设定变化。测量值需在 10 秒内进入 ±1 A 并连续 3 次合格。</p>
          <button className="primary" disabled={busy||!device||!target||!reason.trim()}>生成可审批提案</button>
        </form>:null}
        <section className="panel control-card"><h2>命令记录</h2>
          {!commands.length?<p className="muted">还没有命令。创建提案后可审查具体参数。</p>:commands.map(c=><button key={c.id} className={'command-row '+(selected===c.id?'selected':'')} onClick={()=>{setCommand(null);setSelected(c.id);}}><span>{c.id.slice(0,8)}<small>{when(c.created_at)}</small></span><span>{labels[c.status]||c.status}</span></button>)}
        </section>
      </div>
      <section className="panel control-card control-review" aria-live="polite">
        {!command?<div className="empty"><ShieldCheck size={28}/><h2>审查命令与执行反馈</h2><p>选择一条命令，查看精确参数、审批记录和设备回读。</p></div>:<>
          <div className="control-heading"><h2>命令审查</h2><span className={'status '+(command.status==='VERIFIED'?'completed':command.status==='FAILED'?'failed':'')}>{labels[command.status]||command.status}</span></div>
          {command.payload.request.targets.map(t=><div className="command-target" key={t.point_id}><span>{pointNames[t.point_id]||t.point_id}</span><strong>{fmt(t.value)} {t.unit}</strong></div>)}
          <p>{command.payload.request.reason}</p>
          <dl className="command-facts"><dt>绑定控制版本 / 代次</dt><dd>{command.payload.expected_state_version} / {command.payload.expected_device_epoch}</dd><dt>允许发送至</dt><dd>{when(command.payload.expires_at)}{waiting&&expired?' · 已过期':''}</dd><dt>回读条件</dt><dd>±{command.payload.request.tolerance} A · 连续 3 次 · {command.payload.request.settling_deadline_seconds} 秒</dd><dt>审批者</dt><dd>{command.approval?.approver_id||'待审批'}</dd><dt>已发送次数</dt><dd>{command.outbox.attempts||0}</dd></dl>
          {waiting&&info?.can_control?<div className="control-actions"><button className="primary" disabled={busy||expired} onClick={()=>action('approvals',{payload_hash:command.payload_hash})}><Check size={16}/>批准并执行此提案</button><button className="secondary" disabled={busy} onClick={()=>action('approvals',{payload_hash:command.payload_hash,decision:'reject',reason:'人工拒绝'})}><X size={16}/>拒绝</button></div>:null}
          {!waiting&&!terminal.has(command.status)&&info?.can_control?<div className="control-actions"><button className="secondary" disabled={busy} onClick={()=>action('reconcile')}><RefreshCw size={16}/>查询执行结果</button><button className="secondary" disabled={busy||!!command.cancel_requested} onClick={()=>action('cancel')}>{command.cancel_requested?'已请求取消':'取消后续步骤'}</button></div>:null}
          {command.cancel_requested?<p className="muted">已提交的写入继续回读；取消不会恢复此前设定值。</p>:null}
          {command.result.reason?<div className="control-result"><strong>{command.result.reason}</strong><p>ACK {command.result.ack_received?'已收到':command.result.ack_confirmed_by_status?'由设备状态确认':'未确认'}</p>
            {Object.entries(command.result.points||{}).map(([id,p])=><p key={id}>{pointNames[id]||id}：{p.status==='rejected'?'设备拒绝':<>SP {fmt(p.sp)} / PV {fmt(p.pv)} A · 连续合格 {p.good_samples||0}/3</>}</p>)}
          </div>:null}
          <ol className="command-timeline">{command.events.map(event=><li key={event.seq}><Clock3 size={14}/><span>{labels[event.kind]||({PROPOSED:'提案已创建',CANCEL_REQUESTED:'已请求取消',APPROVAL_REVOKED:'审批已撤销'}[event.kind]||event.kind)}</span><time>{when(event.occurred_at)}</time></li>)}</ol>
          <details><summary>完整提案与审批哈希</summary><p className="command-hash">{command.payload_hash}</p><pre>{JSON.stringify(command.payload,null,2)}</pre></details>
        </>}
      </section>
    </div>
  </div>;
}
