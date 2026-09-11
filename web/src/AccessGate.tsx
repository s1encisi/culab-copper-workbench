import {useEffect,useState,type FormEvent,type ReactNode} from 'react';
import {KeyRound,LoaderCircle,LogOut} from 'lucide-react';
import {api} from './api';

type Access={authenticated:boolean;user:string|null;role:string|null;access_code_location:string};
export function AccessGate({children}:{children:ReactNode}){
  const [access,setAccess]=useState<Access|null>(null);
  const [code,setCode]=useState('');
  const [error,setError]=useState('');
  const [submitting,setSubmitting]=useState(false);
  useEffect(()=>{
    const controller=new AbortController();
    api<Access>('/auth/status',undefined,controller.signal).then(setAccess).catch(e=>{if(!controller.signal.aborted)setError(e.message);});
    const expired=()=>setAccess(v=>v?{...v,authenticated:false}:v);
    addEventListener('culab:auth-required',expired);
    return()=>{controller.abort();removeEventListener('culab:auth-required',expired);};
  },[]);
  async function login(event:FormEvent){
    event.preventDefault();setSubmitting(true);setError('');
    try{await api('/auth/session',{access_code:code});setCode('');setAccess(await api<Access>('/auth/status'));}
    catch(e){setError(e instanceof Error?e.message:'登录失败');}
    finally{setSubmitting(false);}
  }
  if(access?.authenticated)return <>{children}<button className="access-logout" aria-label="退出本机访问" onClick={async()=>{await api('/auth/logout',{});setAccess({...access,authenticated:false});}}><LogOut size={15}/>{access.user}</button></>;
  return <div className="access-screen"><section className="access-card">
    <div className="access-brand">CuLab<span>本机研究工作台</span></div>
    <div className="access-icon"><KeyRound size={25}/></div><h1>进入研究工作台</h1>
    <p>输入本机访问码，继续查看数据、模型与研究任务。</p>
    <form onSubmit={login}><label className="field">本机访问码<input type="password" autoComplete="current-password" value={code} onChange={e=>setCode(e.target.value)} required autoFocus/></label>
      <button className="primary" disabled={submitting||!code.trim()}>{submitting?<LoaderCircle size={17} className="spin"/>:null}进入工作台</button></form>
    {error?<p className="error-box" role="alert">{error}</p>:null}
    <small>访问码文件：runs/mvp/owner_access.key<br/>自定义运行目录时，使用该目录中的同名文件。</small>
    {!access&&!error?<span className="muted">正在连接本机服务…</span>:null}
  </section></div>;
}
