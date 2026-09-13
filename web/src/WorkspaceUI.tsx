import {useCallback,useEffect,useRef,useState,type ReactNode} from 'react';
import {AlertCircle,ChevronRight,LoaderCircle,RefreshCw,X} from 'lucide-react';
import {date,fmt} from './api';

export type RecordValue=Record<string,unknown>;
export const object=(value:unknown):RecordValue=>value&&typeof value==='object'&&!Array.isArray(value)?value as RecordValue:{};
export const items=(value:unknown):RecordValue[]=>Array.isArray(value)?value.map(object):[];
export const text=(value:unknown,fallback='—')=>value===null||value===undefined?fallback:String(value);
export const labels:Record<string,string>={queued:'排队中',running:'运行中',completed:'已完成',failed:'失败',cancelled:'已取消',paused:'已暂停',pausing:'暂停请求中',cancelling:'取消请求中',needs_attention:'需要处理',partial:'部分完成',partial_failure:'部分失败',registered:'已登记',parsed:'已解析',indexed:'已索引',needs_review:'待审核',candidate:'候选',runnable:'可运行',tested:'已测试',benchmarked:'已比较',shadow:'影子验证',approved:'已批准',retired:'已退休',quarantined:'已隔离',revoked:'已撤销',heading:'标题',paragraph:'正文',list_item:'条款',table:'表格',equation:'公式',image:'图像',ocr_page:'OCR 页面',page:'页面',not_captured:'未创建',expired:'已过期',current:'有效',references_changed:'来源已变更'};
export async function request<T>(path:string,options:{method?:string;body?:unknown;signal?:AbortSignal}={}):Promise<T>{
  const binary=options.body instanceof Blob;
  const response=await fetch('/api'+path,{method:options.method||(options.body===undefined?'GET':'POST'),signal:options.signal,
    headers:options.body===undefined?undefined:{'Content-Type':binary?'application/octet-stream':'application/json'},
    body:options.body===undefined?undefined:binary?options.body as Blob:JSON.stringify(options.body)});
  if(response.status===401)window.dispatchEvent(new Event('culab:auth-required'));
  const value=await response.json();
  if(!response.ok)throw new Error(value.error?.message||(Array.isArray(value.detail)?value.detail.map((v:{msg:string})=>v.msg).join('；'):value.detail)||'请求未完成');
  return value as T;
}
export function useRemote<T>(path:string|null,revision=0){
  const [data,setData]=useState<T|null>(null);const[error,setError]=useState('');const[loading,setLoading]=useState(false);const[tick,setTick]=useState(0);
  const last=useRef<string|null>(null);
  useEffect(()=>{if(!path){setData(null);setLoading(false);return;}const controller=new AbortController();
    if(last.current!==path)setData(null);last.current=path;setLoading(true);setError('');
    request<T>(path,{signal:controller.signal}).then(v=>{if(!controller.signal.aborted)setData(v);})
      .catch(e=>{if(!controller.signal.aborted){setError(e.message);setData(null);}}).finally(()=>{if(!controller.signal.aborted)setLoading(false);});
    return()=>controller.abort();},[path,revision,tick]);
  return {data,error,loading,reload:useCallback(()=>setTick(v=>v+1),[])};
}
export function Status({value}:{value:unknown}){const key=text(value,'unknown');return <span className={'workspace-status '+key}><i/>{labels[key]||key}</span>;}
export function Notice({children,tone='neutral'}:{children:ReactNode;tone?:'neutral'|'error'|'success'}){return <div className={'workspace-notice '+tone} role={tone==='error'?'alert':undefined}><AlertCircle size={16}/><div>{children}</div></div>;}
export function EmptyState({title,children}:{title:string;children?:ReactNode}){return <div className="workspace-empty"><h3>{title}</h3>{children?<p>{children}</p>:null}</div>;}
export function Loading(){return <div className="workspace-loading"><LoaderCircle size={19} className="spin"/>正在读取…</div>;}
export function Toolbar({title,children,onRefresh}:{title?:string;children?:ReactNode;onRefresh?:()=>void}){return <div className="workspace-toolbar">{title?<h2>{title}</h2>:null}<div>{children}{onRefresh?<button className="secondary" onClick={onRefresh} aria-label="刷新内容"><RefreshCw size={15}/></button>:null}</div></div>;}
export function Tabs({value,onChange,options}:{value:string;onChange:(v:string)=>void;options:{id:string;label:string}[]}){return <div className="workspace-tabs" role="tablist">{options.map(v=><button role="tab" aria-selected={v.id===value} className={v.id===value?'selected':''} key={v.id} tabIndex={v.id===value?0:-1} onKeyDown={e=>{const i=options.findIndex(x=>x.id===v.id);const n=e.key==='ArrowRight'?(i+1)%options.length:e.key==='ArrowLeft'?(i+options.length-1)%options.length:e.key==='Home'?0:e.key==='End'?options.length-1:-1;if(n>=0){e.preventDefault();onChange(options[n].id);(e.currentTarget.parentElement?.querySelectorAll('button')[n] as HTMLButtonElement|undefined)?.focus();}}} onClick={()=>onChange(v.id)}>{v.label}</button>)}</div>;}
export function Facts({values}:{values:{label:string;value:ReactNode}[]}){return <dl className="workspace-facts">{values.map(v=><div key={v.label}><dt>{v.label}</dt><dd>{v.value}</dd></div>)}</dl>;}
export function JsonDetails({value,label='查看完整记录'}:{value:unknown;label?:string}){return <details className="workspace-json"><summary>{label}</summary><pre>{JSON.stringify(value,null,2)}</pre></details>;}
export function Drawer({title,onClose,children}:{title:string;onClose:()=>void;children:ReactNode}){
  const panel=useRef<HTMLElement>(null);
  useEffect(()=>{const old=document.activeElement as HTMLElement|null;panel.current?.focus();const key=(event:KeyboardEvent)=>{if(event.key==='Escape')onClose();if(event.key==='Tab'&&panel.current){const all=Array.from(panel.current.querySelectorAll<HTMLElement>('button,a[href],input,select,textarea,[tabindex="0"]')).filter(e=>!e.hasAttribute('disabled')&&e.getClientRects().length>0);const first=all[0],last=all.at(-1);if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus();}else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus();}}};document.addEventListener('keydown',key);return()=>{document.removeEventListener('keydown',key);old?.focus();};},[onClose]);
  return <div className="workspace-drawer-backdrop" onMouseDown={e=>{if(e.target===e.currentTarget)onClose();}}><aside className="workspace-drawer" ref={panel} tabIndex={-1} role="dialog" aria-modal="true" aria-label={title}><header><h2>{title}</h2><button onClick={onClose} aria-label="关闭详情"><X size={19}/></button></header>{children}</aside></div>;
}
export function DataTable({columns,rows,onRow}:{columns:{key:string;label:string;render?:(row:RecordValue)=>ReactNode}[];rows:RecordValue[];onRow?:(row:RecordValue)=>void}){
  return <div className="workspace-table-wrap"><table className="workspace-table"><thead><tr>{columns.map(c=><th key={c.key} scope="col">{c.label}</th>)}{onRow?<th scope="col"><span className="sr-only">详情</span></th>:null}</tr></thead><tbody>{rows.map((row,index)=><tr key={text(row.id??row.doc_id??row.method_id??row.run_id,index.toString())}>{columns.map(c=><td key={c.key}>{c.render?c.render(row):text(row[c.key])}</td>)}{onRow?<td><button className="workspace-row-action" onClick={()=>onRow(row)} aria-label={'查看 '+text(row.title??row.method_id??object(row.metadata).title??row.id??row.doc_id,'记录')}><ChevronRight size={17}/></button></td>:null}</tr>)}</tbody></table></div>;
}
export {date,fmt};
