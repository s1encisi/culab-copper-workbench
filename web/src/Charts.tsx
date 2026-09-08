import { useEffect,useRef,useState } from 'react';
import { ChartNoAxesCombined } from 'lucide-react';
import { fmt } from './api';
type Point={x:number;y:number;id?:string;label?:string};
type Line={name:string;color:string;points:Point[]};
export function Chart({lines=[],scatter=[],reference,onSelect,selected,xLabel='',yLabel='',height=290,empty='暂无可展示的数据',xDate=false}:{lines?:Line[];scatter?:Point[];reference?:Point;onSelect?:(id:string)=>void;selected?:string;xLabel?:string;yLabel?:string;height?:number;empty?:string;xDate?:boolean}){
 const ref=useRef<HTMLDivElement>(null);const[width,setWidth]=useState(600);const[hover,setHover]=useState<Point|null>(null);
 useEffect(()=>{const observer=new ResizeObserver(([entry])=>setWidth(Math.max(260,entry.contentRect.width)));if(ref.current)observer.observe(ref.current);return()=>observer.disconnect();},[]);
 const all=[...lines.flatMap(l=>l.points),...scatter,...(reference?[reference]:[])].filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y));
 const pad={l:62,r:22,t:25,b:46};const w=width-pad.l-pad.r;const h=height-pad.t-pad.b;
 let xmin=Math.min(...all.map(p=>p.x)),xmax=Math.max(...all.map(p=>p.x)),ymin=Math.min(...all.map(p=>p.y)),ymax=Math.max(...all.map(p=>p.y));
 if(!all.length){xmin=0;xmax=1;ymin=0;ymax=1;}if(xmax===xmin){xmin-=.5;xmax+=.5;}if(ymax===ymin){ymin-=.5;ymax+=.5;}
 const yr=(ymax-ymin)*.12;ymin-=yr;ymax+=yr;
 const sx=(x:number)=>pad.l+(x-xmin)/(xmax-xmin)*w;const sy=(y:number)=>pad.t+h-(y-ymin)/(ymax-ymin)*h;
 return <div className="chart" ref={ref}>
 <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} aria-label={`${xLabel}与${yLabel}图`}>
 <title>{xLabel}与{yLabel}</title>
 {[0,.25,.5,.75,1].map(t=><g key={t}><line x1={pad.l} y1={pad.t+h*t} x2={width-pad.r} y2={pad.t+h*t} className="grid-line"/>{all.length>0&&<text x={pad.l-10} y={pad.t+h*t+4} textAnchor="end">{fmt(ymax-(ymax-ymin)*t,Math.abs(ymax)>100?0:2)}</text>}</g>)}
 <line x1={pad.l} y1={pad.t} x2={pad.l} y2={height-pad.b} className="axis"/><line x1={pad.l} y1={height-pad.b} x2={width-pad.r} y2={height-pad.b} className="axis"/>
 {[0,.5,1].map(t=>all.length>0&&<text key={t} x={pad.l+w*t} y={height-pad.b+19} textAnchor="middle">{xDate?new Date(xmin+(xmax-xmin)*t).toLocaleDateString('zh-CN',{month:'2-digit',day:'2-digit'}):fmt(xmin+(xmax-xmin)*t,Math.abs(xmax)>100?0:2)}</text>)}
 <text className="axis-title" x={pad.l+w/2} y={height-5} textAnchor="middle">{xLabel}</text><text className="axis-title" transform={`translate(16 ${pad.t+h/2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
 {lines.map(l=><polyline key={l.name} points={l.points.filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)).map(p=>`${sx(p.x)},${sy(p.y)}`).join(' ')} fill="none" stroke={l.color} strokeWidth={2} strokeLinejoin="round"/>)}
 {scatter.map((p,i)=><circle key={p.id||i} cx={sx(p.x)} cy={sy(p.y)} r={selected===p.id?7:4.5} fill={selected===p.id?'#b95125':'#21888a'} stroke="white" strokeWidth="1.5" tabIndex={onSelect?0:undefined} role={onSelect?'button':undefined} aria-label={p.label||`候选 ${i+1}`} onMouseEnter={()=>setHover(p)} onMouseLeave={()=>setHover(null)} onFocus={()=>setHover(p)} onBlur={()=>setHover(null)} onClick={()=>p.id&&onSelect?.(p.id)} onKeyDown={e=>{if((e.key==='Enter'||e.key===' ')&&p.id){e.preventDefault();onSelect?.(p.id);}}}/>)}
 {reference&&<g><path d={`M${sx(reference.x)} ${sy(reference.y)-7}l7 7-7 7-7-7Z`} fill="#b95125" stroke="white" strokeWidth="1.5"/><text x={Math.min(width-55,sx(reference.x)+10)} y={Math.max(14,sy(reference.y)-9)} fill="#a8532c">参考点</text></g>}
 </svg>
 {!all.length&&<div className="chart-empty"><ChartNoAxesCombined size={32}/><span>{empty}</span></div>}
 {hover&&<div className="chart-tooltip" style={{left:Math.min(width-175,Math.max(6,sx(hover.x))),top:Math.max(0,sy(hover.y)-72)}}>{hover.label}<br/>{fmt(hover.x,3)} · {fmt(hover.y,2)}</div>}
 {lines.length>0&&<div className="chart-legend">{lines.map(l=><span key={l.name}><i style={{background:l.color}}/>{l.name}</span>)}</div>}
 </div>
}
