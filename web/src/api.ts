export type Metric = {model:string;target:string;n:number;mae:number;rmse:number;r2:number;skill_mae?:number;fold_id?:string;experiment?:string};
export type Bundle = {bundle_id:string;created_at:string;train_count:number;oof_events:number;metrics:Metric[];fold_metrics:Metric[];default_response:Record<string,string>};
export type EventRow = {event_id:string;decision_at:string;year:number;cu:number|null;as:number|null;mode:string[];mode_code:string;warnings:string[];fold_id:string|null;primary:boolean;supported_stages:number[];optimization_ready:boolean};
export type Signal = {tag:string;chinese_name:string;unit:string;group:string;current:number|null;mean:number|null;slope:number|null;count:number;anchors:{hours:number;value:number|null}[]};
export type Context = EventRow & {signals:Signal[];history:{time:string;cu:number|null;as:number|null}[];mode_evidence:{warnings:string[];evidence_observation_ids:string[];confidence:number}};
export type Candidate = {id:string;f1:number;f2:number;as:number|null;as_margin:number|null;support_distance:number|null;delta_cu:number|null;delta_power:number|null;variables:Record<string,number>;constraint_violation:number};
export type Prediction = {current:number|null;value:number|null;delta:number|null;model:string;unit:string};
export type Result = {kind:string;mode?:string;event_id?:string;decision_at?:string;predictions?:Record<string,Prediction>;mode_evidence?:Context['mode_evidence'];warnings?:string[];model_scope?:string;fold_id?:string;fit_cutoff_at?:string;bundle_id?:string;validation_metrics?:Metric[];metrics?:Metric[];train_count?:number;oof_events?:number;candidates?:Candidate[];reference?:{f1:number;f2:number;as:number|null;variables:Record<string,number>;feasible:boolean};representatives?:Record<string,string>;total_front_points?:number;objective_labels?:string[];objective_units?:string[];search_evaluations?:number;audit_evaluations?:number;feasible_rate?:number;elapsed_ms?:number;stop_reason?:string;note?:string;solution_status?:string;audit?:{passed:boolean;checks:string[]};models?:Record<string,string>;ranges?:{stage:number;current:number;lower:number;upper:number;historical_n:number}[];passed?:boolean;blocked?:boolean;code?:string;message?:string;scenario?:string};
export type Run = {run_id:string;task_type:string;request_key:string;status:string;created_at:string;started_at?:string;duration_ms?:number;request:Record<string,unknown>;result:Result|null;error:{message:string;code:string}|null;trace:{node:string;status:string;duration_ms:number;detail:Record<string,unknown>}[];selections:{candidate_id:string;label:string}[];reused?:boolean};
export type Explanation = {mode:string;text:string;sources:string[];cached?:boolean;fallback_reason?:string;estimated_cost_cny:number;input_tokens:number;output_tokens:number};
export type AgentResult = {kind:'agent_diagnosis';source_run_id:string;model:string;question:string;report:{summary:string;findings:{claim:string;status:'supported'|'inconclusive';evidence_ids:string[]}[];next_checks:string[]}|null;evidence:{evidence_id:string;tool:string;data:Record<string,unknown>}[];steps:{tool:string;purpose:string;evidence_id?:string;cached?:boolean;status:string;error?:{message:string}}[];usage:{api_calls:number;tool_calls:number;tool_cache_hits:number;input_tokens:number;output_tokens:number;cache_hit_tokens:number;estimated_cost_cny:number;unsettled_reservation_cny:number};elapsed_ms?:number};
export type AgentRun = Omit<Run,'result'> & {result:AgentResult|null};
export async function api<T>(path:string, body?:unknown, signal?:AbortSignal):Promise<T> {
  const response = await fetch('/api'+path,{method:body===undefined?'GET':'POST',headers:body===undefined?undefined:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body),signal});
  const value=await response.json();
  if(!response.ok){throw new Error(value.error?.message || (Array.isArray(value.detail)?value.detail.map((x:{msg:string})=>x.msg).join('；'):value.detail) || '请求失败');}
  return value;
}
export const fmt=(value:unknown,digits=2):string=>value===null||value===undefined||!Number.isFinite(Number(value))?'—':Number(value).toLocaleString('zh-CN',{maximumFractionDigits:digits,minimumFractionDigits:digits});
export const date=(value:string|undefined):string=>{
  if(!value)return '—';
  if(/(?:Z|[+-]\d{2}:\d{2})$/.test(value)){
    return new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hourCycle:'h23'}).format(new Date(value));
  }
  return value.replace('T',' ').slice(0,16);
};
export const statusLabel:Record<string,string>={queued:'排队中',running:'计算中',completed:'已完成',failed:'失败',cancelled:'已取消'};
export const taskLabel:Record<string,string>={train:'模型实验',predict:'事件预测',optimize:'多目标优化',diagnostic:'异常验证',agent_diagnostic:'智能体诊断'};
