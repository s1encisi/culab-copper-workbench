"""Model artifact lifecycle, exact release approvals and immutable prediction records."""
from __future__ import annotations

from datetime import datetime,timezone
import json
import sqlite3
import time
import uuid

from copper_mvp.common import WorkbenchError,digest,dumps
from copper_mvp.data_contracts import source_time
from copper_mvp.release_contracts import BASELINE_ID,TASK_ID,TARGET_UNITS,ArtifactRequest,ShadowRequest,ReleaseRequest,ReleaseApproval,RollbackRequest,ForecastRequest


class ModelLifecycle:
    def __init__(self,store,access,sources,*,clock=time.time):
        self.store,self.access,self.sources,self.clock=store,access,sources,clock
        self.connection=store.connection
        with self.connection() as c:
            if not c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_artifacts'").fetchone():
                backup=store.root/"migrations"/("before_g5c_"+uuid.uuid4().hex+".sqlite")
                backup.parent.mkdir(parents=True,exist_ok=True)
                with sqlite3.connect(backup) as target:c.backup(target)
            c.executescript("""
                CREATE TABLE IF NOT EXISTS model_artifacts (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL, owner_id TEXT NOT NULL,
                    request_key TEXT NOT NULL, fingerprint TEXT NOT NULL, descriptor TEXT NOT NULL,
                    status TEXT NOT NULL, version INTEGER NOT NULL, created_at REAL NOT NULL,
                    UNIQUE(project_id,owner_id,request_key));
                CREATE TABLE IF NOT EXISTS model_lifecycle_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, resource_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL, event TEXT NOT NULL, detail TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS model_shadows (
                    id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL, owner_id TEXT NOT NULL,
                    request_key TEXT NOT NULL, fingerprint TEXT NOT NULL, status TEXT NOT NULL,
                    evidence TEXT, evidence_hash TEXT, created_at REAL NOT NULL,
                    UNIQUE(candidate_id,owner_id,request_key));
                CREATE TABLE IF NOT EXISTS model_release_proposals (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL, owner_id TEXT NOT NULL, request_key TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL, payload TEXT NOT NULL, payload_hash TEXT NOT NULL,
                    proposer_auth TEXT NOT NULL, status TEXT NOT NULL, approval TEXT, created_at REAL NOT NULL,
                    UNIQUE(project_id,owner_id,request_key));
                CREATE TABLE IF NOT EXISTS model_pointers (
                    project_id TEXT NOT NULL, target TEXT NOT NULL, purpose TEXT NOT NULL,
                    candidate_id TEXT NOT NULL, release_id TEXT, version INTEGER NOT NULL,
                    PRIMARY KEY(project_id,target,purpose));
                CREATE TABLE IF NOT EXISTS release_predictions (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL, owner_id TEXT NOT NULL,
                    request_key TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    result TEXT NOT NULL, created_at REAL NOT NULL,
                    UNIQUE(project_id,owner_id,request_key));
            """)

    def now(self):
        return datetime.fromtimestamp(self.clock(),timezone.utc).isoformat()

    @staticmethod
    def auth(actor):
        return {"key_hash":actor.key_hash,"user_id":actor.user_id,"role":actor.role}

    def event(self,c,resource,actor,event,detail):
        c.execute("INSERT INTO model_lifecycle_events(resource_id,actor_id,event,detail,created_at) VALUES(?,?,?,?,?)",
                  (resource,actor,event,dumps(detail),self.clock()))

    @staticmethod
    def _artifact(c,identifier):
        row=c.execute("SELECT * FROM model_artifacts WHERE id=?",(identifier,)).fetchone()
        if not row:raise WorkbenchError("模型工件记录不存在","ARTIFACT_NOT_FOUND")
        result=dict(row);result["descriptor"]=json.loads(result["descriptor"])
        descriptor=result["descriptor"]
        if digest({k:v for k,v in descriptor.items() if k!="source_fingerprint"})!=descriptor.get("source_fingerprint"):
            raise WorkbenchError("工件登记内容哈希不符","SOURCE_CHANGED")
        return result

    def artifact(self,actor,identifier):
        actor.require("read")
        with self.connection() as c:
            result=self._artifact(c,identifier)
            if result["project_id"]!=actor.project_id or (result["owner_id"]!=actor.user_id and actor.role!="owner"):
                raise WorkbenchError("无权访问该工件","FORBIDDEN")
            events=[dict(r) for r in c.execute("SELECT * FROM model_lifecycle_events WHERE resource_id=? ORDER BY seq",(identifier,))]
        for event in events:event["detail"]=json.loads(event["detail"])
        return {**result,"events":events}

    def artifacts(self,actor):
        actor.require("read")
        with self.connection() as c:
            rows=c.execute("SELECT id FROM model_artifacts WHERE project_id=? AND (owner_id=? OR ?='owner') ORDER BY created_at DESC",
                           (actor.project_id,actor.user_id,actor.role)).fetchall()
        return [self.artifact(actor,row["id"]) for row in rows]

    def register(self,actor,request):
        actor.require("compute")
        request=ArtifactRequest.model_validate(request)
        fingerprint=digest(request.model_dump())
        with self.connection() as c:
            old=c.execute("SELECT * FROM model_artifacts WHERE project_id=? AND owner_id=? AND request_key=?",
                          (actor.project_id,actor.user_id,request.request_key)).fetchone()
        if old:
            if old["fingerprint"]!=fingerprint:raise WorkbenchError("请求键已用于不同工件","REQUEST_CONFLICT")
            return self.artifact(actor,old["id"])
        descriptor=self.sources.load(actor,request)
        identifier=uuid.uuid4().hex
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            old=c.execute("SELECT * FROM model_artifacts WHERE project_id=? AND owner_id=? AND request_key=?",
                          (actor.project_id,actor.user_id,request.request_key)).fetchone()
            if old:
                if old["fingerprint"]!=fingerprint:raise WorkbenchError("请求键已用于不同工件","REQUEST_CONFLICT")
                identifier=old["id"]
            else:
                c.execute("INSERT INTO model_artifacts VALUES(?,?,?,?,?,?,'benchmarked',1,?)",
                    (identifier,actor.project_id,actor.user_id,request.request_key,fingerprint,dumps(descriptor),self.clock()))
                for stage in ("candidate","registered","runnable","tested","benchmarked"):
                    self.event(c,identifier,actor.user_id,stage,{"source_fingerprint":descriptor["source_fingerprint"],
                        "basis":"verified existing training/evaluation artifacts and dependency versions"})
        return self.artifact(actor,identifier)

    def start_shadow(self,actor,candidate_id,request,executor=None):
        actor.require("compute")
        candidate=self.artifact(actor,candidate_id)
        if candidate["status"] not in ("benchmarked","shadow","approved"):
            raise WorkbenchError("工件状态不能开始影子验证","ARTIFACT_STATE")
        request=ShadowRequest.model_validate(request)
        fingerprint=digest({"request":request.model_dump(),"source":candidate["descriptor"]["source_fingerprint"]})
        identifier=uuid.uuid4().hex
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            old=c.execute("SELECT * FROM model_shadows WHERE candidate_id=? AND owner_id=? AND request_key=?",
                          (candidate_id,actor.user_id,request.request_key)).fetchone()
            if old:
                if old["fingerprint"]!=fingerprint:raise WorkbenchError("影子请求键冲突","REQUEST_CONFLICT")
                identifier=old["id"]
            else:
                c.execute("INSERT INTO model_shadows VALUES(?,?,?,?,?,'queued',NULL,NULL,?)",
                          (identifier,candidate_id,actor.user_id,request.request_key,fingerprint,self.clock()))
        if not old:
            if executor:executor.submit(self._execute_shadow,identifier,candidate,request,self.auth(actor))
            else:self._execute_shadow(identifier,candidate,request,self.auth(actor))
        return self.shadow(actor,identifier)

    def _execute_shadow(self,identifier,candidate,request,actor_auth):
        with self.connection() as c:c.execute("UPDATE model_shadows SET status='running' WHERE id=?",(identifier,))
        try:
            self.access.current(**actor_auth).require("compute")
            evidence=self.sources.probe(candidate["descriptor"],request.samples_per_fold)
            status="completed" if evidence["parity_passed"] else "failed"
        except Exception as exc:
            evidence={"error":{"code":getattr(exc,"code",type(exc).__name__),"message":str(exc)[:300]},"parity_passed":False}
            status="failed"
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            current=self._artifact(c,candidate["id"])
            if current["version"]!=candidate["version"] or current["status"] in ("retired","quarantined"):
                status="stale"
            c.execute("UPDATE model_shadows SET status=?,evidence=?,evidence_hash=? WHERE id=?",
                      (status,dumps(evidence),digest(evidence),identifier))
            if status=="completed" and current["status"]!="approved":
                c.execute("UPDATE model_artifacts SET status='shadow',version=version+1 WHERE id=?",(candidate["id"],))
            self.event(c,candidate["id"],actor_auth["user_id"],"shadow_"+status,{"shadow_id":identifier,"evidence_hash":digest(evidence)})

    def shadow(self,actor,identifier):
        with self.connection() as c:
            row=c.execute("SELECT * FROM model_shadows WHERE id=?",(identifier,)).fetchone()
        if not row:raise WorkbenchError("影子记录不存在","SHADOW_NOT_FOUND")
        result=dict(row)
        self.artifact(actor,result["candidate_id"])
        result["evidence"]=json.loads(result["evidence"]) if result["evidence"] else None
        if result["evidence"] and digest(result["evidence"])!=result["evidence_hash"]:
            raise WorkbenchError("影子证据哈希不符","SOURCE_CHANGED")
        return result

    def pointers(self,actor):
        actor.require("read")
        return {t:self.pointer(actor.project_id,t) for t in ("cu","as")}

    def pointer(self,project,target,purpose="forecast",c=None):
        if c is None:
            with self.connection() as connection:return self.pointer(project,target,purpose,connection)
        row=c.execute("SELECT * FROM model_pointers WHERE project_id=? AND target=? AND purpose=?",(project,target,purpose)).fetchone()
        return dict(row) if row else {"project_id":project,"target":target,"purpose":purpose,"candidate_id":BASELINE_ID,
                                     "release_id":None,"version":0}

    @staticmethod
    def quality_gate(descriptor,shadow,purpose):
        metric=descriptor["benchmark"]
        failures=[]
        if purpose=="scenario_prediction" and not descriptor.get("proxy_approved"):
            failures.append("PROXY_QUALIFICATION_REQUIRED")
        if descriptor.get("causal_control"):failures.append("CAUSAL_CONTROL_NOT_SUPPORTED")
        if metric["n"]<60 or metric["coverage"]<.95:failures.append("INSUFFICIENT_COMMON_COVERAGE")
        if metric["negative_predictions"]:failures.append("NEGATIVE_PREDICTIONS")
        if metric.get("p95_ms") is None or metric["p95_ms"]>100:failures.append("INFERENCE_BUDGET")
        improvement=1-metric["mae"]/metric["baseline_mae"] if metric["baseline_mae"]>0 else 0.
        if improvement<.05:failures.append("MINIMUM_IMPROVEMENT")
        interval=metric["paired_ci95"]
        if interval.get("high") is None or interval["high"]>=0:failures.append("PAIRED_INTERVAL")
        for mode,base in metric.get("baseline_modes",{}).items():
            current=metric.get("mode_metrics",{}).get(mode)
            if current and base["n"]>=30 and current["mae_mean_over_seeds"]>base["mae_mean_over_seeds"]*1.05:
                failures.append("WORST_MODE_DEGRADATION");break
        if not shadow or shadow["status"]!="completed" or not shadow["evidence"].get("parity_passed"):
            failures.append("RUNTIME_SHADOW_REQUIRED")
        return {"passed":not failures,"failures":failures,"relative_improvement":improvement,
                "policy":"release.forecast.fixed.v1","scope":"historical research forecast; shadow adds runtime verification only"}

    def qualification(self,actor,candidate_id,shadow_id=None,purpose="forecast"):
        candidate=self.artifact(actor,candidate_id)
        shadow=self.shadow(actor,shadow_id) if shadow_id else None
        if shadow and shadow["candidate_id"]!=candidate_id:
            raise WorkbenchError("影子证据不属于该工件","REQUEST_CONFLICT")
        return self.quality_gate(candidate["descriptor"],shadow,purpose)

    def proposal(self,actor,identifier):
        actor.require("read")
        with self.connection() as c:
            row=c.execute("SELECT * FROM model_release_proposals WHERE id=?",(identifier,)).fetchone()
        if not row:raise WorkbenchError("发布提案不存在","RELEASE_NOT_FOUND")
        row=dict(row)
        if row["project_id"]!=actor.project_id or (row["owner_id"]!=actor.user_id and actor.role!="owner"):
            raise WorkbenchError("无权访问该发布提案","FORBIDDEN")
        row["payload"]=json.loads(row["payload"]);row["approval"]=json.loads(row["approval"]) if row["approval"] else None
        row.pop("proposer_auth")
        return row

    def propose(self,actor,request,*,rollback=False):
        actor.require("compute")
        request=ReleaseRequest.model_validate(request)
        request_fingerprint=digest({"request":request.model_dump(),"rollback":rollback})
        with self.connection() as c:
            old=c.execute("SELECT * FROM model_release_proposals WHERE project_id=? AND owner_id=? AND request_key=?",
                          (actor.project_id,actor.user_id,request.request_key)).fetchone()
        if old:
            if old["request_fingerprint"]!=request_fingerprint:raise WorkbenchError("发布请求键冲突","REQUEST_CONFLICT")
            return self.proposal(actor,old["id"])
        pointer=self.pointer(actor.project_id,request.target,request.purpose)
        if pointer["version"]!=request.expected_version:
            raise WorkbenchError("模型指针版本已变化","VERSION_CONFLICT")
        candidate=None;shadow=None
        if request.candidate_id!=BASELINE_ID:
            candidate=self.artifact(actor,request.candidate_id)
            descriptor=candidate["descriptor"]
            if descriptor["target"]!=request.target or descriptor["task_id"]!=TASK_ID:
                raise WorkbenchError("工件目标或任务不匹配","RELEASE_SCOPE")
            if candidate["status"] not in ("shadow","approved"):
                raise WorkbenchError("工件尚未经过影子验证","ARTIFACT_STATE")
            self.sources.verify(descriptor)
            if not rollback:
                shadow=self.shadow(actor,request.shadow_id) if request.shadow_id else None
                if shadow and shadow["candidate_id"]!=candidate["id"]:
                    raise WorkbenchError("影子证据不属于该工件","REQUEST_CONFLICT")
                gate=self.quality_gate(descriptor,shadow,request.purpose)
                if not gate["passed"]:
                    raise WorkbenchError("未满足发布条件: "+", ".join(gate["failures"]),"RELEASE_NOT_QUALIFIED")
            elif candidate["status"]!="approved":
                raise WorkbenchError("回退目标未获批准","RELEASE_NOT_QUALIFIED")
        elif request.purpose!="forecast":
            raise WorkbenchError("Persistence 未取得响应代理资格","RELEASE_SCOPE")
        now=self.clock();identifier=uuid.uuid4().hex
        payload={"schema_version":"model-release.g5c.v1","id":identifier,"project_id":actor.project_id,"task_id":TASK_ID,
            "operation":"rollback" if rollback else "promote","target":request.target,"unit":TARGET_UNITS[request.target],
            "purpose":request.purpose,"scope":"historical_research","candidate_id":request.candidate_id,
            "candidate_version":candidate["version"] if candidate else 0,
            "source_fingerprint":candidate["descriptor"]["source_fingerprint"] if candidate else "builtin-persistence-v1",
            "shadow_id":shadow["id"] if shadow else None,"shadow_hash":shadow["evidence_hash"] if shadow else None,
            "previous":pointer,"expected_version":request.expected_version,"created_at":now,"expires_at":now+request.ttl_seconds,
            "policy":"release.forecast.fixed.v1","nonce":uuid.uuid4().hex,"reason":request.reason}
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            if self.pointer(actor.project_id,request.target,request.purpose,c)["version"]!=request.expected_version:
                raise WorkbenchError("模型指针版本已变化","VERSION_CONFLICT")
            c.execute("INSERT INTO model_release_proposals VALUES(?,?,?,?,?,?,?,?, 'pending',NULL,?)",
                (identifier,actor.project_id,actor.user_id,request.request_key,request_fingerprint,dumps(payload),
                 digest(payload),dumps(self.auth(actor)),now))
            self.event(c,identifier,actor.user_id,"release_proposed",{"payload_hash":digest(payload)})
        return self.proposal(actor,identifier)

    def approve(self,actor,identifier,decision):
        actor.require("approve")
        public=self.proposal(actor,identifier)
        decision=ReleaseApproval.model_validate(decision)
        if decision.decision=="reject":
            return self.reject(actor,identifier,decision)
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            row=dict(c.execute("SELECT * FROM model_release_proposals WHERE id=?",(identifier,)).fetchone())
            payload=json.loads(row["payload"])
            if digest(payload)!=row["payload_hash"] or decision.payload_hash!=row["payload_hash"]:
                raise WorkbenchError("审批哈希与提案不一致","REQUEST_CONFLICT")
            if row["status"]!="pending":
                if row["approval"] and json.loads(row["approval"])["decision"]==decision.decision:
                    return self.proposal(actor,identifier)
                raise WorkbenchError("发布提案状态不允许审批","RELEASE_STATE")
            if self.clock()>=payload["expires_at"]:raise WorkbenchError("发布提案已过期","RELEASE_EXPIRED")
            self.access.current(**json.loads(row["proposer_auth"])).require("compute")
            self.access.current(**self.auth(actor)).require("approve")
            pointer=self.pointer(actor.project_id,payload["target"],payload["purpose"],c)
            if pointer["version"]!=payload["expected_version"]:
                raise WorkbenchError("模型指针已改变，原提案失效","VERSION_CONFLICT")
            if payload["candidate_id"]!=BASELINE_ID:
                candidate=self._artifact(c,payload["candidate_id"])
                if candidate["version"]!=payload["candidate_version"] or candidate["status"] not in ("shadow","approved"):
                    raise WorkbenchError("工件状态或版本已变化","VERSION_CONFLICT")
                if candidate["descriptor"]["source_fingerprint"]!=payload["source_fingerprint"]:
                    raise WorkbenchError("工件内容已变化","SOURCE_CHANGED")
                self.sources.verify(candidate["descriptor"])
                if payload["operation"]=="promote":
                    shadow_row=c.execute("SELECT * FROM model_shadows WHERE id=?",(payload["shadow_id"],)).fetchone()
                    if not shadow_row or shadow_row["candidate_id"]!=candidate["id"] or shadow_row["evidence_hash"]!=payload["shadow_hash"]:
                        raise WorkbenchError("影子证据绑定已变化","SOURCE_CHANGED")
                    shadow=dict(shadow_row);shadow["evidence"]=json.loads(shadow["evidence"])
                    if digest(shadow["evidence"])!=shadow["evidence_hash"] or not self.quality_gate(candidate["descriptor"],shadow,payload["purpose"])["passed"]:
                        raise WorkbenchError("发布资格已变化","RELEASE_NOT_QUALIFIED")
            approval={"approved_by":actor.user_id,"role":actor.role,"at":self.now(),"decision":decision.decision,
                      "reason":decision.reason,"payload_hash":row["payload_hash"]}
            status="applied" if decision.decision=="approve" else "rejected"
            if status=="applied":
                c.execute("INSERT OR REPLACE INTO model_pointers VALUES(?,?,?,?,?,?)",
                    (actor.project_id,payload["target"],payload["purpose"],payload["candidate_id"],identifier,pointer["version"]+1))
                if payload["candidate_id"]!=BASELINE_ID:
                    c.execute("UPDATE model_artifacts SET status='approved',version=version+1 WHERE id=?",(payload["candidate_id"],))
                    self.event(c,payload["candidate_id"],actor.user_id,"approved",{"release_id":identifier})
            c.execute("UPDATE model_release_proposals SET status=?,approval=? WHERE id=?",(status,dumps(approval),identifier))
            self.event(c,identifier,actor.user_id,"release_"+status,{"pointer_version":pointer["version"]+(status=="applied")})
        return self.proposal(actor,identifier)

    def reject(self,actor,identifier,decision):
        actor.require("approve")
        self.access.current(**self.auth(actor)).require("approve")
        self.proposal(actor,identifier)
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            row=dict(c.execute("SELECT * FROM model_release_proposals WHERE id=?",(identifier,)).fetchone())
            payload=json.loads(row["payload"])
            if digest(payload)!=row["payload_hash"] or decision.payload_hash!=row["payload_hash"]:
                raise WorkbenchError("审批哈希与提案不一致","REQUEST_CONFLICT")
            if row["status"] not in ("pending","rejected"):
                raise WorkbenchError("已生效的发布需另行回退","RELEASE_STATE")
            if row["status"]=="pending":
                approval={"approved_by":actor.user_id,"role":actor.role,"at":self.now(),"decision":"reject",
                          "reason":decision.reason,"payload_hash":row["payload_hash"]}
                c.execute("UPDATE model_release_proposals SET status='rejected',approval=? WHERE id=?",(dumps(approval),identifier))
                self.event(c,identifier,actor.user_id,"release_rejected",{})
        return self.proposal(actor,identifier)

    def rollback(self,actor,request):
        actor.require("approve")
        request=RollbackRequest.model_validate(request)
        pointer=self.pointer(actor.project_id,request.target)
        if request.destination_release_id:
            destination=self.proposal(actor,request.destination_release_id)
            if destination["status"]!="applied" or destination["payload"]["target"]!=request.target or destination["payload"]["purpose"]!="forecast":
                raise WorkbenchError("回退记录不属于该目标的已发布版本","RELEASE_SCOPE")
            candidate_id=destination["payload"]["candidate_id"]
        elif pointer["release_id"]:
            candidate_id=self.proposal(actor,pointer["release_id"])["payload"]["previous"]["candidate_id"]
        else:
            candidate_id=BASELINE_ID
        return self.propose(actor,ReleaseRequest(request_key=request.request_key,candidate_id=candidate_id,
            target=request.target,expected_version=request.expected_version,reason=request.reason),rollback=True)

    def retire(self,actor,identifier,expected_version):
        actor.require("approve")
        self.artifact(actor,identifier)
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            candidate=self._artifact(c,identifier)
            if candidate["version"]!=expected_version:raise WorkbenchError("工件版本已改变","VERSION_CONFLICT")
            if c.execute("SELECT 1 FROM model_pointers WHERE candidate_id=?",(identifier,)).fetchone():
                raise WorkbenchError("请先通过发布提案回退当前指针","ARTIFACT_STATE")
            c.execute("UPDATE model_artifacts SET status='retired',version=version+1 WHERE id=?",(identifier,))
            self.event(c,identifier,actor.user_id,"retired",{})
        return self.artifact(actor,identifier)


    def pointer_at(self,project,target,cutoff,c):
        releases=c.execute("SELECT * FROM model_release_proposals WHERE project_id=? AND status='applied' ORDER BY created_at",
                           (project,)).fetchall()
        selected={"project_id":project,"target":target,"purpose":"forecast","candidate_id":BASELINE_ID,"release_id":None,"version":0}
        for row in releases:
            payload=json.loads(row["payload"]);approval=json.loads(row["approval"])
            if payload["target"]==target and payload["purpose"]=="forecast" and source_time(approval["at"])<=cutoff:
                version=payload["expected_version"]+1
                if version>selected["version"]:
                    selected={**selected,"candidate_id":payload["candidate_id"],"release_id":row["id"],"version":version}
        return selected

    @staticmethod
    def read_prediction_record(serialized):
        result=json.loads(serialized)
        expected=result.get("content_hash")
        if expected is None or digest({k:v for k,v in result.items() if k!="content_hash"})!=expected:
            raise WorkbenchError("预测记录内容哈希不符","SOURCE_CHANGED")
        return result

    def forecast(self,actor,request):
        actor.require("compute")
        request=ForecastRequest.model_validate(request)
        input_data=self.sources.event(request.event_id)
        fingerprint=digest({"request":request.model_dump(),"dataset":input_data["dataset_version"]})
        with self.connection() as c:
            existing=c.execute("SELECT * FROM release_predictions WHERE project_id=? AND owner_id=? AND request_key=?",
                               (actor.project_id,actor.user_id,request.request_key)).fetchone()
            if existing:
                if existing["fingerprint"]!=fingerprint:raise WorkbenchError("预测请求键冲突","REQUEST_CONFLICT")
                return self.read_prediction_record(existing["result"])
            c.execute("BEGIN")
            cutoff=source_time(input_data["decision_at"])
            pointers={t:self.pointer_at(actor.project_id,t,cutoff,c) if request.selection=="as_of_event"
                      else self.pointer(actor.project_id,t,c=c) for t in ("cu","as")}
        predictions={};warnings=[]
        for target,pointer in pointers.items():
            value=input_data["current"][target]
            result={"value":value,"unit":TARGET_UNITS[target],"model":"Persistence","artifact_id":BASELINE_ID,
                    "release_id":None,"pointer_version":pointer["version"],"fallback_reason":None}
            if value is None or value<0:
                result.update(value=None,fallback_reason="CURRENT_RESULT_MISSING_OR_INVALID")
                warnings.append({"target":target,"code":"CURRENT_RESULT_MISSING_OR_INVALID","message":"当前化验结果缺失或无效"})
                predictions[target]=result
                continue
            if pointer["candidate_id"]!=BASELINE_ID:
                try:
                    with self.connection() as c:candidate=self._artifact(c,pointer["candidate_id"])
                    allowed=("approved","retired") if request.selection=="as_of_event" else ("approved",)
                    if candidate["status"] not in allowed:
                        raise WorkbenchError("发布工件当前不可用","ARTIFACT_STATE")
                    descriptor=candidate["descriptor"]
                    if descriptor["target"]!=target:raise WorkbenchError("发布目标不匹配","RELEASE_SCOPE")
                    self.sources.verify(descriptor)
                    value,evidence=self.sources.predict(descriptor,request.event_id)
                    import math
                    if not math.isfinite(value) or value<0:
                        raise WorkbenchError("模型输出无效，使用参考模型","MODEL_OUTPUT_VALUES")
                    result.update(value=value,model=descriptor["method_id"],artifact_id=candidate["id"],
                                  release_id=pointer["release_id"],model_evidence=evidence)
                except WorkbenchError as exc:
                    result["fallback_reason"]=exc.code
                    warnings.append({"target":target,"code":exc.code,"message":str(exc)})
                    if exc.code in ("MODEL_HASH_MISMATCH","SOURCE_CHANGED","MODEL_VERSION_MISMATCH"):
                        with self.connection() as c:
                            c.execute("UPDATE model_artifacts SET status='quarantined',version=version+1 WHERE id=? AND status!='quarantined'",
                                      (pointer["candidate_id"],))
                            self.event(c,pointer["candidate_id"],"runtime_verifier","quarantined",{"reason":exc.code})
            predictions[target]=result
        identifier=uuid.uuid4().hex
        result={"schema_version":"release-prediction.g5c.v1","id":identifier,"project_id":actor.project_id,"task_id":TASK_ID,
                "trace_id":uuid.uuid4().hex,"created_by":actor.user_id,"source_kind":"registered-model-research","created_at":self.now(),
                "event_id":request.event_id,"snapshot_id":input_data["snapshot_id"],"input_decision_at":input_data["decision_at"],
                "computed_at":self.now(),"selection":request.selection,
                "evaluation_mode":"historical_replay","online_prediction_claim":False,
                "selection_timing_note":"Current research uses today's approved pointer; as_of_event uses only approvals already effective at the event time.",
                "pointer_snapshot":pointers,"predictions":predictions,"warnings":warnings}
        result["content_hash"]=digest(result)
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            existing=c.execute("SELECT * FROM release_predictions WHERE project_id=? AND owner_id=? AND request_key=?",
                               (actor.project_id,actor.user_id,request.request_key)).fetchone()
            if existing:
                if existing["fingerprint"]!=fingerprint:raise WorkbenchError("预测请求键冲突","REQUEST_CONFLICT")
                return self.read_prediction_record(existing["result"])
            c.execute("INSERT INTO release_predictions VALUES(?,?,?,?,?,?,?)",
                      (identifier,actor.project_id,actor.user_id,request.request_key,fingerprint,dumps(result),self.clock()))
        return result

    def prediction(self,actor,identifier):
        actor.require("read")
        with self.connection() as c:
            row=c.execute("SELECT * FROM release_predictions WHERE id=?",(identifier,)).fetchone()
        if not row:raise WorkbenchError("预测记录不存在","PREDICTION_NOT_FOUND")
        if row["project_id"]!=actor.project_id or (row["owner_id"]!=actor.user_id and actor.role!="owner"):
            raise WorkbenchError("无权访问该预测记录","FORBIDDEN")
        return self.read_prediction_record(row["result"])
