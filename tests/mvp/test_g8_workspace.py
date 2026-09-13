from fastapi.testclient import TestClient
from copper_mvp.api import create_app
from copper_mvp.data import DataRepository


def test_workspace_summaries_use_current_actor_and_real_records(tmp_path):
    with TestClient(create_app(tmp_path,DataRepository()),base_url="http://127.0.0.1") as client:
        wb=client.app.state.workbench
        owner_code=wb.access.owner_key_path.read_text().strip()
        assert client.post("/api/auth/session",json={"access_code":owner_code}).status_code==200
        owner=wb.access.authenticate(key=owner_code)
        key=client.post("/api/v2/access-keys",json={"user_id":"reader","role":"researcher"}).json()["access_code"]
        reader=wb.access.authenticate(key=key)
        identities=[]
        for actor,label in ((owner,"owner-private-topic"),(reader,"reader-topic")):
            session=wb.research.store.create_session(actor,label,{})
            task,_=wb.research.store.create_task(actor,session["id"],{"question":label,"request_key":label,
                "context":{},"source_version":"synthetic","role":actor.role,"auth_ref":actor.key_hash})
            fence=wb.research.store.claim(task["id"],"fixture")
            wb.research.store.finish(task["id"],"fixture",fence,{"answer":"saved","model_answer":"saved"},1)
            identities.append(task["id"])
        headers={"Authorization":"Bearer "+key}
        summary=client.get("/api/v2/workspace",headers=headers)
        assert summary.status_code==200
        assert summary.json()["budget"]["scope"]=="own_research"
        assert summary.json()["user"]=={"id":"reader","role":"researcher"}
        tasks=client.get("/api/v2/workspace/tasks",headers=headers).json()["items"]
        assert any(item["id"]==identities[1] for item in tasks)
        assert all(item["id"]!=identities[0] for item in tasks)
        assert "owner-private-topic" not in str(tasks)
        assert client.get("/api/v2/workspace/research-tasks/"+identities[0]+"/events",headers=headers).status_code==404
        assert client.get("/api/v2/workspace/research-tasks/"+identities[1]+"/events",headers=headers).status_code==200
        audit=client.get("/api/v2/workspace/governance",headers=headers).json()
        assert all(row["task_id"]!=identities[0] for row in audit["events"])
        client.cookies.clear()
        assert client.get("/api/v2/workspace").status_code==401
