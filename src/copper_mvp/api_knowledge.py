"""Authenticated local document and retrieval endpoints."""
from fastapi import APIRouter, Request, Response
from pydantic import AwareDatetime
from starlette.concurrency import run_in_threadpool

from copper_mvp.common import WorkbenchError
from copper_mvp.knowledge_contracts import DocumentSpec, DocumentAccess, DocumentReview, KnowledgeQuery, OCRRequest, DocumentCorrections
from copper_mvp.knowledge_parsing import MAX_BYTES
from copper_mvp.research_documents import DisclosureConsent


def knowledge_router():
    router = APIRouter(prefix="/api/v2/knowledge")

    def service(request):
        actor = getattr(request.state, "principal", None)
        if actor is None:
            raise WorkbenchError("需要本机访问码", "UNAUTHENTICATED")
        return request.app.state.workbench.knowledge, actor

    @router.get("/documents")
    def documents(request: Request, as_of: AwareDatetime | None = None):
        store, actor = service(request)
        return {"items": store.list(actor, as_of)}

    @router.post("/documents", status_code=201)
    def register(request: Request, payload: DocumentSpec):
        store, actor = service(request)
        return store.register(actor, payload)

    @router.get("/documents/{doc_id}/versions/{version}")
    def inspect(request: Request, doc_id: str, version: int):
        store, actor = service(request)
        return store.inspect(actor, doc_id, version)

    @router.put("/documents/{doc_id}/versions/{version}/content")
    async def upload(request: Request, doc_id: str, version: int):
        store, actor = service(request)
        store.authorize_upload(actor, doc_id, version)
        payload = bytearray()
        async for part in request.stream():
            if len(payload) + len(part) > MAX_BYTES:
                raise WorkbenchError("文档大小不能超过 16 MB", "DOCUMENT_SIZE")
            payload.extend(part)
        return await run_in_threadpool(store.upload, actor, doc_id, version, bytes(payload))

    @router.post("/documents/{doc_id}/versions/{version}/review")
    def review(request: Request, doc_id: str, version: int, payload: DocumentReview):
        store, actor = service(request)
        return store.review(actor, doc_id, version, payload.accepted, payload.note, payload.expected_parse_hash)

    @router.post("/documents/{doc_id}/versions/{version}/ocr")
    def ocr(request: Request, doc_id: str, version: int, payload: OCRRequest):
        store, actor = service(request)
        return store.processing.run_ocr(actor, doc_id, version, **payload.model_dump())

    @router.post("/documents/{doc_id}/versions/{version}/corrections")
    def corrections(request: Request, doc_id: str, version: int, payload: DocumentCorrections):
        store, actor = service(request)
        return store.processing.correct(actor, doc_id, version, payload)

    @router.get("/documents/{doc_id}/versions/{version}/parses")
    def parse_history(request: Request, doc_id: str, version: int):
        store, actor = service(request)
        return {"items": store.processing.history(actor, doc_id, version)}

    @router.get("/documents/{doc_id}/versions/{version}/parses/{revision}")
    def historical_parse(request: Request, doc_id: str, version: int, revision: int):
        store, actor = service(request)
        return store.processing.historical_parse(actor, doc_id, version, revision)

    @router.get("/documents/{doc_id}/versions/{version}/pages/{page}.png")
    def page_image(request: Request, doc_id: str, version: int, page: int, as_of: AwareDatetime | None = None):
        store, actor = service(request)
        raw = store.processing.page_image(actor, doc_id, version, page, as_of)
        return Response(raw, media_type="image/png", headers={"Cache-Control": "no-store"})

    @router.get("/inventory")
    def inventory(request: Request):
        store, actor = service(request)
        return {"items": store.processing.inventory(actor)}

    @router.post("/documents/{doc_id}/versions/{version}/index")
    def index(request: Request, doc_id: str, version: int):
        store, actor = service(request)
        return store.build_index(actor, doc_id, version)

    @router.put("/documents/{doc_id}/access")
    def access(request: Request, doc_id: str, payload: DocumentAccess):
        store, actor = service(request)
        return store.change_access(actor, doc_id, payload)

    @router.post("/documents/{doc_id}/versions/{version}/disclosures", status_code=201)
    def grant_disclosure(request: Request, doc_id: str, version: int, payload: DisclosureConsent):
        _, actor = service(request)
        return request.app.state.workbench.research.documents.grant(actor, doc_id, version, payload)

    @router.get("/documents/{doc_id}/versions/{version}/disclosures")
    def disclosures(request: Request, doc_id: str, version: int):
        _, actor = service(request)
        return {"items": request.app.state.workbench.research.documents.consents(actor, doc_id, version)}

    @router.post("/documents/{doc_id}/disclosures/{consent_id}/revoke")
    def revoke_disclosure(request: Request, doc_id: str, consent_id: str):
        _, actor = service(request)
        return request.app.state.workbench.research.documents.revoke(actor, doc_id, consent_id)

    @router.delete("/documents/{doc_id}")
    def delete(request: Request, doc_id: str):
        store, actor = service(request)
        return store.delete(actor, doc_id)

    @router.get("/documents/{doc_id}/versions/{version}/source")
    def source(request: Request, doc_id: str, version: int, as_of: AwareDatetime | None = None):
        store, actor = service(request)
        raw, format = store.source(actor, doc_id, version, as_of)
        media = {"md": "text/plain; charset=utf-8", "pdf": "application/pdf",
                 "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}[format]
        return Response(raw, media_type=media, headers={"Content-Disposition": f'attachment; filename="document-v{version}.{format}"',
                                                       "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @router.post("/search")
    def search(request: Request, payload: KnowledgeQuery):
        store, actor = service(request)
        return store.search(actor, **payload.model_dump())

    @router.get("/citations/{chunk_id}")
    def citation(request: Request, chunk_id: str, as_of: AwareDatetime | None = None):
        store, actor = service(request)
        return store.resolve(actor, chunk_id, as_of)

    return router
