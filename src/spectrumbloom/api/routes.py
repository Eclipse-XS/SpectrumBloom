"""REST API endpoints for SpectrumBloom."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile, status

from spectrumbloom.core.security import verify_api_token
from spectrumbloom.domain.schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    DatasetPreviewResponse,
    FullEDAReportResponse,
    MLExperimentRequest,
    MLExperimentResponse,
    QueryRequest,
    QueryResponse,
    SessionStatusResponse,
    UpdateProviderRequest,
)
from spectrumbloom.services.session_service import SpectrumBloomService

router = APIRouter(prefix="/api/v1", dependencies=[Depends(verify_api_token)])


def get_service(request: Request) -> SpectrumBloomService:
    return request.app.state.service


@router.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Sessions"],
)
def create_session(
    body: CreateSessionRequest | None = None,
    service: SpectrumBloomService = Depends(get_service),
) -> CreateSessionResponse:
    return service.create_session(body or CreateSessionRequest())


@router.get("/sessions", response_model=list[SessionStatusResponse], tags=["Sessions"])
def list_sessions(
    service: SpectrumBloomService = Depends(get_service),
) -> list[SessionStatusResponse]:
    return service.list_sessions()


@router.get("/sessions/{session_id}", response_model=SessionStatusResponse, tags=["Sessions"])
def get_session(
    session_id: str,
    service: SpectrumBloomService = Depends(get_service),
) -> SessionStatusResponse:
    return service.get_session_status(session_id)


@router.patch(
    "/sessions/{session_id}/provider",
    response_model=SessionStatusResponse,
    tags=["Sessions"],
)
def update_provider(
    session_id: str,
    body: UpdateProviderRequest,
    service: SpectrumBloomService = Depends(get_service),
) -> SessionStatusResponse:
    return service.update_provider(session_id, body)


@router.post(
    "/sessions/{session_id}/clear",
    response_model=SessionStatusResponse,
    tags=["Sessions"],
)
def clear_session(
    session_id: str,
    service: SpectrumBloomService = Depends(get_service),
) -> SessionStatusResponse:
    return service.clear_session_chat(session_id)


@router.delete("/sessions/{session_id}", tags=["Sessions"])
def delete_session(
    session_id: str,
    service: SpectrumBloomService = Depends(get_service),
) -> dict[str, str]:
    service.delete_session(session_id)
    return {"detail": "Session deleted."}


@router.get("/sessions/{session_id}/history", tags=["Sessions"])
def get_session_history(
    session_id: str,
    service: SpectrumBloomService = Depends(get_service),
) -> dict:
    return service.get_session_history(session_id)


@router.post(
    "/sessions/{session_id}/upload",
    response_model=FullEDAReportResponse,
    tags=["Data & EDA"],
)
async def upload_dataset(
    session_id: str,
    file: UploadFile = File(...),
    service: SpectrumBloomService = Depends(get_service),
) -> FullEDAReportResponse:
    content = await file.read()
    return service.upload_dataset(
        session_id=session_id,
        filename=file.filename or "dataset.csv",
        content=content,
    )


@router.get(
    "/sessions/{session_id}/data/preview",
    response_model=DatasetPreviewResponse,
    tags=["Data & EDA"],
)
def preview_dataset(
    session_id: str,
    rows: int = Query(default=10, ge=1, le=100),
    service: SpectrumBloomService = Depends(get_service),
) -> DatasetPreviewResponse:
    return service.get_preview(session_id=session_id, rows=rows)


@router.get(
    "/sessions/{session_id}/eda",
    response_model=FullEDAReportResponse,
    tags=["Data & EDA"],
)
def full_eda_report(
    session_id: str,
    service: SpectrumBloomService = Depends(get_service),
) -> FullEDAReportResponse:
    return service.get_full_eda(session_id)


@router.post(
    "/sessions/{session_id}/ml/experiment",
    response_model=MLExperimentResponse,
    tags=["ML Experiments"],
)
def run_ml(
    session_id: str,
    body: MLExperimentRequest,
    service: SpectrumBloomService = Depends(get_service),
) -> MLExperimentResponse:
    return service.run_ml(session_id=session_id, request=body)


@router.post(
    "/sessions/{session_id}/query",
    response_model=QueryResponse,
    tags=["Agentic EDA"],
)
def query_agent(
    session_id: str,
    body: QueryRequest,
    service: SpectrumBloomService = Depends(get_service),
) -> QueryResponse:
    return service.run_query(session_id=session_id, query=body.query)
