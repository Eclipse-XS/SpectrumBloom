import { HttpClient, HttpErrorResponse, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, TimeoutError, catchError, map, throwError, timeout } from 'rxjs';
import { AuthTokenService } from '../auth/auth-token.service';
import {
  ApiError,
  CreateSessionRequest,
  CreateSessionResponse,
  DatasetPreviewResponse,
  FullEDAReportResponse,
  HealthResponse,
  MLExperimentRequest,
  MLExperimentResponse,
  QueryRequest,
  QueryResponse,
  SessionHistoryResponse,
  SessionStatusResponse,
  UpdateProviderRequest,
} from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class SpectrumBloomApiService {
  private readonly http = inject(HttpClient);
  private readonly auth = inject(AuthTokenService);

  checkHealth(overrideBaseUrl?: string): Observable<HealthResponse> {
    const cleanOverride = overrideBaseUrl?.trim().replace(/\/+$/, '');
    const url = cleanOverride ? `${cleanOverride}/health` : this.auth.resolveUrl('/health');
    return this.http.get<HealthResponse>(url).pipe(
      timeout(8000),
      catchError((err) => throwError(() => this.normalizeError(err))),
    );
  }

  listSessions(): Observable<SessionStatusResponse[]> {
    return this.http
      .get<SessionStatusResponse[]>(this.auth.resolveUrl('/api/v1/sessions'))
      .pipe(
        timeout(15000),
        catchError((err) => throwError(() => this.normalizeError(err))),
      );
  }

  createSession(body: CreateSessionRequest = {}): Observable<CreateSessionResponse> {
    return this.http
      .post<CreateSessionResponse>(this.auth.resolveUrl('/api/v1/sessions'), body)
      .pipe(
        timeout(15000),
        catchError((err) => throwError(() => this.normalizeError(err))),
      );
  }

  getSession(sessionId: string): Observable<SessionStatusResponse> {
    return this.http
      .get<SessionStatusResponse>(
        this.auth.resolveUrl(`/api/v1/sessions/${encodeURIComponent(sessionId)}`),
      )
      .pipe(
        timeout(15000),
        catchError((err) => throwError(() => this.normalizeError(err))),
      );
  }

  updateProvider(sessionId: string, body: UpdateProviderRequest): Observable<SessionStatusResponse> {
    return this.http
      .patch<SessionStatusResponse>(
        this.auth.resolveUrl(`/api/v1/sessions/${encodeURIComponent(sessionId)}/provider`),
        body,
      )
      .pipe(
        timeout(15000),
        catchError((err) => throwError(() => this.normalizeError(err))),
      );
  }

  clearSession(sessionId: string): Observable<SessionStatusResponse> {
    return this.http
      .post<SessionStatusResponse>(
        this.auth.resolveUrl(`/api/v1/sessions/${encodeURIComponent(sessionId)}/clear`),
        {},
      )
      .pipe(
        timeout(15000),
        catchError((err) => throwError(() => this.normalizeError(err))),
      );
  }

  deleteSession(sessionId: string): Observable<{ detail: string }> {
    return this.http
      .delete<{ detail: string }>(
        this.auth.resolveUrl(`/api/v1/sessions/${encodeURIComponent(sessionId)}`),
      )
      .pipe(
        timeout(15000),
        catchError((err) => throwError(() => this.normalizeError(err))),
      );
  }

  getHistory(sessionId: string): Observable<SessionHistoryResponse> {
    return this.http
      .get<SessionHistoryResponse>(
        this.auth.resolveUrl(`/api/v1/sessions/${encodeURIComponent(sessionId)}/history`),
      )
      .pipe(
        timeout(15000),
        catchError((err) => throwError(() => this.normalizeError(err))),
      );
  }

  uploadDataset(sessionId: string, file: File): Observable<FullEDAReportResponse> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    return this.http
      .post<FullEDAReportResponse>(
        this.auth.resolveUrl(`/api/v1/sessions/${encodeURIComponent(sessionId)}/upload`),
        formData,
      )
      .pipe(
        timeout(120000),
        catchError((err) => throwError(() => this.normalizeError(err))),
      );
  }

  fetchSampleEcommerceFile(): Observable<File> {
    return this.http
      .get('samples/sample_ecommerce.csv', { responseType: 'blob' })
      .pipe(
        timeout(15000),
        map((blob) => new File([blob], 'sample_ecommerce.csv', { type: 'text/csv' })),
        catchError((err) => throwError(() => this.normalizeError(err))),
      );
  }

  getDataPreview(sessionId: string, rows = 100): Observable<DatasetPreviewResponse> {
    const clamped = Math.max(1, Math.min(100, Math.round(rows)));
    const params = new HttpParams().set('rows', String(clamped));
    return this.http
      .get<DatasetPreviewResponse>(
        this.auth.resolveUrl(`/api/v1/sessions/${encodeURIComponent(sessionId)}/data/preview`),
        { params },
      )
      .pipe(
        timeout(30000),
        catchError((err) => throwError(() => this.normalizeError(err))),
      );
  }

  getFullEda(sessionId: string): Observable<FullEDAReportResponse> {
    return this.http
      .get<FullEDAReportResponse>(
        this.auth.resolveUrl(`/api/v1/sessions/${encodeURIComponent(sessionId)}/eda`),
      )
      .pipe(
        timeout(60000),
        catchError((err) => throwError(() => this.normalizeError(err))),
      );
  }

  queryAgent(
    sessionId: string,
    query: string,
    clientTurnId?: string | null,
  ): Observable<QueryResponse> {
    const body: QueryRequest = {
      query,
      ...(clientTurnId ? { client_turn_id: clientTurnId } : {}),
    };
    return this.http
      .post<QueryResponse>(
        this.auth.resolveUrl(`/api/v1/sessions/${encodeURIComponent(sessionId)}/query`),
        body,
      )
      .pipe(
        timeout(200000),
        catchError((err) => throwError(() => this.normalizeError(err))),
      );
  }

  runMlExperiment(sessionId: string, body: MLExperimentRequest): Observable<MLExperimentResponse> {
    return this.http
      .post<MLExperimentResponse>(
        this.auth.resolveUrl(`/api/v1/sessions/${encodeURIComponent(sessionId)}/ml/experiment`),
        body,
      )
      .pipe(
        timeout(120000),
        catchError((err) => throwError(() => this.normalizeError(err))),
      );
  }

  normalizeError(err: unknown): ApiError {
    if (err instanceof TimeoutError) {
      return {
        status: 408,
        detail: 'Request timed out waiting for the analytical backend.',
        kind: 'timeout',
      };
    }

    if (err instanceof HttpErrorResponse) {
      const serverDetail = this.extractDetail(err.error) || err.message || 'Request failed.';

      if (err.status === 0) {
        return {
          status: 0,
          detail:
            'Could not reach the SpectrumBloom FastAPI backend. Check Settings or ensure the API server is running.',
          kind: 'network',
        };
      }
      if (err.status === 401 || err.status === 403) {
        return {
          status: err.status,
          detail: serverDetail || 'Bearer authentication required or invalid token.',
          kind: 'auth',
        };
      }
      if (err.status === 404) {
        return {
          status: 404,
          detail: serverDetail,
          kind: 'not_found',
        };
      }
      if (err.status === 408 || err.status === 504) {
        return {
          status: err.status,
          detail: serverDetail,
          kind: 'timeout',
        };
      }
      if (err.status === 503) {
        return {
          status: 503,
          detail: serverDetail,
          kind: 'sandbox',
        };
      }
      if (err.status === 400 || err.status === 422) {
        return {
          status: err.status,
          detail: serverDetail,
          kind: 'validation',
        };
      }
      return {
        status: err.status,
        detail: serverDetail,
        kind: 'server',
      };
    }

    if (err && typeof err === 'object' && 'kind' in err && 'detail' in err) {
      return err as ApiError;
    }

    return {
      status: 500,
      detail: err instanceof Error ? err.message : 'Unexpected client error.',
      kind: 'server',
    };
  }

  private extractDetail(payload: unknown): string {
    if (!payload) {
      return '';
    }
    if (typeof payload === 'string') {
      return payload;
    }
    if (typeof payload === 'object' && 'detail' in payload) {
      const detail = (payload as { detail: unknown }).detail;
      if (typeof detail === 'string') {
        return detail;
      }
      if (Array.isArray(detail)) {
        return detail
          .map((item) =>
            item && typeof item === 'object' && 'msg' in item
              ? String((item as { msg: unknown }).msg)
              : JSON.stringify(item),
          )
          .join('; ');
      }
      return JSON.stringify(detail);
    }
    return '';
  }
}
