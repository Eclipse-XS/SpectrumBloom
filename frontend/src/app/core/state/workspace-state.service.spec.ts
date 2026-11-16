import { provideHttpClient, withInterceptors } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { authInterceptor } from '../auth/auth.interceptor';
import {
  ChatMessageRecord,
  FullEDAReportResponse,
  SessionStatusResponse,
} from '../models/api.models';
import { WorkspaceStateService } from './workspace-state.service';

function makeSession(
  sessionId: string,
  hasData = false,
  sha256 = 'sha-a',
  messageCount = 0,
): SessionStatusResponse {
  return {
    session_id: sessionId,
    thread_id: `thread-${sessionId}`,
    provider: 'ollama',
    model: 'qwen3:4b-instruct',
    has_data: hasData,
    message_count: messageCount,
    figure_count: 0,
    dataset: hasData
      ? {
          filename: `${sessionId}.csv`,
          size_bytes: 1200,
          sha256,
          rows: 100,
          columns: 5,
          storage_backend: 'local',
          storage_ref: `/tmp/${sessionId}.csv`,
          uploaded_at: '2026-11-15T12:00:00Z',
        }
      : null,
    created_at: '2026-11-15T12:00:00Z',
    updated_at: '2026-11-15T12:00:00Z',
  };
}

function makeEdaReport(filename = 's1.csv', figureId = 'fig-1'): FullEDAReportResponse {
  return {
    profile: {
      filename,
      rows: 100,
      columns: 5,
      duplicate_rows: 0,
      duplicate_pct: 0,
      memory_mb: 0.01,
      completeness_pct: 100,
      column_profiles: [],
      warnings: [],
    },
    statistics: {
      missingness: {
        total_cells: 500,
        total_missing_cells: 0,
        overall_missing_pct: 0,
        rows_with_any_missing: 0,
        rows_with_any_missing_pct: 0,
        columns_with_missing: [],
        missing_cooccurrence_pairs: [],
      },
      numeric_columns: ['revenue'],
      categorical_columns: ['region'],
      pearson_matrix: {},
      spearman_matrix: {},
      top_correlations: [],
      categorical_associations: [],
      outliers: [],
      multivariate_anomaly_count: 0,
      normality_checks: [],
      group_tests: [],
    },
    figures: [
      {
        id: figureId,
        title: `Distribution for ${filename}`,
        chart_kind: 'histogram',
        figure_json: JSON.stringify({
          data: [{ type: 'histogram', x: [1, 2, 3] }],
          layout: { title: filename },
        }),
      },
    ],
  };
}

function loadSessionWithData(
  service: WorkspaceStateService,
  httpMock: HttpTestingController,
  sessionId: string,
  sha256 = 'sha-v1',
): void {
  service.selectSession(sessionId);
  httpMock.expectOne(`/api/v1/sessions/${sessionId}`).flush(makeSession(sessionId, true, sha256));
  httpMock.expectOne(`/api/v1/sessions/${sessionId}/history`).flush({
    session_id: sessionId,
    messages: [],
    last_tool_traces: [],
  });
  httpMock.expectOne(`/api/v1/sessions/${sessionId}/eda`).flush(makeEdaReport(`${sessionId}.csv`));
  httpMock.expectOne(`/api/v1/sessions/${sessionId}/data/preview?rows=100`).flush({
    filename: `${sessionId}.csv`,
    shape: [100, 5],
    columns: ['revenue', 'region'],
    dtypes: { revenue: 'float64', region: 'object' },
    preview: [],
  });
}

describe('WorkspaceStateService', () => {
  let service: WorkspaceStateService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(withInterceptors([authInterceptor])),
        provideHttpClientTesting(),
      ],
    });
    service = TestBed.inject(WorkspaceStateService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('cancels stale in-flight session loads when switching sessions rapidly', () => {
    service.selectSession('s1');
    const req1 = httpMock.expectOne('/api/v1/sessions/s1');
    expect(req1.cancelled).toBeFalse();

    service.selectSession('s2');
    expect(req1.cancelled).toBeTrue();

    const req2 = httpMock.expectOne('/api/v1/sessions/s2');
    req2.flush(makeSession('s2', false));

    const histReq = httpMock.expectOne('/api/v1/sessions/s2/history');
    histReq.flush({ session_id: 's2', messages: [], last_tool_traces: [] });

    expect(service.activeSessionId()).toBe('s2');
    expect(service.loadingDataset()).toBeFalse();
  });

  it('scopes ML experiment results to the active dataset SHA-256 and rolls back failed queries', () => {
    loadSessionWithData(service, httpMock, 's1', 'sha-v1');

    service.runMlExperiment({ target_column: 'churned', task_type: 'auto', cv_folds: 3 });
    const mlReq = httpMock.expectOne('/api/v1/sessions/s1/ml/experiment');
    mlReq.flush({
      target_column: 'churned',
      task_type: 'classification',
      train_rows: 80,
      test_rows: 20,
      feature_count: 3,
      best_model: 'RandomForest',
      models: [],
      feature_importances: [],
      figures: [],
      notes: [],
    });
    expect(service.mlResult()?.best_model).toBe('RandomForest');
    expect(service.mlRunning()).toBeFalse();

    // Reloading session s1 with a newly uploaded file (sha-v2) invalidates sha-v1 ML results.
    loadSessionWithData(service, httpMock, 's1', 'sha-v2');
    expect(service.mlResult()).toBeNull();

    // Failed query rolls back optimistic user bubble and preserves failedPromptDraft.
    service.sendQuery('Calculate mean revenue');
    expect(service.messages().length).toBe(1);
    const qReq = httpMock.expectOne('/api/v1/sessions/s1/query');
    qReq.flush({ detail: 'Backend offline' }, { status: 503, statusText: 'Unavailable' });
    expect(service.messages().length).toBe(0);
    expect(service.querying()).toBeFalse();
    expect(service.failedPromptDraft()).toBe('Calculate mean revenue');
  });

  it('resets mlRunning when switching sessions during an active ML experiment', () => {
    loadSessionWithData(service, httpMock, 's1', 'sha-v1');

    service.runMlExperiment({ target_column: 'churned', task_type: 'auto', cv_folds: 3 });
    expect(service.mlRunning()).toBeTrue();
    const mlReq = httpMock.expectOne('/api/v1/sessions/s1/ml/experiment');
    expect(mlReq.cancelled).toBeFalse();

    // Switch session while ML experiment is in flight
    service.selectSession('s2');
    expect(mlReq.cancelled).toBeTrue();
    expect(service.mlRunning()).toBeFalse();

    httpMock.expectOne('/api/v1/sessions/s2').flush(makeSession('s2', false));
    httpMock.expectOne('/api/v1/sessions/s2/history').flush({
      session_id: 's2',
      messages: [],
      last_tool_traces: [],
    });
    expect(service.mlRunning()).toBeFalse();
    expect(service.mlResult()).toBeNull();
  });

  it('resets uploading when switching sessions during an upload or sample dataset load', () => {
    loadSessionWithData(service, httpMock, 's1', 'sha-v1');

    const file = new File(['a,b\n1,2\n'], 'new.csv', { type: 'text/csv' });
    service.uploadFile(file);
    expect(service.uploading()).toBeTrue();
    expect(service.uploadingFilename()).toBe('new.csv');

    const uploadReq = httpMock.expectOne('/api/v1/sessions/s1/upload');
    expect(uploadReq.cancelled).toBeFalse();

    // Switch session mid-upload
    service.selectSession('s2');
    expect(uploadReq.cancelled).toBeTrue();
    expect(service.uploading()).toBeFalse();
    expect(service.uploadingFilename()).toBe('');

    httpMock.expectOne('/api/v1/sessions/s2').flush(makeSession('s2', false));
    httpMock.expectOne('/api/v1/sessions/s2/history').flush({
      session_id: 's2',
      messages: [],
      last_tool_traces: [],
    });

    // Also test switching during sample dataset fetch
    service.loadSampleDataset();
    expect(service.uploading()).toBeTrue();
    const sampleReq = httpMock.expectOne('samples/sample_ecommerce.csv');
    expect(sampleReq.cancelled).toBeFalse();

    service.selectSession('s1');
    expect(sampleReq.cancelled).toBeTrue();
    expect(service.uploading()).toBeFalse();
    expect(service.uploadingFilename()).toBe('');

    httpMock.expectOne('/api/v1/sessions/s1').flush(makeSession('s1', false));
    httpMock.expectOne('/api/v1/sessions/s1/history').flush({
      session_id: 's1',
      messages: [],
      last_tool_traces: [],
    });
  });

  it('resets querying and cancels active AI query when clearMessages() is called', () => {
    loadSessionWithData(service, httpMock, 's1', 'sha-v1');

    service.sendQuery('What is the average revenue by region?');
    expect(service.querying()).toBeTrue();
    expect(service.pendingQueryText()).toBe('What is the average revenue by region?');

    const qReq = httpMock.expectOne('/api/v1/sessions/s1/query');
    expect(qReq.cancelled).toBeFalse();

    // Clear conversation while query is in flight
    service.clearMessages();
    expect(qReq.cancelled).toBeTrue();
    expect(service.querying()).toBeFalse();
    expect(service.pendingQueryText()).toBe('');

    const clearReq = httpMock.expectOne('/api/v1/sessions/s1/clear');
    clearReq.flush(makeSession('s1', true, 'sha-v1', 0));
    httpMock.expectOne('/api/v1/sessions').flush([
      makeSession('s1', true, 'sha-v1', 0),
    ]);

    expect(service.messages().length).toBe(0);
    expect(service.querying()).toBeFalse();
  });

  it('cancels pending analytical requests and prevents stale figures when changing datasets', () => {
    loadSessionWithData(service, httpMock, 's1', 'sha-v1');
    expect(service.allExportableFigures().map((f) => f.id)).toEqual(['fig-1']);

    service.sendQuery('Plot revenue distribution');
    const qReq = httpMock.expectOne('/api/v1/sessions/s1/query');
    service.runMlExperiment({ target_column: 'revenue', task_type: 'regression', cv_folds: 3 });
    const mlReq = httpMock.expectOne('/api/v1/sessions/s1/ml/experiment');
    expect(service.querying()).toBeTrue();
    expect(service.mlRunning()).toBeTrue();

    // Upload a replacement dataset while both analytical requests are pending
    const newFile = new File(['x,y\n10,20\n'], 'replacement.csv', { type: 'text/csv' });
    service.uploadFile(newFile);
    expect(qReq.cancelled).toBeTrue();
    expect(mlReq.cancelled).toBeTrue();
    expect(service.querying()).toBeFalse();
    expect(service.mlRunning()).toBeFalse();

    const upReq = httpMock.expectOne('/api/v1/sessions/s1/upload');
    upReq.flush(makeEdaReport('replacement.csv', 'fig-replacement'));
    httpMock.expectOne('/api/v1/sessions/s1').flush(makeSession('s1', true, 'sha-v2'));
    httpMock.expectOne('/api/v1/sessions/s1/data/preview?rows=100').flush({
      filename: 'replacement.csv',
      shape: [1, 2],
      columns: ['x', 'y'],
      dtypes: { x: 'int64', y: 'int64' },
      preview: [{ x: 10, y: 20 }],
    });
    httpMock.expectOne('/api/v1/sessions').flush([makeSession('s1', true, 'sha-v2')]);

    expect(service.uploading()).toBeFalse();
    expect(service.messages().length).toBe(0);
    expect(service.allExportableFigures().map((f) => f.id)).toEqual(['fig-replacement']);
  });

  it('reconciles backend history on timeout when the backend actually completed the turn, and reuses client_turn_id on retry', () => {
    loadSessionWithData(service, httpMock, 's1', 'sha-v1');

    // 1. Timeout where backend DID NOT complete yet -> preserves turn ID for idempotent retry
    service.sendQuery('Compute Spearman correlation');
    const qReq1 = httpMock.expectOne('/api/v1/sessions/s1/query');
    const firstTurnId = qReq1.request.body.client_turn_id as string;
    expect(firstTurnId).toBeTruthy();

    qReq1.flush({ detail: 'Gateway Timeout' }, { status: 504, statusText: 'Gateway Timeout' });
    // Reconciliation check fires on timeout
    const reconcileReq1 = httpMock.expectOne('/api/v1/sessions/s1/history');
    reconcileReq1.flush({ session_id: 's1', messages: [], last_tool_traces: [] });

    expect(service.querying()).toBeFalse();
    expect(service.failedPromptDraft()).toBe('Compute Spearman correlation');

    // Retrying the failed prompt reuses the exact same client_turn_id so backend cannot duplicate turns
    service.retryFailedPrompt();
    const qReq2 = httpMock.expectOne('/api/v1/sessions/s1/query');
    expect(qReq2.request.body.client_turn_id).toBe(firstTurnId);

    // 2. Simulate timeout on retry after backend actually completed the turn
    qReq2.flush({ detail: 'Gateway Timeout' }, { status: 504, statusText: 'Gateway Timeout' });
    const completedMessages: ChatMessageRecord[] = [
      {
        role: 'user',
        content: 'Compute Spearman correlation',
        timestamp: '2026-11-15T12:01:00Z',
        figures: [],
        tool_traces: [],
        evaluation: null,
        status: 'success',
        error: null,
        latency_ms: null,
        retries_used: 0,
        client_turn_id: firstTurnId,
      },
      {
        role: 'assistant',
        content: 'Spearman rho is 0.8794.',
        timestamp: '2026-11-15T12:01:02Z',
        figures: [],
        tool_traces: [],
        evaluation: null,
        status: 'success',
        error: null,
        latency_ms: 1450,
        retries_used: 0,
        client_turn_id: firstTurnId,
      },
    ];
    const reconcileReq2 = httpMock.expectOne('/api/v1/sessions/s1/history');
    reconcileReq2.flush({
      session_id: 's1',
      messages: completedMessages,
      last_tool_traces: [],
    });
    httpMock.expectOne('/api/v1/sessions/s1').flush(makeSession('s1', true, 'sha-v1', 2));
    httpMock.expectOne('/api/v1/sessions').flush([makeSession('s1', true, 'sha-v1', 2)]);

    expect(service.queryError()).toBeNull();
    expect(service.failedPromptDraft()).toBe('');
    expect(service.messages().length).toBe(2);
    expect(service.messages()[1].content).toBe('Spearman rho is 0.8794.');
  });
});
