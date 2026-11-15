import { provideHttpClient, withInterceptors } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { authInterceptor } from '../auth/auth.interceptor';
import { FullEDAReportResponse, SessionStatusResponse } from '../models/api.models';
import { WorkspaceStateService } from './workspace-state.service';

function makeSession(
  sessionId: string,
  hasData = false,
  sha256 = 'sha-a',
): SessionStatusResponse {
  return {
    session_id: sessionId,
    thread_id: `thread-${sessionId}`,
    provider: 'ollama',
    model: 'qwen3:4b-instruct',
    has_data: hasData,
    message_count: 0,
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

function makeEdaReport(): FullEDAReportResponse {
  return {
    profile: {
      filename: 's1.csv',
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
    figures: [],
  };
}

describe('WorkspaceStateService', () => {
  let service: WorkspaceStateService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    localStorage.clear();
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
    service.selectSession('s1');
    httpMock.expectOne('/api/v1/sessions/s1').flush(makeSession('s1', true, 'sha-v1'));
    httpMock.expectOne('/api/v1/sessions/s1/history').flush({
      session_id: 's1',
      messages: [],
      last_tool_traces: [],
    });
    httpMock.expectOne('/api/v1/sessions/s1/eda').flush(makeEdaReport());
    httpMock.expectOne('/api/v1/sessions/s1/data/preview?rows=100').flush({
      filename: 's1.csv',
      shape: [100, 5],
      columns: [],
      dtypes: {},
      preview: [],
    });

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

    // Reloading session s1 with a newly uploaded file (sha-v2) invalidates sha-v1 ML results.
    service.selectSession('s1');
    httpMock.expectOne('/api/v1/sessions/s1').flush(makeSession('s1', true, 'sha-v2'));
    httpMock.expectOne('/api/v1/sessions/s1/history').flush({
      session_id: 's1',
      messages: [],
      last_tool_traces: [],
    });
    httpMock.expectOne('/api/v1/sessions/s1/eda').flush(makeEdaReport());
    httpMock.expectOne('/api/v1/sessions/s1/data/preview?rows=100').flush({
      filename: 's1.csv',
      shape: [100, 5],
      columns: [],
      dtypes: {},
      preview: [],
    });
    expect(service.mlResult()).toBeNull();

    // Failed query rolls back optimistic user bubble and preserves failedPromptDraft.
    service.sendQuery('Calculate mean revenue');
    expect(service.messages().length).toBe(1);
    const qReq = httpMock.expectOne('/api/v1/sessions/s1/query');
    qReq.flush({ detail: 'Backend offline' }, { status: 503, statusText: 'Unavailable' });
    expect(service.messages().length).toBe(0);
    expect(service.failedPromptDraft()).toBe('Calculate mean revenue');
  });
});
