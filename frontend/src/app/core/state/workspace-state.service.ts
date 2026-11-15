import { Injectable, computed, inject, signal } from '@angular/core';
import { Subject, forkJoin, of, switchMap, takeUntil, tap } from 'rxjs';
import { SpectrumBloomApiService } from '../api/spectrum-bloom-api.service';
import { AuthTokenService } from '../auth/auth-token.service';
import {
  ApiError,
  ChatMessageRecord,
  CreateSessionRequest,
  DatasetPreviewResponse,
  FigurePayload,
  FullEDAReportResponse,
  HealthResponse,
  MLExperimentRequest,
  MLExperimentResponse,
  SessionStatusResponse,
  UpdateProviderRequest,
} from '../models/api.models';
import { ExpandedChartState, ToastMessage } from '../models/workspace.models';

const SESSION_ID_STORAGE_KEY = 'sb-active-session-id';

interface DatasetScopedMlEntry {
  datasetSha256: string;
  result: MLExperimentResponse;
}

@Injectable({ providedIn: 'root' })
export class WorkspaceStateService {
  private readonly api = inject(SpectrumBloomApiService);
  private readonly auth = inject(AuthTokenService);

  private readonly cancelSessionLoads$ = new Subject<void>();
  private readonly cancelQuery$ = new Subject<void>();
  private toastSeq = 0;

  private readonly _health = signal<HealthResponse | null>(null);
  private readonly _healthError = signal<string | null>(null);
  private readonly _initializing = signal<boolean>(true);
  private readonly _sessions = signal<SessionStatusResponse[]>([]);
  private readonly _activeSession = signal<SessionStatusResponse | null>(null);
  private readonly _edaReport = signal<FullEDAReportResponse | null>(null);
  private readonly _preview = signal<DatasetPreviewResponse | null>(null);
  private readonly _messages = signal<ChatMessageRecord[]>([]);
  private readonly _mlBySession = signal<Record<string, DatasetScopedMlEntry>>({});

  private readonly _loadingDataset = signal<boolean>(false);
  private readonly _uploading = signal<boolean>(false);
  private readonly _uploadingFilename = signal<string>('');
  private readonly _uploadError = signal<string | null>(null);

  private readonly _querying = signal<boolean>(false);
  private readonly _pendingQueryText = signal<string>('');
  private readonly _queryError = signal<string | null>(null);
  private readonly _failedPromptDraft = signal<string>('');

  private readonly _mlRunning = signal<boolean>(false);
  private readonly _mlError = signal<string | null>(null);

  private readonly _selectedCanvasMessageIndex = signal<number | null>(null);
  private readonly _expandedChart = signal<ExpandedChartState | null>(null);
  private readonly _toast = signal<ToastMessage | null>(null);

  readonly health = this._health.asReadonly();
  readonly healthError = this._healthError.asReadonly();
  readonly initializing = this._initializing.asReadonly();
  readonly sessions = this._sessions.asReadonly();
  readonly activeSession = this._activeSession.asReadonly();
  readonly edaReport = this._edaReport.asReadonly();
  readonly preview = this._preview.asReadonly();
  readonly messages = this._messages.asReadonly();

  readonly loadingDataset = this._loadingDataset.asReadonly();
  readonly uploading = this._uploading.asReadonly();
  readonly uploadingFilename = this._uploadingFilename.asReadonly();
  readonly uploadError = this._uploadError.asReadonly();

  readonly querying = this._querying.asReadonly();
  readonly pendingQueryText = this._pendingQueryText.asReadonly();
  readonly queryError = this._queryError.asReadonly();
  readonly failedPromptDraft = this._failedPromptDraft.asReadonly();

  readonly mlRunning = this._mlRunning.asReadonly();
  readonly mlError = this._mlError.asReadonly();

  readonly selectedCanvasMessageIndex = this._selectedCanvasMessageIndex.asReadonly();
  readonly expandedChart = this._expandedChart.asReadonly();
  readonly toast = this._toast.asReadonly();

  readonly activeSessionId = computed(() => this._activeSession()?.session_id ?? null);
  readonly hasData = computed(() => Boolean(this._activeSession()?.has_data));
  readonly datasetMeta = computed(() => this._activeSession()?.dataset ?? null);
  readonly profile = computed(() => this._edaReport()?.profile ?? null);
  readonly statistics = computed(() => this._edaReport()?.statistics ?? null);
  readonly edaFigures = computed(() => this._edaReport()?.figures ?? []);

  readonly maxUploadMb = computed(() => this._health()?.max_upload_mb ?? 25);
  readonly activeProvider = computed(
    () => this._activeSession()?.provider ?? this._health()?.default_provider ?? 'ollama',
  );
  readonly activeModel = computed(
    () =>
      this._activeSession()?.model ??
      this._health()?.llm?.configured_model ??
      this._health()?.default_model ??
      'qwen3:4b-instruct',
  );

  readonly mlResult = computed<MLExperimentResponse | null>(() => {
    const sid = this._activeSession()?.session_id;
    const sha = this._activeSession()?.dataset?.sha256;
    if (!sid || !sha) {
      return null;
    }
    const entry = this._mlBySession()[sid];
    if (!entry || entry.datasetSha256 !== sha) {
      return null;
    }
    return entry.result;
  });

  readonly allExportableFigures = computed<FigurePayload[]>(() => {
    const collected: FigurePayload[] = [];
    const seen = new Set<string>();
    const add = (fig: FigurePayload) => {
      const key = fig.id || `${fig.chart_kind}:${fig.title}`;
      if (!seen.has(key)) {
        seen.add(key);
        collected.push(fig);
      }
    };
    for (const fig of this.edaFigures()) add(fig);
    for (const msg of this._messages()) {
      for (const fig of msg.figures ?? []) add(fig);
    }
    for (const fig of this.mlResult()?.figures ?? []) add(fig);
    return collected;
  });

  initialize(): void {
    this._initializing.set(true);
    this.api.checkHealth().subscribe({
      next: (health) => {
        this._health.set(health);
        this._healthError.set(null);
        if (health.auth_required && !this.auth.hasToken()) {
          this.auth.markAuthChallenge(true);
          this._initializing.set(false);
          return;
        }
        this.bootstrapSession();
      },
      error: (err: ApiError) => {
        this._health.set(null);
        this._healthError.set(err.detail);
        this._initializing.set(false);
      },
    });
  }

  retryBootstrapAfterAuth(): void {
    this.auth.clearAuthChallenge();
    this._initializing.set(true);
    this.refreshHealth(() => this.bootstrapSession());
  }

  refreshHealth(onReady?: () => void): void {
    this.api.checkHealth().subscribe({
      next: (health) => {
        this._health.set(health);
        this._healthError.set(null);
        onReady?.();
      },
      error: (err: ApiError) => {
        this._healthError.set(err.detail);
        this._initializing.set(false);
      },
    });
  }

  private bootstrapSession(): void {
    this.api.listSessions().subscribe({
      next: (sessions) => {
        this._sessions.set(sessions);
        const savedId = this.readSavedSessionId();
        const match =
          (savedId && sessions.find((s) => s.session_id === savedId)) || sessions[0] || null;
        if (match) {
          this.selectSession(match.session_id, () => this._initializing.set(false));
        } else {
          this.createNewSession({}, () => this._initializing.set(false));
        }
      },
      error: (err: ApiError) => {
        if (err.kind === 'auth') {
          this.auth.markAuthChallenge(true);
        }
        this._initializing.set(false);
      },
    });
  }

  refreshSessions(): void {
    this.api.listSessions().subscribe({
      next: (sessions) => {
        this._sessions.set(sessions);
        const currentId = this._activeSession()?.session_id;
        if (currentId) {
          const updated = sessions.find((s) => s.session_id === currentId);
          if (updated) {
            this._activeSession.set(updated);
          }
        }
      },
      error: () => {
        // Keep existing sessions snapshot on transient polling error
      },
    });
  }

  createNewSession(req: CreateSessionRequest = {}, onDone?: () => void): void {
    this.cancelInFlightSessionWork();
    this.api.createSession(req).subscribe({
      next: (created) => {
        this.saveSessionId(created.session_id);
        this.clearDatasetViewState();
        this.api.getSession(created.session_id).subscribe({
          next: (status) => {
            this._activeSession.set(status);
            this.refreshSessions();
            this.showToast('Started a new workspace session.');
            onDone?.();
          },
          error: () => {
            this._activeSession.set({
              session_id: created.session_id,
              thread_id: created.thread_id,
              provider: created.provider,
              model: created.model,
              has_data: false,
              message_count: 0,
              figure_count: 0,
              dataset: null,
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString(),
            });
            this.refreshSessions();
            onDone?.();
          },
        });
      },
      error: (err: ApiError) => {
        this.showToast(err.detail, 'error');
        onDone?.();
      },
    });
  }

  selectSession(sessionId: string, onDone?: () => void): void {
    this.cancelInFlightSessionWork();
    this.saveSessionId(sessionId);
    this.clearDatasetViewState();
    this._loadingDataset.set(true);

    this.api
      .getSession(sessionId)
      .pipe(
        takeUntil(this.cancelSessionLoads$),
        switchMap((status) => {
          this._activeSession.set(status);
          return forkJoin({
            history: this.api.getHistory(sessionId),
            eda: status.has_data ? this.api.getFullEda(sessionId) : of(null),
            preview: status.has_data ? this.api.getDataPreview(sessionId, 100) : of(null),
          });
        }),
      )
      .subscribe({
        next: ({ history, eda, preview }) => {
          this._messages.set(history.messages);
          this.selectLatestAssistantWithArtifacts(history.messages);
          this._edaReport.set(eda);
          this._preview.set(preview);
          this._loadingDataset.set(false);
          onDone?.();
        },
        error: (err: ApiError) => {
          this._loadingDataset.set(false);
          if (err.kind === 'not_found') {
            this.clearSavedSessionId();
            this.bootstrapSession();
          } else {
            this.showToast(err.detail, 'error');
          }
          onDone?.();
        },
      });
  }

  uploadFile(file: File, onSuccess?: () => void): void {
    this._uploadError.set(null);
    const maxMb = this.maxUploadMb();
    const maxBytes = maxMb * 1024 * 1024;
    if (file.size > maxBytes) {
      this._uploadError.set(`File exceeds the configured backend limit of ${maxMb} MB.`);
      return;
    }

    const ensureSession = this._activeSession()?.session_id;
    if (!ensureSession) {
      this.api.createSession().subscribe({
        next: (created) => {
          this.saveSessionId(created.session_id);
          this.executeUpload(created.session_id, file, onSuccess);
        },
        error: (err: ApiError) => {
          this._uploadError.set(err.detail);
        },
      });
      return;
    }
    this.executeUpload(ensureSession, file, onSuccess);
  }

  loadSampleDataset(onSuccess?: () => void): void {
    this._uploadError.set(null);
    this._uploading.set(true);
    this._uploadingFilename.set('sample_ecommerce.csv');
    this.api.fetchSampleEcommerceFile().subscribe({
      next: (sampleFile) => {
        const sid = this._activeSession()?.session_id;
        if (!sid) {
          this.api.createSession().subscribe({
            next: (created) => {
              this.saveSessionId(created.session_id);
              this.executeUpload(created.session_id, sampleFile, onSuccess);
            },
            error: (err: ApiError) => {
              this._uploading.set(false);
              this._uploadError.set(err.detail);
            },
          });
        } else {
          this.executeUpload(sid, sampleFile, onSuccess);
        }
      },
      error: (err: ApiError) => {
        this._uploading.set(false);
        this._uploadError.set(err.detail);
      },
    });
  }

  private executeUpload(sessionId: string, file: File, onSuccess?: () => void): void {
    this.cancelInFlightSessionWork();
    this._uploading.set(true);
    this._uploadingFilename.set(file.name);
    this._uploadError.set(null);
    this._queryError.set(null);
    this._mlError.set(null);

    this.api
      .uploadDataset(sessionId, file)
      .pipe(
        takeUntil(this.cancelSessionLoads$),
        switchMap((edaReport) => {
          this._edaReport.set(edaReport);
          this._messages.set([]);
          this._selectedCanvasMessageIndex.set(null);
          return forkJoin({
            session: this.api.getSession(sessionId),
            preview: this.api.getDataPreview(sessionId, 100),
          });
        }),
      )
      .subscribe({
        next: ({ session, preview }) => {
          this._activeSession.set(session);
          this._preview.set(preview);
          this._uploading.set(false);
          this.refreshSessions();
          this.showToast(`Profiled ${file.name} (${session.dataset?.rows ?? 0} rows).`);
          onSuccess?.();
        },
        error: (err: ApiError) => {
          this._uploading.set(false);
          this._uploadError.set(err.detail);
          this.showToast(err.detail, 'error');
        },
      });
  }

  refreshEda(): void {
    const sid = this._activeSession()?.session_id;
    if (!sid || !this.hasData()) {
      return;
    }
    this._loadingDataset.set(true);
    forkJoin({
      eda: this.api.getFullEda(sid),
      preview: this.api.getDataPreview(sid, 100),
    })
      .pipe(takeUntil(this.cancelSessionLoads$))
      .subscribe({
        next: ({ eda, preview }) => {
          this._edaReport.set(eda);
          this._preview.set(preview);
          this._loadingDataset.set(false);
          this.showToast('Automated EDA diagnostics refreshed.');
        },
        error: (err: ApiError) => {
          this._loadingDataset.set(false);
          this.showToast(err.detail, 'error');
        },
      });
  }

  sendQuery(promptText: string): void {
    const trimmed = promptText.trim();
    const sid = this._activeSession()?.session_id;
    if (!trimmed || !sid || !this.hasData() || this._querying()) {
      return;
    }

    this.cancelQuery$.next();
    this._querying.set(true);
    this._pendingQueryText.set(trimmed);
    this._queryError.set(null);
    this._failedPromptDraft.set('');

    const optimisticUser: ChatMessageRecord = {
      role: 'user',
      content: trimmed,
      timestamp: new Date().toISOString(),
      figures: [],
      tool_traces: [],
      evaluation: null,
      status: 'success',
      error: null,
      latency_ms: null,
      retries_used: 0,
    };
    this._messages.update((prev) => [...prev, optimisticUser]);

    this.api
      .queryAgent(sid, trimmed)
      .pipe(takeUntil(this.cancelQuery$), takeUntil(this.cancelSessionLoads$))
      .subscribe({
        next: (response) => {
          const assistantMsg: ChatMessageRecord = {
            role: 'assistant',
            content: response.answer,
            timestamp: new Date().toISOString(),
            figures: response.figures ?? [],
            tool_traces: response.tool_traces ?? [],
            evaluation: response.evaluation ?? null,
            status: response.status,
            error: response.error,
            latency_ms: response.latency_ms,
            retries_used: response.retries_used,
          };
          this._messages.update((prev) => [...prev, assistantMsg]);
          this._selectedCanvasMessageIndex.set(this._messages().length - 1);
          this._querying.set(false);
          this._pendingQueryText.set('');
          if (response.status === 'failed' && response.error) {
            this._queryError.set(response.error);
          }
          this.api.getSession(sid).subscribe({
            next: (updated) => {
              this._activeSession.set(updated);
              this.refreshSessions();
            },
          });
        },
        error: (err: ApiError) => {
          this._messages.update((prev) => prev.slice(0, -1));
          this._querying.set(false);
          this._pendingQueryText.set('');
          this._queryError.set(err.detail);
          this._failedPromptDraft.set(trimmed);
          this.showToast(err.detail, 'error');
        },
      });
  }

  clearMessages(): void {
    const sid = this._activeSession()?.session_id;
    if (!sid) {
      return;
    }
    this.cancelQuery$.next();
    this.api.clearSession(sid).subscribe({
      next: (updated) => {
        this._activeSession.set(updated);
        this._messages.set([]);
        this._selectedCanvasMessageIndex.set(null);
        this._queryError.set(null);
        this._failedPromptDraft.set('');
        this.refreshSessions();
        this.showToast('Conversation cleared while keeping the active dataset.');
      },
      error: (err: ApiError) => {
        this.showToast(err.detail, 'error');
      },
    });
  }

  deleteSession(sessionId: string): void {
    const deletingActive = this._activeSession()?.session_id === sessionId;
    if (deletingActive) {
      this.cancelInFlightSessionWork();
    }
    this.api.deleteSession(sessionId).subscribe({
      next: () => {
        this._mlBySession.update((map) => {
          const copy = { ...map };
          delete copy[sessionId];
          return copy;
        });
        this.api.listSessions().subscribe({
          next: (remaining) => {
            this._sessions.set(remaining);
            if (deletingActive) {
              if (remaining.length > 0) {
                this.selectSession(remaining[0].session_id);
              } else {
                this.clearSavedSessionId();
                this.createNewSession();
              }
            }
            this.showToast('Session deleted.');
          },
        });
      },
      error: (err: ApiError) => {
        this.showToast(err.detail, 'error');
      },
    });
  }

  updateProvider(req: UpdateProviderRequest, onDone?: () => void): void {
    const sid = this._activeSession()?.session_id;
    if (!sid) {
      this.createNewSession({ provider: req.provider, model: req.model, api_key: req.api_key }, onDone);
      return;
    }
    this.api
      .updateProvider(sid, req)
      .pipe(
        tap((updated) => {
          this._activeSession.set(updated);
          this.refreshSessions();
        }),
      )
      .subscribe({
        next: () => {
          this.showToast(`Model updated to ${req.provider} / ${req.model}.`);
          onDone?.();
        },
        error: (err: ApiError) => {
          this.showToast(err.detail, 'error');
          onDone?.();
        },
      });
  }

  runMlExperiment(req: MLExperimentRequest): void {
    const sid = this._activeSession()?.session_id;
    const sha = this._activeSession()?.dataset?.sha256 ?? '';
    if (!sid || !this.hasData() || this._mlRunning()) {
      return;
    }
    this._mlRunning.set(true);
    this._mlError.set(null);

    this.api
      .runMlExperiment(sid, req)
      .pipe(takeUntil(this.cancelSessionLoads$))
      .subscribe({
        next: (result) => {
          this._mlBySession.update((map) => ({
            ...map,
            [sid]: { datasetSha256: sha, result },
          }));
          this._mlRunning.set(false);
          this.showToast(`Baseline experiment completed (${result.best_model}).`);
        },
        error: (err: ApiError) => {
          this._mlRunning.set(false);
          this._mlError.set(err.detail);
          this.showToast(err.detail, 'error');
        },
      });
  }

  selectCanvasMessage(index: number | null): void {
    this._selectedCanvasMessageIndex.set(index);
  }

  openExpandedChart(chart: ExpandedChartState): void {
    this._expandedChart.set(chart);
  }

  closeExpandedChart(): void {
    this._expandedChart.set(null);
  }

  showToast(text: string, kind: 'success' | 'error' = 'success'): void {
    const id = ++this.toastSeq;
    this._toast.set({ id, text, kind });
    setTimeout(() => {
      if (this._toast()?.id === id) {
        this._toast.set(null);
      }
    }, 3200);
  }

  private cancelInFlightSessionWork(): void {
    this.cancelQuery$.next();
    this.cancelSessionLoads$.next();
    this._querying.set(false);
    this._pendingQueryText.set('');
  }

  private clearDatasetViewState(): void {
    this._edaReport.set(null);
    this._preview.set(null);
    this._messages.set([]);
    this._selectedCanvasMessageIndex.set(null);
    this._uploadError.set(null);
    this._queryError.set(null);
    this._mlError.set(null);
    this._failedPromptDraft.set('');
  }

  private selectLatestAssistantWithArtifacts(messages: ChatMessageRecord[]): void {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const m = messages[i];
      if (m.role === 'assistant' && ((m.figures?.length ?? 0) > 0 || (m.tool_traces?.length ?? 0) > 0)) {
        this._selectedCanvasMessageIndex.set(i);
        return;
      }
    }
    this._selectedCanvasMessageIndex.set(messages.length > 0 ? messages.length - 1 : null);
  }

  private readSavedSessionId(): string | null {
    try {
      return sessionStorage.getItem(SESSION_ID_STORAGE_KEY);
    } catch {
      return null;
    }
  }

  private saveSessionId(id: string): void {
    try {
      sessionStorage.setItem(SESSION_ID_STORAGE_KEY, id);
    } catch {
      // Ignore storage access errors
    }
  }

  private clearSavedSessionId(): void {
    try {
      sessionStorage.removeItem(SESSION_ID_STORAGE_KEY);
    } catch {
      // Ignore storage access errors
    }
  }
}
