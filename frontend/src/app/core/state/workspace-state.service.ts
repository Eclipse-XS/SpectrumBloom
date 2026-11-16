import { Injectable, computed, inject, signal } from '@angular/core';
import { Observable, Subject, finalize, forkJoin, map, of, switchMap, takeUntil, throwError } from 'rxjs';
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
  private readonly cancelUpload$ = new Subject<void>();
  private readonly cancelQuery$ = new Subject<void>();
  private readonly cancelMl$ = new Subject<void>();

  private sessionLoadOpId = 0;
  private uploadOpId = 0;
  private queryOpId = 0;
  private mlOpId = 0;
  private lastFailedTurnId: string | null = null;
  private lastFailedPromptText = '';
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
    const sid = this._activeSession()?.session_id;
    const sha = this._activeSession()?.dataset?.sha256;
    if (!sid || !sha || !this.hasData()) {
      return [];
    }
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
    const bootEpoch = this.sessionLoadOpId;
    this.api.listSessions().subscribe({
      next: (sessions) => {
        this._sessions.set(sessions);
        // Never overwrite an active session or cancel a user-initiated upload that started
        // while initial session discovery was still in flight.
        if (this._uploading() || this._activeSession() || this.sessionLoadOpId !== bootEpoch) {
          this._initializing.set(false);
          return;
        }
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
    const opId = ++this.sessionLoadOpId;
    this.api
      .createSession(req)
      .pipe(takeUntil(this.cancelSessionLoads$))
      .subscribe({
        next: (created) => {
          if (this.sessionLoadOpId !== opId) {
            return;
          }
          this.saveSessionId(created.session_id);
          this.clearDatasetViewState();
          this.api
            .getSession(created.session_id)
            .pipe(takeUntil(this.cancelSessionLoads$))
            .subscribe({
              next: (status) => {
                if (this.sessionLoadOpId !== opId) {
                  return;
                }
                this._activeSession.set(status);
                this.refreshSessions();
                this.showToast('Started a new workspace session.');
                onDone?.();
              },
              error: () => {
                if (this.sessionLoadOpId !== opId) {
                  return;
                }
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
          if (this.sessionLoadOpId !== opId) {
            return;
          }
          this.showToast(err.detail, 'error');
          onDone?.();
        },
      });
  }

  selectSession(sessionId: string, onDone?: () => void): void {
    this.cancelInFlightSessionWork();
    const opId = ++this.sessionLoadOpId;
    this.saveSessionId(sessionId);
    this.clearDatasetViewState();
    this._loadingDataset.set(true);

    this.api
      .getSession(sessionId)
      .pipe(
        takeUntil(this.cancelSessionLoads$),
        switchMap((status) => {
          if (this.sessionLoadOpId === opId) {
            this._activeSession.set(status);
          }
          return forkJoin({
            history: this.api.getHistory(sessionId),
            eda: status.has_data ? this.api.getFullEda(sessionId) : of(null),
            preview: status.has_data ? this.api.getDataPreview(sessionId, 100) : of(null),
          });
        }),
        finalize(() => {
          if (this.sessionLoadOpId === opId) {
            this._loadingDataset.set(false);
          }
        }),
      )
      .subscribe({
        next: ({ history, eda, preview }) => {
          if (this.sessionLoadOpId !== opId || this._activeSession()?.session_id !== sessionId) {
            return;
          }
          this._messages.set(history.messages);
          this.selectLatestAssistantWithArtifacts(history.messages);
          this._edaReport.set(eda);
          this._preview.set(preview);
          onDone?.();
        },
        error: (err: ApiError) => {
          if (this.sessionLoadOpId !== opId) {
            return;
          }
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
    this.startUploadWorkflow(of(file), file.name, onSuccess);
  }

  loadSampleDataset(onSuccess?: () => void): void {
    this.startUploadWorkflow(
      this.api.fetchSampleEcommerceFile(),
      'sample_ecommerce.csv',
      onSuccess,
    );
  }

  private startUploadWorkflow(
    fileSource$: Observable<File>,
    initialFilename: string,
    onSuccess?: () => void,
  ): void {
    const opId = ++this.uploadOpId;
    this.cancelUpload$.next();
    this.cancelActiveAnalyticalWork();
    this._uploading.set(true);
    this._uploadingFilename.set(initialFilename);
    this._uploadError.set(null);
    this._queryError.set(null);
    this._mlError.set(null);

    fileSource$
      .pipe(
        switchMap((file) => {
          const maxMb = this.maxUploadMb();
          if (file.size > maxMb * 1024 * 1024) {
            return throwError(
              () =>
                ({
                  status: 413,
                  kind: 'validation',
                  detail: `File exceeds the configured backend limit of ${maxMb} MB.`,
                }) satisfies ApiError,
            );
          }
          this._uploadingFilename.set(file.name);
          const existingSid = this._activeSession()?.session_id;
          const sessionId$ = existingSid
            ? of(existingSid)
            : this.api.createSession().pipe(
                map((created) => {
                  if (this.uploadOpId === opId) {
                    this.saveSessionId(created.session_id);
                  }
                  return created.session_id;
                }),
              );
          return sessionId$.pipe(
            switchMap((sessionId) =>
              this.api.uploadDataset(sessionId, file).pipe(
                switchMap((edaReport) =>
                  forkJoin({
                    sessionId: of(sessionId),
                    fileName: of(file.name),
                    edaReport: of(edaReport),
                    session: this.api.getSession(sessionId),
                    preview: this.api.getDataPreview(sessionId, 100),
                  }),
                ),
              ),
            ),
          );
        }),
        takeUntil(this.cancelUpload$),
        takeUntil(this.cancelSessionLoads$),
        finalize(() => {
          if (this.uploadOpId === opId) {
            this._uploading.set(false);
            this._uploadingFilename.set('');
          }
        }),
      )
      .subscribe({
        next: ({ sessionId, fileName, edaReport, session, preview }) => {
          if (this.uploadOpId !== opId) {
            return;
          }
          const currentSid = this._activeSession()?.session_id;
          if (currentSid && currentSid !== sessionId) {
            return;
          }
          this._edaReport.set(edaReport);
          this._messages.set([]);
          this._selectedCanvasMessageIndex.set(null);
          this._activeSession.set(session);
          this._preview.set(preview);
          this.refreshSessions();
          this.showToast(`Profiled ${fileName} (${session.dataset?.rows ?? 0} rows).`);
          onSuccess?.();
        },
        error: (err: ApiError) => {
          if (this.uploadOpId !== opId) {
            return;
          }
          this._uploadError.set(err.detail);
          this.showToast(err.detail, 'error');
        },
      });
  }

  refreshEda(): void {
    const sid = this._activeSession()?.session_id;
    const sha = this._activeSession()?.dataset?.sha256;
    if (!sid || !this.hasData()) {
      return;
    }
    const opId = ++this.sessionLoadOpId;
    this._loadingDataset.set(true);
    forkJoin({
      eda: this.api.getFullEda(sid),
      preview: this.api.getDataPreview(sid, 100),
    })
      .pipe(
        takeUntil(this.cancelSessionLoads$),
        finalize(() => {
          if (this.sessionLoadOpId === opId) {
            this._loadingDataset.set(false);
          }
        }),
      )
      .subscribe({
        next: ({ eda, preview }) => {
          if (
            this.sessionLoadOpId !== opId ||
            this._activeSession()?.session_id !== sid ||
            this._activeSession()?.dataset?.sha256 !== sha
          ) {
            return;
          }
          this._edaReport.set(eda);
          this._preview.set(preview);
          this.showToast('Automated EDA diagnostics refreshed.');
        },
        error: (err: ApiError) => {
          if (this.sessionLoadOpId !== opId) {
            return;
          }
          this.showToast(err.detail, 'error');
        },
      });
  }

  sendQuery(promptText: string, explicitTurnId?: string): void {
    const trimmed = promptText.trim();
    const sid = this._activeSession()?.session_id;
    const sha = this._activeSession()?.dataset?.sha256 ?? '';
    if (!trimmed || !sid || !this.hasData() || this._querying()) {
      return;
    }

    const opId = ++this.queryOpId;
    const clientTurnId =
      explicitTurnId ||
      (this.lastFailedTurnId && this.lastFailedPromptText === trimmed
        ? this.lastFailedTurnId
        : `turn_${Date.now()}_${opId}`);

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
      client_turn_id: clientTurnId,
    };
    this._messages.update((prev) => [...prev, optimisticUser]);

    this.api
      .queryAgent(sid, trimmed, clientTurnId)
      .pipe(
        takeUntil(this.cancelQuery$),
        takeUntil(this.cancelSessionLoads$),
        finalize(() => {
          if (this.queryOpId === opId) {
            this._querying.set(false);
            this._pendingQueryText.set('');
          }
        }),
      )
      .subscribe({
        next: (response) => {
          if (
            this.queryOpId !== opId ||
            this._activeSession()?.session_id !== sid ||
            (this._activeSession()?.dataset?.sha256 ?? '') !== sha
          ) {
            return;
          }
          this.lastFailedTurnId = null;
          this.lastFailedPromptText = '';
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
            client_turn_id: clientTurnId,
          };
          this._messages.update((prev) => [...prev, assistantMsg]);
          this._selectedCanvasMessageIndex.set(this._messages().length - 1);
          if (response.status === 'failed' && response.error) {
            this._queryError.set(response.error);
          }
          this.api
            .getSession(sid)
            .pipe(takeUntil(this.cancelSessionLoads$))
            .subscribe({
              next: (updated) => {
                if (this._activeSession()?.session_id === sid) {
                  this._activeSession.set(updated);
                  this.refreshSessions();
                }
              },
            });
        },
        error: (err: ApiError) => {
          if (
            this.queryOpId !== opId ||
            this._activeSession()?.session_id !== sid ||
            (this._activeSession()?.dataset?.sha256 ?? '') !== sha
          ) {
            return;
          }
          // Roll back optimistic user bubble pending reconciliation or retry
          this._messages.update((prev) =>
            prev.length > 0 && prev[prev.length - 1].client_turn_id === clientTurnId
              ? prev.slice(0, -1)
              : prev.slice(0, -1),
          );
          this.lastFailedTurnId = clientTurnId;
          this.lastFailedPromptText = trimmed;
          this._queryError.set(err.detail);
          this._failedPromptDraft.set(trimmed);

          if (err.kind === 'timeout') {
            this.reconcileHistoryAfterUncertainQuery(sid, sha, trimmed, clientTurnId, opId, err);
          } else {
            this.showToast(err.detail, 'error');
          }
        },
      });
  }

  retryFailedPrompt(): void {
    const draft = this._failedPromptDraft().trim();
    if (!draft) {
      return;
    }
    const turnId =
      this.lastFailedTurnId && this.lastFailedPromptText === draft
        ? this.lastFailedTurnId
        : undefined;
    this.sendQuery(draft, turnId);
  }

  private reconcileHistoryAfterUncertainQuery(
    sessionId: string,
    datasetSha256: string,
    promptText: string,
    clientTurnId: string,
    opId: number,
    originalErr: ApiError,
  ): void {
    this.api
      .getHistory(sessionId)
      .pipe(takeUntil(this.cancelQuery$), takeUntil(this.cancelSessionLoads$))
      .subscribe({
        next: (history) => {
          if (
            this.queryOpId !== opId ||
            this._activeSession()?.session_id !== sessionId ||
            (this._activeSession()?.dataset?.sha256 ?? '') !== datasetSha256
          ) {
            return;
          }
          const msgs = history.messages ?? [];
          const matchedByTurnId = msgs.some(
            (m) => m.role === 'assistant' && m.client_turn_id === clientTurnId,
          );
          const lastMsg = msgs.length >= 2 ? msgs[msgs.length - 1] : null;
          const prevMsg = msgs.length >= 2 ? msgs[msgs.length - 2] : null;
          const matchedByTrailingPair =
            lastMsg?.role === 'assistant' &&
            prevMsg?.role === 'user' &&
            prevMsg.content.trim() === promptText;

          if (matchedByTurnId || matchedByTrailingPair) {
            this._messages.set(msgs);
            this.selectLatestAssistantWithArtifacts(msgs);
            this._queryError.set(null);
            this._failedPromptDraft.set('');
            this.lastFailedTurnId = null;
            this.lastFailedPromptText = '';
            this.showToast('Recovered completed analytical turn from backend history.');
            this.api
              .getSession(sessionId)
              .pipe(takeUntil(this.cancelSessionLoads$))
              .subscribe({
                next: (updated) => {
                  if (this._activeSession()?.session_id === sessionId) {
                    this._activeSession.set(updated);
                    this.refreshSessions();
                  }
                },
              });
          } else {
            this.showToast(originalErr.detail, 'error');
          }
        },
        error: () => {
          if (this.queryOpId === opId) {
            this.showToast(originalErr.detail, 'error');
          }
        },
      });
  }

  clearMessages(): void {
    const sid = this._activeSession()?.session_id;
    if (!sid) {
      return;
    }
    this.queryOpId += 1;
    this.cancelQuery$.next();
    this._querying.set(false);
    this._pendingQueryText.set('');
    this._queryError.set(null);
    this._failedPromptDraft.set('');
    this.lastFailedTurnId = null;
    this.lastFailedPromptText = '';

    this.api
      .clearSession(sid)
      .pipe(takeUntil(this.cancelSessionLoads$))
      .subscribe({
        next: (updated) => {
          if (this._activeSession()?.session_id !== sid) {
            return;
          }
          this._activeSession.set(updated);
          this._messages.set([]);
          this._selectedCanvasMessageIndex.set(null);
          this._queryError.set(null);
          this._failedPromptDraft.set('');
          this.refreshSessions();
          this.showToast('Conversation cleared while keeping the active dataset.');
        },
        error: (err: ApiError) => {
          if (this._activeSession()?.session_id !== sid) {
            return;
          }
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
      .pipe(takeUntil(this.cancelSessionLoads$))
      .subscribe({
        next: (updated) => {
          if (this._activeSession()?.session_id === sid) {
            this._activeSession.set(updated);
          }
          this.refreshSessions();
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
    const opId = ++this.mlOpId;
    this.cancelMl$.next();
    this._mlRunning.set(true);
    this._mlError.set(null);

    this.api
      .runMlExperiment(sid, req)
      .pipe(
        takeUntil(this.cancelMl$),
        takeUntil(this.cancelSessionLoads$),
        finalize(() => {
          if (this.mlOpId === opId) {
            this._mlRunning.set(false);
          }
        }),
      )
      .subscribe({
        next: (result) => {
          if (
            this.mlOpId !== opId ||
            this._activeSession()?.session_id !== sid ||
            (this._activeSession()?.dataset?.sha256 ?? '') !== sha
          ) {
            return;
          }
          this._mlBySession.update((map) => ({
            ...map,
            [sid]: { datasetSha256: sha, result },
          }));
          this.showToast(`Baseline experiment completed (${result.best_model}).`);
        },
        error: (err: ApiError) => {
          if (
            this.mlOpId !== opId ||
            this._activeSession()?.session_id !== sid ||
            (this._activeSession()?.dataset?.sha256 ?? '') !== sha
          ) {
            return;
          }
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

  private cancelActiveAnalyticalWork(): void {
    this.queryOpId += 1;
    this.mlOpId += 1;
    this.cancelQuery$.next();
    this.cancelMl$.next();
    this._querying.set(false);
    this._pendingQueryText.set('');
    this._mlRunning.set(false);
  }

  private cancelInFlightSessionWork(): void {
    this.sessionLoadOpId += 1;
    this.uploadOpId += 1;
    this.queryOpId += 1;
    this.mlOpId += 1;
    this.cancelUpload$.next();
    this.cancelQuery$.next();
    this.cancelMl$.next();
    this.cancelSessionLoads$.next();
    this._loadingDataset.set(false);
    this._uploading.set(false);
    this._uploadingFilename.set('');
    this._querying.set(false);
    this._pendingQueryText.set('');
    this._mlRunning.set(false);
    this.lastFailedTurnId = null;
    this.lastFailedPromptText = '';
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
    this.lastFailedTurnId = null;
    this.lastFailedPromptText = '';
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
