import { ChangeDetectionStrategy, Component, computed, effect, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import {
  ChatMessageRecord,
  EvaluationSummary,
  FigurePayload,
  ToolTraceItem,
} from '../../core/models/api.models';
import { StatusKind } from '../../core/models/workspace.models';
import { WorkspaceStateService } from '../../core/state/workspace-state.service';
import { PlotlyChartComponent } from '../../shared/components/plotly-chart.component';
import { SafeMarkdownPipe } from '../../shared/components/safe-markdown.pipe';
import { SbCardComponent } from '../../shared/components/sb-card.component';
import { StatusBadgeComponent } from '../../shared/components/status-badge.component';
import { SvgIconComponent } from '../../shared/components/svg-icon.component';
import { downloadPlotlyHtml } from '../../shared/utils/export.utils';
import { formatDurationMs, formatTimestamp, slugifyFilename } from '../../shared/utils/format.utils';

@Component({
  selector: 'sb-ai-analyst',
  standalone: true,
  imports: [
    FormsModule,
    SafeMarkdownPipe,
    StatusBadgeComponent,
    SbCardComponent,
    PlotlyChartComponent,
    SvgIconComponent,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styles: [
    `
      .analyst {
        height: calc(100vh - 64px);
        display: grid;
        grid-template-columns: minmax(420px, 48%) 1fr;
      }
      .conversation {
        min-width: 0;
        min-height: 0;
        background: var(--surface);
        border-right: 1px solid var(--border);
        display: flex;
        flex-direction: column;
      }
      .analyst-head {
        min-height: 68px;
        padding: 10px 20px;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        border-bottom: 1px solid var(--border);
      }
      .analyst-head > div:first-child {
        min-width: 0;
      }
      .analyst-head .eyebrow {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .analyst-head h1 {
        font-size: 19px;
        margin: 0;
        letter-spacing: -0.02em;
      }
      .analyst-head-actions {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
      }
      .mobile-canvas-btn {
        display: none;
      }
      .messages {
        flex: 1;
        overflow-y: auto;
        padding: 22px 20px;
      }
      .chat-empty {
        max-width: 580px;
        margin: 7vh auto 0;
        text-align: center;
      }
      .analyst-orb {
        width: 48px;
        height: 48px;
        margin: 0 auto 14px;
        border-radius: 12px;
        background: var(--primary-soft);
        color: var(--primary);
        display: grid;
        place-items: center;
      }
      .chat-empty h2 {
        font-size: 19px;
        margin: 0 0 8px;
      }
      .chat-empty p {
        font-size: 12px;
        color: var(--muted);
        margin: 0 auto 18px;
        max-width: 440px;
        line-height: 1.5;
      }
      .prompt-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        text-align: left;
      }
      .prompt-grid button {
        min-height: 48px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 11px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
      }
      .prompt-grid button:hover {
        border-color: var(--primary);
        color: var(--primary);
        background: var(--primary-soft);
      }
      .message {
        display: grid;
        grid-template-columns: 30px 1fr;
        gap: 10px;
        max-width: 720px;
        margin: 0 auto 20px;
      }
      .avatar {
        width: 28px;
        height: 28px;
        border-radius: 7px;
        background: var(--surface-2);
        display: grid;
        place-items: center;
        font-size: 9px;
        font-weight: 700;
      }
      .message.assistant .avatar {
        background: var(--primary-soft);
        color: var(--primary);
      }
      .msg-body {
        min-width: 0;
      }
      .answer-status {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        margin-bottom: 6px;
        font-size: 10px;
        color: var(--faint);
      }
      .message-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 10px;
      }
      .message-actions button {
        border: 0;
        background: none;
        color: var(--muted);
        font-size: 11px;
        display: inline-flex;
        gap: 5px;
        align-items: center;
        padding: 4px 2px;
      }
      .message-actions button:hover {
        color: var(--primary);
      }
      .thinking {
        min-height: 36px;
        display: flex;
        align-items: center;
        gap: 5px;
        color: var(--muted);
        font-size: 12px;
      }
      .thinking > span {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--primary);
        animation: pulse 1s infinite;
      }
      .thinking > span:nth-child(2) {
        animation-delay: 0.15s;
      }
      .thinking > span:nth-child(3) {
        animation-delay: 0.3s;
      }
      .composer {
        margin: 0 18px 16px;
        border: 1px solid var(--border-strong);
        border-radius: 9px;
        background: var(--surface);
        padding: 10px 12px;
      }
      .composer:focus-within {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px color-mix(in srgb, var(--primary) 12%, transparent);
      }
      .composer textarea {
        width: 100%;
        border: 0;
        outline: 0;
        background: none;
        color: var(--text);
        resize: none;
        font-size: 13px;
        line-height: 1.45;
      }
      .composer-foot {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 4px;
      }
      .composer-foot span {
        font-size: 10px;
        color: var(--faint);
      }
      .send-btn {
        border: 0;
        background: var(--primary);
        color: #fff;
        width: 34px;
        height: 34px;
        border-radius: 7px;
        display: grid;
        place-items: center;
      }
      .analysis-canvas {
        min-width: 0;
        min-height: 0;
        background: var(--bg);
        display: flex;
        flex-direction: column;
      }
      .canvas-head {
        min-height: 51px;
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 18px;
      }
      .canvas-body {
        flex: 1;
        overflow-y: auto;
        padding: 18px;
        display: flex;
        flex-direction: column;
        gap: 14px;
      }
      .canvas-empty {
        margin: 18vh auto 0;
        max-width: 300px;
        text-align: center;
      }
      .canvas-empty-icon {
        width: 48px;
        height: 48px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: var(--surface);
        display: grid;
        place-items: center;
        margin: 0 auto 12px;
        color: var(--faint);
      }
      @media (max-width: 1100px) {
        .analyst {
          grid-template-columns: minmax(380px, 52%) 1fr;
        }
      }
      @media (max-width: 800px) {
        .analyst {
          height: calc(100vh - 58px);
          grid-template-columns: 1fr;
        }
        .mobile-canvas-btn {
          display: inline-flex;
        }
        .analysis-canvas {
          display: none;
        }
        .analyst.show-canvas .conversation {
          display: none;
        }
        .analyst.show-canvas .analysis-canvas {
          display: flex;
        }
        .prompt-grid {
          grid-template-columns: 1fr;
        }
        .analyst-head {
          padding: 10px 14px;
        }
        .messages {
          padding: 16px 14px;
        }
        .composer {
          margin: 0 12px 12px;
        }
      }
    `,
  ],
  template: `
    @if (!state.hasData()) {
      <div class="page">
        <div class="empty-panel">
          <h2>Upload a dataset to start chatting</h2>
          <p>The AI Analyst operates on the active session dataset using deterministic tools and the isolated sandbox.</p>
          <button type="button" class="button primary" (click)="goToOnboarding()">
            Upload dataset
          </button>
        </div>
      </div>
    } @else {
      <div class="analyst" [class.show-canvas]="mobileShowCanvas()">
        <div class="conversation">
          <div class="analyst-head">
            <div>
              <div class="eyebrow">{{ state.datasetMeta()?.filename }}</div>
              <h1>AI Analyst</h1>
            </div>
            <div class="analyst-head-actions">
              <sb-status-badge kind="neutral">{{ state.activeModel() }}</sb-status-badge>
              @if (state.messages().length > 0 || state.querying()) {
                @if (!confirmingClear()) {
                  <button
                    type="button"
                    class="button sm"
                    (click)="confirmingClear.set(true)"
                  >
                    Clear chat
                  </button>
                } @else {
                  <button
                    type="button"
                    class="button sm danger"
                    (click)="confirmClearChat()"
                  >
                    Confirm clear
                  </button>
                  <button
                    type="button"
                    class="button sm"
                    (click)="confirmingClear.set(false)"
                  >
                    Cancel
                  </button>
                }
              }
              <button
                type="button"
                class="button sm mobile-canvas-btn"
                (click)="mobileShowCanvas.set(true)"
              >
                Canvas ({{ selectedTurnFigures().length }})
              </button>
            </div>
          </div>

          <div class="messages">
            @if (state.messages().length === 0 && !state.querying()) {
              <div class="chat-empty">
                <div class="analyst-orb">
                  <sb-svg-icon name="spark" [size]="24" />
                </div>
                <h2>Ask a question about {{ state.datasetMeta()?.filename }}</h2>
                <p>
                  Requests execute deterministically first and fall back to the isolated
                  {{ state.health()?.sandbox_backend ?? 'configured' }} sandbox for custom plots or calculations.
                </p>
                <div class="prompt-grid">
                  @for (prompt of suggestedPrompts(); track prompt) {
                    <button type="button" (click)="submitPrompt(prompt)">
                      <span>{{ prompt }}</span>
                      <sb-svg-icon name="chevron" [size]="15" />
                    </button>
                  }
                </div>
              </div>
            } @else {
              @for (msg of state.messages(); track $index; let idx = $index) {
                <div [class]="'message ' + msg.role">
                  <div class="avatar">
                    @if (msg.role === 'user') {
                      You
                    } @else {
                      <sb-svg-icon name="spark" [size]="15" />
                    }
                  </div>
                  <div class="msg-body">
                    @if (msg.role === 'assistant') {
                      <div class="answer-status">
                        <sb-status-badge [kind]="msg.status === 'success' ? 'success' : 'error'">
                          {{ msg.status === 'success' ? 'Completed' : 'Failed' }}
                        </sb-status-badge>
                        @if (msg.latency_ms !== null) {
                          <span>{{ formatDurationMs(msg.latency_ms) }}</span>
                        }
                        @if (msg.retries_used > 0) {
                          <span>{{ msg.retries_used }} repair retry</span>
                        }
                      </div>
                    }

                    @if (msg.role === 'user') {
                      <p style="margin: 2px 0 4px">{{ msg.content }}</p>
                      <small style="color: var(--faint); font-size: 10px">
                        {{ formatTimestamp(msg.timestamp) }}
                      </small>
                    } @else {
                      <div class="markdown-body" [innerHTML]="msg.content | safeMarkdown"></div>

                      @if (msg.error) {
                        <div class="notice error" style="margin-top: 8px">
                          <sb-svg-icon name="warning" [size]="15" />
                          <div class="notice-content"><span>{{ msg.error }}</span></div>
                        </div>
                      }

                      <div class="message-actions">
                        <button type="button" (click)="copyAnswer(msg.content)">
                          <sb-svg-icon name="copy" [size]="14" />
                          Copy
                        </button>
                        @if ((msg.tool_traces.length || 0) > 0 || msg.evaluation) {
                          <button type="button" (click)="toggleEvidence(idx)">
                            <sb-svg-icon name="code" [size]="14" />
                            {{ expandedEvidence().has(idx) ? 'Hide evidence' : 'Inspect evidence' }}
                          </button>
                        }
                        @if ((msg.figures.length || 0) > 0 || (msg.tool_traces.length || 0) > 0) {
                          <button type="button" (click)="focusCanvasTurn(idx)">
                            <sb-svg-icon name="chart" [size]="14" />
                            Focus canvas
                            @if (msg.figures.length > 0) {
                              ({{ msg.figures.length }} chart)
                            }
                          </button>
                        }
                      </div>

                      @if (expandedEvidence().has(idx)) {
                        <div class="evidence">
                          @for (trace of msg.tool_traces; track $index) {
                            <div class="trace-item">
                              <div class="trace-head">
                                <span>
                                  <sb-svg-icon
                                    [name]="trace.error ? 'warning' : 'check'"
                                    [size]="14"
                                  />
                                  {{ trace.tool }}
                                </span>
                                <b class="mono">{{ formatDurationMs(trace.duration_ms) }}</b>
                              </div>
                              @if (trace.code) {
                                <pre>{{ trace.code }}</pre>
                              }
                              @if (trace.stdout) {
                                <pre>{{ trace.stdout }}</pre>
                              }
                              @if (trace.error) {
                                <pre style="color: var(--error)">{{ trace.error }}</pre>
                              }
                              @if (trace.result !== null && trace.result !== undefined) {
                                <pre>{{ stringifyResult(trace.result) }}</pre>
                              }
                            </div>
                          }

                          @if (msg.evaluation; as ev) {
                            <div
                              class="eval-row"
                              title="Trace provenance checks compare reported numbers against executed tool outputs and query context; they do not prove standalone mathematical correctness."
                            >
                              <sb-status-badge [kind]="evalOverallKind(ev)">
                                {{ evalOverallLabel(ev) }}
                              </sb-status-badge>
                              @if (ev.numeric_grounding_ratio !== undefined) {
                                <sb-status-badge [kind]="ev.numeric_grounding_passed ? 'success' : 'warning'">
                                  Trace match: {{ (ev.numeric_grounding_ratio * 100).toFixed(0) }}%
                                </sb-status-badge>
                              }
                              @if (ev.execution_pass_rate !== undefined) {
                                <sb-status-badge [kind]="ev.tool_execution_ok ? 'success' : 'error'">
                                  Tool exec: {{ (ev.execution_pass_rate * 100).toFixed(0) }}%
                                </sb-status-badge>
                              }
                              @if (ev.judge_score !== null && ev.judge_score !== undefined) {
                                <sb-status-badge kind="neutral">
                                  Judge: {{ ev.judge_score.toFixed(2) }}
                                </sb-status-badge>
                              }
                            </div>
                          }
                        </div>
                      }
                    }
                  </div>
                </div>
              }

              @if (state.querying()) {
                <div class="message assistant">
                  <div class="avatar"><sb-svg-icon name="spark" [size]="15" /></div>
                  <div class="thinking">
                    <span></span><span></span><span></span>
                    <span>Running deterministic tools &amp; agent analysis…</span>
                  </div>
                </div>
              }
            }

            @if (state.queryError()) {
              <div class="notice error" style="max-width: 720px; margin: 12px auto 0">
                <sb-svg-icon name="warning" [size]="16" />
                <div class="notice-content">
                  <strong>Query error</strong>
                  <span>{{ state.queryError() }}</span>
                  @if (state.failedPromptDraft()) {
                    <div style="margin-top: 6px">
                      <button type="button" class="button sm" (click)="retryFailedPrompt()">
                        Retry failed query
                      </button>
                    </div>
                  }
                </div>
              </div>
            }
          </div>

          <form class="composer" (ngSubmit)="onComposerSubmit()">
            <textarea
              aria-label="Ask a question about this dataset"
              name="queryInput"
              rows="2"
              [ngModel]="promptText()"
              (ngModelChange)="promptText.set($event)"
              (keydown)="onComposerKeydown($event)"
              placeholder="Ask about trends, segments, correlations, or request a Plotly chart…"
              [disabled]="state.querying()"
            ></textarea>
            <div class="composer-foot">
              <span>Enter to send · Shift+Enter for new line</span>
              <button
                type="submit"
                class="send-btn"
                aria-label="Send question"
                [disabled]="state.querying() || !promptText().trim()"
              >
                <sb-svg-icon name="send" [size]="16" />
              </button>
            </div>
          </form>
        </div>

        <aside class="analysis-canvas">
          <div class="canvas-head">
            <div style="display: flex; align-items: center; gap: 10px">
              <strong style="font-size: 13px">Analysis canvas</strong>
              @if (selectedTurn()) {
                <sb-status-badge kind="neutral">Selected turn</sb-status-badge>
              }
            </div>
            @if (mobileShowCanvas()) {
              <button
                type="button"
                class="button sm"
                (click)="mobileShowCanvas.set(false)"
              >
                Back to chat
              </button>
            }
          </div>

          <div class="canvas-body">
            @if (selectedTurn(); as turn) {
              @if (selectedTurnFigures().length === 0 && selectedTurnTraces().length === 0) {
                <div class="canvas-empty">
                  <div class="canvas-empty-icon"><sb-svg-icon name="chart" [size]="22" /></div>
                  <h3>No chart or tabular artifact on this turn</h3>
                  <p style="font-size: 12px; color: var(--muted)">
                    This response answered directly from dataset metadata. Ask for a distribution, comparison, or scatter plot to render an interactive Plotly chart here.
                  </p>
                </div>
              } @else {
                @for (fig of selectedTurnFigures(); track fig.id) {
                  <sb-card [title]="fig.title" [subtitle]="'Chart type: ' + fig.chart_kind">
                    <button
                      card-actions
                      type="button"
                      class="icon-button"
                      aria-label="Download HTML"
                      (click)="downloadHtml(fig)"
                    >
                      <sb-svg-icon name="download" [size]="15" />
                    </button>
                    <button
                      card-actions
                      type="button"
                      class="icon-button"
                      aria-label="Expand chart"
                      (click)="expandFigure(fig)"
                    >
                      <sb-svg-icon name="expand" [size]="15" />
                    </button>
                    <sb-plotly-chart [figureJson]="fig.figure_json" [title]="fig.title" [height]="320" />
                  </sb-card>
                }

                @for (trace of selectedTurnTraces(); track $index) {
                  <sb-card [title]="'Tool output · ' + trace.tool" [subtitle]="formatDurationMs(trace.duration_ms)">
                    @if (trace.stdout) {
                      <pre style="margin: 0; font-size: 11px; overflow: auto; max-height: 220px">{{ trace.stdout }}</pre>
                    } @else if (trace.result !== null && trace.result !== undefined) {
                      <pre style="margin: 0; font-size: 11px; overflow: auto; max-height: 220px">{{ stringifyResult(trace.result) }}</pre>
                    } @else if (trace.error) {
                      <p style="color: var(--error); margin: 0; font-size: 12px">{{ trace.error }}</p>
                    }
                  </sb-card>
                }
              }
            } @else {
              <div class="canvas-empty">
                <div class="canvas-empty-icon"><sb-svg-icon name="chart" [size]="22" /></div>
                <h3>Analysis results appear here</h3>
                <p style="font-size: 12px; color: var(--muted)">
                  Charts and execution outputs returned by the AI Analyst stay focused in this canvas beside the conversation.
                </p>
              </div>
            }
          </div>
        </aside>
      </div>
    }
  `,
})
export class AiAnalystComponent {
  readonly state = inject(WorkspaceStateService);
  private readonly router = inject(Router);

  readonly promptText = signal<string>('');
  readonly expandedEvidence = signal<Set<number>>(new Set());
  readonly mobileShowCanvas = signal<boolean>(false);
  readonly confirmingClear = signal<boolean>(false);

  readonly formatDurationMs = formatDurationMs;
  readonly formatTimestamp = formatTimestamp;

  constructor() {
    effect(() => {
      const draft = this.state.failedPromptDraft();
      if (draft && !this.promptText()) {
        this.promptText.set(draft);
      }
    });
  }

  readonly suggestedPrompts = computed(() => {
    const prof = this.state.profile();
    const stats = this.state.statistics();
    if (!prof || !stats) {
      return [
        'Which columns contain missing values?',
        'Summarize the dataset distributions.',
      ];
    }
    const prompts: string[] = [];
    const firstNum = stats.numeric_columns[0];
    const secondNum = stats.numeric_columns[1];
    const firstCat = stats.categorical_columns[0];

    if (stats.missingness.columns_with_missing.length > 0) {
      prompts.push('Which columns contain the most missing values and their percentages?');
    } else {
      prompts.push('Provide a statistical summary of key numeric columns.');
    }
    if (firstCat && firstNum) {
      prompts.push(`Compare average ${firstNum} across ${firstCat}.`);
    }
    if (firstNum) {
      prompts.push(`Create a distribution plot for ${firstNum}.`);
    }
    if (firstNum && secondNum) {
      prompts.push(`Which numeric variables are most strongly correlated with ${firstNum}?`);
    } else {
      prompts.push('Are there notable outliers in this dataset?');
    }
    return prompts.slice(0, 4);
  });

  readonly selectedTurn = computed<ChatMessageRecord | null>(() => {
    const msgs = this.state.messages();
    const idx = this.state.selectedCanvasMessageIndex();
    if (idx !== null && msgs[idx]?.role === 'assistant') {
      return msgs[idx];
    }
    for (let i = msgs.length - 1; i >= 0; i -= 1) {
      if (msgs[i].role === 'assistant') {
        return msgs[i];
      }
    }
    return null;
  });

  readonly selectedTurnFigures = computed(() => this.selectedTurn()?.figures ?? []);
  readonly selectedTurnTraces = computed<ToolTraceItem[]>(() => this.selectedTurn()?.tool_traces ?? []);

  submitPrompt(text: string): void {
    this.promptText.set('');
    this.state.sendQuery(text);
  }

  onComposerSubmit(): void {
    const text = this.promptText().trim();
    if (!text) return;
    this.submitPrompt(text);
  }

  onComposerKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.onComposerSubmit();
    }
  }

  retryFailedPrompt(): void {
    const draft = this.state.failedPromptDraft();
    if (draft) {
      this.promptText.set('');
      this.state.retryFailedPrompt();
      return;
    }
    const text = this.promptText().trim();
    if (text) {
      this.submitPrompt(text);
    }
  }

  confirmClearChat(): void {
    this.confirmingClear.set(false);
    this.expandedEvidence.set(new Set());
    this.state.clearMessages();
  }

  focusCanvasTurn(idx: number): void {
    this.state.selectCanvasMessage(idx);
    this.mobileShowCanvas.set(true);
  }

  toggleEvidence(idx: number): void {
    this.expandedEvidence.update((set) => {
      const next = new Set(set);
      if (next.has(idx)) {
        next.delete(idx);
      } else {
        next.add(idx);
      }
      return next;
    });
  }

  copyAnswer(text: string): void {
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      void navigator.clipboard.writeText(text);
      this.state.showToast('Response copied to clipboard.');
    }
  }

  stringifyResult(val: unknown): string {
    if (typeof val === 'string') return val;
    try {
      return JSON.stringify(val, null, 2);
    } catch {
      return String(val);
    }
  }

  evalOverallKind(ev: EvaluationSummary): StatusKind {
    if (ev.passed === true) return 'success';
    if (ev.passed === false) return 'warning';
    return 'neutral';
  }

  evalOverallLabel(ev: EvaluationSummary): string {
    if (ev.passed === true) return 'Trace checks passed';
    if (ev.passed === false) return 'Trace check flagged';
    return 'Not evaluated';
  }

  expandFigure(fig: FigurePayload): void {
    this.state.openExpandedChart({
      title: fig.title,
      subtitle: 'Generated during AI Analyst session',
      figureJson: fig.figure_json,
      filename: `${slugifyFilename(fig.title)}.html`,
    });
  }

  downloadHtml(fig: FigurePayload): void {
    downloadPlotlyHtml(fig, {
      sessionId: this.state.activeSessionId(),
      datasetFilename: this.state.datasetMeta()?.filename ?? null,
    });
  }

  goToOnboarding(): void {
    void this.router.navigate(['/onboarding']);
  }
}
