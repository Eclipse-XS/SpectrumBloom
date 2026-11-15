import { ChangeDetectionStrategy, Component, ElementRef, inject, signal, viewChild } from '@angular/core';
import { Router } from '@angular/router';
import { WorkspaceStateService } from '../../core/state/workspace-state.service';
import { SvgIconComponent } from '../../shared/components/svg-icon.component';

const ALLOWED_EXTENSIONS = new Set(['csv', 'tsv', 'txt', 'xlsx', 'xls', 'xlsm', 'ods']);

@Component({
  selector: 'sb-onboarding',
  standalone: true,
  imports: [SvgIconComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styles: [
    `
      .onboarding {
        min-height: calc(100vh - 64px);
        max-width: 1020px;
        margin: 0 auto;
        padding: 5vh 28px 40px;
      }
      .onboard-intro {
        text-align: center;
        margin-bottom: 26px;
      }
      .onboard-intro h1 {
        font-size: 30px;
        letter-spacing: -0.04em;
        margin: 0 0 10px;
      }
      .onboard-intro p {
        font-size: 13px;
        line-height: 1.6;
        color: var(--muted);
        max-width: 520px;
        margin: 0 auto;
      }
      .upload-zone {
        min-height: 260px;
        border: 1.5px dashed var(--border-strong);
        border-radius: 12px;
        background: var(--surface);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        box-shadow: var(--shadow);
        transition: 0.18s ease;
        padding: 26px;
        text-align: center;
      }
      .upload-zone.drag {
        border-color: var(--primary);
        background: var(--primary-soft);
        transform: scale(1.005);
      }
      .upload-icon {
        width: 46px;
        height: 46px;
        border-radius: 10px;
        background: var(--primary-soft);
        color: var(--primary);
        display: grid;
        place-items: center;
        margin-bottom: 13px;
      }
      .upload-zone h2 {
        font-size: 15px;
        margin: 0 0 6px;
      }
      .upload-zone p {
        font-size: 12px;
        color: var(--muted);
        margin: 0 0 14px;
      }
      .upload-zone small {
        font-size: 10px;
        color: var(--faint);
        margin-top: 12px;
      }
      .upload-progress {
        width: min(260px, 90%);
        margin-top: 12px;
      }
      .sample-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-top: 12px;
        border: 1px solid var(--border);
        background: var(--surface);
        border-radius: 10px;
        padding: 14px 16px;
      }
      .sample-info {
        display: flex;
        align-items: center;
        gap: 11px;
        color: var(--primary);
      }
      .sample-info div {
        display: flex;
        flex-direction: column;
        gap: 2px;
        color: var(--text);
      }
      .sample-info strong {
        font-size: 12px;
      }
      .sample-info small {
        font-size: 10px;
        color: var(--muted);
      }
      .onboard-error {
        margin-top: 12px;
      }
      .onboard-steps {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 22px;
        margin-top: 30px;
      }
      .onboard-steps > div {
        display: flex;
        gap: 11px;
      }
      .onboard-steps b {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: var(--primary);
        padding-top: 2px;
      }
      .onboard-steps span {
        display: flex;
        flex-direction: column;
        gap: 4px;
      }
      .onboard-steps strong {
        font-size: 12px;
      }
      .onboard-steps small {
        font-size: 11px;
        line-height: 1.5;
        color: var(--muted);
      }
      @media (max-width: 800px) {
        .onboarding {
          padding: 4vh 16px 32px;
        }
        .onboard-steps {
          grid-template-columns: 1fr;
          gap: 14px;
        }
        .sample-row {
          flex-direction: column;
          align-items: stretch;
        }
      }
    `,
  ],
  template: `
    <main class="onboarding">
      <div class="onboard-intro">
        <div class="brand-mark large" aria-hidden="true">
          <span></span>
          <span></span>
          <span></span>
        </div>
        <div class="eyebrow">Local-first analytical workspace</div>
        <h1>Bring your data into focus.</h1>
        <p>
          Import a structured table to profile its quality, explore its distributions, and begin a
          grounded analysis with your configured FastAPI backend.
        </p>
      </div>

      <div
        class="upload-zone"
        [class.drag]="dragActive()"
        (dragover)="onDragOver($event)"
        (dragleave)="dragActive.set(false)"
        (drop)="onDrop($event)"
      >
        @if (state.uploading()) {
          <div class="loader-ring" aria-hidden="true"></div>
          <h2>Profiling {{ state.uploadingFilename() }}</h2>
          <p>Validating schema limits and running deterministic statistics…</p>
          <div class="indeterminate upload-progress" role="progressbar" aria-label="Uploading dataset">
            <i></i>
          </div>
          <small>Synchronous server profiling in progress</small>
        } @else {
          <div class="upload-icon">
            <sb-svg-icon name="upload" [size]="24" />
          </div>
          <h2>Drop a dataset here</h2>
          <p>or select a file from your computer</p>
          <button type="button" class="button primary" (click)="openFilePicker()">
            Browse files
          </button>
          <input
            #fileInput
            type="file"
            accept=".csv,.tsv,.txt,.xlsx,.xls,.xlsm,.ods"
            hidden
            (change)="onFileSelected($event)"
          />
          <small>
            CSV, TSV, TXT, XLSX, XLS, XLSM, or ODS · Maximum {{ state.maxUploadMb() }} MB
          </small>
        }
      </div>

      @if (localError() || state.uploadError()) {
        <div class="notice error onboard-error" role="alert">
          <sb-svg-icon name="warning" [size]="17" />
          <div class="notice-content">
            <strong>Upload could not be completed</strong>
            <span>{{ localError() || state.uploadError() }}</span>
          </div>
        </div>
      }

      <div class="sample-row">
        <div class="sample-info">
          <sb-svg-icon name="database" [size]="20" />
          <div>
            <strong>Not ready with a file?</strong>
            <small>
              Upload the bundled 180-row e-commerce CSV through the real FastAPI upload endpoint.
            </small>
          </div>
        </div>
        <button
          type="button"
          class="button"
          [disabled]="state.uploading()"
          (click)="useSampleDataset()"
        >
          Load sample dataset
        </button>
      </div>

      <div class="onboard-steps">
        <div>
          <b>01</b>
          <span>
            <strong>Hardened import</strong>
            <small>
              Validated server-side against upload size limits, delimiters, and formula injection.
            </small>
          </span>
        </div>
        <div>
          <b>02</b>
          <span>
            <strong>Deterministic profile</strong>
            <small>
              Semantic types, missingness, normality tests, outliers, and Plotly charts compute first.
            </small>
          </span>
        </div>
        <div>
          <b>03</b>
          <span>
            <strong>Grounded AI &amp; ML</strong>
            <small>
              Inspect tool traces, verification checks, and leak-safe sklearn baseline comparisons.
            </small>
          </span>
        </div>
      </div>
    </main>
  `,
})
export class OnboardingComponent {
  readonly state = inject(WorkspaceStateService);
  private readonly router = inject(Router);

  private readonly fileInput = viewChild<ElementRef<HTMLInputElement>>('fileInput');
  readonly dragActive = signal<boolean>(false);
  readonly localError = signal<string | null>(null);

  openFilePicker(): void {
    this.fileInput()?.nativeElement.click();
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.dragActive.set(true);
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragActive.set(false);
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      this.handleFile(file);
    }
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    input.value = '';
    if (file) {
      this.handleFile(file);
    }
  }

  useSampleDataset(): void {
    this.localError.set(null);
    this.state.loadSampleDataset(() => {
      void this.router.navigate(['/overview']);
    });
  }

  private handleFile(file: File): void {
    this.localError.set(null);
    const ext = file.name.includes('.') ? file.name.split('.').pop()?.toLowerCase() ?? '' : '';
    if (!ALLOWED_EXTENSIONS.has(ext)) {
      this.localError.set(
        `Unsupported file extension ".${ext || 'unknown'}". Use CSV, TSV, TXT, XLSX, XLS, XLSM, or ODS.`,
      );
      return;
    }
    this.state.uploadFile(file, () => {
      void this.router.navigate(['/overview']);
    });
  }
}
