import { provideHttpClient, withInterceptors } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { AppComponent } from './app.component';
import { routes } from './app.routes';
import { authInterceptor } from './core/auth/auth.interceptor';

describe('AppComponent', () => {
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    localStorage.clear();
    await TestBed.configureTestingModule({
      imports: [AppComponent],
      providers: [
        provideRouter(routes),
        provideHttpClient(withInterceptors([authInterceptor])),
        provideHttpClientTesting(),
      ],
    }).compileComponents();
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('creates app shell and bootstraps /health then /api/v1/sessions', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();

    const healthReq = httpMock.expectOne('/health');
    healthReq.flush({
      status: 'ok',
      version: '0.1.0',
      session_repository: 'connected',
      dataset_storage: { backend: 'local', status: 'connected' },
      sandbox_backend: 'docker',
      sandbox_dynamic_python_enabled: true,
      auth_required: false,
      max_upload_mb: 25,
      default_provider: 'ollama',
      default_model: 'qwen3:4b-instruct',
      llm: {
        provider: 'ollama',
        status: 'connected',
        configured_model: 'qwen3:4b-instruct',
        available_models: ['qwen3:4b-instruct'],
      },
      observability: { opentelemetry: true, langfuse: false },
    });

    const listReq = httpMock.expectOne('/api/v1/sessions');
    listReq.flush([]);

    const createReq = httpMock.expectOne('/api/v1/sessions');
    expect(createReq.request.method).toBe('POST');
    createReq.flush({
      session_id: 's1',
      thread_id: 't1',
      provider: 'ollama',
      model: 'qwen3:4b-instruct',
      has_data: false,
    });

    const getReq = httpMock.expectOne('/api/v1/sessions/s1');
    getReq.flush({
      session_id: 's1',
      thread_id: 't1',
      provider: 'ollama',
      model: 'qwen3:4b-instruct',
      has_data: false,
      message_count: 0,
      figure_count: 0,
      dataset: null,
      created_at: '2026-11-15T12:00:00Z',
      updated_at: '2026-11-15T12:00:00Z',
    });

    const refreshReq = httpMock.expectOne('/api/v1/sessions');
    refreshReq.flush([]);

    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('SpectrumBloom');
  });
});
