import { HttpClient, provideHttpClient, withInterceptors } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { AuthTokenService } from './auth-token.service';
import { authInterceptor } from './auth.interceptor';

describe('authInterceptor', () => {
  let http: HttpClient;
  let httpMock: HttpTestingController;
  let auth: AuthTokenService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(withInterceptors([authInterceptor])),
        provideHttpClientTesting(),
      ],
    });
    http = TestBed.inject(HttpClient);
    httpMock = TestBed.inject(HttpTestingController);
    auth = TestBed.inject(AuthTokenService);
    auth.clearToken();
    auth.clearAuthChallenge();
    auth.setApiBaseUrl('');
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('attaches Bearer token for /api/v1 requests and never writes to localStorage', () => {
    auth.setToken('secret-token-123');

    http.get('/api/v1/sessions').subscribe();
    const req = httpMock.expectOne('/api/v1/sessions');
    expect(req.request.headers.get('Authorization')).toBe('Bearer secret-token-123');
    req.flush([]);

    expect(localStorage.getItem('secret-token-123')).toBeNull();
  });

  it('does not attach Bearer token to external URLs', () => {
    auth.setToken('secret-token-123');

    http.get('https://external.example.com/api/v1/data').subscribe();
    const req = httpMock.expectOne('https://external.example.com/api/v1/data');
    expect(req.request.headers.has('Authorization')).toBeFalse();
    req.flush({});
  });

  it('triggers auth challenge on 401 responses', () => {
    http.get('/api/v1/sessions').subscribe({
      error: () => {
        /* expected */
      },
    });
    const req = httpMock.expectOne('/api/v1/sessions');
    req.flush({ detail: 'Unauthorized' }, { status: 401, statusText: 'Unauthorized' });

    expect(auth.authChallenge()).toBeTrue();
  });
});
