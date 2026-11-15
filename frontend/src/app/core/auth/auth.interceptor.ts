import { HttpErrorResponse, HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { catchError, throwError } from 'rxjs';
import { AuthTokenService } from './auth-token.service';

export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const auth = inject(AuthTokenService);
  const base = auth.apiBaseUrl();

  const isSameOriginRelative = req.url.startsWith('/api/v1/') && !req.url.startsWith('//');
  const isExplicitBase = Boolean(base) && req.url.startsWith(`${base}/api/v1/`);
  const isWindowOrigin =
    typeof window !== 'undefined' && req.url.startsWith(`${window.location.origin}/api/v1/`);

  const isBackendApiRequest = isSameOriginRelative || isExplicitBase || isWindowOrigin;

  const token = auth.token();
  const outboundReq =
    isBackendApiRequest && token
      ? req.clone({
          setHeaders: {
            Authorization: `Bearer ${token}`,
          },
        })
      : req;

  return next(outboundReq).pipe(
    catchError((err: unknown) => {
      if (err instanceof HttpErrorResponse && (err.status === 401 || err.status === 403)) {
        auth.markAuthChallenge(true);
      }
      return throwError(() => err);
    }),
  );
};
