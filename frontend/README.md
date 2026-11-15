# SpectrumBloom Frontend (Angular 19 + TypeScript)

Production standalone Angular web workspace integrated with the SpectrumBloom FastAPI backend (`src/spectrumbloom/`).

## Local Development (with FastAPI Backend on `:8000`)

1. Start the FastAPI backend:
   ```bash
   uvicorn spectrumbloom.main:app --host 127.0.0.1 --port 8000
   ```
2. Start the Angular dev server (same-origin `/api` and `/health` proxy configured in `proxy.conf.json`):
   ```bash
   npm start
   ```
   Open `http://127.0.0.1:4200`.

## Checks & Builds

```bash
npm run typecheck
npx tsc --noEmit -p tsconfig.spec.json
npm run test:ci
npm run build
npm run e2e
```

## Security Design

- API Bearer tokens and optional external LLM API keys are held strictly in Angular signal memory (`AuthTokenService`) and never written to `localStorage`, URLs, or built assets.
- Markdown agent answers are rendered via `marked` + `DOMPurify` (`SafeMarkdownPipe`) with stripped scripts, frames, and dangerous links.
- In Docker Compose, `nginxinc/nginx-unprivileged:1.27-alpine` serves the built SPA on port `8501` and reverse-proxies `/api/` and `/health` directly to `http://api:8000`, forwarding `Authorization` headers unchanged.
