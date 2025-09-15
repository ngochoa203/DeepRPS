# DeepRPS – Deploying BE to Render, FE to Vercel

This repo has three folders: `AI/`, `BE/` (FastAPI backend), and `FE/` (Vite + React frontend).

## Backend (Render)

- Stack: FastAPI + Uvicorn, Python 3.11.
- Requirements pinned in `BE/requirements.txt`.
- App entry: `BE.server.main:app`.
- Health endpoint: `/healthz`.
- CORS: controlled by `ALLOW_ORIGINS` env (comma-separated list; default `*`).
- State storage: writes to `STATE_DIR` (default `/var/tmp/rps_state`). On Render free tier this is ephemeral. If you need persistence across deploys, upgrade the plan and attach a disk, then set `STATE_DIR` to the mount path (e.g., `/data/rps_state`) and add a `disk` section to `render.yaml`.

### One-click Render

Render will detect `render.yaml` at repo root. The provided config targets the free tier (no disks).
- Service name: `deep-rps-be`
- Build: `pip install -r requirements.txt`
- Start: `uvicorn BE.server.main:app --host 0.0.0.0 --port $PORT`
- Env vars: `PYTHON_VERSION=3.11`, `STATE_DIR=/data/rps_state`, `ALLOW_ORIGINS=*`
  Note: Disks are not supported on free tier; for persistence, upgrade and add a disk.

If you prefer manual setup, create a Web Service with:
- Root directory: `BE`
- Start command: as above
- Health check path: `/healthz`

## Frontend (Vercel)

- Stack: Vite + React (TypeScript).
- Build via Vercel config at repo root: `vercel.json` builds from `FE/package.json` and serves `FE/dist`.
- Configure backend URL: set `VITE_API_BASE` env in Vercel project settings to your Render URL, e.g. `https://deep-rps-be.onrender.com`.
- Local dev uses Vite proxy (`vite.config.ts`) to forward `/api` and `/ws` to `http://localhost:8000`.

### Local run

- Backend:
  - Create venv and install deps
  - Run server

- Frontend:
  - `npm i` in `FE/`
  - `npm run dev`

### Environment

- `ALLOW_ORIGINS` (BE): CORS origins, comma-separated. Default `*`.
- `STATE_DIR` (BE): Directory for persisted JSON state (defaults to `/data/rps_state`).
- `VITE_API_BASE` (FE): Full URL to backend base when deployed on Vercel.

## Notes

- The backend stores per-user game state JSON files under `STATE_DIR` using `BE/gamebrain/storage.py`.
- WebSocket endpoint for simple relay is `/ws`; hosting WS on Vercel is not required as it talks directly to the backend.
- If you change expert list size, existing user/global state files may have older weight dimensions; the code already guards and re-seeds from the global prior when mismatched.
