# Deployment

## Backend (Render)

The Flask API lives in `backend/`. `backend/render.yaml` is a Render Blueprint —
easiest path:

1. Push this repo to GitHub.
2. On Render: **New → Blueprint** → select the repo. Render reads
   `backend/render.yaml` and creates the service.
3. Use at least the **Starter** plan — the ViT detector + TensorFlow need ~2 GB RAM.
   The free tier (512 MB) will OOM.

Manual alternative: create a Web Service, root directory `backend`, runtime
Python 3.11, build command `pip install -r requirements.txt`, start command
`gunicorn app:app --workers 1 --threads 4 --timeout 300`.

Notes:
- The first request after a deploy downloads the ViT weights (~350 MB) from
  Hugging Face unless the build step cached them. `render.yaml` already does
  this in `buildCommand`.
- `LOAD_TF_MODEL=0` keeps the weak LSTM model off; the ensemble uses forensic
  features + the pretrained ViT frame detector.
- Health check: `GET /health`.

## Frontend (Vercel)

The React app lives in `frontend/` (CRA). `frontend/vercel.json` already has
SPA rewrites.

1. Push repo to GitHub.
2. On Vercel: **Add New → Project** → import repo → set **Root Directory** to
   `frontend`.
3. Framework preset: Create React App (auto-detected).
4. Point the app at your backend — two options:
   - **Build-time (recommended):** add env var `REACT_APP_API_URL` =
     `https://deepfake-detector-api.onrender.com`, then deploy.
   - **Runtime (no rebuild):** after deploying, edit `config.js` in the
     `frontend/public/` folder (it ships to the site root) and set
     `window.API_URL = 'https://deepfake-detector-api.onrender.com';`,
     then redeploy. This wins over the env var.
5. Deploy.

The API client resolves the backend URL in this order:
`window.API_URL` (config.js) → `REACT_APP_API_URL` → `http://localhost:5000`.

## Local development

```bash
# backend (from repo root)
venv/Scripts/python backend/app.py   # Windows
# or: cd backend && gunicorn app:app

# frontend
cd frontend
npm start                            # talks to localhost:5000 by default
```

Set `REACT_APP_API_URL=http://localhost:5000` in `frontend/.env` if the API
runs elsewhere.
