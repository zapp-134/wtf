# Brain Tumor Detector MLOps Sandbox

Quick-and-dirty automation stack for the brain tumor classifier (EfficientNetB0). The repo focusses on the back-end + MLOps bits: Python training code, Flask inference service, Postgres logging, container builds, Jenkins CI stages, Kubeflow retraining pipeline, and Kubernetes manifests. Everything runs locally so you do not need GCP credits.

## Layout
- `ml/` - data pipeline + training (`train.py`), retrain orchestrator (`retrain.py`), evaluation guard (`check_metrics.py`), model promotion helper (`promote_model.py`).
- `api/` - Flask API (`app.py`) with optional Postgres logging (`db.py`).
- `data/` - dataset split into `train/`, `val/`, `test/` and feedback bucket `weak_feedback/` (`yes/`, `no/`).
- `scripts/` - helper shell loop for periodic retraining.
- `pipelines/` - Kubeflow pipeline definition + compiler script.
- `k8s/`, `argocd/` - manifests for GKE/K8s + Argo CD.
- `Dockerfile`, `docker-compose.yml`, `Jenkinsfile`, `Makefile` - automation glue.

## Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Initial training run
```bash
# uses data/* splits + weak feedback bucket
deactivate 2>/dev/null || true
source .venv/bin/activate
python ml/train.py --epochs 3 --batch_size 16 --out_dir ml/artifacts/local --weak_dir data/weak_feedback
```
Artifacts land inside `ml/artifacts/local/` (`model.keras`, `saved_model/`, `metrics.json`, etc.) and the best threshold is saved in `threshold.txt`.

## Flask API (local)
```bash
source .venv/bin/activate
export MODEL_DIR=ml/artifacts/local
export WEAK_FEEDBACK_DIR=data/weak_feedback
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres  # optional
python -m api.app
```
Endpoints:
- `GET /health` - returns `{status, model_loaded}`.
- `POST /predict` - body `{ "image": "<base64>" }` (accepts plain base64 or data URI). Response includes `prediction`, `probability`, `threshold`.
- `POST /feedback` - body `{ "image": "<base64>", "predicted_label": "yes", "correct_label": "no", "probability": 0.87 }`. Stores the image under `data/weak_feedback/<correct_label>/...` and logs to Postgres (if configured).
- `POST /trigger-retrain` - kicks `ml/retrain.py` (supports payload `{ "threshold": 5, "train_args": ["--epochs 2"] }`).

### Creating base64 payloads
```bash
python - <<'PY'
import base64, sys
with open(sys.argv[1], 'rb') as fh:
    print(base64.b64encode(fh.read()).decode())
PY path/to/image.jpg
```
Use the output inside the JSON request body.

## Automated retraining flow
- Every downvote stores the corrected image under `data/weak_feedback/<label>/`.
- `ml/retrain.py` counts new feedback items. When at least 5 new samples exist (default threshold) it runs `ml/train.py`, checks metrics with `ml/check_metrics.py`, and promotes the run into `ml/artifacts/current/` via `ml/promote_model.py`.
- State is tracked inside `ml/artifacts/state.json`.

Manual trigger:
```bash
python ml/retrain.py --threshold 5 --metric_min 0.72 --metric_drop_tol 0.03
```

Long-running watcher (used by Docker/K8s trainer sidecar):
```bash
./scripts/retrain_loop.sh  # respects THRESHOLD / INTERVAL env vars
```

## Docker + Compose (local 3-tier stack)
```bash
docker compose up --build
```
Services:
- `db` - Postgres 14.
- `api` - Gunicorn + Flask inference.
- `trainer` - loop that periodically runs `ml/retrain.py` (default threshold 5, every 5 minutes).
Volumes keep model artifacts + feedback synced with the host.

## Jenkins pipeline
`Jenkinsfile` mirrors the required flow:
1. Checkout + virtualenv bootstrap.
2. `python -m py_compile` smoke tests.
3. One-epoch training run (`ml/train.py`) with artifacts archived.
4. Docker build & simulated push.
5. Deploy manifests to a temporary review namespace.
6. Argo CD sync (stage).
7. Promote to `stage` namespace.
8. Manual gate.
9. Deploy to `prod` namespace.

Hook this pipeline to PRs and optionally wire Slack/email notifications on failures.

## Kubeflow retraining pipeline
Pipeline definition lives in `pipelines/brain_tumor_pipeline.py`. Compile it to a YAML spec:
```bash
python -m pipelines.compile
```
Upload `brain_tumor_pipeline.yaml` to Kubeflow. Pass the container image tag (e.g. `brain-tumor/api:<git-sha>`). The pipeline performs:
1. Training inside the Docker image (re-using `ml/train.py`).
2. Metric guard via `ml/check_metrics.py` (enforces minimum AUC + tolerance vs last promoted run).
3. Promotion using `ml/promote_model.py`.
Artifacts are shared via PVCs (`brain-artifacts`, `brain-feedback`).

## Kubernetes manifests + Argo CD
- `k8s/` contains deployments for the API, Postgres, PVCs, and a CronJob for periodic retraining.
- Apply manually with `kubectl apply -k k8s/` or let Argo CD own the folder via `argocd/application.yaml`.
- Pods mount persistent volumes so feedback + models survive restarts.

For local clusters (kind/minikube), create matching StorageClasses or tweak the PVC sizes.

## Make targets
- `make train` - 3-epoch local training.
- `make retrain` - checks feedback bucket + retrains if threshold met.
- `make serve` - launches Flask dev server.
- `make docker-build` - builds the API image.
- `make compose-up` / `make compose-down` - bring up/down the Compose stack.
- `make kubeflow` - compiles the pipeline spec.

## Presenting the flow (suggested demo order)
1. Show the dataset layout and baseline training metrics.
2. Run the Flask API; send a sample prediction.
3. Submit a "downvote" via `/feedback` and watch the image land in `data/weak_feedback/`.
4. Trigger `python ml/retrain.py` (or wait for the trainer loop) and highlight metric validation.
5. Re-run `/predict` after `POST /trigger-retrain` + `/reload` to prove the model swapped.
6. Walk through Jenkins/Argo/Kubeflow configs to satisfy the assignment deliverables.

Everything here is intentionally lightweight - tune epochs, batch sizes, or deployment knobs as needed for the demo.
