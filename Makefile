.PHONY: train retrain serve docker-build compose-up compose-down kubeflow

train:
	python ml/train.py --epochs 3 --batch_size 16 --out_dir ml/artifacts/local --weak_dir data/weak_feedback

retrain:
	python ml/retrain.py --threshold 5

serve:
	FLASK_APP=api.app:create_app FLASK_RUN_PORT=8000 flask run

docker-build:
	docker build -t brain-tumor/api:latest .

compose-up:
	docker-compose up --build

compose-down:
	docker-compose down

kubeflow:
	python -m pipelines.compile
