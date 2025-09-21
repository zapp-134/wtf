#!/bin/bash

echo "🔗 Getting permanent service URLs..."

# Get minikube service URLs (these work without port-forwarding!)
echo "🧠 Brain Tumor API:"
minikube service brain-api --url

echo ""
echo "🔬 Kubeflow UI:"
kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8081:80 > /dev/null 2>&1 &
echo "http://localhost:8081"

echo ""
echo "💡 The brain-api URL above works directly - no port-forwarding needed!"
echo "💡 Keep this terminal open for Kubeflow UI access"
