#!/bin/bash

echo "🚀 Starting Brain Tumor Detection Demo..."

# Start minikube if not running
if ! minikube status > /dev/null 2>&1; then
    echo "📦 Starting minikube..."
    minikube start
fi

# Set docker environment
eval $(minikube docker-env)

# Deploy if not already deployed
echo "🔧 Deploying application..."
kubectl apply -k k8s/

# Wait for pods to be ready
echo "⏳ Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=brain-api --timeout=300s

# Kill any existing port forwards
pkill -f "port-forward" 2>/dev/null || true

# Start port forwarding in background
echo "🌐 Setting up access..."
kubectl port-forward svc/brain-api 8080:80 > /dev/null 2>&1 &
kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8081:80 > /dev/null 2>&1 &

# Give it a moment to start
sleep 3

# Test if everything is working
echo "🧪 Testing API..."
if curl -s http://localhost:8080/health | grep -q "model_loaded.*true"; then
    echo "✅ SUCCESS! Demo is ready!"
    echo ""
    echo "🧠 Brain Tumor Detection API: http://localhost:8080"
    echo "🔬 Kubeflow Pipelines UI:     http://localhost:8081"
    echo ""
    echo "📊 Available endpoints:"
    echo "  • GET  /health          - Check status"
    echo "  • POST /predict         - Detect tumors"
    echo "  • POST /feedback        - Submit feedback"
    echo "  • POST /trigger-retrain - Start retraining"
    echo ""
    echo "🎉 Ready for your presentation!"
else
    echo "❌ Something went wrong. Check the logs:"
    echo "kubectl logs -l app=brain-api"
fi
