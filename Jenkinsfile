pipeline {
  agent any

  environment {
    VENV = "venv"
    IMAGE = "brain-tumor/api:${env.BUILD_NUMBER}"
    REVIEW_NS = "review-${env.BUILD_NUMBER}"
    STAGE_NS = "stage"
    PROD_NS = "prod"
  }

  options {
    timestamps()
    ansiColor('xterm')
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Setup Python') {
      steps {
        sh 'python3 -m venv ${VENV}'
        sh '. ${VENV}/bin/activate && pip install --upgrade pip && pip install -r requirements.txt'
      }
    }

    stage('Lint & Smoke Tests') {
      steps {
        sh '. ${VENV}/bin/activate && python -m py_compile ml/*.py api/*.py'
      }
    }

    stage('Train Model') {
      steps {
        sh '. ${VENV}/bin/activate && python ml/train.py --epochs 1 --batch_size 8 --out_dir ml/artifacts/cicd --weak_dir data/weak_feedback'
      }
      post {
        always {
          archiveArtifacts artifacts: 'ml/artifacts/cicd/**', allowEmptyArchive: true
        }
      }
    }

    stage('Build API Image') {
      steps {
        sh 'docker build -t ${IMAGE} .'
      }
    }

    stage('Push Image') {
      steps {
        sh 'echo "Simulating push to registry..."'
      }
    }

    stage('Deploy Temp Namespace') {
      steps {
        sh 'kubectl create namespace ${REVIEW_NS} || true'
        sh 'kubectl apply -n ${REVIEW_NS} -f k8s/'
      }
    }

    stage('Argo CD Sync Stage') {
      steps {
        sh 'argocd app sync brain-tumor-stage --grpc-web || echo "ArgoCD CLI not configured"'
      }
    }

    stage('Promote to Stage') {
      steps {
        sh 'kubectl apply -n ${STAGE_NS} -f k8s/'
      }
    }

    stage('Manual Gate') {
      steps {
        input message: 'Deploy to prod?', ok: 'Ship it'
      }
    }

    stage('Deploy to Prod') {
      steps {
        sh 'kubectl apply -n ${PROD_NS} -f k8s/'
      }
    }
  }

  post {
    always {
      sh 'docker image ls ${IMAGE} || true'
    }
  }
}
