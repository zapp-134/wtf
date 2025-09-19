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
    }

    stages {
        stage('Checkout') {
            steps {
                ansiColor('xterm') {
                    checkout scm
                }
            }
        }

        stage('Setup Python') {
            steps {
                ansiColor('xterm') {
                    sh 'python3 -m venv ${VENV}'
                    sh '. ${VENV}/bin/activate && pip install --upgrade pip && pip install -r requirements.txt'
                }
            }
        }

        stage('Lint & Smoke Tests') {
            steps {
                ansiColor('xterm') {
                    sh '. ${VENV}/bin/activate && python -m py_compile ml/*.py api/*.py'
                }
            }
        }

        stage('Train Model') {
            steps {
                ansiColor('xterm') {
                    sh '. ${VENV}/bin/activate && python ml/train.py --epochs 1 --batch_size 8 --out_dir ml/artifacts/cicd --weak_dir data/weak_feedback'
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'ml/artifacts/cicd/**', allowEmptyArchive: true
                }
            }
        }

        stage('Build API Image') {
            steps {
                ansiColor('xterm') {
                    sh 'docker build -t ${IMAGE} .'
                }
            }
        }

        stage('Push Image') {
            steps {
                ansiColor('xterm') {
                    sh 'echo "Simulating push to registry..."'
                }
            }
        }

        stage('Deploy Temp Namespace') {
            steps {
                ansiColor('xterm') {
                    sh 'kubectl create namespace ${REVIEW_NS} || true'
                    sh 'kubectl apply -n ${REVIEW_NS} -f k8s/'
                }
            }
        }

        stage('Argo CD Sync Stage') {
            steps {
                ansiColor('xterm') {
                    sh 'argocd app sync brain-tumor-stage --grpc-web || echo "ArgoCD CLI not configured"'
                }
            }
        }

        stage('Promote to Stage') {
            steps {
                ansiColor('xterm') {
                    sh 'kubectl apply -n ${STAGE_NS} -f k8s/'
                }
            }
        }

        stage('Manual Gate') {
            steps {
                input message: 'Deploy to prod?', ok: 'Ship it'
            }
        }

        stage('Deploy to Prod') {
            steps {
                ansiColor('xterm') {
                    sh 'kubectl apply -n ${PROD_NS} -f k8s/'
                }
            }
        }
    }

    post {
        always {
            ansiColor('xterm') {
                sh 'docker image ls ${IMAGE} || true'
            }
        }
    }
}
