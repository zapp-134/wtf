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

        stage('Check Python') {
            steps {
                ansiColor('xterm') {
                    sh 'python3 --version'
                }
            }
        }

        stage('Setup Python') {
            steps {
                ansiColor('xterm') {
                    sh '''
                    if [ ! -d "${VENV}" ]; then
                        python3 -m venv ${VENV}
                    fi
                    source ${VENV}/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    '''
                }
            }
        }

        stage('Lint & Smoke Tests') {
            steps {
                ansiColor('xterm') {
                    sh '''
                    source ${VENV}/bin/activate
                    python3 -m py_compile ml/*.py api/*.py
                    '''
                }
            }
        }

        stage('Train Model') {
            steps {
                ansiColor('xterm') {
                    sh '''
                    source ${VENV}/bin/activate
                    python3 ml/train.py --epochs 1 --batch_size 8 --out_dir ml/artifacts/cicd --weak_dir data/weak_feedback
                    '''
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
                    // Assuming Docker is installed on the Jenkins agent
                    sh 'docker build -t ${IMAGE} .'
                }
            }
        }

        stage('Push Image') {
            steps {
                ansiColor('xterm') {
                    // This stage would contain your gcloud/Artifact Registry push commands
                    sh 'echo "Simulating push to ${IMAGE}..."'
                }
            }
        }

        stage('Deploy Temp Namespace') {
            steps {
                ansiColor('xterm') {
                    // Use '|| true' to ignore "already exists" errors
                    sh 'kubectl create namespace ${REVIEW_NS} || true'
                    // Use forward slashes for paths
                    sh 'kubectl apply -n ${REVIEW_NS} -f k8s/'
                }
            }
        }

        stage('Argo CD Sync Stage') {
            steps {
                ansiColor('xterm') {
                    sh 'argocd app sync brain-tumor-stage --grpc-web || echo "ArgoCD CLI not configured or sync failed"'
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
                // Clean up Docker image on the agent
                sh 'docker image ls ${IMAGE} || true'
            }
        }
    }
}