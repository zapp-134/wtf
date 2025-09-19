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
                    bat '''
                    python --version || (
                        echo ERROR: Python not found! Please install Python from python.org and add it to PATH.
                        exit 1
                    )
                    '''
                }
            }
        }

        stage('Setup Python') {
            steps {
                ansiColor('xterm') {
                    bat '''
                    if not exist %VENV% (
                        python -m venv %VENV%
                    )
                    call %VENV%\\Scripts\\activate && pip install --upgrade pip && pip install -r requirements.txt
                    '''
                }
            }
        }

        stage('Lint & Smoke Tests') {
            steps {
                ansiColor('xterm') {
                    bat 'call %VENV%\\Scripts\\activate && python -m py_compile ml\\*.py api\\*.py'
                }
            }
        }

        stage('Train Model') {
            steps {
                ansiColor('xterm') {
                    bat 'call %VENV%\\Scripts\\activate && python ml\\train.py --epochs 1 --batch_size 8 --out_dir ml\\artifacts\\cicd --weak_dir data\\weak_feedback'
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
                    bat 'docker build -t %IMAGE% .'
                }
            }
        }

        stage('Push Image') {
            steps {
                ansiColor('xterm') {
                    bat 'echo Simulating push to registry...'
                }
            }
        }

        stage('Deploy Temp Namespace') {
            steps {
                ansiColor('xterm') {
                    bat 'kubectl create namespace %REVIEW_NS% || exit 0'
                    bat 'kubectl apply -n %REVIEW_NS% -f k8s\\'
                }
            }
        }

        stage('Argo CD Sync Stage') {
            steps {
                ansiColor('xterm') {
                    bat 'argocd app sync brain-tumor-stage --grpc-web || echo ArgoCD CLI not configured'
                }
            }
        }

        stage('Promote to Stage') {
            steps {
                ansiColor('xterm') {
                    bat 'kubectl apply -n %STAGE_NS% -f k8s\\'
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
                    bat 'kubectl apply -n %PROD_NS% -f k8s\\'
                }
            }
        }
    }

    post {
        always {
            ansiColor('xterm') {
                bat 'docker image ls %IMAGE% || exit 0'
            }
        }
    }
}
