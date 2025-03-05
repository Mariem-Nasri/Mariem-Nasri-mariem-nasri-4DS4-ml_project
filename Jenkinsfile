pipeline {
    agent any

    environment {
        PYTHON = "python3"
        ENV_NAME = "venv"
        REQUIREMENTS = "requirements.txt"
        MAIN = "src.main"
        TEST = "pytest tests/"
        APP = "app.py"
        DATA_DIR = "data"
        MODEL_DIR = "models"
        MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
        DOCKER_USERNAME = "mariem773"
        IMAGE_NAME = "flask_predict"
        TAG = "latest"
        DOCKERFILE = "Dockerfile"
    }

    triggers {
        githubPush()
    }

    stages {
        stage('Clean Workspace') {
            steps {
                echo "Cleaning workspace..."
                sh 'rm -rf venv/'
                sh 'find . -type d -name "_pycache_" -exec rm -rf {} +'
                sh "rm -rf ${DATA_DIR}/*.pkl ${MODEL_DIR}/*.pkl"
            }
        }

        stage('Setup Environment') {
            steps {
                echo "Setting up virtual environment..."
                sh """
                set -x
                ${PYTHON} -m venv ${ENV_NAME}
                . ${ENV_NAME}/bin/activate && pip install --default-timeout=100 -r ${REQUIREMENTS}
                """
            }
        }
        stage('Start MLflow Server') {
            steps {
                echo "Starting MLflow server..."
                sh ". ${ENV_NAME}/bin/activate && nohup mlflow server --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &"
            }
        }

        stage('Code Quality') {
            steps {
                echo "Running code quality checks..."
                sh ". ${ENV_NAME}/bin/activate && flake8 --max-line-length=120 src/"
                sh ". ${ENV_NAME}/bin/activate && black src/"          // Automatically format files
                sh ". ${ENV_NAME}/bin/activate && black --check src/"  // Check if files need reformatting
                sh ". ${ENV_NAME}/bin/activate && bandit -r src/"
            }
        }

        stage('Run Unit Tests') {
            steps {
                echo "Running unit tests..."
                sh ". ${ENV_NAME}/bin/activate && ${PYTHON} -m ${TEST}"
            }
        }

        stage('Data Preparation') {
            steps {
                echo "Preparing data..."
                sh ". ${ENV_NAME}/bin/activate && ${PYTHON} -m ${MAIN} --prepare"
            }
        }

        stage('Train Model with MLflow') {
            steps {
                echo "Training model..."
                sh ". ${ENV_NAME}/bin/activate && ${PYTHON} -m ${MAIN} --train"
            }
        }

        stage('Evaluate Model with MLflow') {
            steps {
                echo "Evaluating model..."
                sh ". ${ENV_NAME}/bin/activate && ${PYTHON} -m ${MAIN} --evaluate"
            }
        }

        stage('Save Model to Production') {
            steps {
                echo "Saving model to production..."
                sh ". ${ENV_NAME}/bin/activate && ${PYTHON} -m ${MAIN} --save"
            }
        }

        stage('Predict') {
            steps {
                echo "Running prediction..."
                sh ". ${ENV_NAME}/bin/activate && ${PYTHON} -m ${MAIN} --predict"
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "Building Docker image..."
                script {
                    docker.build("${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}")
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                echo "Pushing Docker image..."
                script {
                    docker.withRegistry('https://registry.hub.docker.com', 'docker-hub-credentials') {
                        docker.image("${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}").push()
                    }
                }
            }
        }

        stage('Deploy with Docker') {
            steps {
                script {
                    def containerName = "${IMAGE_NAME}-${BUILD_ID}"

                    // Check if container exists, then stop and remove it if necessary
                    def containerExists = sh(script: "docker ps -a --filter 'name=${containerName}' -q", returnStdout: true).trim()
                    if (containerExists) {
                        sh "docker stop ${containerName} || true"
                        sh "docker rm ${containerName} || true"
                    }

                    // Run container
                    sh "docker run -d --name ${containerName} -p 5001:5000 ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
                }
            }
        }
    }

    post {
        always {
            script {
                def containerName = "${IMAGE_NAME}-${BUILD_ID}"

                // Check if container exists, then stop and remove it if necessary
                def containerExists = sh(script: "docker ps -a --filter 'name=${containerName}' -q", returnStdout: true).trim()
                if (containerExists) {
                    sh "docker stop ${containerName} || true"
                    sh "docker rm ${containerName} || true"
                }
            }

            sh 'sleep 40'
            sh 'pkill -f "python3 app.py" || true'
        }
    }
}
