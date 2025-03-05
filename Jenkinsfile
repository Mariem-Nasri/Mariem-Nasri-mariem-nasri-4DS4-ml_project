pipeline {
    agent any

    environment {
        PYTHON = "python3"
        ENV_NAME = "venv"
        REQUIREMENTS = "requirements.txt"
        MAIN = "src.main"
        TEST = "pytest tests/"
        APP = "deployment/app.py"
        DATA_DIR = "data"
        MODEL_DIR = "models"
        MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
        DOCKER_USERNAME = "haifabenyoussef"
        IMAGE_NAME = "mariem773/mariem_nasri_4ds4_mlops"
        TAG = "latest"
        DOCKERFILE = "Dockerfile"
        CONTAINER_NAME = "${IMAGE_NAME}_${BUILD_NUMBER}"
    }

    triggers {
        githubPush()
    }

    stages {
        stage('Clean Workspace') {
            steps {
                sh 'rm -rf venv/'
                sh 'find . -type d -name "_pycache_" -exec rm -rf {} +'
                sh "rm -rf ${DATA_DIR}/*.pkl ${MODEL_DIR}/*.pkl"
            }
        }

        stage('Setup Environment') {
            steps {
                sh """
                ${PYTHON} -m venv ${ENV_NAME}
                . ${ENV_NAME}/bin/activate && pip install --default-timeout=100 -r ${REQUIREMENTS}
                """
            }
        }

        stage('Start MLflow Server') {
            steps {
                sh ". ${ENV_NAME}/bin/activate && nohup mlflow server --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000 &> mlflow.log &"
            }
        }

        stage('Code Quality') {
            steps {
                sh ". ${ENV_NAME}/bin/activate && flake8 --max-line-length=120 src/"
                sh ". ${ENV_NAME}/bin/activate && black --check src/"
                sh ". ${ENV_NAME}/bin/activate && black src/"
                sh ". ${ENV_NAME}/bin/activate && bandit -r src/"
            }
        }

        stage('Run Unit Tests') {
            steps {
                sh ". ${ENV_NAME}/bin/activate && ${PYTHON} -m ${TEST}"
            }
        }

        stage('Data Preparation') {
            steps {
                sh ". ${ENV_NAME}/bin/activate && ${PYTHON} -m ${MAIN} --prepare"
            }
        }

        stage('Train Model with MLflow') {
            steps {
                sh ". ${ENV_NAME}/bin/activate && ${PYTHON} -m ${MAIN} --train"
            }
        }

        stage('Evaluate Model with MLflow') {
            steps {
                sh ". ${ENV_NAME}/bin/activate && ${PYTHON} -m ${MAIN} --evaluate"
            }
        }

        stage('Save Model to Production') {
            steps {
                sh ". ${ENV_NAME}/bin/activate && ${PYTHON} -m ${MAIN} --save"
            }
        }

        stage('Predict') {
            steps {
                sh ". ${ENV_NAME}/bin/activate && ${PYTHON} -m ${MAIN} --predict"
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}")
                }
            }
        }

        stage('Push Docker Image') {
            steps {
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
                    sh "docker stop ${CONTAINER_NAME} || true"
                    sh "docker rm ${CONTAINER_NAME} || true"
                    sh "docker run -d --name ${CONTAINER_NAME} -p 5001:5000 ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
                }
            }
        }
    }

    post {
        always {
            sh "docker stop ${CONTAINER_NAME} || true"
            sh "docker rm ${CONTAINER_NAME} || true"
            sh 'sleep 40'
            sh 'pkill -f "python3 deployment/app.py" || true'
        }
    }
}
