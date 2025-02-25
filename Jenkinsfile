pipeline {
    agent any

    environment {
        PYTHON = "python3"
        ENV_NAME = "venv"
        REQUIREMENTS = "requirements.txt"
        MAIN = "src.main"
        APP = "deployment/app.py"
        DATA_DIR = "data"
        MODEL_DIR = "models"
        MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
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
                sh ". ${ENV_NAME}/bin/activate && nohup mlflow server --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &"
            }
        }

        stage('Code Quality') {
            steps {
                sh ". ${ENV_NAME}/bin/activate && flake8 --max-line-length=120 src/"
                sh ". ${ENV_NAME}/bin/activate && black --check src/"
                sh ". ${ENV_NAME}/bin/activate && bandit -r src/"
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

        stage('Deploy') {
            steps {
                sh ". ${ENV_NAME}/bin/activate && nohup ${PYTHON} ${APP} > flask.log 2>&1 &"
            }
        }
    }

    post {
        always {
            sh 'sleep 30'
            sh 'pkill -f "python3 deployment/app.py" || true'
        }
    }
}