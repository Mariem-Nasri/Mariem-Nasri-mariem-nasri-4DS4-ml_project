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
        PYTHONPATH = "${WORKSPACE}"  // Ensure Python can find the `src` directory
    }

    triggers {
        pollSCM('H/5 * * * *')  // Poll SCM every 5 minutes
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
                    . ${ENV_NAME}/bin/activate && pip install --upgrade pip
                    pip install --default-timeout=100 -r ${REQUIREMENTS}
                """
            }
        }

        stage('Start MLflow Server') {
            steps {
                sh """
                    . ${ENV_NAME}/bin/activate
                    nohup mlflow server --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &
                    sleep 10  # Wait for MLflow server to start
                    curl --retry 5 --retry-delay 10 --retry-connrefused http://127.0.0.1:5000
                """
            }
        }

        stage('Code Quality') {
            steps {
                sh """
                    . ${ENV_NAME}/bin/activate
                    black src/
                    flake8 --max-line-length=120 src/
                    bandit -r src/
                """
            }
        }

        stage('Run Tests') {
            steps {
                sh """
                    . ${ENV_NAME}/bin/activate
                    pip install pytest-cov
                    pytest tests/ --cov=src --cov-report=xml --junitxml=test-results.xml
                """
            }
            post {
                always {
                    junit 'test-results.xml'  // Publish test results
                }
            }
        }

        stage('Data Preparation') {
            steps {
                sh """
                    . ${ENV_NAME}/bin/activate
                    ${PYTHON} -m ${MAIN} --prepare
                """
            }
        }

        stage('Train Model with MLflow') {
            steps {
                catchError(buildResult: 'FAILURE', stageResult: 'FAILURE') {
                    sh """
                        . ${ENV_NAME}/bin/activate
                        ${PYTHON} -m ${MAIN} --train || (echo 'Training failed'; exit 1)
                    """
                }
            }
        }

        stage('Evaluate Model with MLflow') {
            steps {
                catchError(buildResult: 'FAILURE', stageResult: 'FAILURE') {
                    sh """
                        . ${ENV_NAME}/bin/activate
                        ${PYTHON} -m ${MAIN} --evaluate || (echo 'Evaluation failed'; exit 1)
                    """
                }
            }
        }

        stage('Save Model to Production') {
            steps {
                sh """
                    . ${ENV_NAME}/bin/activate
                    ${PYTHON} -m ${MAIN} --save
                """
            }
        }

        stage('Deploy') {
            steps {
                sh """
                    . ${ENV_NAME}/bin/activate
                    nohup ${PYTHON} ${APP} > flask.log 2>&1 &
                    sleep 10  # Wait for the Flask app to start
                """
            }
        }
    }

    post {
        always {
            sh 'pkill -f "mlflow server" || true'  // Stop MLflow server
            sh 'pkill -f "python3 deployment/app.py" || true'  // Stop Flask app
            sh 'sleep 10'  // Wait for processes to stop
            archiveArtifacts artifacts: 'mlflow.log, flask.log, coverage.xml', allowEmptyArchive: true  // Archive logs and coverage report
            junit 'test-results.xml'  // Publish test results
        }
        failure {
            emailext body: 'Build failed. Please check: ${BUILD_URL}', subject: 'Build Failed: ${JOB_NAME}', to: 'your-email@example.com'
        }
        success {
            emailext body: 'Build succeeded: ${BUILD_URL}', subject: 'Build Succeeded: ${JOB_NAME}', to: 'your-email@example.com'
        }
    }
}
