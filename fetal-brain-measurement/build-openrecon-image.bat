@echo off
REM Build script for OpenRecon Fetal Brain Measurement Docker image
REM This script builds the integrated Docker image that combines the MRD server with fetal brain measurements

echo ========================================================
echo Building OpenRecon Fetal Brain Measurement Docker Image
echo ========================================================

REM Check if we're in the correct directory
if not exist "Dockerfile.openrecon.integrated" (
    echo Error: Dockerfile.openrecon.integrated not found in current directory
    echo Please run this script from the fetal-brain-measurement directory
    exit /b 1
)

REM Check if required files exist
if not exist "fetalbrainmeasure.py" (
    echo Warning: Required file not found: fetalbrainmeasure.py
)
if not exist "fetalbrainmeasure.json" (
    echo Warning: Required file not found: fetalbrainmeasure.json
)
if not exist "Code\FetalMeasurements-master\requirements.txt" (
    echo Warning: Required file not found: Code\FetalMeasurements-master\requirements.txt
)
if not exist "Models\" (
    echo Warning: Required directory not found: Models\
)

REM Build the image
set IMAGE_NAME=openrecon-fetal-brain:latest
echo Building Docker image: %IMAGE_NAME%
echo This may take several minutes...

docker build --file Dockerfile.openrecon.integrated --tag "%IMAGE_NAME%" --progress=plain .

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo ✅ Build completed successfully!
    echo ========================================================
    echo Docker image: %IMAGE_NAME%
    echo.
    echo To run the server:
    echo   .\run-openrecon-server.bat
    echo.
    echo To run with custom port:
    echo   docker run -p 9003:9002 --gpus all %IMAGE_NAME%
    echo.
    echo To run with data volume mounted:
    echo   docker run -p 9002:9002 --gpus all -v "%cd%\data:/tmp/share" %IMAGE_NAME%
    echo.
) else (
    echo.
    echo ========================================================
    echo ❌ Build failed!
    echo ========================================================
    echo Please check the error messages above and try again.
    exit /b 1
)
