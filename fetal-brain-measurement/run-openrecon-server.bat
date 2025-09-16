@echo off
REM Run script for OpenRecon Fetal Brain Measurement Server
REM This script starts the integrated MRD server with fetal brain measurement capabilities

set IMAGE_NAME=openrecon-fetal-brain:latest
set CONTAINER_NAME=openrecon-fetal-server
set HOST_PORT=9002
set CONTAINER_PORT=9002

echo ========================================================
echo Starting OpenRecon Fetal Brain Measurement Server
echo ========================================================

REM Check if image exists
docker images | findstr "openrecon-fetal-brain" >nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Docker image 'openrecon-fetal-brain' not found
    echo Please build the image first using: .\build-openrecon-image.bat
    exit /b 1
)

REM Stop existing container if running
docker ps | findstr "%CONTAINER_NAME%" >nul
if %ERRORLEVEL% EQU 0 (
    echo Stopping existing container...
    docker stop "%CONTAINER_NAME%"
)

REM Remove existing container if exists
docker ps -a | findstr "%CONTAINER_NAME%" >nul
if %ERRORLEVEL% EQU 0 (
    echo Removing existing container...
    docker rm "%CONTAINER_NAME%"
)

REM Create data directory if it doesn't exist
if not exist "%cd%\data\debug" mkdir "%cd%\data\debug"
if not exist "%cd%\data\fetal_measurements" mkdir "%cd%\data\fetal_measurements"
if not exist "%cd%\data\logs" mkdir "%cd%\data\logs"

echo Starting container...
echo Server will be available at: localhost:%HOST_PORT%
echo Data directory: %cd%\data
echo.

REM Run the container
docker run --name "%CONTAINER_NAME%" --gpus all -p "%HOST_PORT%:%CONTAINER_PORT%" -v "%cd%\data:/tmp/share" -v "%cd%\data\logs:/var/log" --restart unless-stopped -d "%IMAGE_NAME%"

REM Check if container started successfully
timeout /t 5 /nobreak >nul
docker ps | findstr "%CONTAINER_NAME%" >nul
if %ERRORLEVEL% EQU 0 (
    echo ✅ Server started successfully!
    echo.
    echo Container name: %CONTAINER_NAME%
    echo Port: %HOST_PORT%
    echo Image: %IMAGE_NAME%
    echo.
    echo To view logs:
    echo   docker logs -f %CONTAINER_NAME%
    echo.
    echo To stop the server:
    echo   docker stop %CONTAINER_NAME%
    echo.
    echo To connect to the container:
    echo   docker exec -it %CONTAINER_NAME% bash
    echo.
    echo To test the connection:
    echo   python test-client.py
    
    REM Show initial logs
    echo.
    echo Initial logs:
    echo ========================================================
    docker logs "%CONTAINER_NAME%"
    
) else (
    echo ❌ Failed to start container!
    echo Checking logs...
    docker logs "%CONTAINER_NAME%"
    exit /b 1
)
