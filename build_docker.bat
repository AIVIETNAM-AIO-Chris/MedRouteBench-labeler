@echo off
REM ========================================================
REM Script build Docker image cho CheXpert Labeler
REM ========================================================

echo [*] Bat dau build Docker image chexpert-labeler:latest ...
cd chexpert-labeler
docker build -t chexpert-labeler:latest .
cd ..

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Build Docker image that bai. Hay dam bao Docker Desktop dang chay!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [OK] Build Docker image thanh cong! San sang chay labeler.
