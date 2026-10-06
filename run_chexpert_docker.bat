@echo off
REM ========================================================
REM Script chay CheXpert Labeler trong Docker Container
REM ========================================================

set CURRENT_DIR=%cd%
REM Chuyen duong dan sang forward slashes neu can
set WORKSPACE=%CURRENT_DIR:\=/%

echo [*] Dang khoi chay CheXpert Labeler container...
docker run --rm -v "%WORKSPACE%":/data chexpert-labeler:latest python label.py --reports_path /data/data_work/chexpert_input.csv --output_path /data/data_work/chexpert_output.csv --verbose

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Loi khi chay Docker container!
    echo [!] Kiem tra xem Docker Desktop da bat chua va image chexpert-labeler:latest da duoc build chua.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [OK] CheXpert Labeler da chay xong thanh cong! Ket qua luu tai data_work/chexpert_output.csv
