@echo off
echo ========================================
echo Correctif TBB System32
echo ========================================
echo.

REM Vérifie les droits administrateur
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERREUR: Ce script necessite les droits administrateur.
    echo Faites un clic droit et selectionnez "Executer en tant qu'administrateur"
    pause
    exit /b 1
)

echo Que voulez-vous faire?
echo [1] Desactiver tbb.dll de System32 (pour utiliser Prophet)
echo [2] Reactiver tbb.dll de System32 (restaurer)
echo.
choice /C 12 /M "Votre choix "

if errorlevel 2 goto restore
if errorlevel 1 goto disable

:disable
echo.
echo Desactivation de C:\Windows\System32\tbb.dll...
if exist C:\Windows\System32\tbb.dll (
    ren C:\Windows\System32\tbb.dll tbb_BACKUP.dll
    echo Succes: tbb.dll renommee en tbb_BACKUP.dll
    echo.
    echo Vous pouvez maintenant utiliser Prophet.
    echo Pour restaurer, relancez ce script et choisissez l'option 2.
) else (
    echo tbb.dll n'existe pas dans System32 ou a deja ete renommee.
)
goto end

:restore
echo.
echo Restauration de C:\Windows\System32\tbb.dll...
if exist C:\Windows\System32\tbb_BACKUP.dll (
    ren C:\Windows\System32\tbb_BACKUP.dll tbb.dll
    echo Succes: tbb_BACKUP.dll restauree en tbb.dll
) else (
    echo tbb_BACKUP.dll n'existe pas. Rien a restaurer.
)
goto end

:end
echo.
echo ========================================
pause
