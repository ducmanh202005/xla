$ErrorActionPreference = 'Stop'
$demoRoot = $PSScriptRoot
$projectRoot = Split-Path -Parent $demoRoot
$environmentRoot = Join-Path $projectRoot '.runtime\demo-web'
$demoPython = Join-Path $environmentRoot 'Scripts\python.exe'
try {
    if (-not (Test-Path -LiteralPath $demoPython)) {
        Write-Host 'Preparing the demo environment (first run only)...'
        & python -m venv $environmentRoot
        if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.10 or newer, then run again.' }
    }
    # Catch missing packages inside Python: PowerShell 5 treats native stderr as an error.
    $dependencyProbe = @'
import sys
try:
    from importlib.metadata import version
    ready = all(version(name) == expected for name, expected in (
        ('streamlit', '1.56.0'), ('gradio-client', '1.13.3'), ('Pillow', '11.3.0')))
except Exception:
    ready = False
sys.exit(0 if ready else 1)
'@
    & $demoPython -c $dependencyProbe
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Installing interface libraries. No GPU model is downloaded.'
        & $demoPython -m pip install --no-cache-dir --disable-pip-version-check -r (Join-Path $demoRoot 'requirements-local.txt')
        if ($LASTEXITCODE -ne 0) { throw 'Library installation failed. Check your connection and run again.' }
    }
    Write-Host 'M3Diff Studio: http://127.0.0.1:8501  (keep this window open)'
    & $demoPython -m streamlit run (Join-Path $demoRoot 'app.py') --server.address 127.0.0.1 --server.port 8501 --server.maxUploadSize 20 --browser.gatherUsageStats false --theme.base light --theme.primaryColor '#0f766e'
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    Read-Host 'Press Enter to close'
    exit 1
}
