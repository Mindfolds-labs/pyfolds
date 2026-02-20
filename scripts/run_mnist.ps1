param(
  [string]$RunId = ("mnist_" + (Get-Date -Format "yyyyMMdd_HHmmss")),
  [string]$Backend = "folds",
  [string]$Model = "py",
  [int]$Epochs = 10,
  [int]$BatchSize = 128,
  [double]$Lr = 0.001,
  [switch]$Resume,
  [switch]$Console,
  [int]$Tail = 120,
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$runDir = Join-Path "runs" $RunId
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$transcript = Join-Path $runDir "powershell.transcript.txt"
Start-Transcript -Path $transcript -Force | Out-Null

try {
  $script = if ($Backend -eq "mind") { "train_mnist_mind.py" } else { "train_mnist_folds.py" }
  $args = @($script, "--model", $Model, "--epochs", $Epochs, "--batch-size", $BatchSize, "--lr", $Lr, "--run-id", $RunId)
  if ($Resume) { $args += "--resume" }
  if ($Console) { $args += "--console" }

  Write-Host "RUN_ID=$RunId"
  Write-Host "RUN_DIR=$runDir"
  Write-Host "CMD: $Python $($args -join ' ')"

  & $Python @args
  if ($LASTEXITCODE -ne 0) {
    throw "Treino falhou com exit code $LASTEXITCODE"
  }

  Write-Host "OK: Treino finalizado. Veja: $runDir\train.log"
}
catch {
  Write-Host "ERRO: $($_.Exception.Message)" -ForegroundColor Red
  $trainLog = Join-Path $runDir "train.log"
  if (Test-Path $trainLog) {
    Write-Host "`n--- TAIL train.log (últimas $Tail linhas) ---" -ForegroundColor Yellow
    Get-Content $trainLog -Tail $Tail
  }

  $bundleDir = Join-Path $runDir "bundle"
  New-Item -ItemType Directory -Force -Path $bundleDir | Out-Null
  foreach ($p in @("train.log", "metrics.jsonl", "summary.json", "checkpoint.pt", "ADR-ERR-*.md", "powershell.transcript.txt")) {
    Get-ChildItem -Path $runDir -Filter $p -ErrorAction SilentlyContinue |
      Copy-Item -Destination $bundleDir -Force -ErrorAction SilentlyContinue
  }
  $zip = Join-Path $runDir ("crash_bundle_" + $RunId + ".zip")
  if (Test-Path $zip) { Remove-Item $zip -Force }
  Compress-Archive -Path (Join-Path $bundleDir "*") -DestinationPath $zip -Force

  Write-Host "`nBundle gerado: $zip" -ForegroundColor Cyan
  Write-Host "Sugestão: anexar esse .zip no Issue/ADR." -ForegroundColor Cyan
  exit 1
}
finally {
  Stop-Transcript | Out-Null
}
