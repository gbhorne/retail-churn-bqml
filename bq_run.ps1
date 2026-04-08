# bq_run.ps1 - Run a SQL file against BigQuery
# Usage: .\bq_run.ps1 sql\filename.sql

param([string]$file)

$sql = Get-Content $file -Raw

# strip single-line comments
$sql = $sql -replace '(?m)--[^\r\n]*', ''

# collapse whitespace
$sql = $sql -replace '\s+', ' '
$sql = $sql.Trim()

bq query --use_legacy_sql=false --project_id=customer-churn-492703 --format=pretty $sql