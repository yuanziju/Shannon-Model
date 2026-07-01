<#
.SYNOPSIS
    Clean up project structure - move all docs to root, keep only core files in spec dir

.DESCRIPTION
    This script:
    1. Copies all .md files from spec directory to project root
    2. Deletes files from spec directory except spec.md, task.md, checklist.md
    3. Renames files to proper case (README.md, AGENTS.md)

.NOTES
    Author: AI Assistant
    Date: 2026-07-01
    Version: 2.0
#>

$rootDir = "e:\Project\Python\Pytorch\Shannon"
$specDir = "e:\Project\Python\Pytorch\Shannon\.trae\specs\shannon-ai-model"

$keepInSpec = @("spec.md", "task.md", "checklist.md")

Write-Host "`n========================================="
Write-Host "Step 1: Copy all .md files to root"
Write-Host "========================================="

$mdFiles = Get-ChildItem -Path $specDir -Filter "*.md" -File

foreach ($file in $mdFiles) {
    $destPath = Join-Path -Path $rootDir -ChildPath $file.Name
    Copy-Item -Path $file.FullName -Destination $destPath -Force
    Write-Host "[OK] Copied: $($file.Name)"
}

Write-Host "`n========================================="
Write-Host "Step 2: Delete non-core files from spec dir"
Write-Host "========================================="

foreach ($file in $mdFiles) {
    if ($keepInSpec -notcontains $file.Name) {
        Remove-Item -Path $file.FullName -Force
        Write-Host "[OK] Deleted from spec: $($file.Name)"
    }
}

Write-Host "`n========================================="
Write-Host "Step 3: Ensure proper file naming"
Write-Host "========================================="

$renameMap = @(
    @{source="$rootDir\readme.md"; dest="$rootDir\README.md"},
    @{source="$rootDir\agents.md"; dest="$rootDir\AGENTS.md"}
)

foreach ($item in $renameMap) {
    if (Test-Path -Path $item.source) {
        Rename-Item -Path $item.source -NewName (Split-Path -Leaf $item.dest) -Force
        Write-Host "[OK] Renamed: $(Split-Path -Leaf $item.source) -> $(Split-Path -Leaf $item.dest)"
    }
}

Write-Host "`n========================================="
Write-Host "ALL OPERATIONS COMPLETED!"
Write-Host "========================================="