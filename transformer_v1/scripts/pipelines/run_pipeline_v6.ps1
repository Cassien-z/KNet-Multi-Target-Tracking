$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Push-Location $projectRoot
try {
Write-Host "Generating Val V6..."
python scripts/data_generation/script_generate_dataset_unified.py --mode val --count_per_model 500 --num_targets 3 --save_dir data/val_data_v6
Write-Host "Registering Val V6..."
python scripts/pipelines/run_register_pipeline.py --in_dir data/val_data_v6 --out_dir data/val_data_v6_registered

Write-Host "Generating Train V6..."
python scripts/data_generation/script_generate_dataset_unified.py --mode train --count_per_model 2000 --num_targets 1 --save_dir data/train_data_v6
Write-Host "Registering Train V6..."
python scripts/pipelines/run_register_pipeline.py --in_dir data/train_data_v6 --out_dir data/train_data_v6_registered

Write-Host "ALL DONE!"
} finally {
    Pop-Location
}
