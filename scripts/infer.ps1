param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,
    [string]$Config = "config.yaml",
    [string]$Device = "cuda:0",
    [string]$Output = "outputs/inference"
)

python infer.py --config $Config --device $Device --input $InputPath --output $Output
