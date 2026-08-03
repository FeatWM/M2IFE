param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,
    [string]$Config = "config.yaml",
    [string]$Device = "cuda:0",
    [string]$Output = "outputs/pipeline"
)

python -m pipeline.infer --config $Config --device $Device --input $InputPath --output $Output
