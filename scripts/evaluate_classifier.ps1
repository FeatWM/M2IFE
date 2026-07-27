param(
    [string]$Config = "config.yaml",
    [string]$Device = "cuda:0",
    [string]$Output = "outputs/evaluation"
)

python -m classifier.evaluate --config $Config --device $Device --output $Output
