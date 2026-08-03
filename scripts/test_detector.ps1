param(
    [string]$Config = "config.yaml",
    [string]$Device = "cuda:0",
    [ValidateSet("val", "test")]
    [string]$Split = "test"
)

python -m detector.test --config $Config --device $Device --split $Split
