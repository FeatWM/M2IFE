param(
    [string]$Config = "config.yaml",
    [string]$Device = "cuda:0"
)

python -m detector.train --config $Config --device $Device
