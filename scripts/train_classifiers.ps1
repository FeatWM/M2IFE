param(
    [string]$Config = "config.yaml",
    [string]$Device = "cuda:0"
)

python -m classifier.train --config $Config --device $Device --backbone all --fold all
