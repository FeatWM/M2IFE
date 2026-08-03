param(
    [string]$Config = "config.yaml",
    [string]$Device = "cuda:0",
    [ValidateSet("all", "vgg16", "resnet18", "convnext_large")]
    [string]$Backbone = "all"
)

python -m classifier.test --config $Config --device $Device --backbone $Backbone
