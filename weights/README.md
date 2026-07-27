# Model weights

The Git repository excludes binary weights. Arrange downloaded checkpoints as:

```text
weights/
├── detector/
│   └── best.pt
└── classifier/
    ├── vgg16/fold0.ckpt ... fold4.ckpt
    ├── resnet18/fold0.ckpt ... fold4.ckpt
    └── convnext_large/fold0.ckpt ... fold4.ckpt
```

The public `config.yaml` points to this relative layout. To use checkpoints
stored elsewhere, create an ignored `config.local.yaml` and pass it explicitly.

Before a public release, publish SHA-256 checksums and a stable download location
for every checkpoint.

The historical Lightning checkpoints contain `addict.Dict` metadata, so
`addict` remains an installation dependency. For a public weight release,
convert each trusted historical checkpoint once to a plain tensor-only
`state_dict` file.
