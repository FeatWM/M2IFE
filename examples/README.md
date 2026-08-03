# Examples

Only add synthetic or institution-approved, irreversibly de-identified examples
to this directory. Do not commit patient identifiers, original workbooks, or
private clinical images.

After adding an approved example, run the complete workflow with:

```bash
python -m pipeline.infer --config config.yaml --input examples/deidentified_example.bmp
```
