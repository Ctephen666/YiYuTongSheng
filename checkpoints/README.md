# Checkpoints

Put real pretrained SVS assets here when you are ready to replace the dry-run backend with real inference.

Expected default paths from `configs/project.yaml`:

```text
checkpoints/diffsinger/acoustic.ckpt
checkpoints/diffsinger/variance.ckpt
checkpoints/diffsinger/vocoder.ckpt
checkpoints/diffsinger/zh_phoneme_dict.txt
checkpoints/diffsinger/acoustic.yaml
checkpoints/diffsinger/variance.yaml
checkpoints/diffsinger/vocoder.yaml
```

The current pipeline only checks whether these files exist and writes `data/svs/checkpoint_status.json`. It does not download, train, or execute a model.
