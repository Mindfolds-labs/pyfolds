# Noetic TF.js Demo

Execute `create_tfjs_demo(output_dir)` para gerar um `index.html` mínimo.

Converta seu SavedModel para TF.js com:

```bash
tensorflowjs_converter --input_format=tf_saved_model /path/to/saved_model /path/to/demo/model
```

Abra o `index.html` em um servidor HTTP local.
