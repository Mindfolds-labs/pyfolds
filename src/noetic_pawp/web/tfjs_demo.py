"""Minimal TF.js demo scaffold generator for Noetic."""

from __future__ import annotations

from pathlib import Path

HTML_TEMPLATE = """<!doctype html>
<html>
  <head><meta charset=\"utf-8\" /><title>Noetic TF.js Demo</title></head>
  <body>
    <h1>Noetic TF.js Demo</h1>
    <p>Converta o SavedModel com tensorflowjs_converter e ajuste modelUrl abaixo.</p>
    <script src=\"https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js\"></script>
    <script>
      async function run() {
        const modelUrl = './model/model.json';
        const model = await tf.loadGraphModel(modelUrl);
        const x = tf.zeros([1, 224, 224, 3]);
        const y = model.predict ? model.predict(x) : model.execute(x);
        console.log('Inference output:', y);
      }
      run();
    </script>
  </body>
</html>
"""


def create_tfjs_demo(output_dir: str) -> Path:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    html_path = out / "index.html"
    html_path.write_text(HTML_TEMPLATE, encoding="utf-8")
    return html_path
