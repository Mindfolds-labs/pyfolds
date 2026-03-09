# Noetic + TensorFlow

Fluxo recomendado:

PyFolds (PyTorch) -> ONNX -> TensorFlow SavedModel -> Serving / TF.js / TFLite.

Pipelines de dados ficam em `noetic_pawp.data.tf_pipeline` com builders para NYUv2/KITTI.
