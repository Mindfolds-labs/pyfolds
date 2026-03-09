from pyfolds.telemetry.events import make_loss_event
from pyfolds.telemetry.exporters import TensorBoardExporter
from pyfolds.utils.compat import has_tensorboard


def test_tensorboard_integration(tmp_path):
    if not has_tensorboard():
        return
    e = TensorBoardExporter(str(tmp_path))
    e.export([make_loss_event("train", 1, 0.1)])
    e.flush(); e.close()
    assert list(tmp_path.glob("events.out.tfevents.*"))
