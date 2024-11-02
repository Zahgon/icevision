import pytest
import torch
import lightning.pytorch as L

from icevision.metrics.coco_metric.coco_metric import COCOMetric


# WARNING: Only works with cuda: https://github.com/rwightman/efficientdet-pytorch/issues/44#issuecomment-662594014
@pytest.mark.cuda
@pytest.mark.parametrize("metrics", [[], [COCOMetric()]])
def test_lightining_efficientdet_test(
    fridge_efficientdet_dls, fridge_efficientdet_model, light_model_cls, metrics
):
    _, valid_dl = fridge_efficientdet_dls
    light_model = light_model_cls(fridge_efficientdet_model, metrics=metrics)
    trainer = L.Trainer(
        max_epochs=1,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.test(light_model, valid_dl)


@pytest.mark.parametrize("metrics", [[], [COCOMetric()]])
def test_lightining_efficientdet_finalizes_metrics_on_test_epoch_end(
    fridge_efficientdet_model, light_model_cls, metrics
):
    with torch.set_grad_enabled(False):
        light_model = light_model_cls(fridge_efficientdet_model, metrics=metrics)

        light_model.on_test_epoch_end()

        assert light_model.was_finalize_metrics_called == True


def test_lightining_efficientdet_logs_losses_during_test_step(
    fridge_efficientdet_dls, fridge_efficientdet_model, light_model_cls
):
    with torch.set_grad_enabled(False):
        train_dl, _ = fridge_efficientdet_dls
        light_model = light_model_cls(model=fridge_efficientdet_model, metrics=None)
        for batch in train_dl:
            break
        light_model.convert_raw_predictions = lambda *args: None
        light_model.compute_loss = lambda *args: None
        light_model.accumulate_metrics = lambda *args: None

        light_model.test_step(batch, 0)

        assert sorted(light_model.logs.keys()) == sorted(
            ["test_loss", "test_box_loss", "test_class_loss"]
        )
