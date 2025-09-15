import json
from tensorflow import keras
from .config import CFG
from .data import discover_videos, build_dataset
from .model import build_model
from .metrics import F1Score
from .callbacks import EvalAndSaveCallback


def prepare_datasets(cfg=CFG):
    tr = discover_videos(cfg.train_dir, cfg.classes)
    va = discover_videos(cfg.val_dir, cfg.classes)
    te = discover_videos(cfg.test_dir, cfg.classes)
    return (build_dataset(tr, cfg, True), build_dataset(va, cfg, False), build_dataset(te, cfg, False))


def train(cfg=CFG):
    ds_tr, ds_va, ds_te = prepare_datasets(cfg)
    model = build_model(cfg)
    opt = keras.optimizers.Adam(cfg.lr)
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.BinaryAccuracy(name='acc'),
            keras.metrics.Precision(name='prec'),
            keras.metrics.Recall(name='rec'),
            keras.metrics.AUC(name='auc'),
            F1Score(name='f1')
        ],
    )
    cbs = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=5, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True, verbose=1),
        EvalAndSaveCallback(ds_va, ds_te, cfg),
    ]
    history = model.fit(ds_tr, validation_data=ds_va, epochs=cfg.epochs, callbacks=cbs)
    model.save(cfg.ckpt_path)
    with open(cfg.history_path,'w') as f:
        json.dump({k:list(map(float,v)) for k,v in history.history.items()}, f, indent=2)
    return model
