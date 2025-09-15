import os, glob, random, cv2, numpy as np
import tensorflow as tf
from .config import CFG
from .utils import uniform_indices

try:
    import albumentations as A
    _HAS_ALB = True
except Exception:
    _HAS_ALB = False

def read_video_uniform(path: str, T: int, H: int, W: int, do_aug: bool=False, aug_prob: float=0.5):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return np.zeros((T, H, W, 3), dtype=np.float32)
    total = int(max(1, cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    indices = uniform_indices(total, T)

    aug = None
    if do_aug and _HAS_ALB:
        aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.GaussianBlur(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        ])

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
        if aug is not None and random.random() < aug_prob:
            frame = aug(image=frame)["image"]
        frames.append(frame)
    cap.release()
    return (np.stack(frames).astype(np.float32) / 255.0)


def discover_videos(root: str, classes):
    pairs = []
    for label, cname in enumerate(classes):
        vids = glob.glob(os.path.join(root, cname, "**", "*.mp4"), recursive=True)
        for v in vids:
            pairs.append((v, label))
    random.shuffle(pairs)
    return pairs


def build_dataset(samples, cfg=CFG, training: bool=False):
    paths = [p for p,_ in samples]
    labels = [l for _,l in samples]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        p = path.numpy().decode("utf-8")
        arr = read_video_uniform(p, cfg.num_frames, cfg.target_h, cfg.target_w,
                                 do_aug=(cfg.use_augment and training), aug_prob=cfg.aug_prob)
        return arr.astype(np.float32), np.int32(label)

    def _tf_load(path, label):
        frames, lab = tf.py_function(_load, [path, label], [tf.float32, tf.int32])
        frames.set_shape((cfg.num_frames, cfg.target_h, cfg.target_w, 3))
        lab.set_shape(())
        return frames, lab

    if training:
        ds = ds.shuffle(min(1024, len(samples)))
    ds = ds.map(_tf_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
