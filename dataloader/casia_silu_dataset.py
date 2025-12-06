import torch
import os
import random
import numpy as np
import cv2


class CASIASiluDataset:
    def __init__(self, silu_root, target_len=8):
        self.silu_root = silu_root
        self.target_len = target_len
        print(f"[CASIA-SILU] Loading from: {silu_root}")

        self.all_sequences = self._scan_silu()
        self.all_subject_ids = sorted(list(self.all_sequences.keys()))
        print(f"[CASIA-SILU] Found {len(self.all_subject_ids)} subjects.")

    def _scan_silu(self):
        data = {}
        if not os.path.exists(self.silu_root):
            print("❌ silu root 不存在！")
            return data

        for sid in sorted(os.listdir(self.silu_root)):
            sid_dir = os.path.join(self.silu_root, sid)
            if not os.path.isdir(sid_dir):
                continue
            data[sid] = {}

            for cond in sorted(os.listdir(sid_dir)):
                cond_dir = os.path.join(sid_dir, cond)
                if not os.path.isdir(cond_dir):
                    continue
                data[sid][cond] = {}

                for view in sorted(os.listdir(cond_dir)):
                    view_dir = os.path.join(cond_dir, view)
                    if not os.path.isdir(view_dir):
                        continue

                    pngs = [os.path.join(view_dir, p)
                            for p in sorted(os.listdir(view_dir))
                            if p.endswith(".png")]

                    if len(pngs) > 0:
                        data[sid][cond][view] = pngs

        return data

    def load_sequence(self, frame_paths):
        imgs = []
        for p in frame_paths[:self.target_len]:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((224, 224), dtype=np.uint8)
            else:
                img = cv2.resize(img, (224, 224))

            img = np.stack([img, img, img], axis=0)  # (3,224,224)
            imgs.append(torch.tensor(img).float() / 255.)

        while len(imgs) < self.target_len:
            imgs.append(torch.zeros(3, 224, 224))

        return torch.stack(imgs)   # (T, 3,224,224)


class FewShotSampler:
    def __init__(self, dataset, n_way, k_shot, q_query):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        ids = dataset.all_subject_ids
        split = int(0.8 * len(ids))
        self.train_ids = ids[:split]
        self.test_ids = ids[split:]

    def get_episode(self, mode="train"):
        pool = self.train_ids if mode == "train" else self.test_ids
        sampled_ids = random.sample(pool, self.n_way)

        X = []
        Y = []

        for cls, sid in enumerate(sampled_ids):
            all_seq = []
            for cond in self.dataset.all_sequences[sid]:
                for view in self.dataset.all_sequences[sid][cond]:
                    all_seq.append(self.dataset.all_sequences[sid][cond][view])

            needed = self.k_shot + self.q_query
            if len(all_seq) < needed:
                all_seq = all_seq * (needed // len(all_seq) + 1)

            selected = random.sample(all_seq, needed)
            for seq in selected:
                imgs = self.dataset.load_sequence(seq)   # (T,3,224,224)
                X.append(imgs)
                Y.append(cls)

        X = torch.stack(X)  # (B, T, 3, 224, 224)
        Y = torch.tensor(Y)

        phase_w = torch.ones(X.shape[0], X.shape[1])

        return X, Y, phase_w
