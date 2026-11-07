import os
import numpy as np
import tensorflow as tf


class VideoSequenceGenerator(tf.keras.utils.Sequence):

    def __init__(
        self,
        real_dir,
        fake_dir,
        batch_size=1,
        shuffle=True
    ):

        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.file_paths = []
        self.labels = []

        # ============================================
        # LOAD REAL DATA
        # ============================================

        if os.path.exists(real_dir):

            for filename in os.listdir(real_dir):

                if filename.lower().endswith(".npy"):

                    path = os.path.join(
                        real_dir,
                        filename
                    )

                    self.file_paths.append(path)
                    self.labels.append(0)

        # ============================================
        # LOAD FAKE DATA
        # ============================================

        if os.path.exists(fake_dir):

            for filename in os.listdir(fake_dir):

                if filename.lower().endswith(".npy"):

                    path = os.path.join(
                        fake_dir,
                        filename
                    )

                    self.file_paths.append(path)
                    self.labels.append(1)

        self.file_paths = np.array(
            self.file_paths
        )

        self.labels = np.array(
            self.labels,
            dtype=np.float32
        )

        self.indices = np.arange(
            len(self.file_paths)
        )

        self.on_epoch_end()

        print(
            f"Generator loaded: "
            f"{len(self.file_paths)} samples"
        )

        print(
            f"Real: {np.sum(self.labels == 0)}"
        )

        print(
            f"Fake: {np.sum(self.labels == 1)}"
        )


    def __len__(self):

        return int(
            np.ceil(
                len(self.file_paths)
                / self.batch_size
            )
        )


    def __getitem__(self, index):

        start = (
            index *
            self.batch_size
        )

        end = min(
            start +
            self.batch_size,
            len(self.file_paths)
        )

        batch_indices = self.indices[
            start:end
        ]

        batch_paths = self.file_paths[
            batch_indices
        ]

        batch_labels = self.labels[
            batch_indices
        ]

        batch_x = []

        for path in batch_paths:

            try:

                sequence = np.load(
                    path
                ).astype(
                    np.float32
                )

                # ====================================
                # EXPECTED SHAPE:
                #
                # (20, 128, 128, 3)
                # ====================================

                if sequence.ndim != 4:

                    raise ValueError(
                        f"Invalid shape {sequence.shape}"
                    )

                # Normalize if stored as uint8
                if sequence.max() > 1.0:

                    sequence = (
                        sequence / 255.0
                    )

                batch_x.append(
                    sequence
                )

            except Exception as e:

                print(
                    f"Error loading {path}: {e}"
                )

                # Fallback
                batch_x.append(
                    np.zeros(
                        (20, 128, 128, 3),
                        dtype=np.float32
                    )
                )

        batch_x = np.asarray(
            batch_x,
            dtype=np.float32
        )

        batch_labels = np.asarray(
            batch_labels,
            dtype=np.float32
        )

        return batch_x, batch_labels


    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )