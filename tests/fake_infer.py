import time
import numpy as np

from models import load_model
from fake_board import FakeBoard

if __name__ == "__main__":
    board = FakeBoard()
    model = load_model()
    while True:
        if board.get_board_data_count() < 8 * 800:
            continue
        fake_data = board.get_current_board_data(8 * 800)[:12, ::8]
        fake_data = fake_data[:, -800:]
        fake_data = np.expand_dims(fake_data.T, axis=0)

        print(fake_data.shape)
        print(model.predict(fake_data))
