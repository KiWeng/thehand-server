import numpy as np

from fake_board import FakeBoard
from model import EMGModel

if __name__ == "__main__":
    board = FakeBoard()
    model = EMGModel()
    while True:
        if board.get_board_data_count() < 8 * 800:
            continue
        fake_data = board.get_current_board_data(8 * 800)[:12, ::8]
        fake_data = fake_data[:, -800:]
        fake_data = np.expand_dims(fake_data.T, axis=0)

        print(fake_data.shape)
        print(model.predict(fake_data))
