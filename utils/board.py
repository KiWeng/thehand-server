from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

from tests.fake_board import FakeBoard


def load_board(fake, port="Port_#0002.Hub_#0002"):
    board = None
    if fake:
        board = FakeBoard
    else:
        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()
        params.serial_port = port  # 每次根据实际情况更改串口号
        board_shim = BoardShim(BoardIds.ANT_NEURO_EE_225_BOARD, params)
        board = board_shim
        board_shim.prepare_session()
        board_shim.start_stream(450000)
    return board
