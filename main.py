from stockfish import Stockfish as sf
from torch import torch

model = torch.load("2023-11-02-fenify-3d-efficientnet-v2-s-95-val-acc.pt")

stockfish = sf("Stock15\\stockfish\\stockfish-windows-x86-64.exe")

def getBestMove():
    stockfish.set_fen_position(fenVal)
    return stockfish.get_best_move()

# get fen as an input
def getTurnFromFen(fen: str) -> str:
    fields = fen.split()
    # The second field represents whose turn it is
    turn = fields[1]
    
    return turn

pressed = False

yourColor = "w"

fenVal = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" # input("Enter FEN: ")

if(getTurnFromFen(fenVal) == yourColor or pressed):
    text = getBestMove()
    print(text)

