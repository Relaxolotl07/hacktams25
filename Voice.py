import pyttsx3
from stockfish import Stockfish as sf

stockfish = sf(
    "Stock15\\stockfish\\stockfish-windows-x86-64.exe"
)


fenVal = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # input("Enter FEN: ")
)

def getBestMove():
    global textOut
    stockfish.set_fen_position(fenVal)
    textOut = str(stockfish.get_best_move())


getBestMove()


# Output
engine = pyttsx3.init()
engine.say(textOut)
engine.runAndWait()