from stockfish import Stockfish as sf
import pyttsx3

stockfish = sf(
    "Stock15\\stockfish\\stockfish-windows-x86-64.exe"
)


fenVal = (
    "4k3/2R5/7Q/8/8/8/8/4K3 w - - 0 1"  # input("Enter FEN: ")
)

def getBestMove():
    global textOut
    stockfish.set_fen_position(fenVal)
    textOut = str(stockfish.get_best_move())


def convertMove(move: str) -> str:  
    moveText = ""
    
    isCapture = stockfish.will_move_be_a_capture(move) != sf.Capture.NO_CAPTURE

    currPiece = str(stockfish.get_what_is_on_square(move[0:2]))
    currPiece = currPiece[12:]

    
    
    if (isCapture):
        moveText = "Capture using "
    else:
        moveText = "Move using "

    moveText += currPiece + " from " + move[0:2] + " to " + move[2:4]

    return moveText


getBestMove()
print(textOut)

stockfish.get_board_visual()

convertedText = convertMove(textOut)

print(convertedText)

engine = pyttsx3.init()
engine.say(convertedText)
engine.runAndWait()