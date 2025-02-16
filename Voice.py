from stockfish import Stockfish as sf
import pyttsx3
from IMG_to_FEN import IMG_to_FEN
import cv2

stockfish = sf("Stock15\\stockfish\\stockfish-windows-x86-64.exe")

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
frame = cv2.flip(frame, 1)
cv2.imshow("frame", frame)
cv2.waitKey(25)
cv2.imwrite(os.path.join(DATA_DIR, str(j), "{}.jpg".format(counter)), frame)


cap.release()
cv2.destroyAllWindows()

img = IMG_to_FEN()
fenVal = img.fenCodeFromImage(frame)

if fenVal == "":
    fenval = "4k3/2R5/7Q/8/8/8/8/4K3 w - - 0 1"


def getBestMove():
    global textOut
    stockfish.set_fen_position(fenVal)
    textOut = str(stockfish.get_best_move())


def convertMove(move: str) -> str:
    moveText = ""

    isCapture = stockfish.will_move_be_a_capture(move) != sf.Capture.NO_CAPTURE

    currPiece = str(stockfish.get_what_is_on_square(move[0:2]))
    currPiece = currPiece[12:]

    if isCapture:
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
engine.setProperty("rate", 135)
engine.say(convertedText)
engine.runAndWait()
