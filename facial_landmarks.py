from imutils import face_utils
from imutils.video import VideoStream
import argparse
import imutils
import dlib
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (0, 255, 0)

ap = argparse.ArgumentParser()
ap.add_argument(
    "-p", "--predictor", required=True, help="path to facial landmark predictor"
)
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["predictor"])

vs = VideoStream(src=0).start()

while True:
    image = vs.read()
    image = imutils.resize(image, width=500)
    grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(grayed, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(grayed, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image, "Face #{}".format(i + 1), (x - 10, y - 10), FONT, 0.5, COLOR, 2
        )
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    cv2.imshow("Output", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

vs.stream.release()
cv2.destroyAllWindows()
