from estimation_tools import load_model, draw, calculatePYR
from argparse import ArgumentParser
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX

ap = ArgumentParser()
ap.add_argument("-p", "--predictor", default="./shape_predictor_68_face_landmarks.dat")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["predictor"])

model_points = load_model()

vid = cv2.VideoCapture(0)
cv2.namedWindow("Head Pose Estimation", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
cv2.moveWindow("Head Pose Estimation", 0, 0)

while True:
    _, image = vid.read()

    image = imutils.resize(image, width=360)
    grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(grayed, 1)

    pyr = (0, 0, 0)

    if len(faces) > 0:
        image_points = predictor(grayed, faces[0])
        image_points = face_utils.shape_to_np(image_points)
        image_points = np.array(image_points, dtype=np.float32)

        (f, cx, cy) = (image.shape[1], image.shape[1] / 2, image.shape[0] / 2)
        camera_matrix = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype="double")

        distortions = np.zeros((4, 1))
        solution = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            distortions,
            rvec=np.array([[0], [0], [0]], dtype=np.float),
            tvec=np.array([[0], [0], [100]], dtype=np.float),
            useExtrinsicGuess=True,
        )
        r_vec = np.array([[-1, -1, 1]]).T * solution[1]
        t_vec = -1 * solution[2]

        draw(image, r_vec, t_vec, camera_matrix, distortions)
        pyr = calculatePYR(r_vec, t_vec)

    image = imutils.resize(image, width=500)
    image = cv2.flip(image, 1)
    color = (255, 0, 255)
    cv2.putText(image, "pitch: {:.2f}".format(pyr[0]), (10, 20), FONT, 0.5, color, 2)
    cv2.putText(image, "yaw:   {:.2f}".format(pyr[1]), (10, 40), FONT, 0.5, color, 2)
    cv2.putText(image, "roll:  {:.2f}".format(pyr[2]), (10, 60), FONT, 0.5, color, 2)

    cv2.imshow("Head Pose Estimation", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
