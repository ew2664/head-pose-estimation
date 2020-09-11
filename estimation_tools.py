def load_model():
    import numpy as np

    raw_value = []
    with open("./model.txt") as file:
        for line in file:
            raw_value.append(line)
    model_points = np.array(raw_value, dtype=np.float32)
    model_points = np.reshape(model_points, (3, -1)).T
    model_points[:, 2] *= -1
    return model_points


def show_model(model_points):
    from matplotlib import pyplot
    from mpl_toolkits.mplot3d import Axes3D

    fig = pyplot.figure()
    ax = Axes3D(fig)
    x = model_points[:, 0]
    y = model_points[:, 1]
    z = model_points[:, 2]
    ax.scatter(x, y, z)
    pyplot.xlabel("x")
    pyplot.ylabel("y")
    pyplot.show()


def draw(image, r_vec, t_vec, cam_mat, dist):
    import cv2
    import numpy as np

    color = (0, 255, 0)
    # r_vec = np.array([[-1, -1, 1]]).T * r_vec
    # t_vec = -1 * t_vec

    size = 50
    depth = 50

    points = np.array(
        [
            (-size, -size, depth),
            (-size, size, depth),
            (size, size, depth),
            (size, -size, depth),
            (-size, -size, 0),
            (-size, size, 0),
            (size, size, 0),
            (size, -size, 0),
        ],
        dtype=np.float,
    )

    (points, _) = cv2.projectPoints(points, r_vec, t_vec, cam_mat, dist)
    points = np.int32(points.reshape(-1, 2))

    cv2.polylines(image, [points[:4]], True, color, 1, cv2.LINE_AA)
    cv2.line(image, tuple(points[0]), tuple(points[2]), color, 1, cv2.LINE_AA)
    cv2.line(image, tuple(points[1]), tuple(points[3]), color, 1, cv2.LINE_AA)
    cv2.line(image, tuple(points[0]), tuple(points[4]), color, 1, cv2.LINE_AA)
    cv2.line(image, tuple(points[1]), tuple(points[5]), color, 1, cv2.LINE_AA)
    cv2.line(image, tuple(points[2]), tuple(points[6]), color, 1, cv2.LINE_AA)
    cv2.line(image, tuple(points[3]), tuple(points[7]), color, 1, cv2.LINE_AA)


def calculatePYR(r_vec, t_vec):
    import cv2
    import numpy as np

    r_mat = cv2.Rodrigues(np.array([[-1, 1, -1]]).T * r_vec)[0]
    p_mat = np.hstack((r_mat, t_vec))
    (pitch, yaw, roll) = cv2.decomposeProjectionMatrix(p_mat)[6][:, 0]
    return (pitch, yaw, roll)
