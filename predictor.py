import json
import uuid

import numpy as np
from flask import Flask, jsonify, request
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from ultralytics import YOLO
from utils import download_image_from_url


class CustomEncoder(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


app = Flask(__name__)
app.json = CustomEncoder(app)

cors = CORS(app)


model = YOLO("weights/yolov8x-pose.pt")


body_connections = np.array(
    [
        {"box": [[1, 2], [2, 3], [3, 5], [5, 7]], "class": 0},
        {"box": [[1, 3]], "class": 0},
        {"box": [[2, 4], [4, 6]], "class": 0},
        {"box": [[6, 7], [7, 9]], "class": 1},
        {"box": [[6, 8], [8, 10]], "class": 1},
        {"box": [[9, 11]], "class": 1},
        {"box": [[16, 14], [14, 12]], "class": 2},
        {"box": [[17, 15], [15, 13]], "class": 2},
        {"box": [[6, 12], [12, 13]], "class": 3},
        {"box": [[7, 13]], "class": 3},
    ]
)


def postprocess_pose_keypoints(keypointss):
    result = []
    for keypoints in keypointss:
        for connections in body_connections:
            boxid = connections["class"]
            group_line = []
            for connection in connections["box"]:
                keypoint_a = keypoints[connection[0] - 1]
                keypoint_b = keypoints[connection[1] - 1]
                x_start, y_start, visibility_start = (
                    keypoint_a[0],
                    keypoint_a[1],
                    keypoint_a[2],
                )
                x_end, y_end, visibility_end = (
                    keypoint_b[0],
                    keypoint_b[1],
                    keypoint_b[2],
                )
                if visibility_start > 0.5 and visibility_end > 0.5:
                    group_line.append([round(x_start), round(y_start)])
                    group_line.append([round(x_end), round(y_end)])
            if len(group_line):
                result.append(
                    {
                        "box": group_line,
                        "pk": str(uuid.uuid4()),
                        "class": boxid,
                        "boxtype": "lines",
                    }
                )
    return result


def init_image_from_url(_url, coordinates=None):
    org_img, _ = download_image_from_url(_url, coordinates)
    return org_img


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@app.route("/api3/regions/pose-detection", methods=["GET", "POST"])
def run_regions_pose():
    content = request.json
    image_url = content["image_url"]

    org_img = init_image_from_url(image_url, content.get("coordinates", None))

    results = model(org_img)
    keypoints = results[0].keypoints
    keypoints = keypoints.detach().cpu().numpy()
    result = postprocess_pose_keypoints(keypoints)

    result = {
        "bboxes": result,
        "image": {
            "rotation": 0,
            "image_width": org_img.shape[1],
            "image_height": org_img.shape[0],
        },
        "meta": {"rotation": 0, "width": org_img.shape[1], "height": org_img.shape[0]},
    }
    return jsonify(result)
