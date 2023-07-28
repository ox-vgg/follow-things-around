## Copyright 2023 David Miguel Susano Pinto <pinto@robots.ox.ac.uk>
##
## Licensed under the Apache License, Version 2.0 (the "License"); you
## may not use this file except in compliance with the License.  You
## may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
## implied.  See the License for the specific language governing
## permissions and limitations under the License.

import logging
import os
import os.path
import shutil
import subprocess
from typing import Dict, List, NamedTuple, Tuple

import cv2
import detectron2.config
import detectron2.engine
import detectron2.model_zoo
import IPython.display
import ipywidgets
import matplotlib.cm
import numpy as np
import pandas as pd
import PIL.Image
import plotly.express
import torch.utils.data
from svt.siamrpn_tracker import siamrpn_tracker

from follow_things_around import svt_patch


_logger = logging.getLogger(__name__)


USE_GPU: bool = True

DEFAULT_DEVICE: str = "cuda"

UNKNOWN_TRACK_ID_MARKER: int = -1

FFMPEG_LOG_LEVEL: str = "info"

MATCH_OVERLAP_THRESHOLD: float = 0.2

NONMATCH_TRACKING_THRESHOLD: float = 0.9

VIDEO_FILE: str = ""


def subprocess_print_stderr(args: List[str]) -> None:
    p = subprocess.Popen(args, text=True, stderr=subprocess.PIPE)

    for line in iter(p.stderr.readline, ""):
        print(line, end="")
    p.wait()
    if p.returncode != 0:
        raise Exception("subprocess failed")


def ffmpeg_video_to_frames(video_fpath: str, frames_dir: str) -> None:
    subprocess_print_stderr(
        [
            "ffmpeg",
            "-i",
            video_fpath,
            "-vsync",
            "vfr",
            "-q:v",
            "1",
            "-start_number",
            "0",
            "-filter:v",
            "scale=iw:ih*(1/sar)",
            "-loglevel",
            FFMPEG_LOG_LEVEL,
            # FIXME: what if %06d.jpg is not enough and rools over?
            os.path.join(FRAMES_DIR, "%06d.jpg"),
        ]
    )


def ffprobe_get_frame_rate(video_fpath: str) -> float:
    ffprobe_p = subprocess.run(
        [
            "ffprobe",
            "-loglevel",
            "panic",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=r_frame_rate",
            "-print_format",
            "csv",
            video_fpath,
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    # we expect something like "stream,25/1\n"
    frame_rate_parts = ffprobe_p.stdout.split(",")[1][:-1].split("/")
    if len(frame_rate_parts) == 1:
        return float(frame_rate_parts[0])
    else:  # format is something such as 25/1
        return float(frame_rate_parts[0]) / float(frame_rate_parts[1])


def ffmpeg_video_from_frames_and_video(
    frames_dir: str, in_video_fpath: str, out_video_fpath: str
) -> None:
    frame_rate = ffprobe_get_frame_rate(in_video_fpath)
    subprocess_print_stderr(
        [
            "ffmpeg",
            "-y",  # overwrite output files without asking
            "-loglevel",
            FFMPEG_LOG_LEVEL,
            "-framerate",
            str(frame_rate),
            "-pattern_type",
            "glob",
            "-i",
            os.path.join(frames_dir, "*.jpg"),
            "-i",
            in_video_fpath,
            "-c:a",
            "aac",  # https://github.com/ox-vgg/chimpanzee-tracking/issues/1
            "-c:v",
            "libx264",
            "-map",
            "0:v:0",  # use video from input 0 / stream 0
            "-map",
            "1:a:0",  # use audio from input 1 / stream 0
            "-pix_fmt",
            "yuv420p",
            out_video_fpath,
        ]
    )


class FramesDirDataset(torch.utils.data.Dataset):
    def __init__(self, frames_dir: str) -> None:
        super().__init__()
        self._frames_dir = frames_dir
        self._frames = sorted(os.listdir(self._frames_dir))

    @property
    def frames_dir(self) -> str:
        return self._frames_dir

    @property
    def frames(self) -> List[str]:
        return self._frames

    def __getitem__(self, idx: int) -> PIL.Image.Image:
        fname = self._frames[idx]
        img = np.array(PIL.Image.open(os.path.join(self._frames_dir, fname)))
        return img

    def __len__(self) -> int:
        return len(self._frames)


class Detection(NamedTuple):
    x: float
    y: float
    w: float
    h: float


FrameDetections = List[Detection]

DatasetDetections = List[FrameDetections]


def detect(
    dataset: FramesDirDataset,
    detection_model_config_path: str,
    class_idx: int,
    visual_threshold: float,
) -> DatasetDetections:
    if visual_threshold < 0.0 or visual_threshold > 1.0:
        raise ValueError(
            "visual_threshold needs to be a number between 0.0 and 1.0"
        )

    d2_cfg = detectron2.config.get_cfg()
    d2_cfg.merge_from_file(detection_model_config_path)
    d2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = visual_threshold
    d2_cfg.MODEL.DEVICE = DEFAULT_DEVICE

    predictor = detectron2.engine.DefaultPredictor(d2_cfg)
    _logger.info("Finished loading model %s", detection_model_config_path)

    _logger.info("Starting detection phase")

    dataset_detections: DatasetDetections = []

    for frame_index, img in enumerate(dataset):
        if frame_index % 1000 == 0:
            _logger.info(
                "Starting to process images %d to %d",
                frame_index,
                min(frame_index + 1000 - 1, len(dataset) - 1),
            )

        img_bgr = img[
            :, :, ::-1
        ]  # predictor wants image in BGR instead of RGB
        outputs = predictor(img_bgr)
        bboxes = outputs["instances"].pred_boxes[
            outputs["instances"].pred_classes == class_idx
        ]

        frame_detections: FrameDetections = []
        for i, bbox in enumerate(bboxes):
            frame_detections.append(
                Detection(
                    x=float(bbox[0]),
                    y=float(bbox[1]),
                    w=float(bbox[2] - bbox[0]),
                    h=float(bbox[3] - bbox[1]),
                )
            )
        dataset_detections.append(frame_detections)

    _logger.info("Finished detections")
    return dataset_detections


def track(
    dataset: FramesDirDataset,
    detections: DatasetDetections,
    tracking_model_path: str,
):
    _logger.info("Starting tracking phase")

    tracker_config = {
        "gpu_id": 0 if USE_GPU else -1,
        "preload_model": True,
    }

    detections_match_config = {
        "nonmatch_tracking_threshold": NONMATCH_TRACKING_THRESHOLD,
        "match_overlap_threshold": MATCH_OVERLAP_THRESHOLD,
        "UNKNOWN_TRACK_ID_MARKER": UNKNOWN_TRACK_ID_MARKER,
        "frame_img_dir": dataset.frames_dir,
        "via_project_name": VIDEO_FILE,
    }

    # SVT uses dicts instead of lists for detections and uses strings
    # even though it expects the keys to be integers (possibly to make
    # it easy to export to JSON).  So make that conversion.
    # We expect videos to be of a single shot, hence only shot_id 0.
    shot_id = "0"
    detections4svt = {shot_id: {}}
    for frame_idx, frame_detections in enumerate(detections):
        frame_key = str(frame_idx)
        detections4svt[shot_id][frame_key] = {}
        for box_idx, detection in enumerate(frame_detections):
            box_key = str(box_idx)
            # SVT uses a [track_id, x, y, w, h] for each detection
            # (list not tuple because it changes `tracker_id`.
            detections4svt[shot_id][frame_key][box_key] = [
                UNKNOWN_TRACK_ID_MARKER,
                *detection,
            ]

    # SVT also needs a separate map of frame id to relative filepaths.
    frame_id_to_fpath = {str(i): f for i, f in enumerate(dataset.frames)}

    tracker = siamrpn_tracker(
        model_path=tracking_model_path, config=tracker_config
    )

    svt_detections = svt_patch.detections(detections4svt, frame_id_to_fpath)
    svt_detections.match(tracker=tracker, config=detections_match_config)

    _logger.info("Finished tracking")
    return svt_detections


def display_detections(
    dataset: FramesDirDataset,
    svt_s0_detections,
):
    figure_output = ipywidgets.Output()
    frame_slider = ipywidgets.IntSlider(
        value=0,
        min=0,
        max=len(dataset) - 1,
        step=1,
        orientation="horizontal",
        # description and readout are disabled because we'll show the
        # frame filename in the label ourselves.
        description="",
        readout=False,
        # Only make changes when user stops moving slider.
        continuous_update=False,
    )
    previous_button = ipywidgets.Button(
        description="⮜",
        disabled=False,
        tooltip="Previous",
    )
    next_button = ipywidgets.Button(
        description="⮞",
        disabled=False,
        tooltip="Next",
    )

    def show_frame(idx):
        img = dataset[idx]
        fig = plotly.express.imshow(img)
        for track, x, y, width, height in svt_s0_detections[str(idx)].values():
            fig.add_shape(
                type="rect",
                x0=x,
                x1=x + width,
                y0=y,
                y1=y + height,
                line_color="red",
            )
            fig.add_annotation(
                x=x,
                y=y,
                text=f"Track {track}",
                font={"color": "red"},
            )
        figure_output.clear_output()
        with figure_output:
            fig.show()

    def on_frame_slider_change(change):
        frame_label.value = dataset.frames[change["new"]]
        show_frame(change["new"])

    def on_click_previous(button):
        del button
        # IntSlider already clamps the value, we just -=1
        frame_slider.value -= 1

    def on_click_next(button):
        del button
        # IntSlider already clamps the value, we just +=1
        frame_slider.value += 1

    previous_button.on_click(on_click_previous)
    next_button.on_click(on_click_next)
    frame_slider.observe(on_frame_slider_change, names="value")

    frame_label = ipywidgets.Label(dataset.frames[0])
    show_frame(0)

    buttons_box = ipywidgets.HBox([previous_button, frame_slider, next_button])
    whole_box = ipywidgets.VBox([buttons_box, frame_label, figure_output])
    IPython.display.display(whole_box)


def draw_tracks_in_img(img, frame_tracks: pd.DataFrame) -> None:
    cmap = list(matplotlib.cm.Pastel1.colors)
    cmap = [(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in cmap]

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 1

    for row in frame_tracks.itertuples():
        # Some detections have no track.  Do not show them on the
        # video.
        if row.track_id == UNKNOWN_TRACK_ID_MARKER:
            continue

        colour = cmap[row.track_id % len(cmap)]
        label = str(row.track_id)

        bbox_x = int(row.x)
        bbox_y = int(row.y)
        bbox_w = int(row.width)
        bbox_h = int(row.height)

        text_width, text_height = cv2.getTextSize(
            label, font_face, font_scale, font_thickness
        )[0]
        text_box_pts = (
            (bbox_x + bbox_w - text_width + 4, bbox_y),
            (bbox_x + bbox_w, bbox_y - text_height - 4),
        )

        # Draw track bounding box
        cv2.rectangle(
            img,
            (bbox_x, bbox_y),
            (bbox_x + bbox_w, bbox_y + bbox_h),
            colour,
            thickness=2,
        )
        # Draw box for bounding box label
        cv2.rectangle(
            img,
            text_box_pts[0],
            text_box_pts[1],
            colour,
            cv2.FILLED,
        )
        # Write label
        cv2.putText(
            img,
            label,
            text_box_pts[0],
            font_face,
            font_scale,
            color=(0, 0, 0),
            lineType=cv2.LINE_AA,
        )


def make_frames_with_tracks(
    csv_fpath: str, in_frames_dir: str, out_frames_dir: str
) -> None:
    track_data = pd.read_csv(csv_fpath)
    track_data.sort_values("frame_filename", inplace=True)
    track_data.set_index(keys=["frame_filename"], drop=False, inplace=True)

    for frame_fname in sorted(os.listdir(in_frames_dir)):
        in_fpath = os.path.join(in_frames_dir, frame_fname)
        out_fpath = os.path.join(out_frames_dir, frame_fname)
        frame_tracks = track_data.loc[track_data.frame_filename == frame_fname]

        if len(frame_tracks) == 0:
            shutil.copy(in_fpath, out_fpath)
        else:
            img = cv2.imread(in_fpath)
            draw_tracks_in_img(img, frame_tracks)
            cv2.imwrite(out_fpath, img)
