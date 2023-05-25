# %% [markdown] id="AR4oetmCFHUa"
# # VGG Chimpanzee Tracking
#
# This project provides an interface to track chimpanzees in the wild.
# It will detect the chimpanzees in video files.  It is based in the
# following projects:
#
# - [Count, Crop and Recognise: Fine-Grained Recognition in the
#   Wild](https://www.robots.ox.ac.uk/~vgg/research/ccr/) -
#   [[paper]](http://www.robots.ox.ac.uk/~vgg/publications/2019/Bain19/bain19.pdf)
#
# - [Chimpanzee face recognition from videos in the wild using deep
#   learning](https://www.robots.ox.ac.uk/~vgg/research/ChimpanzeeFaces/) -
#   [[paper]](https://advances.sciencemag.org/content/advances/5/9/eaaw0736.full.pdf)
#

# %% [markdown] id="O3vImnLxFHUf"
# ## 1 - Read Me First
#
# This project is a [Jupyter](https://jupyter.org/) notebook to track
# chimpanzees in the wild and was designed to run in [Google
# Colab](https://colab.research.google.com/).  If you are not reading
# this notebook in Google Colab, click
# [here](https://colab.research.google.com/github/ox-vgg/chimpanzee-tracking/blob/main/tracking.ipynb).
#

# %% [markdown] id="-tfDPTizFHUi"
# ### 1.1 - What is, and how to use, a Jupyter notebook
#
# A Jupyter notebook is a series of "cells".  Each cell contains either
# text (like this one) or code (like others below).  A cell that
# contains code will have a "Run cell" button on the left side like this
# "<img height="18rem" alt="The 'Run cell' button in Colab"
# src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAQAAAD9CzEMAAABTklEQVRYw+2XMU7DMBRAX6ss3VA7VV25AFNWzsDQXoAzVDlBKw6QDJwhTO3OCVjaka0VXVKJDUVC4jOgiMHYcRx9S0j9f7XfS5x8+xsu8R9iQEpGyY4TgnBiR0lGyqA/fMaaI2LJI2tm4fAxObUV3mRNzjgEP+fcCm/yzLwbPKHwhjdZkPjiR2w64wVhw8jv6bdBeEHY+rxFEYz/WaiWWPTCC8LChZ9Q9RZUTOyCvDdeEHJ71drL6o43b0Ftq+6VYxJc8ciXp2L1F37IwSkAuOXVS3BgaApS55TfInzg00ORmoLMSwBww0urIDMFpbcAEpZ8OMeXpmDfQQBwzbNj/N6cUHUUANzzbi03I+oAAUx5stRCfIH6Eql/ZPXfVL3Q1LcK9c1OfbuOcOCoH5kRDn31tiVC4xWhdVRvfiO07xEuIFGuUBEugVGusZfQj28NImRviDLNnQAAAABJRU5ErkJggg==">".
# When you click the "Run cell" button, the code in that cell will run
# and when it finishes, a green check mark appears next to the "Run
# cell" button".  You need to wait for the code in that cell to finish
# before "running" the next cell.
#

# %% [markdown] id="4kkvU97kDjYh"
# ### 1.2 - Particulars of this notebook
#
# This notebook was designed to run in Google Colab and to analyse
# videos in Google Drive.  It will also save back the analysis results
# in Google Drive.  As such, it requires a Google account.
#
# You must run the cells on this notebook one after the other since each
# cell is dependent on the results of the previous cell.
#
# This notebook also some interactive cells, namely in the options
# sections.  After setting their values, these cells must be run, just
# like other code cells.  Setting their values only has effect after you
# "run" their cell.

# %% [markdown] id="meUaiJq1V-tI"
# ### 1.3 - Testing this notebook
#
# We recommend you first test this notebook with a short video, less
# than 20 seconds long.  Try our own sample video (download it
# [here](https://thor.robots.ox.ac.uk/software/chimpanzee-tracking/test-data/19-mini.mp4)),
# and then a video fragment of your own videos.

# %% [markdown] id="iXk_VjsyDmtS"
# ### 1.4 - Results files
#
# This notebook will save all the results in a single directory.  It
# will generate the following files:
#
# - `frames` - a directory with the individual video frames.  You may
#   want to delete them after validating the results.  They take up a
#   lot of space and can be regenerated later
#
# - `detections.pkl` - the initial detections in [Python's pickle
#   format](https://docs.python.org/3/library/pickle.html).
#
# - `results-via-project.json` - the final detections as a
#   [VIA](https://www.robots.ox.ac.uk/~vgg/software/via/) project.  This
#   requires the images in the `frames` directory.
#
# - `results.csv` - the final detections in CSV format.
#
# - `tracks.mp4` - video with tracks (see Section 6).
#
# Note that none of those files includes the video filename.  As such,
# our recommendation is to create a results directory for each video.
#

# %% [markdown] id="vuE9bu6GDpOv"
# ### 1.5 - GPU access
#
# A GPU is required to run this pipeline in a sensible manner.  For
# example, without a GPU, a two minutes video will take close to two
# hours to process.
#
# By default, this notebook will run with a GPU.  However, it is
# possible that you were not allocated one, typically because you've
# used up all your GPU resources.  You can confirm this, and possibly
# change it, manually.  To do that, navigate to "Edit" -> "Notebook
# Settings" and select "GPU" from the "Hardware Accelerator" menu.
#

# %% [markdown] id="AU4JuuRqDrxQ"
# ### 1.6 - Moving forward
#
# You can run this notebook on Google Colab but if you have a large
# collection of videos or if your videos are particularly long, you may
# use up all of your Colab resources.  It may be worth running this on
# your own computers.  Contact us if you need help to do that.
# Alternatively, you purchase additional "compute units" with [Google
# Colab Plans](https://colab.research.google.com/signup).
#

# %% [markdown] id="wWhspjzoFHUl"
# ## 2 - Setup
#

# %% [markdown] id="z9eagfzTBgMh"
# ### 2.1 - Check for GPU access
#

# %% cellView="form" id="k72IXhXhFHUn"
#@markdown By default, this notebook will run with a GPU.  However, it
#@markdown is possible that you were not allocated one.  If you get a
#@markdown message saying that you do not have access to a GPU,
#@markdown navigate to "Edit" -> "Notebook Settings" and select "GPU"
#@markdown from the "Hardware Accelerator" menu.  Once you change it,
#@markdown you need to run this cell again.

# gpu_info = !nvidia-smi --list-gpus
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
    USE_GPU = False
    print('You are NOT connected to a GPU.  This will run very slow.')
    print('Consider reconnecting to a runtime with GPU access.')
else:
    USE_GPU = True
    print('You are connected to the following GPUs:')
    print(gpu_info)

# %% [markdown] id="0jKiMdsdBpQO"
# ### 2.2 - Install and load dependencies
#

# %% cellView="form" id="y3Nw3Km_FHUp"
#@markdown

#^ this @markdown above is just so that we can set cellView to form

import contextlib
import glob
import io
import logging
import os
import os.path
import pickle
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import google.colab.drive
import google.colab.output
import IPython.display
import ipywidgets
import matplotlib.cm
import numpy as np
import pandas as pd
import PIL.Image
import plotly.express
import requests
import torch
import torch.backends.cudnn as cudnn
import torch.cuda
from torch.autograd import Variable

logging.basicConfig()
_logger = logging.getLogger()


## The SVT package https://www.robots.ox.ac.uk/~vgg/projects/seebibyte/software/svt/
# Despite the `--quiet` flag, it still prints out a mysterious
# "Preparing metadata (setup.py)" message so just redirect stdout.
# Important messages should go to stderr anyway.
# !pip install --quiet git+https://gitlab.com/vgg/svt/ > /dev/null
import svt.detections
from svt.siamrpn_tracker import siamrpn_tracker


## The ssd.pytorch "package" (cloned into ssd_pytorch so it can imported)
# When we import stuff from ssd.pytorch, it import data.coco which reads a
# `HOME/data/coco/coco_labels.txt` (even though we don't need it).  We create
# an empty file so it doesn't fail to import.  We also need to add `HOME` to
# `data/config.py` or it will try to read from `/root`.  See
# https://github.com/amdegroot/ssd.pytorch/issues/571
# !git clone --quiet \
#   --single-branch --branch vgg-colab \
#   https://github.com/carandraug/ssd.pytorch.git ssd_pytorch/
# !echo 'HOME = "ssd_pytorch"' >> ssd_pytorch/data/config.py
# !mkdir ssd_pytorch/data/coco
# !touch ssd_pytorch/data/coco/coco_labels.txt
from ssd_pytorch.data import base_transform
from ssd_pytorch.ssd import build_ssd

# %% [markdown] id="chRTgjgMCJdd"
# ### 2.3 - Mount Google Drive
#

# %% cellView="form" id="iOD4aWrjFHUr"
#@markdown When you run this cell, a dialog will appear about a
#@markdown request for access to your Google Drive Files.  This is
#@markdown required to access the videos for analysis and to then save
#@markdown the results.  Once you click on "Connect to Google Drive",
#@markdown a pop-up window will appear to choose a Google Account and
#@markdown then to allow access to "Google Drive for desktop".

google.colab.drive.mount('/content/drive')

# %% [markdown] id="lb_oICb3CPpC"
# ### 2.4 - Video file and results folder
#

# %% cellView="form" id="Gk90XC-8FHUs"
#@markdown To find the correct path, open the "Files" menu in the left
#@markdown sidebar.  The `drive` directory contains your Google Drive
#@markdown files.  Navigate the files, right click on the wanted file
#@markdown or directory, and select "Copy path".  Then paste the path
#@markdown in this form.  Do not forget to then "run" this cell.

VIDEO_FILE = ''  #@param {type:"string"}
RESULTS_DIRECTORY = ''  #@param {type:"string"}

if not VIDEO_FILE:
    raise Exception('VIDEO_FILE is empty, you must set it.')
if not RESULTS_DIRECTORY:
    raise Exception('RESULTS_DIRECTORY is empty, you must set it.')
if not os.path.isfile(VIDEO_FILE):
    raise Exception('The VIDEO_FILE \'%s\' does not exist' % VIDEO_FILE)
if not os.path.isdir(RESULTS_DIRECTORY):
    raise Exception(
        'The RESULTS_DIRECTORY \'%s\' does not exist' % RESULTS_DIRECTORY
    )

# %% [markdown] id="7c1gbeVrFHUu"
# ### 2.5 - Advanced options
#
# The cells hidden in this section expose the advanced options for this
# pipeline and perform the final setup.  In most cases you do not need
# to change their values.  You can click the "Run cell" button to run
# all the hidden cells inside.
#

# %% cellView="form" id="SH-IeyipFHUv"
#@markdown #### 2.5.1 - Chimpanzee detection

#@markdown The detection step is the first step.  It detects the
#@markdown location of chimpanzees without attempting to identify who
#@markdown they are.

#@markdown A detection model is required.  You can either train
#@markdown your own model, or you can use one of our pre-trained
#@markdown models for
#@markdown [face](https://thor.robots.ox.ac.uk/models/staging/chimp-tracking/face-model-ssd300_CFbootstrap_85000.pth)
#@markdown or
#@markdown [body](https://thor.robots.ox.ac.uk/models/staging/chimp-tracking/body-model-ssd300_BFbootstrapBissau4p5k_prebossou_best.pth).
#@markdown Either way, you will need to upload the model to your
#@markdown Google Drive and specify its path here.
DETECTION_MODEL = 'https://thor.robots.ox.ac.uk/models/staging/chimp-tracking/face-model-ssd300_CFbootstrap_85000.pth'  #@param {type: "string"}

#@markdown When the model detects a face or body, that detection is
#@markdown made with a confidence score.  Detections with a confidence
#@markdown score lower than the threshold are ignored.  If you set the
#@markdown threshold too high, you may loose some tracks but if you
#@markdown set it too low you may gain false tracks that need to be
#@markdown removed later.
DETECTION_THRESHOLD = 0.37  #@param {type: "slider", min: 0.0, max: 1.0, step: 0.01}

# %% cellView="form" id="nmuWC94sFHUw"
#@markdown #### 2.5.2 - Chimpanzee tracking

#@markdown The final step is to track the detected chimpanzees in the
#@markdown video.

#@markdown You will need to provide a model.  We recommend you use
#@markdown [this one](https://thor.robots.ox.ac.uk/models/staging/chimp-tracking/tracking-model-20181031_e45.pth).
#@markdown You will also specify a path in your Google Drive.
TRACKING_MODEL = 'https://thor.robots.ox.ac.uk/models/staging/chimp-tracking/tracking-model-20181031_e45.pth'  #@param {type: "string"}

MATCH_OVERLAP_THRESHOLD = 0.6 #@param {type:"slider", min:0.0, max:1.0, step:0.01}

UNKNOWN_TRACK_ID_MARKER = -1

# %% cellView="form" id="YUIcftRMFHUy"
#@markdown #### 2.5.3 - Verbosity

#@markdown How chatty do you want the notebook to be?  INFO is a good
#@markdown choice if you want to have a feeling for progress.
LOG_LEVEL = 'INFO'  #@param ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]

# %% cellView="form" id="OWKXpxAuFHUy"
#@markdown #### 2.5.4 - The final setup step

#@markdown Run this cell to perform the final pipeline setup based on
#@markdown the given options.

logging.getLogger().setLevel(LOG_LEVEL)

FRAMES_DIR = os.path.join(RESULTS_DIRECTORY, 'frames')
DETECTIONS_FPATH = os.path.join(RESULTS_DIRECTORY, 'detections.pkl')
VIA_PROJECT_FPATH = os.path.join(RESULTS_DIRECTORY, 'results-via-project.json')
CSV_FPATH = os.path.join(RESULTS_DIRECTORY, 'results.csv')
TRACKS_VIDEO_FPATH = os.path.join(RESULTS_DIRECTORY, 'tracks.mp4')


# These should never be true because USE_GPU was set automatically
# based on whether a GPU is available.
if USE_GPU and not torch.cuda.is_available():
    raise Exception('Your runtime does not have a GPU.')
elif torch.cuda.is_available() and not USE_GPU:
    _logger.warn('You have a GPU but chose to not use it.  Are you sure?')

if USE_GPU:
    DEFAULT_DEVICE = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    DEFAULT_DEVICE = 'cpu'
    torch.set_default_tensor_type('torch.FloatTensor')

_logger.info('Will use %s device.', DEFAULT_DEVICE.upper())


# Required to display the tracking results with plotly or matplotlib.
google.colab.output.enable_custom_widget_manager()


def local_path_for_model(path: str) -> str:
    if path.startswith('https://'):
        downloaded_fh = tempfile.NamedTemporaryFile(delete=False)
        r = requests.get(path)
        downloaded_fh.write(r.content)
        downloaded_fh.flush()
        return downloaded_fh.name
    else:
        return path


DETECTION_MODEL_PATH = local_path_for_model(DETECTION_MODEL)
TRACKING_MODEL_PATH = local_path_for_model(TRACKING_MODEL)


logging2ffmpeg_loglevel = {
    'CRITICAL': 'fatal',
    'ERROR': 'error',
    'WARNING': 'warning',
    'INFO': 'info',
    'DEBUG': 'debug',
}

FFMPEG_LOG_LEVEL = logging2ffmpeg_loglevel[LOG_LEVEL]


def subprocess_print_stderr(args: List[str]) -> None:
    p = subprocess.Popen(args, text=True, stderr=subprocess.PIPE)

    for line in iter(p.stderr.readline, ""):
        print(line, end="")
    p.wait()
    if p.returncode != 0:
        raise Exception('subprocess failed')


def ffmpeg_video_to_frames(video_fpath: str, frames_dir: str) -> None:
    subprocess_print_stderr(
        [
            'ffmpeg',
            '-i', video_fpath,
            '-vsync', 'vfr',
            '-q:v', '1',
            '-start_number', '0',
            '-filter:v', 'scale=iw:ih*(1/sar)',
            '-loglevel', FFMPEG_LOG_LEVEL,
            # FIXME: what if %06d.jpg is not enough and rools over?
            os.path.join(FRAMES_DIR, "%06d.jpg"),
        ]
    )


def ffprobe_get_frame_rate(video_fpath: str) -> float:
    ffprobe_p = subprocess.run(
        [
            'ffprobe',
            '-loglevel', 'panic',
            '-select_streams', 'v',
            '-show_entries', 'stream=r_frame_rate',
            '-print_format', 'csv',
            video_fpath,
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    # we expect something like "stream,25/1\n"
    frame_rate_parts = ffprobe_p.stdout.split(',')[1][:-1].split('/')
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
            'ffmpeg',
            '-y',  # overwrite output files without asking
            '-loglevel', FFMPEG_LOG_LEVEL,
            '-framerate', str(frame_rate),
            '-pattern_type', 'glob',
            '-i', os.path.join(frames_dir, '*.jpg'),
            '-i', in_video_fpath,
            '-c:a', 'aac',  # https://github.com/ox-vgg/chimpanzee-tracking/issues/1
            '-c:v', 'libx264',
            '-map', '0:v:0',  # use video from input 0 / stream 0
            '-map', '1:a:0',  # use audio from input 1 / stream 0
            '-pix_fmt', 'yuv420p',
            out_video_fpath,
        ]
    )


# Ideally this would be a NamedTuple and values couldn't change.
# However, in practice this is created in the detect phase and the
# id_score is filled in during the identify phase.  So we use
# dataclass and Optional.  Probably should be refactored.  But
# then, SVT also wants to access the values by position so
# dataclass brings its own problems.
@dataclass
class Detection:
    track_id: int
    x: float
    y: float
    w: float
    h: float
    id_score: Optional[float]


# TODO: why frame_id_to_filename only includes frames with detections?
# Can't we just check on the the list of detections that there's none?
#
# TODO: why do we convert the frame number to a string fo rkeys in the
# dict?  Why not just use an int?
#
# TODO: why return frame_id_to_filename (can't that be deduced?)
def detect(
    video_frames: List[str],
    detection_model_path: str,
    visual_threshold: float,
):
    if visual_threshold < 0.0 or visual_threshold > 1.0:
        raise ValueError(
            'visual_threshold needs to be a number between 0.0 and 1.0'
        )

    num_classes = 2  # +1 background
    net = build_ssd('test', 300, num_classes)
    net.load_state_dict(
        torch.load(detection_model_path, map_location=DEFAULT_DEVICE)
    )
    net.eval()
    net = net.to(DEFAULT_DEVICE)

    # XXX: Can't we do the benchmark just once and then save it?
    if USE_GPU:
        cudnn.benchmark = True

    _logger.info('Finished loading model %s', detection_model_path)

    _logger.info('Starting detection phase')

    # init this important variables
    frame_id_to_filename: Dict[str, str] = {}
    frame_detections: Dict[str, Dict[str, Detection]] = defaultdict(dict)

    # Go through frame list
    for frame_index, frame_fpath in enumerate(video_frames):
        frame_id = str(frame_index)  # XXX: Why?  Why are we not using the int?
        frame_id_to_filename[frame_id] = frame_fpath

        if frame_index % 1000 == 0:
            _logger.info(
                'Starting to process images %d to %d',
                frame_index,
                min(frame_index + 1000 - 1, len(video_frames) - 1),
            )

        # Acquire image
        img = cv2.imread(frame_fpath)

        # Apply transforms to the image
        transformed = base_transform(img, net.size, [104, 117, 123])
        transformed = torch.from_numpy(transformed).permute(2, 0, 1)
        transformed = Variable(transformed.unsqueeze(0))
        transformed = transformed.to(DEFAULT_DEVICE)

        # Silence UserWarnings from ssd_pytorch/layers/box_utils.py "An output
        # with one or more elements was resized since it had shape [X], which
        # does not match the required output shape [Y]. This behavior is
        # deprecated, and in a future PyTorch release outputs will not be
        # resized unless they have zero elements. You can explicitly reuse an
        # out tensor t by resizing it, inplace, to zero elements with
        # t.resize_(0)."
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net_out = net(transformed)  # run detector
        net_detections = net_out.data

        # Scale each detection back up to the image
        scale = torch.Tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
        )
        pred_num = 0
        for i in range(net_detections.size(1)):
            j = 0
            while net_detections[0, i, j, 0] >= visual_threshold:
                score = net_detections[0, i, j, 0]
                pt = (net_detections[0, i, j, 1:] * scale).cpu().numpy()
                coords = (
                    max(float(pt[0]), 0.0),
                    max(float(pt[1]), 0.0),
                    min(float(pt[2]), img.shape[1]),
                    min(float(pt[3]), img.shape[0]),
                )
                if coords[2] - coords[0] >= 1 and coords[3] - coords[1] >= 1:
                    # XXX: original code had the option to load
                    # face_id_score from a previous run but this makes
                    # very little sense now?

                    # Save detections to list ...
                    a_detection = Detection(
                        track_id=UNKNOWN_TRACK_ID_MARKER,
                        x=coords[0],
                        y=coords[1],
                        w=coords[2] - coords[0],
                        h=coords[3] - coords[1],
                        id_score=None,
                    )

                    frame_detections[frame_id][str(pred_num)] = a_detection
                    pred_num += 1
                j += 1

    _logger.info('Finished detections')
    return frame_detections, frame_id_to_filename


def track(
    detections: Dict[str, Dict[str, Detection]],
    frame_id_to_filename: Dict[str, str],
    tracking_model_path: str,
):
    _logger.info('Starting tracking phase')

    tracker_config = {
        'gpu_id': 0 if USE_GPU else -1,
        'verbose': True,
        'preload_model': True,
    }

    detections_match_config = {
        'match_overlap_threshold': MATCH_OVERLAP_THRESHOLD,
        'UNKNOWN_TRACK_ID_MARKER': UNKNOWN_TRACK_ID_MARKER,
        'frame_img_dir': '',
        'verbose': True,
        'via_project_name': VIDEO_FILE,
    }

    # We only use one shot, hence only shot_id 0.
    # XXX: Why are we using strings for int keys?
    shot_id = '0'
    detections4svt = {shot_id: {x: {} for x in frame_id_to_filename.keys()}}
    for frame_id, detections_values in detections.items():
        for box_id, detection in detections_values.items():
            detections4svt[shot_id][frame_id][box_id] = [
                detection.track_id,
                detection.x,
                detection.y,
                detection.w,
                detection.h,
            ]

    # redirect sys.stdout to a buffer to capture the prints() in the code below
    svt_stdout = io.StringIO()
    with contextlib.redirect_stdout(svt_stdout):
        tracker = siamrpn_tracker(
            model_path=tracking_model_path, config=tracker_config
        )

        svt_detections = svt.detections.detections()
        svt_detections.read(detections4svt, frame_id_to_filename)
        svt_detections.match(tracker=tracker, config=detections_match_config)

    _logger.info(svt_stdout.getvalue())
    _logger.info('Finished tracking')
    return svt_detections


def display_detections(
    frame_id_to_filename,
    svt_s0_detections,
):
    frame_id_filename_pair = sorted(
        list(frame_id_to_filename.items()),
        key=lambda kv: kv[1],
    )

    figure_output = ipywidgets.Output()
    frame_slider = ipywidgets.IntSlider(
        value=0,
        min=0,
        max=len(frame_id_filename_pair) - 1,
        step=1,
        orientation='horizontal',
        # description and readout are disabled because we'll show the
        # frame filename in the label ourselves.
        description="",
        readout=False,
        # Only make changes when user stops moving slider.
        continuous_update=False,
    )
    previous_button = ipywidgets.Button(
        description='⮜',
        disabled=False,
        tooltip='Previous',
    )
    next_button = ipywidgets.Button(
        description='⮞',
        disabled=False,
        tooltip='Next',
    )

    def show_frame(idx):
        img = PIL.Image.open(frame_id_filename_pair[idx][1])
        fig = plotly.express.imshow(img)
        for track, x, y, width, height in svt_s0_detections[str(idx)].values():
            fig.add_shape(
                type='rect',
                x0=x,
                x1=x + width,
                y0=y,
                y1=y + height,
                line_color='red',
            )
            fig.add_annotation(
                x=x,
                y=y,
                text=f'Track {track}',
                font={'color': 'red'},
            )
        figure_output.clear_output()
        with figure_output:
            fig.show()

    def on_frame_slider_change(change):
        frame_label.value = frame_id_filename_pair[change['new']][1]
        show_frame(change['new'])

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
    frame_slider.observe(on_frame_slider_change, names='value')

    frame_label = ipywidgets.Label(frame_id_filename_pair[0][1])
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
    track_data.sort_values('frame_filename', inplace=True)
    track_data.set_index(keys=['frame_filename'], drop=False, inplace=True)

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


# %% [markdown] id="XIgX5CS8FHU0"
# ## 3 - Convert video to frames
#
# The pipeline needs the video frames as individual image files.  This
# cell will create a `frames` directory and save the individual images
# there.  You may skip running this cell if you already have a `frames`
# directory with images.  This cell will error if the `frames` directory
# already exists to prevent overwriting any existing data.
#

# %% cellView="form" id="E4xj4YRWFHU0"
#@markdown Skip this cell if you already have the frames.  If you run
#@markdown this cell and the `frames` directory already exists, it
#@markdown errors to avoid overwriting any previous images.

os.makedirs(FRAMES_DIR, exist_ok=False)

ffmpeg_video_to_frames(VIDEO_FILE, FRAMES_DIR)

_logger.info('Finished extracting individual frames to \'%s\'', FRAMES_DIR)

# %% [markdown] id="AAhN5CkMFHU0"
# ## 4 - Detection step
#
# The detection of chimpanzees is the first step in the pipeline.  If
# you have previously run the detection step then you will have a `detections.pkl`
# file in the results directory.  If so, skip the "detection" cell and
# run the "load previous detections results" cell instead (you may need
# to click in "2 cells hidden" to see them).
#

# %% cellView="form" id="jk_beBJTFHU2"
#@markdown ### 4.1 - Run detection (option 1)

video_frames = sorted(glob.glob(os.path.join(FRAMES_DIR, '*.jpg')))

if len(video_frames) == 0:
    raise Exception(
        "No files in '%s'.  Did you run the previous section which converts"
        " the video to frames?" % FRAMES_DIR
    )

detections, frame_id_to_filename = detect(
    video_frames,
    DETECTION_MODEL_PATH,
    DETECTION_THRESHOLD,
)

with open(DETECTIONS_FPATH, 'wb') as fh:
    pickle.dump(
        {
            'detections': detections,
            'frame_id_to_filename': frame_id_to_filename,
        },
        fh,
    )
_logger.info('Detection results saved to \'%s\'', DETECTIONS_FPATH)

# %% cellView="form" id="xHVc5I4EFHU2"
#@markdown ### 4.2 - Load previous detection results (option 2)

with open(DETECTIONS_FPATH, 'rb') as fh:
    loaded_detections = pickle.load(fh)

detections = loaded_detections['detections']
frame_id_to_filename = loaded_detections['frame_id_to_filename']
video_frames = list(frame_id_to_filename.values())

_logger.info('Detection results loaded from \'%s\'', DETECTIONS_FPATH)

# %% [markdown] id="FN5FGZxJFHU3"
# ## 5 - Tracking step
#

# %% cellView="form" id="GodRuQQ4FHU3"
#@markdown The final step in the pipeline is to track the detected
#@markdown chimpanzees in the video.  At the end of this step, the
#@markdown tracking results will be saved in a CSV file and as a
#@markdown [VIA](https://www.robots.ox.ac.uk/~vgg/software/via/)
#@markdown project.

svt_detections = track(
    detections,
    frame_id_to_filename,
    TRACKING_MODEL_PATH,
)

# XXX: So far we've been using absolute filepaths.  However, in the
# exported VIA project we want to use the filename only so that the
# images can be found by setting the project "Default Path".  This
# hack changes the filepath in the SVT internals.
svt_detections.frame_id_to_filename_map = {
    k: os.path.basename(v)
    for k, v in svt_detections.frame_id_to_filename_map.items()
}

svt_detections.export_via_project(
    VIA_PROJECT_FPATH,
    config={'frame_img_dir': FRAMES_DIR, 'via_project_name': ''},
)
svt_detections.export_plain_csv(CSV_FPATH, {})

# %% [markdown] id="1XoTxVlsFHU3"
# ## 6 - Visualise tracking results
#

# %% [markdown] id="LL5er1FZCiBo"
# ### 6.1 - Visualise in Google Colab (option 1)
#

# %% cellView="form" id="4sbn8DsHIi1C"
#@markdown You can see the tracking results right here, inside this
#@markdown Google Colab notebook, but the interface is a bit slow.
#@markdown This is fine if you want to have a quick look at some of
#@markdown of frames only.

#@markdown Run this cell and then click on the arrow buttons to
#@markdown display the next or previous frame, and you can move the
#@markdown slider to move to a specific frame.  When you dragging the
#@markdown slider, the displayed frame is only updated once the slider
#@markdown is released.  Expect a couple of seconds wait for the frame
#@markdown to be updated.

display_detections(frame_id_to_filename, svt_detections.detection_data['0'])

# %% [markdown] id="i5FjIafRC8te"
# ### 6.2 - Visualise locally with VIA (option 2)
#

# %% [markdown] id="QBdj3CToLPVG"
# [VIA](https://www.robots.ox.ac.uk/~vgg/software/via/) is a web
# application to view and perform annotations of image, audio, and
# video.  It is free software and runs locally on the web browser.  You
# can view the tracking results on the individual frames with VIA.
#
# This is much more responsive than viewing inside the notebook but
# requires download the frames locally (either manually or with [Google
# Drive for
# Desktop](https://support.google.com/a/users/answer/13022292)).
#
# 1. Download [VIA
#    2](https://www.robots.ox.ac.uk/~vgg/software/via/downloads/via-2.0.12.zip).
#    This is a zip file.  Open it.  Inside there is a `via.html` file.
#    Open it in your web browser to start VIA.
#
# 2. Download the `results-via-project.json` from your results diretcory
#    and the whole frames directory.  If you are using Google Drive for
#    Desktop sync it now.  The frames directory is pretty large and this
#    step may take a long time.
#
# 3. Navigate to "Project" -> "Load" and select the
#    `results-via-project.json` file.  A "File Not Found" error message
#    will appear.  This means that VIA does not know where the images
#    are.
#
# 4. Navigate to "Project" -> "Settings".  Set the "Default Path" to the
#    `frames` directory in your computer.
#

# %% [markdown] id="5kfdOPAfCpyx"
# ### 6.3 - Create video file with tracks (option 3)
#

# %% cellView="form" id="l44oU0LWA7nw"
#@markdown You may also generate a video file with the detections
#@markdown superimposed.  The video file will be named `tracks.mp4`
#@markdown and saved in the `RESULTS_DIRECTORY` in your Google Drive.

with tempfile.TemporaryDirectory() as out_frames_dir:
    tmp_tracks_fpath = os.path.join(out_frames_dir, 'tracks.mp4')
    make_frames_with_tracks(CSV_FPATH, FRAMES_DIR, out_frames_dir)
    ffmpeg_video_from_frames_and_video(
        out_frames_dir, VIDEO_FILE, tmp_tracks_fpath
    )
    shutil.move(tmp_tracks_fpath, TRACKS_VIDEO_FPATH)

_logger.info('Video file with tracks created \'%s\'', TRACKS_VIDEO_FPATH)
