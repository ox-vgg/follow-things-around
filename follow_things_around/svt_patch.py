#!/usr/bin/env python3

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

## Most of the code in this file is copied from svt which has the
## following copyright notice:
##
## Copyright(c) 2018 Abhishek Dutta and Li Zhang
##
## All rights reserved.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
## Redistributions of source code must retain the above copyright notice, this
## list of conditions and the following disclaimer.
## Redistributions in binary form must reproduce the above copyright notice,
## this list of conditions and the following disclaimer in the documentation
## and/or other materials provided with the distribution.
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
## ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
## LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
## CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
## SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
## INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
## CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
## ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
## POSSIBILITY OF SUCH DAMAGE.

## FIXME: implement a track_forward and track_backward or a merge
## tracks or something in svt.detections.detections.
##
## svt.detections.detections.match makes tracks out of the existing
## detections but it doesn't add any new detections from the actual
## track() step even when the score is really high and it overlaps
## with the template bbox.
##
## We should probably add methods to extend and merge tracks using the
## results of track().  We should do that in the forward and reverse
## direction.  In the mean time, this just replaces the match() with
## something that tracks forward and adds new detections as required.
##
## This version adds a nonmatch_tracking_threshold config option which
## is the score threshold.

import logging
import os.path

import svt.detections


_logger = logging.getLogger(__name__)


class detections(svt.detections.detections):
    def match(self, tracker, config):
        next_track_id = 0  # intialize globally unique track id
        for shot_id in self.detection_data:
            _logger.info("Processing shot_id=%s", shot_id)

            #### retrieve a sorted list of all frame_id for a given shot_id
            frame_id_list = sorted(
                self.detection_data[shot_id], key=int
            )  # key=int ensures frame_id is treated as number

            #### run a forward matching pass for each pair of consecutive frames
            for frame_id_index in range(0, len(frame_id_list) - 1):
                template_frame_id = frame_id_list[frame_id_index]
                search_frame_id = frame_id_list[frame_id_index + 1]
                template_fn = self.frame_id_to_filename_map[template_frame_id]
                search_fn = self.frame_id_to_filename_map[search_frame_id]
                search_bbox_list = self.detection_data[shot_id][
                    search_frame_id
                ]
                _logger.debug(
                    "  Frames: %s -> %s", template_frame_id, search_frame_id
                )

                #### Preload template and search image
                template_abs_fn = os.path.join(
                    config["frame_img_dir"], template_fn
                )
                search_abs_fn = os.path.join(
                    config["frame_img_dir"], search_fn
                )
                template_img = self.load_image(template_abs_fn)
                search_img = self.load_image(search_abs_fn)

                for box_id in self.detection_data[shot_id][template_frame_id]:
                    _logger.debug("    box_id=%s", box_id)
                    b = self.detection_data[shot_id][template_frame_id][box_id]
                    template_bbox = [
                        b[0],
                        int(b[1]),
                        int(b[2]),
                        int(b[3]),
                        int(b[4]),
                    ]  # we don't need float

                    #### initialize tracker using frame k
                    tracker.init_tracker(template_img, template_bbox)
                    #### track the object in frame (k+1)
                    pos, size, score = tracker.track(search_img)
                    tracked_search_bbox = [
                        template_bbox[0],
                        int(pos[0] - size[0] / 2),
                        int(pos[1] - size[1] / 2),
                        int(size[0]),
                        int(size[1]),
                    ]

                    (
                        max_overlap_search_box_id,
                        max_overlap,
                    ) = self.find_most_overlapping_bbox(
                        tracked_search_bbox, search_bbox_list
                    )
                    _logger.debug(
                        "      overlap=%f, search bbox_id=%s",
                        max_overlap,
                        max_overlap_search_box_id,
                    )
                    if max_overlap >= config["match_overlap_threshold"]:
                        # propagate the track_id of template's bbox to the matched search bbox
                        if (
                            template_bbox[0]
                            == config["UNKNOWN_TRACK_ID_MARKER"]
                        ):
                            self.detection_data[shot_id][template_frame_id][
                                box_id
                            ][0] = next_track_id
                            next_track_id = next_track_id + 1

                        self.detection_data[shot_id][search_frame_id][
                            max_overlap_search_box_id
                        ][0] = self.detection_data[shot_id][template_frame_id][
                            box_id
                        ][
                            0
                        ]
                        _logger.debug(
                            "    %s is track %d",
                            search_frame_id,
                            template_bbox[0],
                        )
                    elif score > config["nonmatch_tracking_threshold"]:
                        # There is no match on the next frame
                        # detections but the tracker found one within
                        # the score threshold so we add a new one.
                        # propagate the track_id of template's bbox to the matched search bbox
                        if (
                            template_bbox[0]
                            == config["UNKNOWN_TRACK_ID_MARKER"]
                        ):
                            self.detection_data[shot_id][template_frame_id][
                                box_id
                            ][0] = next_track_id
                            next_track_id = next_track_id + 1
                        next_available_box_id = str(
                            max(
                                [
                                    int(x)
                                    for x in self.detection_data[shot_id][
                                        template_frame_id
                                    ].keys()
                                ],
                                default=-1,
                            )
                            + 1
                        )
                        self.detection_data[shot_id][search_frame_id][
                            next_available_box_id
                        ] = [
                            self.detection_data[shot_id][template_frame_id][
                                box_id
                            ][0],
                            *tracked_search_bbox[1:],
                        ]
                        _logger.debug(
                            "added new detection for shot_id %s frame_id %s box id %s : %s",
                            shot_id,
                            search_frame_id,
                            next_available_box_id,
                            tracked_search_bbox,
                        )
