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

import IPython.display
import ipywidgets
import plotly.express

from follow_thins_around import FramesDirDataset


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
