# Image Processing Interview Project

## Introduction
This project aims to filter out redundant frames from a collection of images based on certain criteria. It is particularly useful for scenarios where continuous image data needs to be analyzed, such as in surveillance systems or video processing pipelines. This method uses image differencing techniques to find similarity between the current image frame and the previous image frame. If the similarity is higher, then the current image frame is considered a non-essential frame; if the similarity is lower (i.e., high difference), then the current image frame is considered an essential frame.

## Features
- Sorting images by camera_ID and timestamp.
- Filtering out non-essential frames based on image differencing method.
- Support for storing non-essential frames if required.
- Detection of damaged images.
- Detailed logging of processing results.

## Installation

```bash
git clone [this repo]
cd kopernikus_interview
pip install -r requirements.txt
python imaging_interview.py --path /path/to/image/folder --store_non_ess_frames
# Replace /path/to/image/folder with the path to the folder containing the image files. Use the --store_non_ess_frames flag to store non-essential frames.
```
