#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import json
import argparse

top = ['rgba_scaledx2']
bottom = ['mask', 'depth_fusion']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', '-i', help="Input folder. Ex: `data/000`", type=str, required=True)
    parser.add_argument('-output', '-o', help="Output file. Ex: `000.png`", type=str, required=True)
    parser.add_argument('-top', '-t', help="Top row of images", type=str, nargs='+')
    parser.add_argument('-bottom', '-b', help="Bottom row of images", type=str, nargs='+')
    args = parser.parse_args()

    if args.top:
        top = args.top

    if args.bottom:
        bottom = args.bottom

    if os.path.isdir( args.input ):
        payload_path = os.path.join( args.input, "payload.json")
        if os.path.isfile(payload_path):
            data = json.load( open(payload_path) )

            top_row = None
            bottom_row = None

            # vertically concatenate the top row
            for band in top:
                img = cv2.imread( os.path.join( args.input, data["bands"][band]["url"] ) )
                if top_row is None:
                    top_row = img
                else:
                    top_row = cv2.hconcat([top_row, img])
            
            # vertically concatenate the bottom row
            for band in bottom:
                img = cv2.imread( os.path.join( args.input, data["bands"][band]["url"] ) )
                if bottom_row is None:
                    bottom_row = img
                else:
                    bottom_row = cv2.hconcat([bottom_row, img])
            
            # horizontally concatenate the top and bottom rows
            if top_row is None:
                conc = bottom_row
            elif bottom_row is None:
                conc = top_row
            else:
                conc = cv2.vconcat([top_row, bottom_row])

            cv2.imwrite(args.output, conc)
