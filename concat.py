#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import json
import argparse


       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', '-i', help="Input folder. Ex: `data/000`", type=str, required=True)
    args = parser.parse_args()

    if os.path.isdir( args.input ):
        payload_path = os.path.join( args.input, "payload.json")
        if os.path.isfile(payload_path):
            data = json.load( open(payload_path) )

            rgba_scaled = cv2.imread( os.path.join( args.input, data["bands"]["rgba_scaled"]["url"] ) )
            mask = cv2.imread( os.path.join( args.input, data["bands"]["mask"]["url"] ) )
            mask_inpaint = cv2.imread( os.path.join( args.input, data["bands"]["mask_inpaint"]["url"] ) )
            depth_marigold = cv2.imread( os.path.join( args.input, data["bands"]["depth_marigold"]["url"] ) )
            depth_fusion = cv2.imread( os.path.join( args.input, data["bands"]["depth_fusion"]["url"] ) )

            lower_row = cv2.hconcat([mask, mask_inpaint, depth_marigold, depth_fusion]) 

            conc = cv2.vconcat([rgba_scaled, lower_row]) 

            cv2.imwrite("test.png", conc)
