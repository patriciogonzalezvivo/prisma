# Copyright (c) 2024, Patricio Gonzalez Vivo
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.

#     * Neither the name of Patricio Gonzalez Vivo nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import os
import cv2
import argparse

from bands.common.meta import load_metadata

top = ['rgba', 'depth_patchfusion']
bottom = ['mask', 'mask_sdf']

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

    data = load_metadata(args.input)
    if data:
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
