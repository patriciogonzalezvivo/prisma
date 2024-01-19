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
import json
from .io import check_overwrite

META_FILE = "metadata.json"

def get_metadata_path(path):
    """Get the path for the metadata file for a given path."""

    if os.path.isfile( path ):
        if path.endswith('.json'):
            return path 
        else:
            return get_metadata_path( os.path.dirname(path) )
        
    if os.path.isdir( path ):
        return os.path.join(path, META_FILE)
    
    return None


def load_metadata(path):
    """Load the metadata from a given path."""
    metadata_path = get_metadata_path(path)
    if os.path.exists(metadata_path):
        return json.load( open(metadata_path) )
    return None


def create_metadata(path):
    """Create the metadata file for a given path."""
    folder = path
    if os.path.isfile( path ):
        folder = os.path.dirname(path)

    # Check if the output folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    metadata_path = get_metadata_path(path)

    if metadata_path == None:
        print("Can't create metadata for {}".format(path))

    else:
        metadata_path = os.path.join(folder, META_FILE)

        if os.path.exists(metadata_path):
            print("Metadata already exists for {}".format(path))
        else:
            print("Creating metadata for {} as {}".format(path, metadata_path))
            with open( metadata_path, 'w') as metadata_file:
                metadata_file.write( json.dumps({ "bands": { } }, indent=4) )
        
    return load_metadata(metadata_path)


def is_video(path):
    """Check if a path is a video."""
    return path.endswith('.mp4')


def get_target(path, metadata, band="rgba", target="", force_extension=None):
    """Get the target path for a given path and band."""

    if os.path.isdir( target ):
        input_folder = target
    else:
        input_folder = os.path.dirname(path)

    input_filename = os.path.basename(path)
    input_extension = input_filename.rsplit( ".", 1 )[ 1 ]

    if force_extension and not is_video(path):
        input_extension = force_extension

    target_filename = band + "." + input_extension

    if target == "" or os.path.isdir( target ):
        target = os.path.join(input_folder, target_filename)

    if metadata:
        add_band(metadata, band, url=target_filename)

    return target


def get_url(path, metadata, band):
    """Get the url for a given path and band."""

    if os.path.isdir( path ):
        if metadata:
            if "bands" in metadata:
                if band in metadata["bands"]:
                    if "url" in metadata["bands"][band]:
                        return os.path.join(path, metadata["bands"][band]["url"])
    return path


def add_band(metadata, band, url="", folder=""):
    """Add a band to the metadata."""

    if "bands" not in metadata:
        metadata["bands"] = { }

    if band not in metadata["bands"]:
        metadata["bands"][band] = { }

    if url != "":
        metadata["bands"][band]["url"] = url

    if folder != "":
        metadata["bands"][band]["folder"] = folder


def write_metadata(path, metadata):
    """Write the metadata to a given path."""
    
    if metadata == None:
        return
    
    metadata_path = get_metadata_path(path)
    if os.path.exists(metadata_path):
        with open( metadata_path, 'w') as metadata_file:
            metadata_file.write( json.dumps(metadata, indent=4) )