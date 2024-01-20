# Copyright (c) 2024, Patricio Gonzalez Vivo
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (the "License"). 
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#


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

    if force_extension:
        if not is_video(path) or force_extension == "csv":
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