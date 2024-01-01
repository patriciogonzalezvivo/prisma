import os
import json
from .io import check_overwrite

META_FILE = "payload.json"

def get_metadata_path(path):
    if os.path.isfile( path ):
        if path.endswith('.json'):
            return path 
        else:
            return get_metadata_path( os.path.dirname(path) )
        
    if os.path.isdir( path ):
        return os.path.join(path, META_FILE)
    
    return None


def load_metadata(path):
    metadata_path = get_metadata_path(path)
    if os.path.exists(metadata_path):
        return json.load( open(metadata_path) )
    return None


def create_metadata(path):
    if not os.path.exists(path):
        os.makedirs(path)
        write_metadata(path, { "bands": { } })
    return load_metadata(path)


def is_video(path):
    return path.endswith('.mp4')


def get_target(path, metadata, band="rgba", target="", force_image_extension=None):
    if os.path.isdir( target ):
        input_folder = target
    else:
        input_folder = os.path.dirname(path)

    input_filename = os.path.basename(path)
    input_extension = input_filename.rsplit( ".", 1 )[ 1 ]

    if force_image_extension and not is_video(path):
        input_extension = force_image_extension

    target_filename = band + "." + input_extension

    if target == "":
        target = os.path.join(input_folder, target_filename)

    if metadata:
        add_band(metadata, band, url=target_filename)

    return target


def get_url(path, metadata, band):
    if os.path.isdir( path ):
        if metadata:
            if "bands" in metadata:
                if band in metadata["bands"]:
                    if "url" in metadata["bands"][band]:
                        return os.path.join(path, metadata["bands"][band]["url"])
    return path


def add_band(metadata, band, url="", folder=""):
    if "bands" not in metadata:
        metadata["bands"] = { }

    if band not in metadata["bands"]:
        metadata["bands"][band] = { }

    if url != "":
        metadata["bands"][band]["url"] = url

    if folder != "":
        metadata["bands"][band]["folder"] = folder


def write_metadata(path, metadata):

    if os.path.isdir( path ) and metadata is not None:
        metadata_path = os.path.join( path, META_FILE)

        with open( metadata_path, 'w') as metadata_file:
            metadata_file.write( json.dumps(metadata, indent=4) )