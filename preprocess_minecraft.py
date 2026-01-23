from array_record.python import array_record_module
import glob
import os
import subprocess
import numpy as np
import json
import tarfile
import tempfile
import multiprocessing
import io
import pickle

# Configuration
INPUT_TARS = "/scratch/vpt/*.tar"
OUTPUT_DIR = "/scratch/vpt_arrayrecord"
CHUNK_FRAMES = 256
FPS = 20

def process_tar_shard(tar_path, output_filename):
    """Reads a single TAR, chunks all videos inside, writes one ArrayRecord file."""
    
    writer = array_record_module.ArrayRecordWriter(output_filename, 'group_size:1')
    
    try:
        # Stream the tar file
        with tarfile.open(tar_path, "r|*") as tar:
            
            # We need to pair .mp4 and .jsonl files. 
            # WebDataset tars usually group them by filename key.
            # We'll use a simple buffer to hold one until the other arrives.
            buffer = {} # key -> { 'mp4': bytes, 'jsonl': bytes }
            
            for member in tar:
                if not member.isfile(): continue
                
                # Extract file key (e.g. "data/video_001")
                fname = os.path.basename(member.name)
                key = os.path.splitext(fname)[0]
                ext = os.path.splitext(fname)[1]
                
                if key not in buffer: buffer[key] = {}
                
                # Read file content into memory (Files are usually <500MB, fine for RAM)
                f_obj = tar.extractfile(member)
                if f_obj:
                    content = f_obj.read()
                    buffer[key][ext] = content
                
                # If we have both parts, process immediately
                if '.mp4' in buffer[key] and '.jsonl' in buffer[key]:
                    _process_pair(key, buffer[key]['.mp4'], buffer[key]['.jsonl'], writer)
                    del buffer[key] # Free RAM

    except Exception as e:
        print(f"Error processing {tar_path}: {e}")
    finally:
        writer.close()

def _process_pair(key, mp4_bytes, jsonl_bytes, writer):
    """Chunks a single video/action pair and writes to ArrayRecord"""
    
    # 1. Parse Actions
    actions = []
    try:
        json_str = jsonl_bytes.decode('utf-8')
        for line in json_str.strip().split('\n'):
            actions.append(json.loads(line))
    except:
        print(f"Skipping corrupt jsonl: {key}")
        return

    total_frames = len(actions)
    
    # 2. Write MP4 to Temp (FFMPEG needs a file to seek efficiently)
    # We use /dev/shm for speed if available (RAM disk), else /tmp
    temp_dir = '/dev/shm' if os.path.exists('/dev/shm') else None
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=temp_dir) as tmp_vid:
        tmp_vid.write(mp4_bytes)
        tmp_vid.flush()
        
        # 3. Iterate & Chunk
        for start_frame in range(0, total_frames - CHUNK_FRAMES, CHUNK_FRAMES):
            # A. Slice Actions
            action_chunk = actions[start_frame : start_frame + CHUNK_FRAMES]
            # Convert to numpy/bytes (customize your serialization here)
            # Assuming you want raw list of dicts pickled for flexibility, or numpy for speed
            # Using pickle for safety with the VPT dict structure:
            action_bytes = pickle.dumps(action_chunk)
            
            # B. Slice Video (Re-encode)
            start_time = start_frame / FPS
            duration = CHUNK_FRAMES / FPS
            
            # FFMPEG piping: We pipe the output bytes directly to Python to avoid 2nd temp file
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start_time:.3f}",   # Seek start
                "-i", tmp_vid.name,           # Input temp file
                "-t", f"{duration:.3f}",      # Duration
                "-c:v", "libx264",            # Output codec
                "-preset", "ultrafast",       # Speed over compression ratio
                "-f", "mp4",                  # Container
                "-movflags", "frag_keyframe+empty_moov", # Critical for streaming mp4 bytes
                "pipe:1"                      # Output to stdout
            ]
            
            try:
                # Run ffmpeg and capture stdout
                process = subprocess.run(cmd, capture_output=True, check=True)
                video_chunk_bytes = process.stdout
                
                # C. Write Record
                record_struct = {
                    'video': video_chunk_bytes,
                    'actions': action_bytes,
                    'key': f"{key}_{start_frame}" # Helpful for debugging
                }
                writer.write(pickle.dumps(record_struct))
                
            except subprocess.CalledProcessError:
                # FFMPEG failed (maybe end of file issues), skip chunk
                pass

def worker(args):
    tar_path, output_path = args
    if os.path.exists(output_path): return # Skip existing
    process_tar_shard(tar_path, output_path)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    all_tars = sorted(glob.glob(INPUT_TARS))
    tasks = []
    
    for i, tar_path in enumerate(all_tars):
        # file-001.tar -> file-001.array_record
        basename = os.path.basename(tar_path).replace('.tar', '.array_record')
        out_path = os.path.join(OUTPUT_DIR, basename)
        tasks.append((tar_path, out_path))
        
    print(f"Found {len(tasks)} shards to process.")
    
    # Use standard multiprocessing
    # Don't use too many workers! FFMPEG spawns threads too.
    # Start with cpu_count / 2
    num_workers = max(1, multiprocessing.cpu_count() // 2)
    
    with multiprocessing.Pool(num_workers) as pool:
        # Use imap_unordered to see progress
        for _ in pool.imap_unordered(worker, tasks):
            print(".", end="", flush=True)

if __name__ == "__main__":
    main()