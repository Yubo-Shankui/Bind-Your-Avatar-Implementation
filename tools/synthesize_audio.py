import argparse
import subprocess
import os

def merge_audio_files(input_file1, input_file2, output_file):
    """
    Merge two WAV audio files using ffmpeg.
    The merged audio will contain sounds from both original audio files.
    If the input audio durations are different, the output audio duration will match the shorter one.

    Parameters:
    input_file1 (str): Path to the first input WAV file.
    input_file2 (str): Path to the second input WAV file.
    output_file (str): Path to the output WAV file.
    """
    if not os.path.exists(input_file1):
        print(f"Error: Input file {input_file1} does not exist.")
        return
    if not os.path.exists(input_file2):
        print(f"Error: Input file {input_file2} does not exist.")
        return

    command = [
        'ffmpeg',
        '-i', input_file1,
        '-i', input_file2,
        '-filter_complex', '[0:a][1:a]amix=inputs=2:duration=shortest[aout]',
        '-map', '[aout]',
        '-y',
        output_file
    ]

    try:
        print(f"Executing ffmpeg command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Audio files successfully merged to {output_file}")
        if result.stdout:
            print("ffmpeg output:")
            print(result.stdout)
        if result.stderr:
            print("ffmpeg error output (may contain warnings or detailed information):")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg command execution failed. Return code: {e.returncode}")
        print("ffmpeg output (stdout):")
        print(e.stdout)
        print("ffmpeg error output (stderr):")
        print(e.stderr)
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please ensure ffmpeg is installed and added to your system PATH.")
    except Exception as e:
        print(f"Unknown error occurred while merging audio: {e}")

if __name__ == "__main__":
    input_file1 = "datasets/benchmark/step1/audios/left_audio/new6.wav"
    input_file2 = "datasets/benchmark/step1/audios/right_audio/new6.wav"
    output_file = "demo_examples/audio/6.wav"

    merge_audio_files(input_file1, input_file2, output_file)

    # print("\nUsage example:")
    # print(f"python tools/synthesize_audio.py input1.wav input2.wav output.wav")
