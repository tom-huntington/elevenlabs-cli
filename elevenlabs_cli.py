import asyncio
import os
from elevenlabs.client import AsyncElevenLabs
from elevenlabs import Voice, VoiceSettings, save
import subprocess
from typing import AsyncIterator # Iterator, Optional, Union, Optional, 
import argparse
from pathlib import Path
from nltk.tokenize import sent_tokenize
import re
import json
import hashlib
from lxml import etree
from more_itertools import bucket
from dotenv import load_dotenv

# async def print_models() -> None:
#     models = await client.models.get_all()
#     print(models)


def concatenate_audio_files(input_files : list[Path], lists_txt : Path, output_file, keep = True):
    with open(lists_txt, "w") as f:
        for i in input_files:
            if i is not None:
                f.write(f"file '{i.relative_to(lists_txt.parent)}'\n")
        list_file_path = f.name

    # assert False

    # ffmpeg_command = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', str(list_file_path), '-pass', '1', '-af', 'loudnorm=print_format=json', '-f', 'null', '/dev/null']
    # result = subprocess.run(ffmpeg_command, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, check=True)

    # pattern = r"\[Parsed_loudnorm_0 @ 0x[0-9a-f]+\]"
    # _, json_string = re.split(pattern, result.stderr, 1)
    # stats = json.loads(json_string)

    # af = f"loudnorm=linear=true:I=-23:TP=0:measured_I={stats['input_i']}:measured_LRA={stats['input_lra']}:measured_TP={stats['input_tp']}:measured_thresh={stats['input_thresh']}"
    ffmpeg_cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', str(list_file_path), '-acodec', 'aac', str(output_file)]
    print(subprocess.list2cmdline(ffmpeg_cmd))
    subprocess.run(ffmpeg_cmd)
    

    # if not keep:
    #     for f in (*input_files, list_file_path):
    #         os.remove(f)


async def stream_to_file(client: AsyncElevenLabs, chunk : tuple[list[str], list[str], Path, str], sample_rate, semaphore, output_format): # async_iter_bytes : AsyncIterator[bytes]):

    # key = hashlib.md5(text.encode('utf-8')).hexdigest()
    sentences, hashes, bufname, voice_id = chunk
    # new_name = bufname.with_name(key + bufname.suffix)
    # os.rename(bufname, new_name)
    # return bufname
    if bufname.exists():
        return voice_id, bufname

    text = " ".join(sentences)
    async with semaphore:
        response = client.text_to_speech.convert(
                voice_id=voice_id,
                model_id="eleven_turbo_v2",
                voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.75),
                # optimize_streaming_latency=optimize_streaming_latency,
                output_format=output_format,
                text=text,
                # request_options=request_options,
                # pronunciation_dictionary_locators=pronunciation_dictionary_locators
            )
        # audio = client.generate(
        #     text="Hello world",
        #     voice=Voice(
        #             voice_id='EXAVITQu4vr4xnSDxMaL',
        #             settings=VoiceSettings(stability=0.5, similarity_boost=0.75) # , style=0.0, use_speaker_boost=True)
        #     ),
        #     model="eleven_turbo_v2",
        #     output_format="pcm_16000",
        # )

        command = [
            'ffmpeg', '-y',
            '-f', 's16le', 
            '-ar', str(sample_rate),  # sample rate
            '-ac', '1',  # number of audio channels
            '-i', '-',  # The input comes from stdin
            '-acodec', 'copy',  # audio codec for M4A
            str(bufname),
        ]
        print(" ".join(command))

        process = subprocess.Popen(command, stdin=subprocess.PIPE)
    
        async for data in response:
            process.stdin.write(data)

        process.stdin.close()
        process.wait()
        print(f"Finished writing to {bufname}")
        return voice_id, bufname


def starts_with_list(list1, list2):
    if len(list1) < len(list2):
        return False
    return list1[:len(list2)] == list2

def chunk_strings(chunk_text, target_length, first_hashes, parent : Path, speaker):
    sentences = split_sentences(chunk_text) 
    hashes = [hash(s) for s in sentences] 

    def make_chunk(sentences, hashes_):
        voice_id = voices[speaker]
        key = "-".join((*hashes_, voice_id))
        bufname = parent / (key + '.wav')
        return (sentences, hashes_, bufname, voice_id)
        
    chunks = []
    current_chunk = []
    current_hash_chunk = []
    current_length = 0

    while(sentences):
        first_matches = first_hashes.get(hashes[0], None)
        if first_matches:
            if starts_with_list(hashes, first_matches):
                if current_chunk:
                    chunks.append(make_chunk(current_chunk, current_hash_chunk))
                    current_chunk = []
                    current_hash_chunk = []
                    current_length = 0

                chunks.append(make_chunk(sentences[:len(first_matches)], first_matches))
                sentences = sentences[len(first_matches):]
                hashes = hashes[len(first_matches):]
                continue
        
        string = sentences[0]
        if not current_chunk:
            current_chunk.append(string)
            current_hash_chunk.append(hashes[0])
            current_length += len(string)
        else:
            # Calculate the length if we add this string
            new_length = current_length + len(string)
            # Compare the difference from the target for current and new length
            current_diff = abs(current_length - target_length)
            new_diff = abs(new_length - target_length)
            # Decide whether to add the string to the current chunk
            if new_diff < current_diff:
                current_chunk.append(string)
                current_hash_chunk.append(hashes[0])
                current_length = new_length
            else:
                # Finalize the current chunk and start a new one
                chunks.append(make_chunk(current_chunk, current_hash_chunk))
                current_chunk = [string]
                current_hash_chunk = [hashes[0]]
                current_length = len(string)
        
        sentences = sentences[1:]
        hashes = hashes[1:]

    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(make_chunk(current_chunk, current_hash_chunk))

    return chunks



def split_sentences(text):
    return sent_tokenize(text)
    # pattern = r'(?<=[.!?])\s+'
    # parts = re.split(pattern, text)
    # parts = [part for part in parts if part]
    # return parts


def hash(s : str):
    return hashlib.shake_256(s.encode('utf-8')).hexdigest(4)
    # return [hashlib.shake_256(s.encode('utf-8')).hexdigest(4) for s in list_sentences]

async def main():
    parser = argparse.ArgumentParser(description="Eleven labs cli")

    # Add the subscription argument
    parser.add_argument(
        '--sub',
        type=str,
        choices=['Free', 'Starter', 'Creator', 'Pro', 'Scale'],
        help='Choose your subscription type',
        default='Creator'
    )
    parser.add_argument("input_file", type=str, help="The input text file")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    temp_path = Path(".temp") / input_path.stem
    temp_path.mkdir(parents=True, exist_ok=True)

    sample_rate, output_format, concurrency_limit = {
            'Free': (16000, "pcm_16000", 2),
            'Starter': (22050, "pcm_22050", 3),
            'Creator': (24000, "pcm_24000", 5),
            'Pro': (44100, "pcm_44100", 10),
            'Scale': (44100, "pcm_44100", 15),
        }[args.sub]

    
    load_dotenv()
    api_key = os.getenv("11LABS_API")
    client = AsyncElevenLabs(
      api_key=api_key,
    )
    global voices
    voices = {
            'default': os.getenv("11LABS_VOICE1"), 
            'two': os.getenv("11LABS_VOICE2") 
            }

    # with open('testm.txt', 'r') as f:
    with open(args.input_file, 'r') as f:
        text = f"<default>{f.read()}</default>"

        
    root : etree._Element = etree.fromstring(text)

    by_speaker = [(s, t) for child in root.iter()
                  for s, t in ((child.tag, child.text), ('default', child.tail))
                  if t]

    # sentences = split_sentences(text)
    


    # hashes = [hash(c) for c in chunked_sentences]
    # for list_h in hashes:
    #     open(f"test/{'-'.join(list_h)}.wav", 'a').close()
    def get_hashes_of_cached(cache_dir : Path):
        return [re.split('-', name.stem) for name in cache_dir.glob('*.wav')]
            
    hashes_of_cached = get_hashes_of_cached(temp_path)
    first_hashes = {hl[0]: hl for hl in hashes_of_cached}
    chunked_sentences = [chunk
                         for speaker, text in by_speaker
                         for chunk in chunk_strings(text, 300, first_hashes, temp_path, speaker)]

    assert temp_path.exists()
    with open(temp_path / 'mapping.txt', 'a') as f:
        f.write("------------------------------\n")
        for s, h, _, speaker in chunked_sentences:
            f.write(f"sp: {speaker}\n{h}\n{s}\n")
    
    with open(temp_path / "transcript.txt", 'w') as f:
        _t = "\n".join(text for _, text in by_speaker)
        f.write(_t)
        f.write("\n")

    # texts = [" ".join(c) for c in chunked_sentences]

    # if any(len(t) > 300 for t in texts):
    #     lengths = list(len(t) for t in texts)
    #     print(f"Max length: {max(lengths)}, All:")
    #     print(lengths)
    #     print("Do you want to continue? (y/n)")
    #     if 'y' != input():
    #         exit(0)
    

    semaphore = asyncio.Semaphore(concurrency_limit)

    tasks = [stream_to_file(client, c, sample_rate, semaphore, output_format) for c in chunked_sentences]
    vid_path : list[tuple[str, Path]] = await asyncio.gather(*tasks)

    print("Volume normalization")
    normed_wavs_dir = temp_path / 'normed'
    normed_wavs_dir.mkdir(exist_ok=True)
    bucket_ = bucket(vid_path, lambda t: t[0])
    for id in bucket_:
        id_paths : list[Path] = list(bucket_[id])
        voice_list_unnormed = temp_path / f"lists-{id}.txt"
        with open(voice_list_unnormed, 'w') as f:
            for _, p in id_paths:
                f.write(f"file '{p.relative_to(temp_path )}'\n")
        
        ffmpeg_command = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', str(voice_list_unnormed), '-pass', '1', '-af', 'loudnorm=print_format=json', '-f', 'null', '/dev/null']
        result = subprocess.run(ffmpeg_command, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, check=True)

        pattern = r"\[Parsed_loudnorm_0 @ 0x[0-9a-f]+\]"
        _, json_string = re.split(pattern, result.stderr, 1)
        stats = json.loads(json_string)

        af = f"loudnorm=linear=true:I=-23:TP=0:measured_I={stats['input_i']}:measured_LRA={stats['input_lra']}:measured_TP={stats['input_tp']}:measured_thresh={stats['input_thresh']}"

        for _, p in id_paths:
            ffmpeg_cmd = ['ffmpeg', '-y', '-i', str(p), '-af', af, '-acodec', 'pcm_s16le', '-ar','24000', str(normed_wavs_dir / p.name)]
            print(subprocess.list2cmdline(ffmpeg_cmd))
            subprocess.run(ffmpeg_cmd)

    normed_files = [normed_wavs_dir / path.name for _, path in vid_path ]

    concatenate_audio_files(normed_files, temp_path / "lists.txt", f"{input_path.stem}.m4a")

asyncio.run(main())
