import logging
import time


import subprocess
from google.cloud import speech

import ffmpeg
import re

# Audio recording parameters
STREAMING_LIMIT = 240000  # 4 minutes
SAMPLE_RATE = 44100

# speech context - word classes
speech_context = speech.SpeechContext(phrases=['공','신발','숟가락','그릇','포크','버스','자전거','물고기','강아지','고양이','거울','칫솔','양말','선물','꽃','텔레비전'])


# CHUNK_SIZE = int(SAMPLE_RATE / 5)  # 200ms
CHUNK_SIZE = 8192  # 100ms





def run(url, queue, barrier):
    """start bidirectional streaming from microphone input to speech API"""

    speech_recognizer = ResumableSpeechRecognizer(url, SAMPLE_RATE, CHUNK_SIZE)

    if barrier is not None: # debug
        barrier.wait()
    
    with speech_recognizer as recognizer:
        recognizer.transcribe_stream(queue)



def get_current_time():
    """Return Current Time in MS."""

    return int(round(time.time() * 1000))


class ResumableSpeechRecognizer:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, url, rate, chunk_size):
        self.client = speech.SpeechClient()
        self._rate = rate
        self.chunk_size = chunk_size
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self.closed = True
        self.url = url
        
    def __enter__(self):
        self.closed = False
        self.ffmpeg_process = self.start_decoder_subprocess(self.url)
        return self

    def __exit__(self, type, value, traceback):
        self.closed = True
        self.ffmpeg_process.terminate()
        print("audial exit")
        
        
    def transcribe_stream (self, queue=None, language_code='ko-KR'):
        ''' Main function of the class. Writes transcription outputs to `queue`.

            frame_size: of a single frame, in bytes
        '''
        # ffmpeg_process = self.start_decoder_subprocess(url)
        
        while not self.closed:
            try:
                stream = self.frame_generator(self.ffmpeg_process, self.chunk_size)

                requests = (speech.StreamingRecognizeRequest(audio_content=chunk)
                            for chunk in stream)


                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self._rate, 
                    language_code=language_code,
                    speech_contexts=[speech_context],
                )

                streaming_config = speech.StreamingRecognitionConfig(
                        config=config, 
                        interim_results=True)

                responses = self.client.streaming_recognize(streaming_config, requests)


                self.listen_print_loop(responses, queue)

                if self.result_end_time > 0:
                    self.final_request_end_time = self.is_final_end_time
                    self.result_end_time = 0
                    self.last_audio_input = []
                    self.last_audio_input = self.audio_input
                    self.audio_input = []
                    self.restart_counter = self.restart_counter + 1

                    if not self.last_transcript_was_final:
                        continue
                    self.new_stream = True
            except Exception:
                print('end of stream')
                self.ffmpeg_process.terminate()
                self.closed = True
                return
                
                
        
        
    def start_decoder_subprocess(self, url):
        
        args = (
            ffmpeg
            .input(url)
            .output('pipe:', format='s16le', acodec='pcm_s16le')
            .compile()
        )
        
        return subprocess.Popen(args, stdout=subprocess.PIPE)


    def read_single_frame(self, process, frame_size):
        ''' Reads a single frame from the stdout associated with the process,
            and returns it in bytes form.

            Since process.stdout.read() is blocking, this function should
            theoretically not read "partial" frames that don't fill
            `frame_size`. But right now the timeout of read() is too long such
            that the "end of audio stream" never gets called. Therefore, the
            timeout of the streaming recognition API should be used to trigger
            graceful shutdown.

            Audio encoding: mono signed 16-bit PCM (little endian)
        '''
        in_bytes = process.stdout.read(frame_size)
        
        if len(in_bytes) < frame_size:
            return None

        return in_bytes


    def frame_generator(self, process, frame_size):
        ''' A generator method is any function with the keyword "yield".
            Calling generator method returns a generator object!
        '''
        
            
        while not self.closed:
            data = []

            if self.new_stream and self.last_audio_input:

                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)

                if chunk_time != 0:

                    if self.bridging_offset < 0:
                        self.bridging_offset = 0

                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time

                    chunks_from_ms = round(
                        (self.final_request_end_time - self.bridging_offset)
                        / chunk_time
                    )

                    self.bridging_offset = round(
                        (len(self.last_audio_input) - chunks_from_ms) * chunk_time
                    )

                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])
                    yield b"".join(data)
                    

                self.new_stream = False
                
            
            frame = self.read_single_frame(process, frame_size)
            self.audio_input.append(frame)
            
            while frame is not None:
                yield frame
                frame = self.read_single_frame(process, frame_size)
                self.audio_input.append(frame)
           
            
            



    def listen_print_loop(self, responses, queue):
        """Iterates through server responses and prints them.
        The responses passed is a generator that will block until a response
        is provided by the server.
        Each response may contain multiple results, and each result may contain
        multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
        print only the transcription for the top alternative of the top result.
        In this case, responses are provided for interim results as well. If the
        response is an interim one, print a line feed at the end of it, to allow
        the next result to overwrite it, until the response is a final one. For the
        final one, print a newline to preserve the finalized transcription.
        """
            
        for response in responses:

            if get_current_time() - self.start_time > STREAMING_LIMIT:
                self.start_time = get_current_time()
                break

            if not response.results:
                continue

            result = response.results[0]

            if not result.alternatives:
                continue
            
            transcript = result.alternatives[0].transcript
            confidence = result.alternatives[0].confidence
            

            result_seconds = 0
            result_micros = 0

            if result.result_end_time.seconds:
                result_seconds = result.result_end_time.seconds

            if result.result_end_time.microseconds:
                result_micros = result.result_end_time.microseconds

            self.result_end_time = int((result_seconds * 1000) + (result_micros / 1000))

            corrected_time = (
                self.result_end_time
                - self.bridging_offset
                + (STREAMING_LIMIT * self.restart_counter)
            )
            # Display interim results, but with a carriage return at the end of the
            # line, so subsequent lines will overwrite them.
            
            result_to_queue = {
                'from': 'audio',
                'transcript': transcript,
                'confidence': confidence,
                'time' : corrected_time,
                'is_final' : result.is_final
            }

            if result.is_final:

                if queue:
                    queue.put(result_to_queue)

                self.is_final_end_time = self.result_end_time
                self.last_transcript_was_final = True

                # Exit recognition if any of the transcribed phrases could be
                # one of our keywords.
#                 if re.search(r"\b(exit|quit)\b", transcript, re.I):
#                     print(YELLOW)
#                     print("Exiting...\n")
                    
#                     self.closed = True
#                     break
# 
            else:
                
                if queue:
                    queue.put(result_to_queue)
                

                self.last_transcript_was_final = False




