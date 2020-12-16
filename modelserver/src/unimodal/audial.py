import os
import subprocess

from google.cloud import speech

import ffmpeg



def run(url, queue, barrier):
    ''' Runnable function for an independent process.
        Writes transcription results to `queue`.
    '''
    speech_recognizer = SpeechRecognizer()

    if barrier is not None: # debug
        barrier.wait()

    speech_recognizer.transcribe_stream(url, queue)


class SpeechRecognizer:
    ''' Contains everything to transcribe an audio stream.

        `transcribe_stream()` is the main function, which starts a decoder
        ffmpeg subprocess and reads the decoded frames in fixed lengths.

        TODO:
          - Graceful shutdown: currently client.streaming_recognize() throws a
            timeout error if no requests are sent for a short period of time (a
            few seconds). Need to catch the error so that a new stream after a
            while will continue to transcribe.

          - Currently, the decoder subprocess is tightly coupled to the
            SpeechRecognizer. Is this unwise?

          - Google app credentials should be set by Docker.
    '''

    def __init__(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/workspace/modelserver/data/default-demo-app-c4bc3-b67061a9c4b1.json"
        self.client = speech.SpeechClient()

        print('SpeechRecognizer initialized')


    def transcribe_stream(self, url, queue=None, frame_size=8192, 
                          sampling_rate=44100, language_code='ko-KR'):
        ''' Main function of the class. Writes transcription outputs to `queue`.

            frame_size: of a single frame, in bytes
        '''
        ffmpeg_process = self.start_decoder_subprocess(url)
        # stream is a generator yielding chunks of audio data.
        stream = self.frame_generator(ffmpeg_process, frame_size)

        requests = (speech.StreamingRecognizeRequest(audio_content=chunk)
                    for chunk in stream) # generator expression, lazily created

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sampling_rate, # optimally should be 16000
            language_code=language_code)

        streaming_config = speech.StreamingRecognitionConfig(
                config=config, 
                interim_results=True)

        # streaming_recognize returns a generator.
        responses = self.client.streaming_recognize(streaming_config, requests)

        # throws an error if don't send audio for a long time; should catch it
        for response in responses:
            # Once the transcription has settled, the first result will contain
            # the is_final result. The other results will be for subsequent
            # portions of the audio. 
            # print(response)
            for result in response.results:
                # print('Finished: {}'.format(result.is_final))
                # print('Stability: {}'.format(result.stability))
                alternatives = result.alternatives

                # The alternatives are ordered from most likely to least.
                # -- UNCOMMENT TO DEBUG -- 
                # for alternative in alternatives:
                #     print('Confidence: {}'.format(alternative.confidence))
                #     print(u'Transcript: {}'.format(alternative.transcript))
                # ----

                final_result = {
                    'from': 'audio',
                    'transcript': alternatives[0].transcript,
                    'confidence': alternatives[0].confidence,
                    'is_final': result.is_final,
                    'stability': result.stability,
                }

                if queue:
                    queue.put(final_result)


    def start_decoder_subprocess(self, url):
        ''' Starts an ffmpeg process that decodes the RTMP stream, then pipes
            the output to stdout.

            Actually this should work on files as well.
        '''
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
            print("end of audio stream")
            return None

        return in_bytes


    def frame_generator(self, process, frame_size):
        ''' A generator method is any function with the keyword "yield".
            Calling generator method returns a generator object!
        '''
        frame = self.read_single_frame(process, frame_size)
        while frame is not None:
            yield frame
            frame = self.read_single_frame(process, frame_size)

