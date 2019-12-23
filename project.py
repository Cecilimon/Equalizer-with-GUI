import librosa
import numpy as np
import scipy.stats
import scipy.io.wavfile as wav
import sounddevice as sd
import logging

now_playing = False
## Most of the comments are in Korean special credits to Huh yoon

## Equalizing process managing function
def equalizer(file_path, sl_1, sl_2, sl_3, sl_4,play_bool):
    if play_bool is True:
        audio, sr = librosa.load(file_path, sr=16000)
        audio_seg = short_time_analysis(audio)
        logging.info("Threading comes to equalizer")
        audio_dft = fft(audio_seg)
        dft_mag, dft_phase = separation(audio_dft)
        modified_mag = filter_combination(dft_mag, sl_1, sl_2, sl_3, sl_4)
        modified_dft = combination(modified_mag, dft_phase)
        modified_seg = ifft(modified_dft)
        modified_audio = short_time_synthesis(modified_seg)
        wav.write('output_file.wav', sr, modified_audio.astype(np.float32))
        global now_playing
        now_playing = play_bool
        play_audio(modified_audio, sr)
    else:
        sd.stop()


#Audio playing by using the sound device module
#Allows you to play audio from an input array.
def play_audio(play_song,sr):
    print("came to play_audio")
    #used sound devices to play array
    logging.info("Threading comes to play_audio")
    if now_playing is True:
        sd.play(play_song, sr)
        sd.wait()
    else:
        sd.stop()



#short time analysis


def short_time_analysis(x, win_len=320, win_type='rectangular'):
    """
    Parameters:
        x (int or float): 1D ndarray
        win_len (int): Lenght of each short-time frame
        win_type (string): Type of window function to be used
    Returns:
        x_seg (int or float): 2D ndarray
    """

    # Assume 50% overlap
    overlap_len = win_len // 2
    shift_len = win_len - overlap_len

    if win_type == 'rectangular':
        window = np.ones(
            win_len)  # 만약 win_type이 rectangular이면 win_len의 길이를 가지고 모든 원소가 1인 array를 생성해준다. 결국 x_seg를 결정하는 데 있어서 곱해주는 window는 rectangular일 때는 가중치가 없다.
    elif win_type == 'hanning':
        window = np.hanning(
            win_len)  # 만약 win_type이 hanning이면 win_leg의 길이를 가지는 han functuon이 window array로 설정되고, win_type을 hanning으로 했을 때는 x_seg를 결정하는 데 있어서 가중치가 있게 된다.
    else:
        raise Exception("Wrong window type")

    sig_len = len(x)  # sig_len을 input의 길이로 지정해준다.
    if sig_len % win_len != 0:  # sin_len를 win_len로 나눈 나머지가 0이 아닐 때
        x_padded = np.pad(x, (0, win_len - (sig_len % win_len)),
                          'constant')  # sig_len(len(x))가 win_len로 딱 나눠떨어지지 않기 때문에 길이를 맞춰주기 위해 win_len에서 나머지를 빼준 개수만큼 0을 x에 뒷부분에 추가한다.
    else:
        x_padded = x  # sig_len를 win_len로 나눈 나머지가 0이면 input은 그대로 x_padded가 된다.
    x_padded = np.pad(x_padded, (win_len, win_len),
                      'constant')  # convolution함에 있어서 alaising을 방지하기 위해 x_padded array의 앞쪽 끝과 뒤쪽 끝 부분에 win_len 길이의 0을 각각 삽입한다.

    num_frames = (len(x_padded) - win_len) // shift_len + 1  # input을 구간별로 나눠줄 때 그 구간의 개수가 num_frames가 된다.
    x_seg = np.zeros((num_frames, win_len))  # x_seg를 numframes X win_len 크기의 0행렬로 만들어준다.
    for k in range(num_frames):
        x_seg[k] = x_padded[
                   k * shift_len:k * shift_len + win_len] * window  # window를 shift하며 x와 각각 곱하여 x_seg의 각 행에 저장한다.

    return x_seg

# audio_seg = short_time_analysis(audio)

#dft

def fft(x_seg):
    num_frames, win_len = x_seg.shape # num_frames는 x_seg의 행의 개수, win_len은 x_seg의 열의 개수로 지정해준다.
    size = win_len // 2 # size는 win_len의 반으로 지정해준다.
    fft = np.zeros((num_frames, win_len), dtype = "complex_") # fft를 numframes X win_len 크기의 0행렬로 만들어준다.
    x_seg_even = np.zeros((size, num_frames), dtype = "complex_") # x_seg_even을 size X num_frames 크기의 0행렬로 만들어준다.
    x_seg_odd = np.zeros((size, num_frames), dtype = "complex_") # x_seg_odd를 size X num_frames 크기의 0행렬로 만들어준다.
    multiplier = np.zeros((size, size), dtype = "complex_") # multiplier를 size X size 크기의 0행렬로 만들어준다.
    w_0 = np.complex(np.exp(-1j*2*np.pi/(win_len))) # multiplier를 만들기위한 w_0를 계산한다.
    for i in range(num_frames):
        for j in range(size):
            x_seg_even[j][i] = x_seg[i][2*j] # x_seg_even은 x_seg의 짝수 자리에 있는 숫자들의 행렬을 transpose한 행렬이다.
    for i in range(num_frames):
        for j in range(size):
            x_seg_odd[j][i] = x_seg[i][2*j+1] # x_seg_odd는 x_seg의 홀수 자리에 있는 숫자들의 행렬을 transpose한 행렬이다.
    for i in range(size):
        for j in range(size):
            multiplier[i][j] = w_0 ** (2*i*j) # fft 연산을 위한 multiplier 행렬을 계산한다.
    x_seg_even_r = np.zeros((size, num_frames), dtype = "complex_") # multiplier와 x_seg_even을 행렬곱한 행렬을 저장한다.
    x_seg_odd_r = np.zeros((size, num_frames), dtype = "complex_") # multiplier와 x_seg_odd를 행렬곱한 행렬을 저장한다.
    x_seg_even_r = np.dot(multiplier, x_seg_even)
    x_seg_odd_r = np.dot(multiplier, x_seg_odd)
    for i in range(num_frames): # x_seg_even_r과 x_seg_odd_r을 이용하여 fft 행렬을 계산한다.
        for j in range(size):
            fft[i][j] = x_seg_even_r[j][i] + (w_0**j)*x_seg_odd_r[j][i]
            fft[i][j+size] = x_seg_even_r[j][i] - (w_0**j)*x_seg_odd_r[j][i]
    return fft

# audio_dft = dft(audio_seg)

#separation

def separation(fft):
    num_frames, win_len = fft.shape  # num_frames는 fft의 행의 개수, win_len은 fft의 열의 개수로 지정해준다.
    dft_mag = np.zeros((num_frames, win_len))  # dft_mag를 numframes X win_len 크기의 0행렬로 만들어준다.
    dft_phase = np.zeros((num_frames, win_len))  # dft_phase를 numframes X win_len 크기의 0행렬로 만들어준다.
    for i in range(num_frames):
        for j in range(win_len):
            a = np.real(fft[i][j])  # fft[i][j]의 행렬에서의 실수값들을 a로 지정한다.
            b = np.imag(fft[i][j])  # fft[i][j]의 행렬에서의 허수값들을 b로 지정한다.
            dft_mag[i][j] = np.sqrt(a ** 2 + b ** 2)  # 복소수에서의 magnitude는 실수값과 허수값의 제곱의 합을 루트를 씌운 것이다.
            dft_phase[i][j] = np.arctan(b / a)  # 복소수에서의 phase는 허수값을 실수값으로 나눈 것의 탄젠트 역함수를 씌운 것이다.
    return dft_mag, dft_phase  # filtering을 해줄 때에 phase는 그대로, magnitude만 filtering을 해주기 위해 magnitude response와 phase response를 나눠준다.


# dft_mag, dft_phase = separation(audio_dft)


def hpf(dft_mag, slider_val):
    num_frames, size = dft_mag.shape
    x = np.linspace(0, 319, 320)
    y = scipy.stats.norm.pdf(x, 160, 35) # 129-160-191
    #슬라이더 값을 너무 크게 늘어나지 않도록 적절하게 조절해둔 것
    mul_with_dft = 2**(slider_val/6)*90*y
    #필터를 위한 배열
    y_filtered = np.zeros((num_frames, size), dtype="complex_")
    #필터와 dft_mag를 convolution해주는 과정
    for i in range(num_frames):
        y_filtered[i] = dft_mag[i] * mul_with_dft
    return y_filtered


def lpf(dft_mag, slider_val) :
    num_frames, size = dft_mag.shape
    x1 = np.linspace(0, 160, 161)
    y1 = scipy.stats.norm.pdf(x1, 1, 10) # 0-1-2
    y2 = y1.tolist()
    y2 = y2[1:160]
    y2.reverse()
    y2 = np.array(y2)
    y_f = np.concatenate([y1, y2])
    mul_with_dft = 2**(slider_val/6)*20* y_f
    y_filtered = np.zeros((num_frames, size), dtype="complex_")
    for i in range(num_frames):
        y_filtered[i] = dft_mag[i] * mul_with_dft
    return y_filtered

def bpf_1(dft_mag, slider_val) :
    num_frames, size = dft_mag.shape
    x1 = np.linspace(0, 160, 161)
    y1 = scipy.stats.norm.pdf(x1, 30, 15) # 9-20-32
    y2 = y1.tolist()
    y2 = y2[1:160]
    y2.reverse()
    y2 = np.array(y2)
    y_f = np.concatenate([y1, y2])
    mul_with_dft = 2**(slider_val/6)*30*y_f
    y_filtered = np.zeros((num_frames, size), dtype="complex_")
    for i in range(num_frames):
        y_filtered[i] = dft_mag[i] * mul_with_dft
    return y_filtered


def bpf_2(dft_mag,slider_val) :
    num_frames, size = dft_mag.shape
    x1 = np.linspace(0, 160, 161)
    y1 = scipy.stats.norm.pdf(x1, 80, 30) # 33-80-110
    y2 = y1.tolist()
    y2 = y2[1:160]
    y2.reverse()
    y2 = np.array(y2)
    y_f = np.concatenate([y1, y2])
    mul_with_dft = 2**(slider_val/6)*75*y_f
    y_filtered = np.zeros((num_frames, size), dtype="complex_")
    for i in range(num_frames):
        y_filtered[i] = dft_mag[i] * mul_with_dft
    return y_filtered


#combinding the filters

def filter_combination(dft_mag, sl_1, sl_2, sl_3, sl_4): # combinding the filters
    num_frames, win_len = dft_mag.shape
    modified_mag = np.zeros((num_frames, win_len), dtype = "complex_")
    modified_mag = lpf(dft_mag, sl_1) + bpf_1(dft_mag, sl_2) \
                   + bpf_2(dft_mag,sl_3) + hpf(dft_mag, sl_4)
    return modified_mag

# combination

def combination(dft_mag, dft_phase):
    num_frames, win_len = dft_mag.shape # num_frames는 dft_mag의 행의 개수, win_len은 dft_mag의 열의 개수로 지정해준다.
    dft = np.zeros((num_frames, win_len), dtype = "complex_") # dft를 numframes X win_len 크기의 0행렬로 만들어준다.
    for i in range(num_frames):
        for j in range(win_len):
            a = dft_mag[i][j]
            b = dft_phase[i][j]
            dft[i][j] = a * np.exp(1j*b) # 복소수의 magnitude와 phase값으로 복소수를 표현할 때, magnitude*exp(j*phase)로 표현할 수 있다.
    return dft


# modified_dft = combination(modified_mag, modified_phase)

#Inverse Discrete transform of filtered signal



def ifft(fft):
    num_frames, win_len = fft.shape # num_frames는 fft의 행의 개수, win_len은 fft의 열의 개수로 지정해준다.
    size = win_len // 2 # size는 win_len의 반으로 지정해준다.
    y_seg = np.zeros((num_frames, win_len)) # y_seg를 numframes X win_len 크기의 0행렬로 만들어준다.
    multiplier = np.zeros((size, size), dtype = "complex_") # multiplier를 size X size 크기의 0행렬로 만들어준다.
    w_0 = np.complex(np.exp(-1j*2*np.pi/(win_len))) # multiplier를 만들기위한 w_0를 계산한다.
    for i in range(size):
        for j in range(size):
            multiplier[i][j] = w_0 ** (2*i*j) # fft 연산을 위한 multiplier 행렬을 계산한다.
    multiplier_i = np.zeros((size, size), dtype = "complex_") # multiplier_i를 size X size 크기의 0행렬로 만들어준다.
    multiplier_i = np.linalg.inv(multiplier) # multiplier_i는 multiplier의 역행렬이다.
    y_seg_even_r = np.zeros((size, num_frames), dtype = "complex_") # x_seg_even_r을 size X num_frames 크기의 0행렬로 만들어준다.
    y_seg_odd_r = np.zeros((size, num_frames), dtype = "complex_") # x_seg_odd_r을 size X num_frames 크기의 0행렬로 만들어준다.
    for i in range(num_frames):
        for j in range(size):
            y_seg_even_r[j][i] = (fft[i][j] + fft[i][j+size]) / 2
            y_seg_odd_r[j][i] = (fft[i][j] - fft[i][j+size]) / 2 / (w_0**j)
    y_seg_even = np.zeros((size, num_frames), dtype = "complex_") # x_seg_even을 size X num_frames 크기의 0행렬로 만들어준다.
    y_seg_odd = np.zeros((size, num_frames), dtype = "complex_") # x_seg_odd을 size X num_frames 크기의 0행렬로 만들어준다.
    y_seg_even = np.dot(multiplier_i, y_seg_even_r) # multiplier_i와 x_seg_even_r을 행렬곱한 행렬을 저장한다.
    y_seg_odd = np.dot(multiplier_i, y_seg_odd_r) # multiplier_i와 x_seg_odd_r을 행렬곱한 행렬을 저장한다.
    for i in range(num_frames): # x_seg_even과 x_seg_odd을 이용하여 y_seg 행렬을 계산한다.
        for j in range(size):
            y_seg[i][2*j] = y_seg_even[j][i]
            y_seg[i][2*j+1] = y_seg_odd[j][i]
    return y_seg

# modified_seg = idft(modified_dft)

# synthesis
def short_time_synthesis(y_seg, win_type='rectangular'):
    """
    Parameters:
        y_seg (int or float): 2D ndarray
        win_type (string): Type of applied window function
    Returns:
        y (int or float): 1D ndarray
    """

    # Assume 50% overlap
    num_frames, win_len = y_seg.shape  # num_frames는 y_seg의 행의 개수, win_len는 y_seg의 열의 개수로 지정해준다.
    overlap_len = win_len // 2
    shift_len = win_len - overlap_len

    if win_type not in ['rectangular', 'hanning']:
        raise Exception("Wrong window type")

    sig_len = (num_frames - 1) * shift_len + win_len
    y = np.zeros(sig_len)  # sig_len의 길이를 가지고 모든 원소가 0인 array y를 생성한다.
    for k in range(num_frames):
        y[k * shift_len:k * shift_len + win_len] += y_seg[k]  # array y를 y_seg를 이용하여 계산한다.

    y = y[win_len:-win_len]  # 처음에 padding한 양 끝에서 win_len만큼의 0을 없애준다.
    if win_type == 'rectangular': y /= 2  # short_time_analysis에서 shift시키면서 x_seg array를 생성하는 과정에서 항들이 두 번 씩 들어가게 되기 때문에 역으로 변환할 때는 2로 나눠주어야 한다.

    return y
