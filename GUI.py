import tkinter
from tkinter import *
import threading
from tkinter import filedialog
from project import equalizer
# from Global_var import now_playing
# import librosa
import logging


## Eqaulizer with GUI by Kang Keehyuk, Oh Myungchan, Son junyung, Huh Yoon ##
##이퀄라이저는 필터링 처리된 값을 파일로 저장하는 동시에 재생을 하게된다. 그러나 이것이 필터링하는 시간이 조금 오래 걸리므로 import 후
##파일 재생을 누르더라도 실제로 소리가 재생되기까지 시간이 걸리므로 조금 기다리면 된다.
##Equalizer takes some time to play and save the wav file. Since we haven't used much modules and have implemented
##the FFT and the IFFT and the Filters are made as a freqeuncy response of the system so that we could convolve by multiplication
#Enjoy !

file_path = ''
now_playing = False

#Import of files
def file_import():
    try:
        global file_path
        file_path = filedialog.askopenfilename(filetypes=[('Audio Files', '*.wav')])
        file_name_label.config(text=file_path)
    except ValueError:
        pass


def get_slider_values():
    #it puts the value of the sliders in a array
    slider_array = [Value63hz.get(), Value1khz.get(), Value4khz.get(), Value16khz.get()]
    print("get slider value")
    if file_path and now_playing:
        print("slider value changed")
        #used threading to handle the filters
        audio_thread = threading.Thread(target=equalizer, args=(file_path, slider_array[0], slider_array[1],
                                                                slider_array[2], slider_array[3], now_playing), daemon=True)
        audio_thread.start()

def press_button_play():
    global now_playing

    if not now_playing:
        now_playing = True
        get_slider_values()


def press_button_stop():
    global now_playing

    if now_playing:
        now_playing = False
        get_slider_values()

def press_button_change():
    if now_playing:
        get_slider_values()

root = Tk()
root.title("Equalizer")
# Initiation of Frames
slider_frames = Frame(root)
# slider_frames.text = Text(root, width=10, height=10)
# slider_frames.text.insert('1.0', 'Equalizer with realtime adaption')
# slider_frames.text.pack()
slider_frames.pack(padx=20, pady=20)
slider_frames.pack(side=TOP)
slider1_frame = Frame(slider_frames)
slider1_frame.pack(side=LEFT)
slider2_frame = Frame(slider_frames)
slider2_frame.pack(side=LEFT)
slider3_frame = Frame(slider_frames)
slider3_frame.pack(side=LEFT)
slider4_frame = Frame(slider_frames)
slider4_frame.pack(side=LEFT)
change_button = Frame(slider_frames)
change_button.pack(padx=20, pady=30)

import_frame = Frame(root)
import_frame.pack(padx=10, pady=10)
import_frame.pack(side=BOTTOM)

# Variable creation

Value63hz = DoubleVar()
Value1khz = DoubleVar()
Value4khz = DoubleVar()
Value16khz = DoubleVar()

# All slider_frames widgets

# Slider 1
w = Scale(slider1_frame, from_=12, to=-12, variable=Value63hz)  ### slider label =45hz
w.set(0)
w.pack(side=TOP)
# Label for slider 1
label_63hz = Label(slider1_frame, text='63Hz')
label_63hz.pack(side=TOP)

# Slider 2
w2 = Scale(slider2_frame, from_=12, to=-12, variable=Value1khz)  # slider label = 700hz
w2.set(0)
w2.pack(side=TOP)
# Label for slider 2
label_250hz = Label(slider2_frame, text='250Hz')
label_250hz.pack(side=TOP)

# Slider 3
w3 = Scale(slider3_frame, from_=12, to=-12, variable=Value4khz)  # slider label = 2khz
w3.set(0)
w3.pack(side=TOP)
# Label for slider 3
label_1khz = Label(slider3_frame, text='1kHz')
label_1khz.pack(side=TOP)

# Slider 4
w4 = Scale(slider4_frame, from_=12, to=-12, variable=Value16khz)  # slider label = 8k
w4.set(0)
w4.pack(side=TOP)
# Label for slider 4
label_4khz = Label(slider4_frame, text='4kHz')
label_4khz.pack(side=TOP)

#slider value update button
audio_update_button = Button(change_button, text="Update\nchange", command=press_button_change)
audio_update_button.pack()

# Audio Import Button
audio_button = Button(import_frame, text='Import Music', command=file_import)
audio_button.pack(side=LEFT)
file_name_label = Label(import_frame, text='No File Chosen.')
file_name_label.pack(side=LEFT)

# Audio Play button

audio_play_button = Button(root, text="SAVE & PLAY", command=press_button_play)
audio_play_button.pack(padx=20, pady=10)

audio_stop_button = Button(root, text="STOP", command=press_button_stop)
audio_stop_button.pack(padx=20, pady=10)




mainloop()
