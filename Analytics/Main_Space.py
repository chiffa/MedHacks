__author__ = 'andrei'

from scipy.io import wavfile
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.fftpack import fft, ifft, rfft, irfft
from scipy.signal import blackman
import numpy as np
import os
import supporting_functions as SF
from itertools import izip


path = '/home/andrei/Documents/MedHacks'
# name = os.path.join(path, 'audio_1.wav')
name = os.path.join(path, 'audio_5_brent_chest_30s.wav')
# name = os.path.join(path, 'audio_7_katherine_fingers_30s.wav')


timeframe = 30.
cutoff = 110.
inner_cut = 2


def frame_to_freq(_frame):
    return _frame/float(frame)*rate


def freq_to_frame(_freq):
    return int(_freq/float(rate)*frame)


rate, data = wavfile.read(name)

frame = int(rate*timeframe)

print 'rate: %s Hz'%rate
print 'total duration: %s seconds'% (data.shape[0]/float(rate))
print 'capturing %s seconds %s frames' % (timeframe, frame)


x_time = np.linspace(0, timeframe, frame)
x_frequency = np.linspace(0, rate, frame/2)

data = gaussian_filter1d(data, sigma=10, axis=0)
diff = data[0:frame, 0] - data[0:frame, 1]


def filter_hum(array_to_clear):

    frame_fft = rfft(array_to_clear)
    frame_fft[freq_to_frame(110):] = 0

    plt.semilogy(x_frequency, 2.0/frame*np.abs(frame_fft[:frame/2]))
    plt.tight_layout()
    plt.show()

    refactored_array = irfft(frame_fft)
    refactored_array = np.real(refactored_array)

    return refactored_array


def chop_dataset(input_array):
    grad = np.gradient(input_array)


    poi1 = grad < -0.02
    local_minima1 = SF.local_min(grad)
    combined1 = np.logical_and(local_minima1, poi1)

    poi2 = input_array < -5
    local_minima2 = SF.local_min(input_array)
    combined2 = np.logical_and(local_minima2, poi2)

    brps1 = np.nonzero(combined1)[0]
    brps1 = [x_time[brp] for brp in brps1.tolist() if x_time[brp]>inner_cut and x_time[brp]<timeframe-inner_cut]

    brps2 = np.nonzero(combined2)[0]
    brps2 = [x_time[brp] for brp in brps2.tolist() if x_time[brp]>inner_cut and x_time[brp]<timeframe-inner_cut]

    brps1 = np.array(brps1)
    brps2 = np.array(brps2)

    retained_pairs = []
    for brp in brps1:
        candidate_list = brps2[np.logical_and(brps2 > brp, brps2 < brp+0.1)]
        if len(candidate_list) > 0:
            retained_pairs.append(candidate_list[0])

    plt.plot(x_time[inner_cut*rate:-inner_cut*rate], input_array[inner_cut*rate:-inner_cut*rate])
    SF.show_breakpoints(retained_pairs, 'k')
    plt.show()

    return retained_pairs


def fold_line(input_array, chop_points):
    delay_times = 60/(np.array(chop_points)[1:] - np.array(chop_points)[:-1])
    print 'pulse: %s bpm, std: %s bpm' % (np.mean(delay_times), np.std(delay_times, ddof=1))

    lane_bank = []
    l_offsets = []
    l_max = 0
    r_max = 0
    for i in range(1, len(chop_points)-1):
        lane_bank.append(input_array[int(chop_points[i-1]*rate):int(chop_points[i+1]*rate)])
        l_offset = chop_points[i] - chop_points[i-1]
        l_offsets.append(l_offset)

        if chop_points[i+1] - chop_points[i] > r_max:
            r_max = chop_points[i+1] - chop_points[i]
        if l_offset > l_max:
            l_max = l_offset

    # print lane_bank

    support_array = np.zeros((len(lane_bank), int((l_max+r_max)*rate)))
    for i, (l_offset, array) in enumerate(izip(l_offsets, lane_bank)):
        print array.shape[0]
        print array
        print int((l_max-l_offset)*rate), int((l_max-l_offset)*rate)+array.shape[0]
        support_array[i, int((l_max-l_offset)*rate) : int((l_max-l_offset)*rate)+array.shape[0]] = array

    # norm_array = -support_array[:, int(l_max*rate)]
    # print norm_array.shape
    # print norm_array
    # print(support_array.shape)
    # support_array /= norm_array[:, np.newaxis]

    x_range = np.linspace(-l_max, r_max, int((l_max+r_max)*rate))
    for i in range(0, support_array.shape[0]):
        plt.plot(x_range, support_array[i])
    plt.show()

    plt.plot(x_range, np.average(support_array, axis=0))
    plt.show()

    return np.average(support_array, axis=0), support_array


def profile_distance():
    pass


def plot(input_array):
    grad = np.gradient(input_array)

    plt.plot(x_time[inner_cut*rate:-inner_cut*rate], input_array[inner_cut*rate:-inner_cut*rate])
    plt.axhline(0, color='r')
    plt.show()

    plt.plot(input_array[inner_cut*rate:-inner_cut*rate], grad[inner_cut*rate:-inner_cut*rate])
    plt.show()


if __name__ == '__main__':
    new_time_r = filter_hum(diff)
    plot(new_time_r)
    chop_points = chop_dataset(new_time_r)
    average, ref_set = fold_line(new_time_r, chop_points)

    # TODO: sort data to see which one is the peak side

