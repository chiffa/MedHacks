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
from pickle import dump, load


path = '/home/andrei/Documents/MedHacks'
# base_name = 'audio_24_anna_neck_30s.wav'
# # name = os.path.join(path, 'audio_5_brent_chest_30s.wav')
# name = os.path.join(path, 'audio_19_andrei_neck.wav')
# name = os.path.join(path, 'audio_22_brent_chest_3min.wav')
# name = os.path.join(path, 'audio_26_suyi_neck_30s.wav')
# name = os.path.join(path, 'audio_24_anna_neck_30s.wav')
name = os.path.join(path, 'audio_27_katherine_neck_30s.wav')

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

diff = -diff

def filter_hum(array_to_clear):

    frame_fft = rfft(array_to_clear)
    frame_fft[freq_to_frame(110):] = 0

    # plt.semilogy(x_frequency, 2.0/frame*np.abs(frame_fft[:frame/2]), color='k')
    # plt.xlim([0, 250])
    # plt.show()

    refactored_array = irfft(frame_fft)
    refactored_array = np.real(refactored_array)

    return refactored_array


def chop_dataset(input_array, collapse=0.0):
    grad = np.gradient(input_array)

    grad_filter = -0.0022
    inf_filter = -0.6
    sup_filter = 0.4
    filter_window = 0.05

    x_time = np.linspace(0, input_array.shape[0]/float(rate), input_array.shape[0])

    _timeframe = input_array.shape[0]/float(rate)

    poi1 = grad < grad_filter
    local_minima1 = SF.local_min(grad)
    combined1 = np.logical_and(local_minima1, poi1)

    poi2 = input_array < inf_filter
    local_minima2 = SF.local_min(input_array)
    combined2 = np.logical_and(local_minima2, poi2)

    poi3 = input_array > sup_filter
    local_max3 = SF.local_min(-input_array)
    combined3 = np.logical_and(local_max3, poi3)

    brps1 = np.nonzero(combined1)[0]
    brps1 = [x_time[brp] for brp in brps1.tolist() if x_time[brp]>inner_cut and x_time[brp]<_timeframe-inner_cut]

    brps2 = np.nonzero(combined2)[0]
    brps2 = [x_time[brp] for brp in brps2.tolist() if x_time[brp]>inner_cut and x_time[brp]<_timeframe-inner_cut]

    brps3 = np.nonzero(combined3)[0]
    brps3 = [x_time[brp] for brp in brps3.tolist() if x_time[brp]>inner_cut and x_time[brp]<_timeframe-inner_cut]

    brps1 = np.array(brps1)
    brps2 = np.array(brps2)
    brps3 = np.array(brps3)

    retained_pairs = []
    pair_scores = []
    for brp in brps1:
        candidate_list1 = brps2[np.logical_and(brps2 > brp, brps2 < brp+filter_window)]
        candidate_list2 = brps3[np.logical_and(brps3 < brp, brps3 > brp-filter_window)]
        if len(candidate_list1) > 0 and len(candidate_list2) > 0:
            if len(retained_pairs) == 0:
                retained_pairs.append(candidate_list1[0])
                pair_scores.append(input_array[int(candidate_list1[0]*rate)]-input_array[int(candidate_list2[-1]*rate)])
            elif np.abs(candidate_list1[0]-retained_pairs[-1]) > 0.01:
                retained_pairs.append(candidate_list1[0])
                pair_scores.append(input_array[int(candidate_list2[-1]*rate)]-input_array[int(candidate_list1[0]*rate)])

    if collapse:
        pair_dists = np.array(retained_pairs[1:]) - np.array(retained_pairs[:-1])
        flter = pair_dists < collapse
        supression_list = []
        for i in flter.nonzero()[0]:
            if pair_scores[i] < pair_scores[i+1]:
                supression_list.append(i)
            else:
                supression_list.append(i+1)
        keeplist = np.zeros_like(retained_pairs).astype(np.bool)
        keeplist[supression_list] = True

        new_retained_pairs = np.array(retained_pairs)[np.logical_not(keeplist)]
        retained_pairs = new_retained_pairs.tolist()

    plt.plot(x_time[inner_cut*rate:-inner_cut*rate], input_array[inner_cut*rate:-inner_cut*rate], color='k')
    # SF.show_breakpoints(brps1.tolist(), 'g')
    # SF.show_breakpoints(brps2.tolist(), 'y')
    # SF.show_breakpoints(brps3.tolist(), 'r')

    SF.show_breakpoints(retained_pairs, color='r')
    # plt.axhline(sup_filter, 'r')
    # plt.axhline(inf_filter, 'r')
    plt.show()

    return retained_pairs


def splice_n_stitch(input_array, chop_points):
    # print len(chop_points)
    # print input_array.shape[0]

    interval_floor = 0.33

    chop_points = np.array(chop_points+[input_array.shape[0]/float(rate)])
    intervals = chop_points[1:] - chop_points[:-1]
    anomalous = np.array([False]+(intervals < interval_floor).tolist())
    anomalous = np.logical_and(anomalous[1:-1], np.logical_or(anomalous[2:], anomalous[:-2]))
    anomalous = np.array([False] + anomalous.tolist() + [False])
    # cut_points = (chop_points*rate).astype(np.int64)
    # print cut_points
    anomaly_lane = SF.brp_setter(chop_points*rate, anomalous).astype(np.int16)
    # print anomalous
    # print anomaly_lane.shape, anomaly_lane

    plt.plot(x_time[inner_cut*rate:-inner_cut*rate], input_array[inner_cut*rate:-inner_cut*rate], 'k')
    plt.fill_between(x_time[inner_cut*rate:-inner_cut*rate], np.zeros_like(anomaly_lane)[inner_cut*rate:-inner_cut*rate],
                      anomaly_lane[inner_cut*rate:-inner_cut*rate]*5, color='r')
    plt.fill_between(x_time[inner_cut*rate:-inner_cut*rate], np.zeros_like(anomaly_lane)[inner_cut*rate:-inner_cut*rate],
                      -anomaly_lane[inner_cut*rate:-inner_cut*rate]*5, color='r')
    # SF.show_breakpoints(chop_points, 'k')
    plt.show()

    new_input_array = input_array[np.logical_not(anomaly_lane)]

    return new_input_array


def fold_line(input_array, chop_points):
    delay_times = 60./(np.array(chop_points)[1:] - np.array(chop_points)[:-1])
    delay_times = SF.remove_outliers(delay_times, 0.01)

    pulse, std = (np.nanmean(delay_times), np.nanstd(delay_times, ddof=1))

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

    support_array = np.zeros((len(lane_bank), int((l_max+r_max)*rate)))
    for i, (l_offset, array) in enumerate(izip(l_offsets, lane_bank)):
        support_array[i, int((l_max-l_offset)*rate) : int((l_max-l_offset)*rate)+array.shape[0]] = array

    x_range = np.linspace(-l_max, r_max, int((l_max+r_max)*rate))

    return x_range, np.average(support_array, axis=0), delay_times, pulse, std


def plot(input_array):
    grad = np.gradient(input_array)

    plt.plot(x_time[inner_cut*rate:-inner_cut*rate], input_array[inner_cut*rate:-inner_cut*rate])
    plt.axhline(0, color='r')
    plt.show()

    plt.plot(input_array[inner_cut*rate:-inner_cut*rate], grad[inner_cut*rate:-inner_cut*rate])
    plt.show()


def render_analysis(x_range, average, delays, pulse, std):
    title = 'Pulse: %s bpm\n Pulse Standard Deviation: %s bpm' % ("{0:.2f}".format(pulse), "{0:.2f}".format(std))

    plt.plot(x_range, average, color='k')
    plt.show()

    SF.smooth_histogram(delays, 'k', title)


def render_multiple_analyses(x_ranges, average_array, delays_array, pulse_array, std_array):
    pass


if __name__ == '__main__':
    new_time_r = filter_hum(diff)
    plot(new_time_r)
    chop_points = chop_dataset(new_time_r)
    new_time_r = splice_n_stitch(new_time_r, chop_points)
    chop_points = chop_dataset(new_time_r, collapse=0.33)
    x_range, average, delays, pulse, std = fold_line(new_time_r, chop_points)
    render_analysis(x_range, average, delays, pulse, std)

    dump((x_range, average, delays, pulse, std), open(name[:-3].split('/')[-1]+'dmp', 'w'))


