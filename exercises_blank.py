import numpy as np
from scipy.fftpack import dct


def split_meta_line(line, delimiter=' '):
    """
    param line: lines of metadata
    param delimiter: delimeter
    return: speaker_id: speaker IDs: gender: gender: file_path: path to file
    """
    line = line.strip()
    if not line:
        return None, None, None
    parts = line.split(delimiter)
    if len(parts) < 3:
        return None, None, None
    speaker_id = parts[0]
    gender = parts[1]
    file_path = delimiter.join(parts[2:]) if len(parts) > 3 else parts[2]
    return speaker_id, gender, file_path


def preemphasis(signal, pre_emphasis=0.97):
    """
    param signal: input signal
    param pre_emphasis: preemphasis coeffitient
    return: emphasized_signal: signal after pre-emphasis procedure
    """
    if len(signal) == 0:
        return signal
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    return emphasized_signal.astype(np.float64, copy=False)


def framing(emphasized_signal, sample_rate=16000, frame_size=0.025, frame_stride=0.01):
    """
    param emphasized_signal: signal after pre-emphasis procedure
    param sample_rate: signal sampling rate
    param frame_size: sliding window size
    param frame_stride: step
    return: frames: output matrix [nframes x sample_rate*frame_size]
    """
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)

    win = np.hamming(frame_length)
    frames = np.empty((num_frames, frame_length), dtype=np.float64)
    for i in range(num_frames):
        start = i * frame_step
        frames[i] = pad_signal[start:start + frame_length] * win

    return frames


def power_spectrum(frames, NFFT=512):
    """
    param frames: framed signal
    param NFFT: number of fft bins
    return: pow_frames: framed signal power spectrum
    """
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = (mag_frames ** 2) / float(NFFT)
    return pow_frames


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def compute_fbank_filters(nfilt=40, sample_rate=16000, NFFT=512):
    """
    param nfilt: number of filters
    param sample_rate: signal sampling rate
    param NFFT: number of fft bins in power spectrum
    return: fbank [nfilt x (NFFT/2+1)]
    """
    low_freq_mel = 0
    high_freq = sample_rate / 2
    high_freq_mel = _hz_to_mel(high_freq)

    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_edges = np.floor((NFFT + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin_edges[m - 1])
        f_m = int(bin_edges[m])
        f_m_plus = int(bin_edges[m + 1])

        for k in range(f_m_minus, f_m):
            denom = f_m - f_m_minus
            if denom <= 0:
                continue
            fbank[m - 1, k] = (k - bin_edges[m - 1]) / denom
        for k in range(f_m, f_m_plus):
            denom = f_m_plus - f_m
            if denom <= 0:
                continue
            fbank[m - 1, k] = (bin_edges[m + 1] - k) / denom

    return fbank


def compute_fbanks_features(pow_frames, fbank):
    """
    param pow_frames: framed signal power spectrum, matrix [nframes x sample_rate*frame_size]
    param fbank: matrix of the fbank filters [nfilt x (NFFT/2+1)] where NFFT: number of fft bins in power spectrum
    return: filter_banks_features: log mel FB energies matrix [nframes x nfilt]
    """
    filter_banks_features = np.dot(pow_frames, fbank.T)
    filter_banks_features = np.where(
        filter_banks_features == 0, np.finfo(float).eps, filter_banks_features)
    filter_banks_features = np.log(filter_banks_features)

    return filter_banks_features


def compute_mfcc(filter_banks_features, num_ceps=20):
    """
    param filter_banks_features: log mel FB energies matrix [nframes x nfilt]
    param num_ceps: number of cepstral components for MFCCs
    return: mfcc: mel-frequency cepstral coefficients (MFCCs)
    """
    mfcc = dct(filter_banks_features, type=2, axis=1, norm="ortho")[:, :num_ceps]
    return mfcc


def mvn_floating(features, LC, RC, unbiased=False):
    """
    param features: features matrix [nframes x nfeats]
    param LC: the number of frames to the left defining the floating
    param RC: the number of frames to the right defining the floating
    param unbiased: biased or unbiased estimation of normalising sigma
    return: normalised_features: normalised features matrix [nframes x nfeats]
    """
    nframes, dim = features.shape
    LC = min(LC, nframes - 1)
    RC = min(RC, nframes - 1)
    n = (np.r_[np.arange(RC + 1, nframes), np.ones(RC + 1) * nframes] - np.r_[np.zeros(LC), np.arange(nframes - LC)])[
        :, np.newaxis]
    f = np.cumsum(features, 0)
    s = np.cumsum(features ** 2, 0)
    f = (np.r_[f[RC:], np.repeat(f[[-1]], RC, axis=0)] - np.r_[np.zeros((LC + 1, dim)), f[:-LC - 1]]) / n
    s = (np.r_[s[RC:], np.repeat(s[[-1]], RC, axis=0)] - np.r_[np.zeros((LC + 1, dim)), s[:-LC - 1]]
         ) / (n - 1 if unbiased else n) - f ** 2 * (n / (n - 1) if unbiased else 1)

    std = np.sqrt(np.maximum(s, np.finfo(float).eps))
    normalised_features = (features - f) / std
    normalised_features[s == 0] = 0

    return normalised_features
