"""
This code is based on Facebook's HDemucs code: https://github.com/facebookresearch/demucs
"""

"""STFT-based Loss modules."""

import torch
import math
import torch.nn.functional as F


def stft(x, fft_size, hop_size, win_length, window, start_interval=0.0, end_interval=1.0):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
        start_interval (float): where to start (between 0 hz and the nyquist frequency)
        end_interval (float): where to end (between 0 hz and the nyquist frequency)
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.view_as_real(torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True))
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    if start_interval > 0:
        real[:,0:int(start_interval*real.shape[1])-1,:] = 0
        imag[:,0:int(start_interval*real.shape[1])-1,:] = 0
    
    if end_interval < 1:
        real[:,int(end_interval*real.shape[1])-1:,:] = 0
        imag[:,int(end_interval*real.shape[1])-1:,:] = 0

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf

    magnitude = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)
    phase = torch.atan2(prevent_zeroes(imag), prevent_zeroes(real)).transpose(2, 1)

    return magnitude, phase

def prevent_zeroes(x: torch.Tensor, threshold=1e-7):
    y = x.clone()
    y[(x < threshold) & (x > -1.0 * threshold)] = threshold
    return y


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self, magnitude_weight_shift=0.0, phase_weight=1.0):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()
        self.freq_weights = None
        self.magnitude_weight_shift = magnitude_weight_shift
        self.phase_weight = phase_weight

    def forward(self, x_mag, y_mag, x_phase, y_phase):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
            x_phase (Tensor): Phase spectrogram of predicted signal (B, #frames, #freq_bins).
            y_phase (Tensor): Phase spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        if self.magnitude_weight_shift != 0.0:
            mag_diff = torch.where(y_mag > x_mag,
                                   (1+self.magnitude_weight_shift)*(y_mag - x_mag), 
                                   (1-self.magnitude_weight_shift)*(y_mag - x_mag)
                                   )
        else:
            mag_diff = y_mag - x_mag
            
        phase_diff = torch.sin((x_phase - y_phase)/2)

        max_diff = torch.maximum(torch.abs(mag_diff), torch.abs(phase_diff) * self.phase_weight *math.pi)

        conv_loss = torch.mean(max_diff*torch.maximum(x_mag,y_mag))

        return conv_loss


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self, magnitude_weight_shift=0.0):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()
        self.magnitude_weight_shift = magnitude_weight_shift

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        quantile_mag_y = torch.clamp(torch.quantile(y_mag,0.9,dim=2,keepdim=True)[0], 1e-1, None)
        max_mag_y = torch.max(y_mag,dim=2, keepdim=True)[0]
        scale_mag_y = torch.clamp(torch.maximum(quantile_mag_y,max_mag_y/16),1e-1,None)

        mag_diff = torch.abs(torch.where(y_mag > x_mag,
                                (1+self.magnitude_weight_shift)*(y_mag - x_mag), 
                                (1-self.magnitude_weight_shift)*(y_mag - x_mag)
                                ))
        return torch.mean(torch.pow(mag_diff/scale_mag_y, 1.3))

        
class TransientLoss(torch.nn.Module):
    """Transient loss module."""

    def __init__(self):
        """Initilize transient loss module."""
        super(TransientLoss, self).__init__()
        self.freq_weights = None

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Transient loss value.
        """

        x_grad = torch.gradient(x_mag,dim=1)
        y_grad = torch.gradient(y_mag,dim=1)

        quantile_grad_y = torch.quantile(torch.abs(y_grad[0]),0.92,dim=2, keepdim=True)[0]
        max_grad_y = torch.max(torch.abs(y_grad[0]),dim=2, keepdim=True)[0]
        scale_grad_y = torch.clamp(torch.maximum(quantile_grad_y,max_grad_y/16),5e-2,None)

        return F.mse_loss(x_grad[0]/scale_grad_y, y_grad[0]/scale_grad_y)


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window", start_interval=0.0, end_interval=1.0, magnitude_weight_shift=0.0, phase_weight=1.0):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.magnitude_weight_shift = magnitude_weight_shift
        self.spectral_convergenge_loss = SpectralConvergengeLoss(magnitude_weight_shift, phase_weight)
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss(magnitude_weight_shift)
        self.transient_loss = TransientLoss()
        self.start_interval = start_interval
        self.end_interval = end_interval

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag, x_phase = stft(x, self.fft_size, self.shift_size, self.win_length, self.window, self.start_interval, self.end_interval)
        y_mag, y_phase = stft(y, self.fft_size, self.shift_size, self.win_length, self.window, self.start_interval, self.end_interval)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag, x_phase, y_phase)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        trans_loss = self.transient_loss(x_mag, y_mag)

        if torch.isnan(sc_loss) or torch.isnan(mag_loss) or torch.isnan(trans_loss):
            raise ValueError("Got NaN from STFT Loss: sc: " +
                             str(torch.isnan(sc_loss).item()) + " mag: " +
                             str(torch.isnan(mag_loss).item()) + " trans: " +
                             str(torch.isnan(trans_loss).item()))

        return sc_loss, mag_loss, trans_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1531, 2557, 4099, 4999, 3001, 6037],
                 hop_sizes=[127, 307, 241, 401, 547, 577],
                 win_lengths=[601, 1399, 1249, 4999, 1601, 5167],
                 window="hann_window", factor_sc=0.1, factor_mag=0.1, factor_trans=0.1,
                 start_interval=0.0, end_interval=1.0, magnitude_weight_shift=0.0,
                 phase_weight=1.0):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
            start_interval (float): where to start (between 0 hz and the nyquist frequency)
            end_interval (float): where to end (between 0 hz and the nyquist frequency)
            magnitude_weight_shift (float): if non-zero, increases loss for predicted magnititudes that are too low (positive) or high (negative)
            phase_weight (float):  weight of phase portion of spectral convergence loss
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        assert magnitude_weight_shift > -1.0, magnitude_weight_shift < 1.0
        assert phase_weight >= 0.0
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window, start_interval, end_interval, magnitude_weight_shift, phase_weight)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        self.factor_trans = factor_trans

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        trans_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l, trans_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
            trans_loss += trans_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)
        trans_loss /= len(self.stft_losses)

        return self.factor_sc*sc_loss, self.factor_mag*mag_loss, self.factor_trans*trans_loss
