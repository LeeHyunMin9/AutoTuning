import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt


def optcutfreq(y, freq=1, fclim=[], show=False, ax=None, idx_parameter ='', save_dir = ''):
    """ Automatic search of optimal filter cutoff frequency based on residual analysis.

    The 'optimal' cutoff frequency (in the sense that a filter with such cutoff
    frequency removes as much noise as possible without considerably affecting
    the signal) is found by performing a residual analysis of the difference
    between filtered and unfiltered signals over a range of cutoff frequencies.
    The optimal cutoff frequency is the one where the residual starts to change
    very little because it is considered that from this point, it's being
    filtered mostly noise and minimally signal, ideally.

    Parameters
    ----------
    y : 1D array_like
        Data
    freq : float, optional (default = 1)
        sampling frequency of the signal y
    fclim : list with 2 numbers, optional (default = [])
        limit frequencies of the noisy part or the residuals curve
    show : bool, optional (default = False)
        True (1) plots data in a matplotlib figure
        False (0) to not plot
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    fc_opt : float
             optimal cutoff frequency (None if not found)

    Notes
    -----
    A second-order zero-phase digital Butterworth low-pass filter is used.
    # The cutoff frequency is correctyed for the number of passes:
    # C = (2**(1/npasses) - 1)**0.25. C = 0.802 for a dual pass filter. 

    The matplotlib figure with the results will show a plot of the residual
    analysis with the optimal cutoff frequency, a plot with the unfiltered and
    filtered signals at this optimal cutoff frequency (with the RMSE of the
    difference between these two signals), and a plot with the respective
    second derivatives of these signals which should be useful to evaluate
    the quality of the optimal cutoff frequency found.

    This function performs well with data where the signal has frequencies
    considerably below the Niquist frequency and the noise is predominantly
    white in the higher frequency region.

    If the automatic search fails, the lower and upper frequencies of the noisy
    part of the residuals curve can be inputed as a parameter (fclim).
    These frequencies can be chosen by viewing the plot of the residuals (enter
    show=True as input parameter when calling this function).

    Examples
    --------
    y = np.cumsum(np.random.randn(1000))
    # optimal cutoff frequency based on residual analysis and plot:
    fc_opt = optcutfreq(y, freq=1000, show=True)
    # sane analysis but specifying the frequency limits and plot:
    optcutfreq(y, freq=1000, fclim=[200,400], show=True)
    # It's not always possible to find an optimal cutoff frequency
    # or the one found can be wrong (run this example many times):
    y = np.random.randn(100)
    optcutfreq(y, freq=100, show=True)

    """
    from scipy.interpolate import UnivariateSpline

    # Correct the cutoff frequency for the number of passes in the filter
    C = 0.802  # for dual pass; C = (2**(1/npasses)-1)**0.25

    # signal filtering
    freqs = np.linspace((freq/2) / 100, (freq/2)*C, 101, endpoint=False)
    res = []
    for fc in freqs:
        b, a = signal.butter(2, (fc/C) / (freq / 2))    
        yf = signal.filtfilt(b, a, y)
        # residual between filtered and unfiltered signals
        res = np.hstack((res, np.sqrt(np.mean((yf - y)**2))))

    # find the optimal cutoff frequency by fitting an exponential curve
    # y = A*exp(B*x)+C to the residual data and consider that the tail part
    # of the exponential (which should be the noisy part of the residuals)
    # decay starts after 3 lifetimes (exp(-3), 95% drop)
    if not len(fclim): # or np.any(fclim[0] < 0) or np.any(fclim[1] > freq/2):
        fc1 = 0
        fc2 = int(0.95*(len(freqs)-1))
        # log of exponential turns the problem to first order polynomial fit
        # make the data always greater than zero before taking the logarithm
        reslog = np.log(np.abs(res[fc1:fc2 + 1] - res[fc2]) +
                        1000 * np.finfo(np.float32).eps)   
                        # np.finfo(np.float32).eps = 1.1920929e-07, np.finfo(np.float64).eps = 2.220446049250313e-16
                        # np.finfo란? : 부동소수점의 정보를 담고 있는 클래스
                        # np.finfo(dtype).eps : dtype의 유효숫자(가수) 중 가장 작은 수를 반환
                        # 1000을 곱하는 이유는 log(0) = -inf 을 피하기 위함
        Blog, Alog = np.polyfit(freqs[fc1:fc2 + 1], reslog, 1)
        # find the frequency where the exponential decay starts
        fcini = np.nonzero(freqs >= -3 / Blog)  # 3 lifetimes
        fclim = [fcini[0][0], fc2] if np.size(fcini) else []
    else:
        fclim = [np.nonzero(freqs >= fclim[0])[0][0],
                 np.nonzero(freqs >= C*0.95*fclim[1])[0][0]]

    # find fc_opt with linear fit y=A+Bx of the noisy part of the residuals
    if len(fclim) and fclim[0] < fclim[1]:
        B, A = np.polyfit(freqs[fclim[0]:fclim[1]], res[fclim[0]:fclim[1]], 1)
        # optimal cutoff frequency is the frequency where y[fc_opt] = A
        roots = UnivariateSpline(freqs, res - A, s=0).roots() # s=0 : smoothing factor, roots() : Return the zeros of a spline function.
        fc_opt = roots[0] if len(roots) else None
    else:
        fc_opt = None

    # if fc_opt < 67:
    #     fc_opt = 67

    if show:
        show_graph(y, freq, freqs, res, fclim, fc_opt, B, A, ax, fc1, fc2, Blog, Alog, idx_parameter, save_dir)

    return fc_opt


def show_graph(y, freq, freqs, res, fclim, fc_opt, B, A, ax, fc1, fc2, Blog, Alog, idx_parameter, save_dir):   #fc1,fc2,Blog,Alog
    """Plot results of the optcutfreq function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            fig = plt.figure(num=None, figsize=(10, 5))
            ax = np.array([plt.subplot(121),
                           plt.subplot(222),
                           plt.subplot(224)])

        plt.rc('axes', labelsize=12, titlesize=12)
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        ax[0].plot(freqs, res, 'b.', markersize=9)
        time = np.linspace(0, len(y) / freq, len(y))
        ax[1].plot(time, y, 'g', linewidth=1, label='Unfiltered')
        ydd = np.diff(y, n=2) * freq ** 2
        ax[2].plot(time[:-2], ydd, 'g', linewidth=1, label='Unfiltered')
        if fc_opt:
            ylin = np.poly1d([B, A])(freqs)
            ax[0].plot(freqs, ylin, 'r--', linewidth=2)
            # exponential of log fitted curve
            yexp = np.exp(np.poly1d([Blog, Alog])(freqs[fc1+1:fc2]))
            ax[0].plot(freqs[fc1+1:fc2], yexp, 'k-', linewidth=2 )
            ax[0].plot(freqs[fc1+1],0,'k*',markersize=10)
            ax[0].plot(freqs[fc2],0,'ko',markersize=10)

            ax[0].plot(freqs[fclim[0]], res[fclim[0]], 'r>',
                       freqs[fclim[1]], res[fclim[1]], 'r<', ms=9)
            ax[0].set_ylim(ymin=0, ymax=4 * A)
            ax[0].plot([0, freqs[-1]], [A, A], 'r-', linewidth=2)
            ax[0].plot([fc_opt, fc_opt], [0, A], 'r-', linewidth=2)
            ax[0].plot(fc_opt, 0, 'ro', markersize=7, clip_on=False,
                       zorder=9, label='$Fc_{opt}$ = %.1f Hz' % fc_opt)
            ax[0].legend(fontsize=12, loc='best', numpoints=1, framealpha=.5)
            # Correct the cutoff frequency for the number of passes
            C = 0.802  # for dual pass; C = (2**(1/npasses) - 1)**0.25
            b, a = signal.butter(2, (fc_opt/C) / (freq / 2))
            yf = signal.filtfilt(b, a, y)
            ax[1].plot(time, yf, color=[1, 0, 0, .5],
                       linewidth=2, label='Opt. filtered')
            ax[1].legend(fontsize=12, loc='best', framealpha=.5)
            ax[1].set_title('Signals (RMSE = %.3g)' % A)
            yfdd = np.diff(yf, n=2) * freq ** 2
            ax[2].plot(time[:-2], yfdd, color=[1, 0, 0, .5],
                       linewidth=2, label='Opt. filtered')
            ax[2].legend(fontsize=12, loc='best', framealpha=.5)
            resdd = np.sqrt(np.mean((yfdd - ydd) ** 2))
            ax[2].set_title('Second derivatives (RMSE = %.3g)' % resdd)
        else:
            ax[0].text(.5, .5, 'Unable to find optimal cutoff frequency',
                       horizontalalignment='center', color='r', zorder=9,
                       transform=ax[0].transAxes, fontsize=12)
            yexp = np.exp(np.poly1d([Blog, Alog])(freqs[fc1+1:fc2]))
            ax[0].plot(freqs[fc1+1:fc2], yexp, 'k-', linewidth=2 )
            ax[0].plot(freqs[fc1+1],0,'k*',markersize=10)
            ax[0].plot(freqs[fc2],0,'ko',markersize=10)

            ax[0].plot(freqs[fclim[0]], res[fclim[0]], 'r>',
                       freqs[fclim[1]], res[fclim[1]], 'r<', ms=9)
            ax[0].set_ylim(ymin=0, ymax=4 * A)
            ax[1].set_title('Signal')
            ax[2].set_title('Second derivative')

        ax[0].set_xlabel('Cutoff frequency [Hz]')
        ax[0].set_ylabel('Residual RMSE')
        ax[0].set_title('Residual analysis')
        ax[0].grid()
        # ax2.set_xlabel('Time [s]') 
        ax[1].set_xlim(0, time[-1])
        ax[1].grid()
        ax[2].set_xlabel('Time [s]')
        ax[2].set_xlim(0, time[-1])
        ax[2].grid()
        fig.suptitle(f'{idx_parameter}-Frequency Analysis')
        plt.tight_layout()
        plt.savefig(save_dir + f'/optcutfreq.png')
        #plt.show()
        plt.close()


def get_lowcutoff_and_SNR(time_torque_data, idx_parameter, save_dir, show_graph):
    
    def butter_highpass(cutoff, fs, order=3):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_highpass_filter(data, cutoff, fs, order=3):
        b, a = butter_highpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y


    if not isinstance(time_torque_data, type(None)):
        one = time_torque_data # 550, 996
        N = len(one)
        Fs = N
        Ts = 1 / Fs
        te = 1.0
        t = np.arange(0.0, te, Ts) *Fs

        print('Start to Find Optimal Cutoff Frequency and Drawing!!!')
        try:
            lowcutoff = optcutfreq(one, Fs, show = show_graph, idx_parameter = idx_parameter, save_dir = save_dir)#20.
            
            #highcutoff = cutoff[1]#90.
            hpf = butter_highpass_filter(one, lowcutoff, Fs)
            k = np.arange(N)
            T = N / Fs
            freq = k / T
            freq = freq[range(int(N/2))]

            fig = plt.figure(num=None, figsize=(10, 5))
            ax = np.array([plt.subplot(121),
                                plt.subplot(122)])

           
            
            # 3. 필터 적용된 FFT Plot
            non_filter_yfft = np.fft.fft(one)
            non_filter_yf = non_filter_yfft/np.sqrt(N/2)
            non_filter_yf = non_filter_yf[range(int(N/2))]
            high_yfft = np.fft.fft(hpf)
            high_yf = high_yfft / np.sqrt(N/2)
            high_yf = high_yf[range(int(N/2))]

            if show_graph:
                # 1. 원 신호

                ax[0].plot(t, one, 'y', label='origin')

                # 2. 필터 적용된 Plot
                ax[0].plot(t, hpf, 'b', label='(High)filtered data')
                ax[0].legend()

                ax[1].plot(freq, abs(non_filter_yf), 'y', label = 'origin')
                ax[1].plot(freq, abs(high_yf), 'b', label = '(High)filtered data')
                # draw the cutoff frequency
                ax[1].axvline(x = lowcutoff, color = 'r', linestyle = '--', label = 'cutoff')

                ax[1].set_title("HBF")
                ax[1].set_xlim(0, Fs/2)
                ax[1].legend()
                fig.suptitle(f'{idx_parameter}-Frequency Analysis')
                #plt.show()
                plt.savefig(save_dir + f'/hpf.png')
                plt.close()
        except Exception as e:
            print(f'Error : {e}') 
            lowcutoff = None
            pass
        finally:
            plt.close()
        # Calculate SNR
            def SNR(signal, noise):
                return 10*np.log10(np.sum(signal**2)/np.sum(noise**2))
        
            snr = SNR(one, hpf) if lowcutoff else 0

        return lowcutoff, snr

    else:
        return None, 0
    

