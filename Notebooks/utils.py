import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

def load_data(src_dir,name):
    """
    Load .h5 Alphasat data, located in src_dir
    """
    def read_hdf(src_dir,name,n_channel):
        df = pd.read_hdf(os.path.join(src_dir,"{}_ch{}.h5".format(name,n_channel)))
        df = df.set_index('time')
        return df
    
    return read_hdf(src_dir,name,1), read_hdf(src_dir,name,2), read_hdf(src_dir,name,3), read_hdf(src_dir,name,4)



def add_flags(df_event, df_ch1, df_ch2, df_ch3, df_ch4):
    """
    Add flag to the channel dataframes:
        - flag 0: no event
        - flag 1: rain event
        - flag 2: failure
    Add ind_event:
        - ind_event -1: rain or failure
        - ind_event 0->len(df_event): no_event
    """
    def flag(df_event,df):
        df['flag'] = np.zeros_like(df['signal'], dtype=int)
        df['ind_event'] = np.zeros_like(df['signal'], dtype=int)
        for i, event in df_event.iterrows():
            time_start = event.loc['TIME START']
            time_stop = event.loc['TIME STOP']

            cond_idx = (df.index > time_stop)
            df.loc[cond_idx, 'ind_event'] = i+1
            
            cond_idx = (df.index >= time_start) & (df.index <= time_stop)
            if df_event.loc[i]['EVENT'] == 'rain': # set flag 1 for rain
                df.loc[cond_idx, 'flag'] = 1
                df.loc[cond_idx, 'ind_event'] = -1
            if df_event.loc[i]['EVENT'] == 'failure': # set flag 2 for failure
                df.loc[cond_idx, 'flag'] = 2
                df.loc[cond_idx, 'ind_event'] = -1

        return df
    
    return flag(df_event,df_ch1), flag(df_event,df_ch2), flag(df_event,df_ch3), flag(df_event,df_ch4)
    
    
    
def plot_one_day_4ch(date, df_ch1, df_ch2, df_ch3, df_ch4, signal_filt=False, excess_attenuation=False, bool_save=False, show=False):
    """
    Plot time series for one day (midnight to midnight)
    """
    
    # Crop dataframe to the date of interest
    df_ch1 = df_ch1[(df_ch1.index>=date) & (df_ch1.index < (date+timedelta(days = 1)))]
    df_ch2 = df_ch2[(df_ch2.index>=date) & (df_ch2.index < (date+timedelta(days = 1)))]
    df_ch3 = df_ch3[(df_ch3.index>=date) & (df_ch3.index < (date+timedelta(days = 1)))]
    df_ch4 = df_ch4[(df_ch4.index>=date) & (df_ch4.index < (date+timedelta(days = 1)))]
    
    fig, axs = plt.subplots(2,2,figsize=(12,10))
    for i in range(0,4):
        if i==0:
            axs[i%2,i//2].set_title("ch1: x-polar 39.4 GHz")
            df = df_ch1
        elif i==1:
            axs[i%2,i//2].set_title("ch2: co-polar 39.4 GHz")
            df = df_ch2
        elif i==2:
            axs[i%2,i//2].set_title("ch3: x-polar 19.7 GHz")
            df = df_ch3
        elif i==3:
            axs[i%2,i//2].set_title("ch4: co-polar 19.7 GHz")
            df = df_ch4
            
        time = df.index
        if not excess_attenuation:
            signal = df['signal']
            noise = df['noise']

            axs[i%2,i//2].plot(time,signal, color='lightblue')
            axs[i%2,i//2].plot(time,noise,color="gray")
            if signal_filt: axs[i%2,i//2].plot(time, df['signal_filt'], color="darkblue")
            axs[i%2,i//2].grid()
            axs[i%2,i//2].set_ylim(-50,20)
            axs[i%2,i//2].set_ylabel("Power [dB]")
            axs[i%2,i//2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        else:
            axs[i%2,i//2].plot(time, df['excess_attenuation'], color='lightblue')
            axs[i%2,i//2].plot(time, np.zeros_like(time, dtype=float), color='red')
            axs[i%2,i//2].grid()
            axs[i%2,i//2].set_ylabel("Power [dB]")
            axs[i%2,i//2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Show rain events & determine start and stop times of rain event during the day
        idx_start = []
        idx_stop = []
        if df.loc[:, 'flag'][0] == 1: #if rain at the start of the day
            idx_start.append(0)
        for idx in np.where(np.diff(df.loc[:, 'flag'].values) == True)[0]: #find index of 0 to 1 transitions in flags
            idx_start.append(idx) 
        for idx in np.where(-np.diff(df.loc[:, 'flag'].values) == True)[0]: #find index of 1 to 0 transitions in flags
            idx_stop.append(idx) 
        if df.loc[:, 'flag'][-1] == 1: #if rain at the end of the day
            idx_stop.append(len(df.loc[:, 'flag'])-1)
            
        for k in range(0,len(idx_start)):
            axs[i%2,i//2].axvspan(df.index[idx_start[k]],df.index[idx_stop[k]], color='red', alpha=0.25)
            
        # Show failure events
        idx_start = []
        idx_stop = []
        flag_failure = np.copy(df.loc[:, 'flag'].values)
        flag_failure[flag_failure<=1] = 0
        flag_failure = flag_failure / 2
        if flag_failure[0] == 1: 
            idx_start.append(0)
        for idx in np.where(np.diff(flag_failure) == True)[0]: #find index of 0 to 2 transitions in flags
            idx_start.append(idx) 
        for idx in np.where(-np.diff(flag_failure) == True)[0]: #find index of 2 to 0 transitions in flags
            idx_stop.append(idx) 
        if flag_failure[-1] == 1: 
            idx_stop.append(len(df['flag'])-1)
            
        for k in range(0,len(idx_start)):
            axs[i%2,i//2].axvspan(df.index[idx_start[k]],df.index[idx_stop[k]], color='grey', alpha=0.25)
            
    fig.suptitle("Alphasat received data at LLN for "+ date.strftime('%Y-%m-%d'),fontsize=14)
    
    if bool_save:
        fig.savefig(f'figures/{date.strftime('%B%Y')}/{date.strftime('%Y_%m_%d')} ' +
        f'{int(excess_attenuation)*"excess attenuation"+int(signal_filt)*'filtered'}.png')
    
    if not show:
        plt.close()
        
def plot_RAPIDS_outputs(rapids_data,title):
    """
    Plot RAPIDS attenuation from the rapids_data dataframe. The dataframe must contain a "PROBABILITY" and a "ATTENUATION" field.
    """
    plt.figure()
    for i in range(0,18):
        plt.semilogx(rapids_data["PROBABILITY"].values[i*26:(i+1)*26],rapids_data["ATTENUATION"].values[i*26:(i+1)*26],label="{}°".format((i+1)*5))
    plt.grid()
    plt.title(title)
    plt.xlim(1e-3,100)
    plt.legend()