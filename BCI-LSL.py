import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, iirnotch
from scipy.stats import kurtosis,skew
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import model_selection
from time import time,sleep
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
import serial


#LSL receive
fs = 200.0
lowcut = 0.5
highcut = 60
f0 = 50.0 #freq to be removed
Q = 30

w0 = f0/(fs/2)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def feature_compute(y_blink): 
    #attr = []
    time_step=1/200
    #attr_mean = np.mean(y_blink,axis = 0)
    attr_var = np.var(y_blink[:,0],axis=0)
    #attr_min = np.amin(y_blink,axis = 0)
    #attr_max = np.amax(y_blink,axis = 0)  
    #attr_skew = skew(y_blink[:,1],axis=0)
    #attr_kurtosis = kurtosis(y_blink,axis=0)
    attr = np.zeros(1)
    #print(attr_mean)
    #attr[0] = attr_mean
    attr[0] = attr_var 
    #attr[1] = attr_skew
    #attr[2] = attr_min
    #attr[3] = attr_max
    ps=np.abs(np.fft.fft(y_blink[:,1]))**2
    frequencies=np.fft.fftfreq(200,time_step)
    #attr[1] = attr_kurtosis
    
    attr = np.array(attr)
    return ps,frequencies,attr
from pylsl import StreamInlet, resolve_stream
import numpy as np
# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
sample_list=[]
timestamp_list=[]
counter=0
#ser = serial.Serial('/dev/ttyACM0', 9600)
while True:
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    sample, timestamp = inlet.pull_sample()
    #print(sample)
    sample_list.append(sample)
    timestamp_list.append(timestamp)
    if(len(timestamp_list)==200):
      sample_list=np.reshape(np.array(sample_list),(200,4))
      sample_list=sample_list[:,0:2]
      sample_list=sample_list*1000
      y_eeg=butter_bandpass_filter(sample_list,lowcut, highcut, fs, order=9)
      b, a =iirnotch(w0, Q)
      y_fin=lfilter(b, a, y_eeg)
       #plt.plot(y_fin)
       #plt.show()
      psd, freqs, test_attrib=feature_compute(y_fin)
      idx1= np.argsort(freqs)
      #idx2= np.argsort(psd)
      #print(freqs[10],psd[10])
     # print(freq[idx2])
      #plt.plot(freqs[idx], psd[idx])
      #plt.axis([-100,100,0,7])
      #plt.show()
    
      #print(test_attrib[0])
      print((psd[8]+psd[9]+psd[10]+psd[11]/4))
      #print(rf.predict(np.reshape(test_attrib,(1,-1))))
      #if(psd[10]>200 and counter==0):
            print("alpha response ")
            ser.write(b'C')
            counter=2
      if(test_attrib[0]>0.2 and counter==0):
            print("eye activated")
            #Serial Communication with Arduino
            time.sleep(2)
            ser.write(b'B')
            counter=1
      if(test_attrib[0]>=9 and counter==0):
            print("eye activated")
            time.sleep(2)
            ser.write(b'E')
            counter=3
      elif(counter==1):
            ser.write(b'b')
            counter=0
      elif(counter==2):
            ser.write(b'c')
            counter=0
      elif(counter==3):
            ser.write(b'e')
            counter=0
      print(timestamp_list, y_fin)
      timestamp_list=[]
      sample_list=[]	

