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


#Let the filtering begin:
#Implementing a band pass butterworth filter from 0.5 - 45 Hz to account for DC offset and mains noise(50Hz)
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
fs = 200.0
lowcut = 0.5
highcut = 60
plt.figure(1)
plt.clf()
for order in [3, 6, 9]:
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    w, h = freqz(b, a, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],'--', label='sqrt(0.5)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.legend(loc='best')


#filtering data 
y = lfilter(b, a, eeg_data)


#notch filter(50HZ)
f0 = 50.0 #freq to be removed
Q = 30

w0 = f0/(fs/2)

b, a =iirnotch(w0, Q)
# Frequency response
w, h = freqz(b, a)
# Generate frequency axis
freq = w*fs/(2*np.pi)
 # Plot
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
ax[0].set_title("Frequency Response")
ax[0].set_ylabel("Amplitude (dB)", color='blue')
ax[0].set_xlim([0, 100])
ax[0].set_ylim([-25, 10])
ax[0].grid()
ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
ax[1].set_ylabel("Angle (degrees)", color='green')
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_xlim([0, 100])
ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax[1].set_ylim([-90, 90])
ax[1].grid()
plt.show()

y_eeg = lfilter(b, a, y)

#y_blink is basically y_jaw
y_blink=y_eeg[:,1]


#Signal Proccessing begins:
attributes = []
i=0
while i < len(y_blink):
    print(i)
    if i+199>len(y_blink):
        break
    else:
        attr_mean = np.mean(y_blink[i:i+200],axis = 0)
        attr_var = np.var(y_blink[i:i+200],axis=0)
        attr_min = np.amin(y_blink[i:i+200],axis = 0)
        attr_max = np.amax(y_blink[i:i+200],axis = 0)  
        attr_skew = skew(y_blink[i:i+200],axis=0)
        attr_kurtosis = kurtosis(y_blink[i:i+200],axis=0)
        attr = np.zeros(6)
        attr[0] = attr_mean
        attr[1] = attr_var      
        attr[2] = attr_min
        attr[3] = attr_max
        attr[4] = attr_skew
        attr[5] = attr_kurtosis
        attributes.append(attr)
        del attr
        i+=200
attributes = np.array(attributes)


#Machine Learning starts now:
X_train, X_test, y_t, y_test = model_selection.train_test_split(attributes, labels, test_size=0, random_state=0, stratify = labels)
print ("For SVM:")
clf=svm.SVC()
clf.fit(X_train,y_t)
t1=time()
print (clf.score(X_test,y_test))
print ("Time Taken : " + str(time()-t1))
print ()

print ("For Decision Trees:")
d_tree=DecisionTreeClassifier()
d_tree.fit(X_train,y_t)
t1=time()
print (d_tree.score(X_test,y_test))
print ("Time Taken : " + str(time()-t1))
print()

for cri in ["entropy","gini"]:
    print ("Criterion : ",cri)
    rf=RandomForestClassifier(n_estimators=100,max_features=6,criterion=cri)
    rf.fit(X_train,y_t)
    t1=time()
    print (rf.score(X_test,y_test))
    print ("Time Taken : " + str(time()-t1))
    #print (classification_report(y_test,rf.predict(X_test),digits=4))

print ("For Extra Trees:")


for cri in ["entropy","gini"]:
    print ("Criterion : ",cri)	
    rf=ExtraTreesClassifier(n_estimators=100,max_features=6,criterion=cri)
    rf.fit(X_train,y_t)
    t1=time()
    print (rf.score(X_test,y_test))
    print ("Time Taken : " + str(time()-t1))
    #print (classification_report(y_test,rf.predict(X_test),digits=4))

    







