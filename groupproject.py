import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from hmmlearn import hmm
from sklearn.svm import SVC
from CustomHMM import *
pitchTuning = np.load('standardfeatures/DotaMediaFilesPitchTuning.npy')
TempoGram = np.load('standardfeatures/DotaMediaFilesTempoGram.npy')
Labels = np.load('standardfeatures/DotaMediaLabels.npy')
MFCCS = np.load('standardfeatures/DotaMediaMFCCS.npy')
ZEROS = np.load('standardfeatures/DotaMediaZEROS.npy')

# download & install HMM stuff
# practice HMM w/ example
# construct varying n-d matrices
  # average values
  # subsample values


# collapsed MFCC feature matrix based on averaging those 13 MFCC rows
MFCCS_collapsed = np.zeros([MFCCS.shape[0], MFCCS.shape[1]])
for i in np.arange(MFCCS.shape[0]):
  for j in np.arange(MFCCS.shape[1]):
    MFCCS_collapsed[i,j] = np.mean(MFCCS[i,j,:])

# construct ZEROS matrix based on averaging 130 rows
ZEROS_collapsed = np.zeros([ZEROS.shape[0], 1])
for i in np.arange(ZEROS.shape[0]):
    ZEROS_collapsed[i] = np.mean(ZEROS[i,0,:])


#normalize features
MFCCS_collapsed_normalized = (MFCCS_collapsed - np.min(MFCCS_collapsed)) / (np.max(MFCCS_collapsed) - np.min(MFCCS_collapsed))
ZEROS_collapsed_normalized = (ZEROS_collapsed - np.min(ZEROS_collapsed)) / (np.max(ZEROS_collapsed) - np.min(ZEROS_collapsed))
pitchTuning_normalized = (pitchTuning - np.min(pitchTuning)) / (np.max(pitchTuning) - np.min(pitchTuning))
TempoGram_normalized = (TempoGram - np.min(TempoGram)) / (np.max(TempoGram) - np.min(TempoGram))

#reshape matrices
pitchTuning_normalized = np.reshape(pitchTuning_normalized,[len(pitchTuning_normalized),1])
TempoGram_normalized = np.reshape(TempoGram_normalized,[len(TempoGram_normalized),1])

#wrap features in 1 matrix
X = np.concatenate((TempoGram_normalized, pitchTuning_normalized, ZEROS_collapsed_normalized, MFCCS_collapsed_normalized),axis=1)


# extract 2 classes & shuffle data
le = preprocessing.LabelEncoder()
le.fit(Labels)
Labels_transformed = le.transform(Labels)
twoClassIndexes = np.where((Labels_transformed == 1) | (Labels_transformed == 0))[0]
twoX = X[twoClassIndexes,:]
twoY = Labels_transformed[twoClassIndexes]
twoX, twoY = shuffle(twoX, twoY)

# GMM HMM https://github.com/hmmlearn/hmmlearn/issues/123
customHMM = CustomHMM()
customHMM.fit(twoX, twoY)
customHMMPredictions = customHMM.predict(twoX)
res = []
for x in twoX:
  print(np.reshape(x,[1,len(x)]).shape)
print("Custom HMM accuracy: " + str(accuracy_score(twoY, customHMMPredictions)))

# GMM HMM
remodel = hmm.GMMHMM(n_components=2, covariance_type="diag", n_iter=100)
remodel.fit(twoX)
hmmPredictions = remodel.predict(twoX)
print("GMM HMM accuracy: " + str(accuracy_score(twoY, hmmPredictions)))
# compare to SVM
svm = SVC()
svm.fit(twoX, twoY)
predictions = svm.predict(twoX)
print("SVM accuracy: " + str(accuracy_score(twoY, predictions)))