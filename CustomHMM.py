from hmmlearn import hmm
import numpy as np

class CustomHMM:
  def __init__(self):
      def build_hmm():
          model = hmm.GMMHMM(n_components=3, n_mix=3, covariance_type="diag", init_params="t")
          model.transmat_ = np.array([[0.5, 0.5, 0.0],
                                      [0.0, 0.5, 0.5],
                                      [0.0, 0.0, 1.0]])
          return model

      self.hmm_0 = build_hmm()
      self.hmm_1 = build_hmm()

  def fit(self, X_train, y_train):
      # X_train shape(n_instances, n_samples)
      labels = set(y_train)
      if len(labels) != 2:
          raise Exception("y_train doesn't contain 2 classes")
      X_0 = X_train[y_train == 0, :]
      X_1 = X_train[y_train == 1, :]

      self.hmm_0.fit(X_0)
      self.hmm_1.fit(X_1)

  def predict(self, X_test):
      res = []
      for x in X_test:
        x = np.reshape(x,[1,len(x)])
        res.append(0 if self.hmm_0.score(x) > self.hmm_1.score(x) else 1)
      return np.array(res)