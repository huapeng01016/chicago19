*! version 1.0.0  01jul2019
program pysvm
    version 16
    syntax varlist, predict(name)

    gettoken label feature : varlist

    //call the Python function
    python: dosvm("`label'", "`feature'", "`predict'")
end

version 16
python:
from sfi import Data
import numpy as np
from sklearn.svm import SVC

def dosvm(label, features, predict):
    X = np.array(Data.get(features))
    y = np.array(Data.get(label))

    svc_clf = SVC(gamma='auto')
    svc_clf.fit(X, y)

    y_pred = svc_clf.predict(X)

    Data.addVarByte(predict)
    Data.store(predict, None, y_pred)

end
