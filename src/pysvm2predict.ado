*! version 1.0.0  01jul2019
program pysvm2predict
	version 16
	syntax anything [if] [in]

	gettoken newvar rest : anything
	
	if "`rest'" != "" {
		exit 198
	}
	
	confirm new variable `newvar'	
	marksample touse
	qui count if `touse'
	if r(N) == 0 {
		di as text "zero observations"
		exit 2000
	}

	qui replace `touse' = _n if `touse' != 0 	
	python: dosvmpredict2("`newvar'", "`touse'")
end

version 16
python:
import sys
from sfi import Data, Macro
import numpy as np
from sklearn.svm import SVC
from __main__ import svc_clf

def dosvmpredict2(predict, select):
	features = select + " "+ Macro.getGlobal('e(svm_features)')
	X = np.array(Data.get(features, selectvar=select))

	y_pred = svc_clf.predict(X[:,1:])
	y1 = np.reshape(y_pred, (-1,1))

	y = np.concatenate((X, y1), axis=1)
	Data.addVarDouble(predict)
	dim = y.shape[0]
	j = y.shape[1]-1
	for i in range(dim):
		Data.storeAt(predict, y[i,0]-1, y[i,j])
	
end
