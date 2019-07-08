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
from sfi import Data, Macro
import numpy as np
from sklearn.svm import SVC
import __main__

def dosvmpredict2(predict, select):
	features = 	select + " "+ Macro.getGlobal('e(svm_features)')
	X = np.array(Data.get(features, selectvar=select))

	y_pred = svc_clf.predict(X[:,1:])

	Data.addVarByte(predict)
	Data.store(predict, None, y_pred)
	
end
