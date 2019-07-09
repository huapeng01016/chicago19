*! version 1.0.0  01jul2019
program pysvm2
	version 16
	syntax varlist(min=3) [if] [in] 

	gettoken label features : varlist

	marksample touse
	qui count if `touse'
	if r(N) == 0 {
		di as error "no observations"
		exit 2000
	}
	
	qui summarize `label' if `touse' 
	if r(min) >= r(max) {
		di as error "outcome does not vary"
		exit 2000
	}
	
	quietly python: dosvm2("`label'", "`features'", "`touse'")	
	di as text "note: training finished successfully"
end

version 16
python:
import sys
from sfi import Data, Macro
import numpy as np
from sklearn.svm import SVC
import __main__

def dosvm2(label, features, select):
	X = np.array(Data.get(features, selectvar=select))
	y = np.array(Data.get(label, selectvar=select))

	svc_clf = SVC(gamma='auto')
	svc_clf.fit(X, y)
	
	__main__.svc_clf = svc_clf 
	Macro.setGlobal('e(svm_features)', features)
	return svc_clf
end
