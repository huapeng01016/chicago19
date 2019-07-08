*! version 1.0.0  01jul2019
program pysvm2
    version 16
    syntax varlist(min=3) [if] [in] [, verbose]

    gettoken label features : varlist

	marksample touse
	qui count if `touse'
	if r(N) == 0 {
		di as text "no observations"
		exit 2000   
	}
	
	local veb = "no"
	if "`verbose'" != "" {
		local veb = "yes"	
	}

	python: svc_clf=dosvm2("`label'", "`features'", "`touse'", "yes")
end

version 16
python:
from sfi import Data, Macro
import numpy as np
from sklearn.svm import SVC

def dosvm2(label, features, select, verbose="no"):
	X = np.array(Data.get(features, selectvar=select))
	y = np.array(Data.get(label, selectvar=select))

	v = False
	if verbose != "no":
		v = True
	svc_clf = SVC(gamma='auto', verbose=v)
	svc_clf.fit(X, y)

	Macro.setGlobal('e(svm_features)', features)
	return svc_clf
end
