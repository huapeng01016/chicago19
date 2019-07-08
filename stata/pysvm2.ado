*! version 1.0.0  01jul2019
program pysvm2
    version 16
    syntax varlist(min=3) [if] [in] 

    gettoken label features : varlist

	marksample touse
	qui count if `touse'
	if r(N) == 0 {
		di as text "no observations"
		exit 2000   
	}
	
	python script maindata.py, global
	python: svm_dic=dosvm2("`label'", "`features'", "`touse'")
end

version 16
python:
import sys
from sfi import Data, Macro
import numpy as np
from sklearn.svm import SVC

def dosvm2(label, features, select):
	X = np.array(Data.get(features, selectvar=select))
	y = np.array(Data.get(label, selectvar=select))

	svc_clf = SVC(gamma='auto')
	svc_clf.fit(X, y)

	svm_dic = getattr(sys.modules['__main__'], "svm_dic", None)
	if svm_dic == None:
		return None
	
	svm_dic['svm_svc_clf'] = svc_clf 
	Macro.setGlobal('e(svm_features)', features)
	return svm_dic
end
