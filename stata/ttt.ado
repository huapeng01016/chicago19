*! version 1.0.0  01jul2019
program ttt
    version 16

	python script maindata.py, global
	python script side.py, global
	python: a = ttt_p2()
	python: len(a)	
end

version 16
python:

import sys
def ttt_p2():
	svm_dic = getattr(sys.modules['__main__'], "svm_dic", None)
	if svm_dic == None:
		return range(100)
	svm_dic['three'] = 30
	return range(10)

end