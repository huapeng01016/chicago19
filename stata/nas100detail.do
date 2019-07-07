cscript

use nas100ticker, clear
quietly describe

frame create detail

forvalues i = 1/`r(N)' {
	local a = ticker[`i']
	local b detail
	python script nas1detail.py, args(`a' `b')
	sleep 100
}

frame detail : save nasd100detail.dta, replace
frame detail : list in 1/5, clean
