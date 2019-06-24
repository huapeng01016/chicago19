use auto_78_img.dta, clear
desc
cap mkdir cars
cap mkdir images

forvalue i = 1/`r(N)' {
	  dyndoc auto_78.txt `=make[`i']', saving("cars/`=make[`i']'.html") replace
}
