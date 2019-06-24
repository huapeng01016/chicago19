putdocx clear 
putdocx begin

sysuse auto, clear

generate fuel = 100/mpg
label variable fuel "Fuel consumption (Gallons per 100 Miles)"

putdocx paragraph, style("Heading1")
putdocx text ("Produce a table from saved dataset from -esttab-")

preserve
cap erase esttab_ex.csv
eststo clear
eststo, title("Model 1"): quietly regress fuel weight gear turn
eststo, title("Model 2"): quietly regress fuel weight gear turn foreign
esttab using esttab_ex.csv, 		///
	b(3) t(2) r2(2) ar2(2)			/// 
	plain star notes par label 		/// 
	title(Regression table using -esttab-) width(80%)

import delimited using esttab_ex.csv, clear
putdocx table tbl_data = data(_all)
restore

putdocx save data_table.docx, replace

exit
